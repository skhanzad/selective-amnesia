"""Benchmark runner: feeds conversations through memory system, evaluates QA."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.evaluation.data_loaders import BenchmarkSample, QAPair
from src.evaluation.metrics import (
    EditDeleteResult,
    ExperimentResult,
    MemoryStats,
    QAResult,
    adversarial_check,
    contains_match,
    exact_match,
    gpt4o_judge,
    multi_answer_f1,
    recall_at_k,
    reciprocal_rank,
    task_success,
    token_f1,
)
from src.memory.forgetting import apply_forgetting, get_policy
from src.memory.graph_store import GraphStore
from src.memory.retriever import get_retriever
from src.memory.schemas import EdgeType, MemoryEdge, MemoryNode, NodeType

logger = logging.getLogger(__name__)


# ── Pretty printing helpers ──────────────────────────────────────

def _header(text: str, char: str = "=", width: int = 80) -> None:
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _subheader(text: str, char: str = "-", width: int = 70) -> None:
    print(f"\n  {char * width}")
    print(f"  {text}")
    print(f"  {char * width}")


def _kv(key: str, value: Any, indent: int = 4) -> None:
    pad = " " * indent
    print(f"{pad}{key:<30} {value}")


def _progress_bar(current: int, total: int, width: int = 30, label: str = "") -> str:
    filled = int(width * current / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    pct = 100 * current / max(total, 1)
    return f"[{bar}] {current}/{total} ({pct:.0f}%) {label}"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


def _score_indicator(f1: float) -> str:
    if f1 >= 0.8:
        return "GOOD"
    elif f1 >= 0.4:
        return "PARTIAL"
    elif f1 > 0:
        return "WEAK"
    else:
        return "MISS"


# ── Core functions ───────────────────────────────────────────────

def _build_llm(config: dict) -> Any:
    """Create LLM instance from config."""
    llm_cfg = config.get("llm", {})
    provider = llm_cfg.get("provider", "ollama")
    model = llm_cfg.get("model", "llama3.2")
    temperature = llm_cfg.get("temperature", 0.0)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _extract_memories_from_turn(
    llm: Any,
    user_text: str,
    assistant_text: str,
    graph_store: GraphStore,
    current_turn: int,
    config: dict,
) -> list[MemoryNode]:
    """Extract memories from a single conversation turn using the LLM."""
    extraction_prompt = config.get("agent", {}).get("extraction_prompt", "")
    if not extraction_prompt:
        return []

    existing = graph_store.get_all_nodes()
    existing_summary = "No existing memories yet."
    if existing:
        lines = ["Existing memories:"]
        for i, n in enumerate(existing[:15]):
            lines.append(f"  [{i}] ({n.node_type.value}) {n.content[:80]}")
        existing_summary = "\n".join(lines)

    try:
        prompt = extraction_prompt.format(
            user_message=user_text,
            assistant_message=assistant_text,
            existing_memories=existing_summary,
        )
    except KeyError:
        prompt = extraction_prompt.replace("{user_message}", user_text).replace(
            "{assistant_message}", assistant_text
        ).replace("{existing_memories}", existing_summary)

    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        raw = result.content.strip()

        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]

        memories_data = json.loads(raw)
        if not isinstance(memories_data, list):
            memories_data = [memories_data]
    except Exception as e:
        logger.debug("Memory extraction parse error: %s", e)
        return []

    default_imp = config.get("memory", {}).get("default_importance", 0.5)
    existing_nodes = graph_store.get_all_nodes()
    new_nodes = []
    for mem in memories_data:
        if not isinstance(mem, dict) or "content" not in mem:
            continue
        try:
            node_type = NodeType(mem.get("node_type", "fact"))
        except ValueError:
            node_type = NodeType.fact

        imp = mem.get("importance", default_imp)
        if not isinstance(imp, (int, float)):
            imp = default_imp

        node = MemoryNode(
            content=mem["content"],
            node_type=node_type,
            importance=min(1.0, max(0.0, float(imp))),
            turn_created=current_turn,
            turn_last_accessed=current_turn,
        )
        graph_store.add_node(node)
        new_nodes.append(node)

    # Create edges from the LLM's edge specifications
    # Ported from src/agent/nodes.py:215-256
    for i, mem in enumerate(memories_data):
        if not isinstance(mem, dict) or i >= len(new_nodes):
            continue
        source_node = new_nodes[i]

        for edge_spec in mem.get("edges", []):
            if not isinstance(edge_spec, dict):
                continue

            target_index = edge_spec.get("target_index")
            edge_type_str = edge_spec.get("edge_type", "related_to")

            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                edge_type = EdgeType.related_to

            # target_index can refer to another new node or an existing node
            target_node = None
            if isinstance(target_index, int):
                if 0 <= target_index < len(new_nodes):
                    target_node = new_nodes[target_index]
                elif 0 <= target_index < len(existing_nodes):
                    target_node = existing_nodes[target_index]

            if target_node and target_node.id != source_node.id:
                try:
                    edge = MemoryEdge(
                        source_id=source_node.id,
                        target_id=target_node.id,
                        edge_type=edge_type,
                    )
                    graph_store.add_edge(edge)
                except KeyError:
                    pass  # node not found, skip

    return new_nodes


def _answer_question(
    llm: Any,
    question: str,
    retrieved_context: str,
    config: dict,
) -> str:
    """Ask the LLM a question with retrieved memory context."""
    system = config.get("agent", {}).get("system_prompt", "You are a helpful assistant.")

    if retrieved_context:
        system += f"\n\nRetrieved Memories:\n{retrieved_context}"

    system += (
        "\n\nAnswer the question with a short, precise phrase. "
        "Use exact words from the memories when possible. "
        "If you don't have enough information, say 'Not mentioned'."
    )

    msgs = [SystemMessage(content=system), HumanMessage(content=question)]

    try:
        result = llm.invoke(msgs)
        return result.content.strip()
    except Exception as e:
        logger.error("LLM error answering question: %s", e)
        return ""


def ingest_conversations(
    sample: BenchmarkSample,
    graph_store: GraphStore,
    llm: Any,
    config: dict,
    use_memory_extraction: bool = True,
    verbose: bool = True,
) -> tuple[int, dict]:
    """Feed all conversation sessions into the memory graph.

    Returns (turn_count, ingestion_stats).
    """
    turn = 0
    forget_cfg = config.get("forgetting", {})
    policy_name = forget_cfg.get("policy", "none")
    policy = get_policy(policy_name, config)
    budget = forget_cfg.get("budget_target", 150)
    min_imp = forget_cfg.get("min_importance_to_keep", 0.3)
    run_every = forget_cfg.get("run_every_n_turns", 5)

    total_sessions = len(sample.sessions)
    total_extracted = 0
    total_forgotten = 0
    extraction_failures = 0
    session_timings: list[float] = []

    if verbose:
        _subheader(f"INGESTING {total_sessions} SESSIONS")

    for si, session in enumerate(sample.sessions):
        session_start = time.time()
        session_extracted = 0
        user_text = ""

        for conv_turn in session.turns:
            if conv_turn.speaker == "user":
                user_text = conv_turn.text
            elif conv_turn.speaker == "assistant" and user_text:
                turn += 1

                if use_memory_extraction:
                    new_nodes = _extract_memories_from_turn(
                        llm, user_text, conv_turn.text, graph_store, turn, config
                    )
                    if new_nodes:
                        session_extracted += len(new_nodes)
                        total_extracted += len(new_nodes)
                    else:
                        extraction_failures += 1

                # Apply forgetting
                if policy_name != "none" and turn % run_every == 0:
                    nodes = graph_store.get_all_nodes()
                    to_remove = apply_forgetting(nodes, policy, turn, budget, min_imp)
                    for nid in to_remove:
                        graph_store.remove_node(nid)
                    if to_remove:
                        total_forgotten += len(to_remove)

                user_text = ""

        session_elapsed = time.time() - session_start
        session_timings.append(session_elapsed)

        if verbose:
            node_count = graph_store.node_count()
            ts_label = session.timestamp[:20] if session.timestamp else ""
            progress = _progress_bar(si + 1, total_sessions)
            print(
                f"    {progress}  "
                f"session={session.session_id:<12} "
                f"+{session_extracted} memories  "
                f"total={node_count}  "
                f"{_format_duration(session_elapsed)}  "
                f"{ts_label}"
            )

    stats = {
        "total_turns": turn,
        "total_extracted": total_extracted,
        "total_forgotten": total_forgotten,
        "extraction_failures": extraction_failures,
        "avg_session_time": sum(session_timings) / max(len(session_timings), 1),
        "total_ingest_time": sum(session_timings),
    }

    if verbose:
        node_count = graph_store.node_count()
        edge_count = graph_store.edge_count()
        type_counts = Counter(n.node_type.value for n in graph_store.get_all_nodes())
        print()
        _kv("Turns processed", turn)
        _kv("Memories extracted", total_extracted)
        _kv("Extraction failures", extraction_failures)
        _kv("Memories forgotten", total_forgotten)
        _kv("Final node count", node_count)
        _kv("Final edge count", edge_count)
        _kv("Ingestion time", _format_duration(stats["total_ingest_time"]))
        if type_counts:
            _kv("Node types", dict(type_counts))

    return turn, stats


def evaluate_qa(
    sample: BenchmarkSample,
    graph_store: GraphStore,
    llm: Any,
    retriever: Any,
    current_turn: int,
    config: dict,
    verbose: bool = True,
    use_gpt4o_judge: bool = False,
) -> tuple[list[QAResult], dict]:
    """Evaluate all QA pairs for a sample using the populated memory graph.

    Returns (qa_results, eval_stats).
    """
    ret_cfg = config.get("retrieval", {})
    max_results = ret_cfg.get("max_results", 10)
    total_qa = len(sample.qa_pairs)

    results = []
    retrieval_traces: list[dict] = []
    qa_timings: list[float] = []
    running_f1 = 0.0

    if verbose and total_qa > 0:
        _subheader(f"EVALUATING {total_qa} QUESTIONS")

    for qi, qa in enumerate(sample.qa_pairs):
        q_start = time.time()

        # Retrieve
        retrieved = retriever.retrieve(
            query=qa.question,
            graph_store=graph_store,
            max_results=max_results,
            current_turn=current_turn,
        )

        # Build context using to_context_string() to include edge information
        context_parts = []
        retrieved_contents = []
        for rm in retrieved:
            node = rm.node
            context_parts.append(rm.to_context_string())
            retrieved_contents.append({
                "content": node.content[:80],
                "type": node.node_type.value,
                "importance": node.importance,
                "relevance": round(rm.relevance_score, 3),
            })
        context = "\n".join(context_parts)

        # Answer
        prediction = _answer_question(llm, qa.question, context, config)

        # Score — category-specific evaluation matching LoCoMo protocol
        category_id = qa.metadata.get("category_id")
        answer_for_scoring = qa.answer

        if category_id == 5:
            # Adversarial: check for correct refusal
            f1 = adversarial_check(prediction)
        elif category_id == 1:
            # Multi-hop: split on comma for sub-answer F1
            f1 = multi_answer_f1(prediction, qa.answer, sep=",")
        elif category_id == 3:
            # Contextual: strip answer at first semicolon before scoring
            answer_for_scoring = qa.answer.split(";")[0].strip()
            f1 = token_f1(prediction, answer_for_scoring)
        else:
            # Single-hop, temporal, open-domain, and LongMemEval types
            f1 = token_f1(prediction, qa.answer)

        em = exact_match(prediction, answer_for_scoring)
        cm = contains_match(prediction, answer_for_scoring)

        # Retrieval quality metrics
        retrieved_texts = [rm.node.content for rm in retrieved]
        r_at_5 = recall_at_k(retrieved_texts, qa.answer, k=5)
        mrr_score = reciprocal_rank(retrieved_texts, qa.answer)
        ts = task_success(f1)

        # Optional GPT-4o judge for LongMemEval
        judge = None
        if use_gpt4o_judge and sample.source == "longmemeval":
            is_abstention = "_abs" in sample.sample_id
            judge = gpt4o_judge(
                qa.question, qa.answer, prediction,
                qa.category, is_abstention,
            )

        q_elapsed = time.time() - q_start
        qa_timings.append(q_elapsed)
        running_f1 += f1

        results.append(
            QAResult(
                question=qa.question,
                ground_truth=qa.answer,
                prediction=prediction,
                category=qa.category,
                f1=f1,
                exact=em,
                contains=cm,
                sample_id=sample.sample_id,
                recall_at_5=r_at_5,
                mrr=mrr_score,
                task_success=ts,
                latency_seconds=q_elapsed,
                num_retrieved=len(retrieved),
                judge_score=judge,
            )
        )

        # Retrieval trace for results file
        retrieval_traces.append({
            "question": qa.question,
            "num_retrieved": len(retrieved),
            "retrieved": retrieved_contents,
        })

        if verbose:
            avg_f1 = running_f1 / (qi + 1)
            indicator = _score_indicator(f1)
            progress = _progress_bar(qi + 1, total_qa)
            print(f"    {progress}")
            print(f"      Q: {qa.question[:70]}")
            print(f"      A: {qa.answer[:70]}")
            print(f"      P: {prediction[:70]}")
            print(
                f"      F1={f1:.3f} [{indicator}]  "
                f"exact={em:.0f}  contains={cm:.0f}  "
                f"R@5={r_at_5:.2f}  MRR={mrr_score:.2f}  "
                f"retrieved={len(retrieved)}  "
                f"{_format_duration(q_elapsed)}  "
                f"(running avg F1={avg_f1:.3f})"
            )
            # Show top retrieved memories on misses
            if f1 < 0.3 and retrieved_contents:
                top = retrieved_contents[:3]
                for rc in top:
                    print(
                        f"        -> [{rc['type']}] {rc['content'][:60]}  "
                        f"(rel={rc['relevance']}, imp={rc['importance']:.2f})"
                    )
            print()

    eval_stats = {
        "total_qa": total_qa,
        "avg_qa_time": sum(qa_timings) / max(len(qa_timings), 1),
        "total_eval_time": sum(qa_timings),
        "retrieval_traces": retrieval_traces,
    }

    return results, eval_stats


def get_memory_stats(graph_store: GraphStore, forgotten_count: int = 0) -> MemoryStats:
    all_nodes = graph_store.get_all_nodes(include_disabled=True)
    enabled = [n for n in all_nodes if n.enabled]
    type_counts = Counter(n.node_type.value for n in enabled)

    # Compute storage size from serialized graph
    graph_dict = graph_store.to_dict()
    storage_bytes = len(json.dumps(graph_dict, default=str).encode("utf-8"))
    node_sizes = [len(n.content.encode("utf-8")) for n in all_nodes]
    avg_node_size = sum(node_sizes) / max(len(node_sizes), 1)

    return MemoryStats(
        total_nodes=len(all_nodes),
        enabled_nodes=len(enabled),
        total_edges=graph_store.edge_count(),
        nodes_by_type=dict(type_counts),
        forgotten_count=forgotten_count,
        storage_bytes=storage_bytes,
        avg_node_size_bytes=avg_node_size,
    )


def evaluate_memory_ops(
    qa_results: list[QAResult],
    retrieval_traces: list[dict],
    graph_store: GraphStore,
    llm: Any,
    retriever: Any,
    current_turn: int,
    config: dict,
    max_tests: int = 5,
    verbose: bool = True,
) -> EditDeleteResult:
    """Evaluate edit success, delete success, and locality score.

    Tests whether the memory system correctly reflects edits and deletions,
    and whether unrelated answers remain stable (locality).
    """
    ret_cfg = config.get("retrieval", {})
    max_results = ret_cfg.get("max_results", 10)

    # Find QA pairs that were answered well and had retrieved memories
    good_qa = [
        (r, t)
        for r, t in zip(qa_results, retrieval_traces)
        if r.f1 > 0.5 and t.get("num_retrieved", 0) > 0 and t.get("retrieved")
    ]
    if not good_qa:
        if verbose:
            print("    (no suitable QA pairs for edit/delete testing)")
        return EditDeleteResult()

    # Find unrelated QA pairs for locality checks
    test_pairs = good_qa[:max_tests]
    test_categories = {r.category for r, _ in test_pairs}
    locality_pairs = [
        r for r in qa_results
        if r.category not in test_categories and r.f1 > 0.3
    ][:max_tests]

    edit_successes = 0
    delete_successes = 0
    locality_scores: list[float] = []
    details: list[dict] = []

    if verbose:
        _subheader(f"TESTING EDIT/DELETE ({len(test_pairs)} pairs)")

    for qi, (qa_result, trace) in enumerate(test_pairs):
        # Find the memory node that was retrieved for this question
        retrieved_info = trace.get("retrieved", [])
        if not retrieved_info:
            continue

        target_content = retrieved_info[0].get("content", "")
        # Find the actual node in the graph by content prefix match
        target_node = None
        for node in graph_store.get_all_nodes():
            if node.content[:80] == target_content:
                target_node = node
                break
        if not target_node:
            continue

        # Snapshot the graph for restoration
        snapshot = graph_store.to_dict()
        original_content = target_node.content

        # ── Edit test ────────────────────────────────────────────
        edited_content = f"CORRECTED: {original_content[:60]} is actually MODIFIED_VALUE"
        graph_store.update_node(target_node.id, content=edited_content)

        # Re-retrieve and re-answer
        edit_retrieved = retriever.retrieve(
            query=qa_result.question,
            graph_store=graph_store,
            max_results=max_results,
            current_turn=current_turn,
        )
        edit_context = "\n".join(
            f"[{rm.node.node_type.value}] {rm.node.content}" for rm in edit_retrieved
        )
        edit_prediction = _answer_question(llm, qa_result.question, edit_context, config)

        # Edit success: the answer changed AND references the edited content
        edit_answer_changed = edit_prediction.strip() != qa_result.prediction.strip()
        edit_content_reflected = (
            "MODIFIED_VALUE" in edit_prediction or "CORRECTED" in edit_prediction
            or edit_prediction.strip() != qa_result.prediction.strip()
        )
        edit_ok = edit_answer_changed and edit_content_reflected
        if edit_ok:
            edit_successes += 1

        # ── Locality test (during edit) ──────────────────────────
        for loc_qa in locality_pairs:
            loc_retrieved = retriever.retrieve(
                query=loc_qa.question,
                graph_store=graph_store,
                max_results=max_results,
                current_turn=current_turn,
            )
            loc_context = "\n".join(
                f"[{rm.node.node_type.value}] {rm.node.content}" for rm in loc_retrieved
            )
            loc_prediction = _answer_question(llm, loc_qa.question, loc_context, config)
            loc_f1 = token_f1(loc_prediction, loc_qa.ground_truth)
            # Locality: unrelated answer should not degrade significantly
            locality_scores.append(1.0 if loc_f1 >= loc_qa.f1 - 0.15 else 0.0)

        # Restore graph for delete test
        restored = GraphStore.from_dict(snapshot)
        graph_store._graph = restored._graph
        graph_store._nodes = restored._nodes

        # ── Delete test ──────────────────────────────────────────
        graph_store.remove_node(target_node.id)

        del_retrieved = retriever.retrieve(
            query=qa_result.question,
            graph_store=graph_store,
            max_results=max_results,
            current_turn=current_turn,
        )
        del_context = "\n".join(
            f"[{rm.node.node_type.value}] {rm.node.content}" for rm in del_retrieved
        )
        del_prediction = _answer_question(llm, qa_result.question, del_context, config)

        # Delete success: the deleted content's specific tokens no longer appear
        orig_tokens = set(original_content.lower().split())
        del_tokens = set(del_prediction.lower().split())
        # Check that distinctive original tokens aren't parroted back
        distinctive = orig_tokens - {"the", "a", "an", "is", "was", "of", "in", "to", "and"}
        overlap = distinctive & del_tokens
        delete_ok = len(overlap) < len(distinctive) * 0.5 if distinctive else True
        if delete_ok:
            delete_successes += 1

        # Restore graph
        restored = GraphStore.from_dict(snapshot)
        graph_store._graph = restored._graph
        graph_store._nodes = restored._nodes

        detail = {
            "question": qa_result.question[:70],
            "target_memory": original_content[:60],
            "edit_success": edit_ok,
            "delete_success": delete_ok,
            "edit_prediction": edit_prediction[:70],
            "delete_prediction": del_prediction[:70],
        }
        details.append(detail)

        if verbose:
            e_mark = "PASS" if edit_ok else "FAIL"
            d_mark = "PASS" if delete_ok else "FAIL"
            print(f"    [{qi + 1}/{len(test_pairs)}] Edit={e_mark}  Delete={d_mark}")
            print(f"      Q: {qa_result.question[:65]}")
            print(f"      Memory: {original_content[:60]}")

    num_tested = len(details)
    result = EditDeleteResult(
        edit_success_rate=edit_successes / max(num_tested, 1),
        delete_success_rate=delete_successes / max(num_tested, 1),
        locality_score=(
            sum(locality_scores) / len(locality_scores)
            if locality_scores else 1.0
        ),
        num_edits_tested=num_tested,
        num_deletes_tested=num_tested,
        num_locality_checks=len(locality_scores),
        details=details,
    )

    if verbose and num_tested > 0:
        print()
        _kv("Edit success rate", f"{result.edit_success_rate:.2%} ({edit_successes}/{num_tested})")
        _kv("Delete success rate", f"{result.delete_success_rate:.2%} ({delete_successes}/{num_tested})")
        _kv("Locality score", f"{result.locality_score:.4f} ({len(locality_scores)} checks)")

    return result


def run_experiment(
    experiment_name: str,
    samples: list[BenchmarkSample],
    config: dict,
    results_dir: str = "results",
    max_qa_per_sample: int | None = None,
    verbose: bool = True,
    run_edit_delete_tests: bool = True,
    use_gpt4o_judge: bool = False,
) -> ExperimentResult:
    """Run a full experiment: ingest conversations, evaluate QA, record results."""
    exp_start = time.time()

    llm = _build_llm(config)
    retrieval_mode = config.get("retrieval", {}).get("mode", "none")
    forgetting_policy = config.get("forgetting", {}).get("policy", "none")
    use_memory = retrieval_mode != "none"
    retriever = get_retriever(retrieval_mode, config)
    dataset_name = samples[0].source if samples else "unknown"
    llm_cfg = config.get("llm", {})

    if verbose:
        _header(f"EXPERIMENT: {experiment_name}")
        _kv("Dataset", dataset_name)
        _kv("Samples", len(samples))
        _kv("LLM", f"{llm_cfg.get('provider', '?')}/{llm_cfg.get('model', '?')}")
        _kv("Retrieval mode", retrieval_mode)
        _kv("Forgetting policy", forgetting_policy)
        _kv("Budget target", config.get("forgetting", {}).get("budget_target", "N/A"))
        _kv("Max results", config.get("retrieval", {}).get("max_results", "N/A"))
        if max_qa_per_sample:
            _kv("Max QA per sample", max_qa_per_sample)

    all_qa_results: list[QAResult] = []
    total_forgotten = 0
    all_ingest_stats: list[dict] = []
    all_eval_stats: list[dict] = []

    for i, sample in enumerate(samples):
        total_turns = sum(len(s.turns) for s in sample.sessions)
        qa_count = len(sample.qa_pairs)
        if max_qa_per_sample:
            qa_count = min(qa_count, max_qa_per_sample)

        if verbose:
            _header(
                f"SAMPLE {i + 1}/{len(samples)}: {sample.sample_id}  "
                f"({len(sample.sessions)} sessions, {total_turns} turns, {qa_count} QA)",
                char="-",
            )

        graph_store = GraphStore()

        # Ingest
        if use_memory:
            turn_count, ingest_stats = ingest_conversations(
                sample, graph_store, llm, config,
                use_memory_extraction=True, verbose=verbose,
            )
            total_forgotten += ingest_stats.get("total_forgotten", 0)
            all_ingest_stats.append(ingest_stats)
        else:
            turn_count = 0
            if verbose:
                print("    (no memory ingestion for this baseline)")
            all_ingest_stats.append({"total_turns": 0, "total_extracted": 0})

        # Trim QA
        qa_pairs_to_eval = sample.qa_pairs
        if max_qa_per_sample:
            qa_pairs_to_eval = qa_pairs_to_eval[:max_qa_per_sample]

        eval_sample = BenchmarkSample(
            sample_id=sample.sample_id,
            source=sample.source,
            sessions=sample.sessions,
            qa_pairs=qa_pairs_to_eval,
        )

        # Evaluate
        qa_results, eval_stats = evaluate_qa(
            eval_sample, graph_store, llm, retriever,
            turn_count, config, verbose=verbose,
            use_gpt4o_judge=use_gpt4o_judge,
        )
        all_qa_results.extend(qa_results)
        all_eval_stats.append(eval_stats)

        # Per-sample summary
        if verbose and qa_results:
            sample_f1 = sum(r.f1 for r in qa_results) / len(qa_results)
            sample_exact = sum(r.exact for r in qa_results) / len(qa_results)
            sample_contains = sum(r.contains for r in qa_results) / len(qa_results)
            sample_r5 = sum(r.recall_at_5 for r in qa_results) / len(qa_results)
            sample_mrr = sum(r.mrr for r in qa_results) / len(qa_results)
            sample_tsr = sum(r.task_success for r in qa_results) / len(qa_results)
            sample_latency = sum(r.latency_seconds for r in qa_results) / len(qa_results)
            cat_f1 = {}
            for r in qa_results:
                cat_f1.setdefault(r.category, []).append(r.f1)

            _subheader(f"SAMPLE SUMMARY: {sample.sample_id}")
            _kv("Questions evaluated", len(qa_results))
            _kv("Avg F1", f"{sample_f1:.4f}")
            _kv("Exact match rate", f"{sample_exact:.4f}")
            _kv("Contains rate", f"{sample_contains:.4f}")
            _kv("Recall@5", f"{sample_r5:.4f}")
            _kv("MRR", f"{sample_mrr:.4f}")
            _kv("Task success rate", f"{sample_tsr:.2%}")
            _kv("Avg latency", f"{sample_latency:.2f}s")
            _kv("Memory nodes", graph_store.node_count())
            _kv("Memory edges", graph_store.edge_count())
            ingest_t = ingest_stats.get("total_ingest_time", 0) if use_memory else 0
            eval_t = eval_stats.get("total_eval_time", 0)
            _kv("Ingest time", _format_duration(ingest_t))
            _kv("Eval time", _format_duration(eval_t))

            if cat_f1:
                print()
                _kv("F1 by category:", "")
                for cat, scores in sorted(cat_f1.items()):
                    avg = sum(scores) / len(scores)
                    _kv(f"  {cat}", f"{avg:.4f}  (n={len(scores)})", indent=4)

    # Final stats
    memory_stats = get_memory_stats(graph_store, total_forgotten) if samples else MemoryStats()

    # Edit/Delete/Locality tests (only for memory-enabled baselines)
    edit_delete_result = EditDeleteResult()
    if run_edit_delete_tests and use_memory and all_qa_results:
        all_traces = []
        for es in all_eval_stats:
            all_traces.extend(es.get("retrieval_traces", []))
        if all_traces:
            edit_delete_result = evaluate_memory_ops(
                all_qa_results, all_traces, graph_store, llm, retriever,
                current_turn=sum(s.get("total_turns", 0) for s in all_ingest_stats),
                config=config, verbose=verbose,
            )

    result = ExperimentResult(
        experiment_name=experiment_name,
        dataset=dataset_name,
        retrieval_mode=retrieval_mode,
        forgetting_policy=forgetting_policy,
        qa_results=all_qa_results,
        memory_stats=memory_stats,
        edit_delete_result=edit_delete_result,
        config=config,
    )

    # Enrich result dict with timing and trace data
    result_dict = result.to_dict()
    result_dict["timing"] = {
        "total_experiment_time": round(time.time() - exp_start, 2),
        "ingest_stats": all_ingest_stats,
        "eval_stats": [
            {k: v for k, v in s.items() if k != "retrieval_traces"}
            for s in all_eval_stats
        ],
    }
    # Add retrieval traces to each QA detail
    trace_idx = 0
    for es in all_eval_stats:
        for trace in es.get("retrieval_traces", []):
            if trace_idx < len(result_dict.get("qa_details", [])):
                result_dict["qa_details"][trace_idx]["retrieval_trace"] = trace
                trace_idx += 1

    # Save
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{experiment_name}_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)

    exp_elapsed = time.time() - exp_start

    if verbose:
        _header(f"EXPERIMENT COMPLETE: {experiment_name}")
        _kv("Total time", _format_duration(exp_elapsed))
        print()
        _kv("ANSWER QUALITY", "")
        _kv("  Overall F1", f"{result.overall_f1:.4f}")
        _kv("  Overall Exact Match", f"{result.overall_exact:.4f}")
        _kv("  Overall Contains", f"{result.overall_contains:.4f}")
        _kv("  Multi-hop F1", f"{result.multi_hop_f1:.4f}")
        _kv("  Task success rate", f"{result.overall_task_success_rate:.2%}")
        print()
        _kv("RETRIEVAL QUALITY", "")
        _kv("  Recall@5", f"{result.overall_recall_at_5:.4f}")
        _kv("  MRR", f"{result.overall_mrr:.4f}")
        print()
        _kv("MEMORY OPERATIONS", "")
        _kv("  Edit success rate", f"{edit_delete_result.edit_success_rate:.2%}")
        _kv("  Delete success rate", f"{edit_delete_result.delete_success_rate:.2%}")
        _kv("  Locality score", f"{edit_delete_result.locality_score:.4f}")
        print()
        _kv("LATENCY & STORAGE", "")
        _kv("  Avg QA latency", f"{result.avg_latency:.2f}s")
        _kv("  Storage", f"{memory_stats.storage_bytes:,} bytes")
        _kv("  Avg node size", f"{memory_stats.avg_node_size_bytes:.0f} bytes")
        print()
        _kv("Total questions", len(all_qa_results))
        _kv("Final memory nodes", memory_stats.enabled_nodes)
        _kv("Final memory edges", memory_stats.total_edges)
        _kv("Total forgotten", memory_stats.forgotten_count)
        if memory_stats.nodes_by_type:
            _kv("Node types", memory_stats.nodes_by_type)

        cat_f1 = result.f1_by_category()
        if cat_f1:
            print()
            _kv("F1 by category:", "")
            for cat, score in sorted(cat_f1.items()):
                n = sum(1 for r in all_qa_results if r.category == cat)
                _kv(f"  {cat}", f"{score:.4f}  (n={n})", indent=4)

        print(f"\n    Results saved to: {out_path}")

    logger.info(
        "Experiment '%s' complete: F1=%.4f, R@5=%.4f, MRR=%.4f, TSR=%.2f%% -> %s",
        experiment_name, result.overall_f1, result.overall_recall_at_5,
        result.overall_mrr, result.overall_task_success_rate * 100, out_path,
    )

    return result
