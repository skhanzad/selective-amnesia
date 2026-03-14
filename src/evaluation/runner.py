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
    ExperimentResult,
    MemoryStats,
    QAResult,
    contains_match,
    exact_match,
    multi_answer_f1,
    token_f1,
)
from src.memory.forgetting import apply_forgetting, get_policy
from src.memory.graph_store import GraphStore
from src.memory.retriever import get_retriever
from src.memory.schemas import MemoryNode, NodeType

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

        # Build context
        context_parts = []
        retrieved_contents = []
        for rm in retrieved:
            node = rm.node
            context_parts.append(f"[{node.node_type.value}] {node.content}")
            retrieved_contents.append({
                "content": node.content[:80],
                "type": node.node_type.value,
                "importance": node.importance,
                "relevance": round(rm.relevance_score, 3),
            })
        context = "\n".join(context_parts)

        # Answer
        prediction = _answer_question(llm, qa.question, context, config)

        # Score
        if ";" in qa.answer:
            f1 = multi_answer_f1(prediction, qa.answer)
        else:
            f1 = token_f1(prediction, qa.answer)

        em = exact_match(prediction, qa.answer)
        cm = contains_match(prediction, qa.answer)

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
            )
        )

        q_elapsed = time.time() - q_start
        qa_timings.append(q_elapsed)
        running_f1 += f1

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

    return MemoryStats(
        total_nodes=len(all_nodes),
        enabled_nodes=len(enabled),
        total_edges=graph_store.edge_count(),
        nodes_by_type=dict(type_counts),
        forgotten_count=forgotten_count,
    )


def run_experiment(
    experiment_name: str,
    samples: list[BenchmarkSample],
    config: dict,
    results_dir: str = "results",
    max_qa_per_sample: int | None = None,
    verbose: bool = True,
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
        )
        all_qa_results.extend(qa_results)
        all_eval_stats.append(eval_stats)

        # Per-sample summary
        if verbose and qa_results:
            sample_f1 = sum(r.f1 for r in qa_results) / len(qa_results)
            sample_exact = sum(r.exact for r in qa_results) / len(qa_results)
            sample_contains = sum(r.contains for r in qa_results) / len(qa_results)
            cat_f1 = {}
            for r in qa_results:
                cat_f1.setdefault(r.category, []).append(r.f1)

            _subheader(f"SAMPLE SUMMARY: {sample.sample_id}")
            _kv("Questions evaluated", len(qa_results))
            _kv("Avg F1", f"{sample_f1:.4f}")
            _kv("Exact match rate", f"{sample_exact:.4f}")
            _kv("Contains rate", f"{sample_contains:.4f}")
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

    result = ExperimentResult(
        experiment_name=experiment_name,
        dataset=dataset_name,
        retrieval_mode=retrieval_mode,
        forgetting_policy=forgetting_policy,
        qa_results=all_qa_results,
        memory_stats=memory_stats,
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
        _kv("Overall F1", f"{result.overall_f1:.4f}")
        _kv("Overall Exact Match", f"{result.overall_exact:.4f}")
        _kv("Overall Contains", f"{result.overall_contains:.4f}")
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
        "Experiment '%s' complete: F1=%.4f, exact=%.4f, contains=%.4f -> %s",
        experiment_name, result.overall_f1, result.overall_exact,
        result.overall_contains, out_path,
    )

    return result
