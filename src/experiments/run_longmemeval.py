"""Run LongMemEval QA evaluation using graph-based memory.

Pipeline (per instance):
  1. Build a MemoryGraph from the haystack sessions.
  2. Retrieve relevant nodes for the question.
  3. Generate an answer using LLM + retrieved context.
  4. Score with token-F1 heuristic (or LLM judge if configured).
  5. Aggregate by question type.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm import build_llm
from experiments.config import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_TOP_K, RESULTS_DIR
from experiments.data_loaders import LongMemInstance, load_longmemeval
from experiments.graph_builder import build_longmemeval_graph
from experiments.graph_retriever import retrieve_and_format
from experiments.metrics import llm_judge_accuracy

_QA_SYSTEM_PROMPT = """\
You are answering a question based on your memory of past conversations.
Use the provided memory context to answer accurately and concisely.
If the question asks about timing, be specific about dates and order.
If the information was updated, provide the most recent version.
If you cannot answer from the context, say "I don't have that information".\
"""


def _answer_question(
    question: str,
    context: str,
    question_date: str = "",
    llm=None,
    model: str = "",
    provider: str = "",
) -> str:
    if llm is None:
        llm = build_llm(model=model, provider=provider)
    date_line = f"\nCurrent date: {question_date}" if question_date else ""
    messages = [
        SystemMessage(content=_QA_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Memory context:\n{context}{date_line}\n\nQuestion: {question}\nAnswer:"
        ),
    ]
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"  [generation error: {e}]")
        return ""


def evaluate_longmemeval(
    *,
    forget_preset: str = "none",
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    top_k: int = DEFAULT_TOP_K,
    max_instances: int | None = None,
) -> Dict:
    """Run LongMemEval evaluation and return results dict.

    Returns::

        {
            "method": str,
            "per_type": {"single-session-user": float, ...},
            "task_avg": float,
            "overall": float,
            "abstention": float,
            "details": [{question_id, question_type, prediction, label, ...}],
        }
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    instances = load_longmemeval()
    if max_instances:
        instances = instances[:max_instances]

    method_name = f"GraphMem ({forget_preset})"
    type_scores: Dict[str, List[float]] = defaultdict(list)
    abstention_scores: List[float] = []
    all_scores: List[float] = []
    details: List[Dict] = []

    # Single LLM instance for all QA calls
    qa_llm = build_llm(model=model, provider=provider)

    for i, inst in enumerate(instances):
        is_abstention = "_abs" in inst.question_id

        if (i + 1) % 25 == 0:
            running = sum(all_scores) / len(all_scores) if all_scores else 0
            print(f"  [{i+1}/{len(instances)}] running acc={running:.3f}")

        # ---- Step 1: Build graph from all sessions FIRST ----
        graph = build_longmemeval_graph(
            inst, forget_preset=forget_preset,
            model=model, provider=provider,
        )

        # ---- Step 2: Retrieve from pre-built graph ----
        context = retrieve_and_format(inst.question, graph, top_k=top_k)

        # ---- Step 3: LLM predicts answer ----
        prediction = _answer_question(
            inst.question, context,
            question_date=inst.question_date,
            llm=qa_llm,
        )

        # ---- Step 4: Measure metric ----
        if is_abstention:
            low = prediction.lower()
            correct = 1.0 if ("don't have" in low or "not available" in low or "no information" in low or "i don't" in low) else 0.0
            abstention_scores.append(correct)
        else:
            correct = llm_judge_accuracy(prediction, inst.answer)
            type_scores[inst.question_type].append(correct)
            all_scores.append(correct)

        details.append({
            "question_id": inst.question_id,
            "question_type": inst.question_type,
            "question": inst.question,
            "answer": inst.answer,
            "prediction": prediction,
            "correct": correct,
            "is_abstention": is_abstention,
        })

    per_type = {t: round(float(np.mean(s)) * 100, 1) for t, s in type_scores.items()}
    task_avg = round(float(np.mean([np.mean(s) for s in type_scores.values()])) * 100, 1) if type_scores else 0.0
    overall = round(float(np.mean(all_scores)) * 100, 1) if all_scores else 0.0
    abstention = round(float(np.mean(abstention_scores)) * 100, 1) if abstention_scores else 0.0

    results = {
        "method": method_name,
        "forget_preset": forget_preset,
        "model": model,
        "top_k": top_k,
        "per_type": per_type,
        "task_avg": task_avg,
        "overall": overall,
        "abstention": abstention,
        "num_instances": len(all_scores),
        "details": details,
    }

    out_path = RESULTS_DIR / f"longmemeval_{forget_preset}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nLongMemEval results ({method_name}):")
    for t, acc in per_type.items():
        print(f"  {t}: {acc:.1f}%")
    print(f"  Task-Avg: {task_avg:.1f}%  Overall: {overall:.1f}%  Abstention: {abstention:.1f}%")
    print(f"  Saved to {out_path}")

    return results
