"""Run LoCoMo QA evaluation using graph-based memory.

For each conversation:
  1. Build a MemoryGraph (session-by-session extraction + optional forgetting).
  2. For each QA pair: retrieve relevant nodes → generate answer → compute F1.
  3. Aggregate scores by category.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm import build_llm
from experiments.config import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_TOP_K, RESULTS_DIR
from experiments.data_loaders import LoCoMoSample, load_locomo
from experiments.graph_builder import build_locomo_graph
from experiments.graph_retriever import retrieve_and_format
from experiments.metrics import eval_locomo_qa, locomo_context_recall

_QA_SYSTEM_PROMPT = """\
You are answering questions about a long conversation between two people.
Use the provided memory context to answer. Write a short phrase answer.
Answer with exact words from the context whenever possible.
If the information is not available, say "Not mentioned in conversation".\
"""


def _answer_question(
    question: str,
    context: str,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
) -> str:
    llm = build_llm(model=model, provider=provider)
    messages = [
        SystemMessage(content=_QA_SYSTEM_PROMPT),
        HumanMessage(content=f"Memory context:\n{context}\n\nQuestion: {question}\nShort answer:"),
    ]
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"  [generation error: {e}]")
        return ""


def evaluate_locomo(
    *,
    forget_preset: str = "none",
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    top_k: int = DEFAULT_TOP_K,
    max_samples: int | None = None,
    max_qas_per_sample: int | None = None,
) -> Dict:
    """Run LoCoMo evaluation and return results dict.

    Returns::

        {
            "method": str,
            "per_category": {1: float, 2: float, ...},
            "overall": float,
            "per_sample": [{sample_id, qas: [{question, answer, prediction, f1, category}]}],
        }
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_locomo()
    if max_samples:
        samples = samples[:max_samples]

    method_name = f"GraphMem ({forget_preset})"
    all_scores: List[float] = []
    cat_scores: Dict[int, List[float]] = defaultdict(list)
    per_sample_results: List[Dict] = []

    for si, sample in enumerate(samples):
        print(f"\n[LoCoMo {si+1}/{len(samples)}] {sample.sample_id}")

        # Step 1: build graph
        graph = build_locomo_graph(
            sample, forget_preset=forget_preset,
            model=model, provider=provider,
        )
        print(f"  Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        # Step 2: evaluate QA
        qas = sample.qas
        if max_qas_per_sample:
            qas = qas[:max_qas_per_sample]

        sample_results: List[Dict] = []
        for qi, qa in enumerate(qas):
            context = retrieve_and_format(qa.question, graph, top_k=top_k)
            prediction = _answer_question(qa.question, context, model=model, provider=provider)
            f1 = eval_locomo_qa(prediction, qa.answer, qa.category)

            all_scores.append(f1)
            cat_scores[qa.category].append(f1)
            sample_results.append({
                "question": qa.question,
                "answer": qa.answer,
                "prediction": prediction,
                "f1": round(f1, 4),
                "category": qa.category,
            })

            if (qi + 1) % 20 == 0:
                print(f"  QA {qi+1}/{len(qas)}  running F1={sum(all_scores)/len(all_scores):.3f}")

        per_sample_results.append({"sample_id": sample.sample_id, "qas": sample_results})

    per_cat = {cat: round(float(sum(s) / len(s)), 4) for cat, s in sorted(cat_scores.items())}
    overall = round(float(sum(all_scores) / len(all_scores)), 4) if all_scores else 0.0

    results = {
        "method": method_name,
        "forget_preset": forget_preset,
        "model": model,
        "top_k": top_k,
        "per_category": per_cat,
        "overall": overall,
        "num_qas": len(all_scores),
        "per_sample": per_sample_results,
    }

    out_path = RESULTS_DIR / f"locomo_{forget_preset}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nLoCoMo results ({method_name}):")
    for cat, score in per_cat.items():
        print(f"  Cat {cat}: {score:.3f}")
    print(f"  Overall: {overall:.3f}")
    print(f"  Saved to {out_path}")

    return results
