"""Metrics for LoCoMo and LongMemEval evaluation.

LoCoMo: token-level F1 (primary), multi-answer F1, adversarial detection.
LongMemEval: LLM-judged accuracy, NDCG, recall@k.
"""
import re
import string
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np


# =====================================================================
# Text normalisation (matches LoCoMo's evaluation.py)
# =====================================================================

def normalize_answer(s: str) -> str:
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


# =====================================================================
# LoCoMo metrics
# =====================================================================

def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    """Category-1 (multi-hop): each ground-truth sub-answer gets its best
    match among the prediction's comma-separated parts."""
    preds = [p.strip() for p in prediction.split(",")]
    gts = [g.strip() for g in ground_truth.split(",")]
    if not gts:
        return 0.0
    return float(np.mean([max(token_f1(p, gt) for p in preds) for gt in gts]))


def adversarial_score(prediction: str) -> float:
    low = prediction.lower()
    return 1.0 if ("no information available" in low or "not mentioned" in low) else 0.0


def eval_locomo_qa(prediction: str, answer: str, category: int) -> float:
    if category == 5:
        return adversarial_score(prediction)
    if category == 1:
        return multi_answer_f1(prediction, answer)
    if category == 3:
        answer = answer.split(";")[0].strip()
    return token_f1(prediction, answer)


def locomo_context_recall(retrieved_ids: List[str], evidence_ids: List[str]) -> float:
    """Fraction of evidence dialogue IDs that appear in retrieved context."""
    if not evidence_ids:
        return 1.0
    return sum(1 for eid in evidence_ids if eid in retrieved_ids) / len(evidence_ids)


# =====================================================================
# LongMemEval retrieval metrics
# =====================================================================

def dcg(relevances: np.ndarray, k: int) -> float:
    r = np.asfarray(relevances)[:k]
    if r.size == 0:
        return 0.0
    return float(r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1))))


def ndcg_at_k(
    rankings: List[int],
    correct_docs: Set[str],
    corpus_ids: List[str],
    k: int = 10,
) -> float:
    relevances = [1 if corpus_ids[i] in correct_docs else 0 for i in range(len(corpus_ids))]
    sorted_rel = [relevances[idx] for idx in rankings[:k]]
    ideal_rel = sorted(relevances, reverse=True)
    ideal = dcg(np.array(ideal_rel), k)
    actual = dcg(np.array(sorted_rel), k)
    return (actual / ideal) if ideal > 0 else 0.0


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 10,
) -> Tuple[float, float]:
    """Returns ``(recall_any, recall_all)`` at *k*."""
    top_k = set(retrieved_ids[:k])
    recall_any = float(any(rid in top_k for rid in relevant_ids))
    recall_all = float(all(rid in top_k for rid in relevant_ids))
    return recall_any, recall_all


# =====================================================================
# LongMemEval LLM-judged accuracy (simplified — uses token-F1 fallback)
# =====================================================================

def llm_judge_accuracy(prediction: str, answer: str) -> float:
    """Heuristic stand-in for GPT-4o judge: F1 > 0.5 counts as correct."""
    return 1.0 if token_f1(prediction, answer) > 0.5 else 0.0


# =====================================================================
# Aggregation helpers
# =====================================================================

def aggregate_by_key(
    scores: List[Tuple[str, float]],
) -> Dict[str, float]:
    """Given (key, value) pairs, return {key: mean(values)}."""
    buckets: Dict[str, List[float]] = {}
    for key, val in scores:
        buckets.setdefault(key, []).append(val)
    return {k: float(np.mean(v)) for k, v in buckets.items()}
