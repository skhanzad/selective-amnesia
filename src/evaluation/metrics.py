"""Evaluation metrics for the benchmark suite.

Implements token-level F1 (matching LoCoMo's Counter+stemmer approach),
exact match, adversarial check, GPT-4o judge, and memory-specific metrics.
"""

from __future__ import annotations

import logging
import os
import re
import string
from collections import Counter
from dataclasses import dataclass, field

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()
logger = logging.getLogger(__name__)


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 using Counter intersection and Porter stemmer.

    Matches the official LoCoMo evaluation (ext/locomo/task_eval/evaluation.py:126-138).
    """
    pred_tokens = [_stemmer.stem(w) for w in normalize_answer(prediction).split()]
    gt_tokens = [_stemmer.stem(w) for w in normalize_answer(ground_truth).split()]

    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str, sep: str = ",") -> float:
    """F1 for multi-answer questions (answers separated by sep).

    Default separator is comma, matching LoCoMo (ext/locomo/task_eval/evaluation.py:142).
    """
    gt_answers = [a.strip() for a in ground_truth.split(sep) if a.strip()]
    pred_answers = [a.strip() for a in prediction.split(sep) if a.strip()]

    if not gt_answers:
        return 1.0 if not pred_answers else 0.0

    # For each ground truth, find best matching prediction
    scores = []
    for gt in gt_answers:
        best = max((token_f1(p, gt) for p in pred_answers), default=0.0)
        scores.append(best)

    return sum(scores) / len(scores)


def adversarial_check(prediction: str) -> float:
    """Check if prediction correctly identifies an adversarial (unanswerable) question.

    Returns 1.0 if prediction contains refusal phrases, else 0.0.
    Matches LoCoMo (ext/locomo/task_eval/evaluation.py:217-221).
    """
    lower = prediction.lower()
    if "no information available" in lower or "not mentioned" in lower:
        return 1.0
    return 0.0


# ── GPT-4o Judge (LongMemEval) ──────────────────────────────────

_JUDGE_TEMPLATES: dict[str, str] = {
    "single-session-user": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "single-session-assistant": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "multi-session": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "In addition, do not penalize off-by-one errors for the number of days. "
        "If the question asks for the number of days/weeks/months, etc., and the model makes "
        "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
        "response is still correct. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer is the "
        "required answer."
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a "
        "response from a model. Please answer yes if the response satisfies the desired "
        "response. Otherwise, answer no. The model does not need to reflect all the points "
        "in the rubric. The response is correct as long as it recalls and utilizes the user's "
        "personal information correctly."
        "\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a "
        "model. Please answer yes if the model correctly identifies the question as "
        "unanswerable. The model could say that the information is incomplete, or some other "
        "information is given but the asked information is not."
        "\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {response}"
        "\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    ),
}


def gpt4o_judge(
    question: str,
    answer: str,
    prediction: str,
    question_type: str,
    is_abstention: bool = False,
) -> float | None:
    """Use GPT-4o to judge answer correctness (LongMemEval protocol).

    Replicates exact prompt templates from ext/LongMemEval/src/evaluation/evaluate_qa.py:24-42.
    Returns 1.0 if correct, 0.0 if incorrect, None if OPENAI_API_KEY not set.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None

    template_key = "abstention" if is_abstention else question_type
    template = _JUDGE_TEMPLATES.get(template_key)
    if template is None:
        # Fall back to single-session-user for unknown types
        template = _JUDGE_TEMPLATES["single-session-user"]

    prompt = template.format(question=question, answer=answer, response=prediction)

    try:
        from langchain_openai import ChatOpenAI

        judge_llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=10)
        result = judge_llm.invoke([{"role": "user", "content": prompt}])
        return 1.0 if "yes" in result.content.strip().lower() else 0.0
    except Exception as e:
        logger.warning("GPT-4o judge failed: %s", e)
        return None


def exact_match(prediction: str, ground_truth: str) -> float:
    """Binary exact match after normalization."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def contains_match(prediction: str, ground_truth: str) -> float:
    """Check if ground truth is contained in prediction (for looser eval)."""
    return 1.0 if normalize_answer(ground_truth) in normalize_answer(prediction) else 0.0


def _is_relevant(content: str, ground_truth: str, threshold: float = 0.1) -> bool:
    """Check if a memory's content is relevant to the ground truth answer."""
    return token_f1(content, ground_truth) > threshold


def recall_at_k(
    retrieved_contents: list[str], ground_truth: str, k: int = 5
) -> float:
    """Fraction of top-k retrieved memories that are relevant to the ground truth.

    Relevance is determined by token F1 overlap with the answer.
    """
    top_k = retrieved_contents[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for c in top_k if _is_relevant(c, ground_truth))
    return relevant / len(top_k)


def reciprocal_rank(retrieved_contents: list[str], ground_truth: str) -> float:
    """Reciprocal rank of the first relevant retrieved memory.

    Returns 1/rank of the first memory with significant token overlap to the
    ground truth answer, or 0.0 if no relevant memory is retrieved.
    """
    for i, content in enumerate(retrieved_contents):
        if _is_relevant(content, ground_truth):
            return 1.0 / (i + 1)
    return 0.0


def task_success(f1: float, threshold: float = 0.5) -> float:
    """Binary success: 1.0 if F1 meets threshold, else 0.0."""
    return 1.0 if f1 >= threshold else 0.0


# ── Aggregate metrics ────────────────────────────────────────────

@dataclass
class QAResult:
    question: str
    ground_truth: str
    prediction: str
    category: str
    f1: float
    exact: float
    contains: float
    sample_id: str = ""
    recall_at_5: float = 0.0
    mrr: float = 0.0
    task_success: float = 0.0
    latency_seconds: float = 0.0
    num_retrieved: int = 0
    judge_score: float | None = None


@dataclass
class MemoryStats:
    """Stats about the memory graph at evaluation time."""
    total_nodes: int = 0
    enabled_nodes: int = 0
    total_edges: int = 0
    nodes_by_type: dict[str, int] = field(default_factory=dict)
    forgotten_count: int = 0
    storage_bytes: int = 0
    avg_node_size_bytes: float = 0.0


@dataclass
class EditDeleteResult:
    """Results from edit/delete/locality evaluation."""
    edit_success_rate: float = 0.0
    delete_success_rate: float = 0.0
    locality_score: float = 0.0
    num_edits_tested: int = 0
    num_deletes_tested: int = 0
    num_locality_checks: int = 0
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "edit_success_rate": round(self.edit_success_rate, 4),
            "delete_success_rate": round(self.delete_success_rate, 4),
            "locality_score": round(self.locality_score, 4),
            "num_edits_tested": self.num_edits_tested,
            "num_deletes_tested": self.num_deletes_tested,
            "num_locality_checks": self.num_locality_checks,
            "details": self.details,
        }


@dataclass
class ExperimentResult:
    experiment_name: str
    dataset: str
    retrieval_mode: str
    forgetting_policy: str
    qa_results: list[QAResult] = field(default_factory=list)
    memory_stats: MemoryStats = field(default_factory=MemoryStats)
    edit_delete_result: EditDeleteResult = field(default_factory=EditDeleteResult)
    config: dict = field(default_factory=dict)

    @property
    def overall_f1(self) -> float:
        if not self.qa_results:
            return 0.0
        return sum(r.f1 for r in self.qa_results) / len(self.qa_results)

    @property
    def overall_exact(self) -> float:
        if not self.qa_results:
            return 0.0
        return sum(r.exact for r in self.qa_results) / len(self.qa_results)

    @property
    def overall_contains(self) -> float:
        if not self.qa_results:
            return 0.0
        return sum(r.contains for r in self.qa_results) / len(self.qa_results)

    @property
    def overall_recall_at_5(self) -> float:
        if not self.qa_results:
            return 0.0
        return sum(r.recall_at_5 for r in self.qa_results) / len(self.qa_results)

    @property
    def overall_mrr(self) -> float:
        if not self.qa_results:
            return 0.0
        return sum(r.mrr for r in self.qa_results) / len(self.qa_results)

    @property
    def overall_task_success_rate(self) -> float:
        if not self.qa_results:
            return 0.0
        return sum(r.task_success for r in self.qa_results) / len(self.qa_results)

    @property
    def avg_latency(self) -> float:
        if not self.qa_results:
            return 0.0
        return sum(r.latency_seconds for r in self.qa_results) / len(self.qa_results)

    @property
    def overall_judge_accuracy(self) -> float | None:
        judged = [r for r in self.qa_results if r.judge_score is not None]
        if not judged:
            return None
        return sum(r.judge_score for r in judged) / len(judged)

    @property
    def multi_hop_f1(self) -> float:
        mh = [r for r in self.qa_results if r.category == "multi_hop"]
        if not mh:
            return 0.0
        return sum(r.f1 for r in mh) / len(mh)

    def f1_by_category(self) -> dict[str, float]:
        cats: dict[str, list[float]] = {}
        for r in self.qa_results:
            cats.setdefault(r.category, []).append(r.f1)
        return {k: sum(v) / len(v) for k, v in cats.items()}

    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "dataset": self.dataset,
            "retrieval_mode": self.retrieval_mode,
            "forgetting_policy": self.forgetting_policy,
            "overall_f1": round(self.overall_f1, 4),
            "overall_exact_match": round(self.overall_exact, 4),
            "overall_contains": round(self.overall_contains, 4),
            "overall_recall_at_5": round(self.overall_recall_at_5, 4),
            "overall_mrr": round(self.overall_mrr, 4),
            "overall_task_success_rate": round(self.overall_task_success_rate, 4),
            "multi_hop_f1": round(self.multi_hop_f1, 4),
            "overall_judge_accuracy": (
                round(self.overall_judge_accuracy, 4)
                if self.overall_judge_accuracy is not None else None
            ),
            "avg_latency_seconds": round(self.avg_latency, 4),
            "f1_by_category": {k: round(v, 4) for k, v in self.f1_by_category().items()},
            "num_questions": len(self.qa_results),
            "memory_stats": {
                "total_nodes": self.memory_stats.total_nodes,
                "enabled_nodes": self.memory_stats.enabled_nodes,
                "total_edges": self.memory_stats.total_edges,
                "nodes_by_type": self.memory_stats.nodes_by_type,
                "forgotten_count": self.memory_stats.forgotten_count,
                "storage_bytes": self.memory_stats.storage_bytes,
                "avg_node_size_bytes": round(self.memory_stats.avg_node_size_bytes, 1),
            },
            "edit_delete_result": self.edit_delete_result.to_dict(),
            "qa_details": [
                {
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "prediction": r.prediction,
                    "category": r.category,
                    "f1": round(r.f1, 4),
                    "exact": round(r.exact, 4),
                    "contains": round(r.contains, 4),
                    "recall_at_5": round(r.recall_at_5, 4),
                    "mrr": round(r.mrr, 4),
                    "task_success": round(r.task_success, 4),
                    "latency_seconds": round(r.latency_seconds, 4),
                    "num_retrieved": r.num_retrieved,
                    "judge_score": r.judge_score,
                    "sample_id": r.sample_id,
                }
                for r in self.qa_results
            ],
        }
