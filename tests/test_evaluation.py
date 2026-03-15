"""Tests for evaluation metrics and data loaders."""

import pytest

from src.evaluation.metrics import (
    adversarial_check,
    contains_match,
    exact_match,
    multi_answer_f1,
    normalize_answer,
    recall_at_k,
    reciprocal_rank,
    task_success,
    token_f1,
    ExperimentResult,
    QAResult,
)
from src.evaluation.data_loaders import load_locomo, load_longmemeval


# ── Metrics tests ────────────────────────────────────────────────

def test_normalize_answer():
    assert normalize_answer("The quick Brown Fox") == "quick brown fox"
    assert normalize_answer("  hello,  world!  ") == "hello world"


def test_token_f1_exact():
    assert token_f1("hello world", "hello world") == 1.0


def test_token_f1_partial():
    f1 = token_f1("hello world foo", "hello world")
    assert 0.5 < f1 < 1.0


def test_token_f1_no_overlap():
    assert token_f1("foo bar", "baz qux") == 0.0


def test_token_f1_empty():
    assert token_f1("", "") == 1.0
    assert token_f1("hello", "") == 0.0
    assert token_f1("", "hello") == 0.0


def test_token_f1_duplicates():
    """Counter-based F1 handles duplicate tokens correctly."""
    # "cat cat" has 2 "cat" tokens; "cat" has 1
    # common = min(2,1) = 1; precision = 1/2, recall = 1/1
    f1 = token_f1("cat cat", "cat")
    assert f1 == pytest.approx(2/3, abs=0.01)


def test_token_f1_stemming():
    """Stemmer maps 'running' and 'runs' to same stem."""
    f1 = token_f1("running", "runs")
    assert f1 > 0.0  # stems should match


def test_multi_answer_f1_comma():
    """Default separator is comma (matching LoCoMo)."""
    f1 = multi_answer_f1("coffee, tea", "coffee, tea")
    assert f1 == 1.0

    f1 = multi_answer_f1("coffee", "coffee, tea")
    assert 0.0 < f1 < 1.0  # partial match


def test_multi_answer_f1_semicolon():
    """Can still use semicolon if explicitly passed."""
    f1 = multi_answer_f1("coffee; tea", "coffee; tea", sep=";")
    assert f1 == 1.0


def test_adversarial_check():
    assert adversarial_check("No information available about this topic") == 1.0
    assert adversarial_check("This was not mentioned in any conversation") == 1.0
    assert adversarial_check("The answer is 42") == 0.0
    assert adversarial_check("") == 0.0


def test_exact_match():
    assert exact_match("Hello World", "hello world") == 1.0
    assert exact_match("foo", "bar") == 0.0


def test_contains_match():
    assert contains_match("I think the answer is 42", "42") == 1.0
    assert contains_match("no match", "42") == 0.0


def test_recall_at_k():
    retrieved = ["User likes dark roast coffee", "Weather is sunny", "User lives in NYC"]
    assert recall_at_k(retrieved, "dark roast coffee", k=5) > 0
    assert recall_at_k(retrieved, "completely unrelated answer xyz", k=5) == 0.0
    assert recall_at_k([], "coffee", k=5) == 0.0


def test_reciprocal_rank():
    retrieved = ["Weather is sunny", "User likes dark roast coffee", "Random info"]
    # "coffee" is relevant to index 1 (second position) -> RR = 1/2
    rr = reciprocal_rank(retrieved, "dark roast coffee")
    assert rr == pytest.approx(0.5)
    # No relevant results
    assert reciprocal_rank(retrieved, "completely unrelated xyz") == 0.0
    assert reciprocal_rank([], "coffee") == 0.0


def test_reciprocal_rank_first():
    retrieved = ["User likes dark roast coffee", "Other info"]
    rr = reciprocal_rank(retrieved, "dark roast coffee")
    assert rr == pytest.approx(1.0)


def test_task_success():
    assert task_success(0.8) == 1.0
    assert task_success(0.5) == 1.0
    assert task_success(0.3) == 0.0
    assert task_success(0.0) == 0.0
    assert task_success(0.6, threshold=0.7) == 0.0
    assert task_success(0.7, threshold=0.7) == 1.0


def test_experiment_result_aggregation():
    result = ExperimentResult(
        experiment_name="test",
        dataset="test",
        retrieval_mode="none",
        forgetting_policy="none",
        qa_results=[
            QAResult("q1", "a", "a", "cat1", 1.0, 1.0, 1.0),
            QAResult("q2", "b", "c", "cat1", 0.0, 0.0, 0.0),
            QAResult("q3", "d", "d", "cat2", 1.0, 1.0, 1.0),
        ],
    )
    assert result.overall_f1 == pytest.approx(2 / 3, abs=0.01)
    assert result.f1_by_category()["cat1"] == pytest.approx(0.5)
    assert result.f1_by_category()["cat2"] == pytest.approx(1.0)


def test_experiment_result_judge_accuracy():
    result = ExperimentResult(
        experiment_name="test",
        dataset="test",
        retrieval_mode="none",
        forgetting_policy="none",
        qa_results=[
            QAResult("q1", "a", "a", "cat1", 1.0, 1.0, 1.0, judge_score=1.0),
            QAResult("q2", "b", "c", "cat1", 0.0, 0.0, 0.0, judge_score=0.0),
            QAResult("q3", "d", "d", "cat2", 1.0, 1.0, 1.0, judge_score=None),
        ],
    )
    # Only 2 judged results: 1.0 and 0.0 -> avg 0.5
    assert result.overall_judge_accuracy == pytest.approx(0.5)


def test_experiment_result_no_judge():
    result = ExperimentResult(
        experiment_name="test",
        dataset="test",
        retrieval_mode="none",
        forgetting_policy="none",
        qa_results=[
            QAResult("q1", "a", "a", "cat1", 1.0, 1.0, 1.0),
        ],
    )
    assert result.overall_judge_accuracy is None


# ── Data loader tests ────────────────────────────────────────────

def test_load_locomo():
    samples = load_locomo(max_samples=1)
    assert len(samples) == 1
    s = samples[0]
    assert s.source == "locomo"
    assert len(s.sessions) > 0
    assert len(s.qa_pairs) > 0
    assert s.sessions[0].turns[0].speaker in ("user", "assistant")
    # Check categories are normalized
    cats = {q.category for q in s.qa_pairs}
    assert cats.issubset({"multi_hop", "temporal", "contextual", "open_domain", "adversarial"})


def test_load_longmemeval():
    samples = load_longmemeval(max_samples=2)
    assert len(samples) == 2
    s = samples[0]
    assert s.source == "longmemeval"
    assert len(s.sessions) > 0
    assert len(s.qa_pairs) == 1  # one question per LME sample
    assert s.sessions[0].turns[0].speaker in ("user", "assistant")
