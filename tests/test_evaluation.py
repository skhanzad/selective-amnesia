"""Tests for evaluation metrics and data loaders."""

import pytest

from src.evaluation.metrics import (
    contains_match,
    exact_match,
    multi_answer_f1,
    normalize_answer,
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


def test_multi_answer_f1():
    f1 = multi_answer_f1("coffee; tea", "coffee; tea")
    assert f1 == 1.0

    f1 = multi_answer_f1("coffee", "coffee; tea")
    assert 0.0 < f1 < 1.0  # partial match


def test_exact_match():
    assert exact_match("Hello World", "hello world") == 1.0
    assert exact_match("foo", "bar") == 0.0


def test_contains_match():
    assert contains_match("I think the answer is 42", "42") == 1.0
    assert contains_match("no match", "42") == 0.0


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
