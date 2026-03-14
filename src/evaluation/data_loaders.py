"""Loaders for LoCoMo and LongMemEval into a common benchmark format."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QAPair:
    question: str
    answer: str
    category: str  # normalized category label
    evidence_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversationTurn:
    speaker: str  # "user" or "assistant" (normalized)
    text: str
    turn_id: str = ""
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    turns: list[ConversationTurn]
    session_id: str = ""
    timestamp: str = ""


@dataclass
class BenchmarkSample:
    sample_id: str
    source: str  # "locomo" or "longmemeval"
    sessions: list[Session]
    qa_pairs: list[QAPair]
    metadata: dict = field(default_factory=dict)


# ── LoCoMo ───────────────────────────────────────────────────────

LOCOMO_CATEGORIES = {
    1: "multi_hop",
    2: "temporal",
    3: "contextual",
    4: "open_domain",
    5: "adversarial",
}


def load_locomo(
    path: str = "ext/locomo/data/locomo10.json",
    max_samples: int | None = None,
) -> list[BenchmarkSample]:
    """Load LoCoMo dataset into common format."""
    with open(path) as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    samples = []
    for item in data:
        conv = item["conversation"]
        speaker_a = conv.get("speaker_a", "Speaker_A")
        speaker_b = conv.get("speaker_b", "Speaker_B")

        # Extract sessions
        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda x: int(x.split("_")[1]),
        )

        sessions = []
        for sk in session_keys:
            timestamp = conv.get(f"{sk}_date_time", "")
            turns = []
            for utterance in conv[sk]:
                speaker = utterance["speaker"]
                # Normalize: treat speaker_a as user, speaker_b as assistant
                role = "user" if speaker == speaker_a else "assistant"
                turns.append(
                    ConversationTurn(
                        speaker=role,
                        text=utterance["text"],
                        turn_id=utterance.get("dia_id", ""),
                        timestamp=timestamp,
                        metadata={
                            "original_speaker": speaker,
                            "blip_caption": utterance.get("blip_caption", ""),
                        },
                    )
                )
            sessions.append(
                Session(session_id=sk, turns=turns, timestamp=timestamp)
            )

        # Extract QA pairs
        qa_pairs = []
        for qa in item.get("qa", []):
            cat_id = qa.get("category", 0)
            # Category 5 (adversarial) uses "adversarial_answer"
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            qa_pairs.append(
                QAPair(
                    question=qa["question"],
                    answer=str(answer),
                    category=LOCOMO_CATEGORIES.get(cat_id, f"cat_{cat_id}"),
                    evidence_ids=qa.get("evidence", []),
                    metadata={"category_id": cat_id},
                )
            )

        samples.append(
            BenchmarkSample(
                sample_id=item.get("sample_id", f"locomo_{len(samples)}"),
                source="locomo",
                sessions=sessions,
                qa_pairs=qa_pairs,
            )
        )

    return samples


# ── LongMemEval ──────────────────────────────────────────────────

def load_longmemeval(
    path: str = "ext/LongMemEval/data/longmemeval_oracle.json",
    max_samples: int | None = None,
    question_types: list[str] | None = None,
) -> list[BenchmarkSample]:
    """Load LongMemEval dataset into common format.

    Uses oracle variant by default (smaller, contains only evidence sessions).
    """
    with open(path) as f:
        data = json.load(f)

    if question_types:
        data = [d for d in data if d.get("question_type") in question_types]

    if max_samples:
        data = data[:max_samples]

    samples = []
    for item in data:
        sessions = []
        haystack_sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates", [])
        haystack_ids = item.get("haystack_session_ids", [])

        for i, session_turns in enumerate(haystack_sessions):
            timestamp = haystack_dates[i] if i < len(haystack_dates) else ""
            session_id = haystack_ids[i] if i < len(haystack_ids) else f"session_{i}"

            turns = []
            for turn in session_turns:
                turns.append(
                    ConversationTurn(
                        speaker=turn.get("role", "user"),
                        text=turn.get("content", ""),
                        timestamp=timestamp,
                        metadata={"has_answer": turn.get("has_answer", False)},
                    )
                )
            sessions.append(
                Session(session_id=session_id, turns=turns, timestamp=timestamp)
            )

        qa_pairs = [
            QAPair(
                question=item["question"],
                answer=item["answer"],
                category=item.get("question_type", "unknown"),
                evidence_ids=item.get("answer_session_ids", []),
                metadata={"question_date": item.get("question_date", "")},
            )
        ]

        samples.append(
            BenchmarkSample(
                sample_id=item.get("question_id", f"lme_{len(samples)}"),
                source="longmemeval",
                sessions=sessions,
                qa_pairs=qa_pairs,
                metadata={"question_date": item.get("question_date", "")},
            )
        )

    return samples
