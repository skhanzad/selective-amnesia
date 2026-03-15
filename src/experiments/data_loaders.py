"""Load and parse LoCoMo and LongMemEval datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from experiments.config import LOCOMO_DATA, LONGMEMEVAL_ORACLE


# =====================================================================
# LoCoMo
# =====================================================================

@dataclass
class LoCoMoTurn:
    speaker: str
    dia_id: str
    text: str
    blip_caption: Optional[str] = None


@dataclass
class LoCoMoSession:
    session_num: int
    date_time: str
    turns: List[LoCoMoTurn]


@dataclass
class LoCoMoQA:
    question: str
    answer: str
    category: int
    evidence: List[str]


@dataclass
class LoCoMoSample:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: List[LoCoMoSession]
    qas: List[LoCoMoQA]


def load_locomo(path: Path = LOCOMO_DATA) -> List[LoCoMoSample]:
    raw = json.loads(path.read_text())
    samples: List[LoCoMoSample] = []
    for entry in raw:
        conv = entry["conversation"]
        speaker_a = conv.get("speaker_a", "Speaker A")
        speaker_b = conv.get("speaker_b", "Speaker B")

        sessions: List[LoCoMoSession] = []
        for sess_num in range(1, 50):
            key = f"session_{sess_num}"
            if key not in conv or not conv[key]:
                continue
            dt_key = f"{key}_date_time"
            turns = [
                LoCoMoTurn(
                    speaker=t["speaker"],
                    dia_id=t["dia_id"],
                    text=t["text"],
                    blip_caption=t.get("blip_caption"),
                )
                for t in conv[key]
            ]
            sessions.append(LoCoMoSession(
                session_num=sess_num,
                date_time=conv.get(dt_key, ""),
                turns=turns,
            ))

        qas = [
            LoCoMoQA(
                question=q["question"],
                answer=str(q.get("answer", q.get("adversarial_answer", ""))),
                category=q["category"],
                evidence=[e.replace("(", "").replace(")", "") for e in q.get("evidence", [])],
            )
            for q in entry.get("qa", [])
        ]
        samples.append(LoCoMoSample(
            sample_id=entry["sample_id"],
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            qas=qas,
        ))
    return samples


def locomo_conversation_to_text(sample: LoCoMoSample) -> str:
    """Flatten all sessions into a single text block (for full-context eval)."""
    parts: List[str] = []
    for sess in sample.sessions:
        parts.append(f"\n--- {sess.date_time} ---")
        for t in sess.turns:
            line = f"{t.speaker}: {t.text}"
            if t.blip_caption:
                line += f" [shares image: {t.blip_caption}]"
            parts.append(line)
    return "\n".join(parts)


def locomo_session_to_text(session: LoCoMoSession, speaker_a: str = "", speaker_b: str = "") -> str:
    """Render a single session as text."""
    lines = [f"[{session.date_time}]"]
    for t in session.turns:
        line = f"{t.speaker}: {t.text}"
        if t.blip_caption:
            line += f" [shares image: {t.blip_caption}]"
        lines.append(line)
    return "\n".join(lines)


# =====================================================================
# LongMemEval
# =====================================================================

@dataclass
class LongMemTurn:
    role: str
    content: str
    has_answer: bool = False


@dataclass
class LongMemSession:
    session_id: str
    date: str
    turns: List[LongMemTurn]


@dataclass
class LongMemInstance:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: List[LongMemSession]
    answer_session_ids: List[str]


def load_longmemeval(path: Path = LONGMEMEVAL_ORACLE) -> List[LongMemInstance]:
    raw = json.loads(path.read_text())
    instances: List[LongMemInstance] = []
    for entry in raw:
        sessions: List[LongMemSession] = []
        for i, sess_turns in enumerate(entry.get("haystack_sessions", [])):
            sid = entry["haystack_session_ids"][i] if i < len(entry.get("haystack_session_ids", [])) else f"session_{i}"
            date = entry["haystack_dates"][i] if i < len(entry.get("haystack_dates", [])) else ""
            turns = [
                LongMemTurn(
                    role=t["role"],
                    content=t["content"],
                    has_answer=t.get("has_answer", False),
                )
                for t in sess_turns
            ]
            sessions.append(LongMemSession(session_id=sid, date=date, turns=turns))

        instances.append(LongMemInstance(
            question_id=entry["question_id"],
            question_type=entry.get("question_type", "unknown"),
            question=entry["question"],
            answer=entry["answer"],
            question_date=entry.get("question_date", ""),
            sessions=sessions,
            answer_session_ids=entry.get("answer_session_ids", []),
        ))
    return instances


def longmem_session_to_text(session: LongMemSession) -> str:
    lines = [f"[{session.date}]"]
    for t in session.turns:
        lines.append(f"{t.role}: {t.content}")
    return "\n".join(lines)
