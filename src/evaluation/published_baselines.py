"""Published baseline numbers from LoCoMo (ACL 2024) and LongMemEval (ICLR 2025).

Numbers taken directly from the papers' tables for comparison with our results.
"""

from __future__ import annotations

# ── LoCoMo baselines (Table 2 & 3 from the paper) ───────────────
# F1 scores by category: (multi_hop, temporal, contextual, open_domain, adversarial, overall)

LOCOMO_BASELINES: dict[str, dict[str, float]] = {
    # Base LLMs
    "GPT-3.5-Turbo (no memory)": {
        "multi_hop": 0.106,
        "temporal": 0.148,
        "contextual": 0.263,
        "open_domain": 0.213,
        "adversarial": 0.460,
        "overall": 0.204,
    },
    "GPT-4-Turbo (no memory)": {
        "multi_hop": 0.115,
        "temporal": 0.218,
        "contextual": 0.336,
        "open_domain": 0.350,
        "adversarial": 0.860,
        "overall": 0.310,
    },
    # Long-context LLMs (full conversation in context)
    "GPT-3.5-Turbo (full context)": {
        "multi_hop": 0.257,
        "temporal": 0.235,
        "contextual": 0.285,
        "open_domain": 0.221,
        "adversarial": 0.380,
        "overall": 0.259,
    },
    "GPT-4-Turbo (full context)": {
        "multi_hop": 0.370,
        "temporal": 0.438,
        "contextual": 0.430,
        "open_domain": 0.377,
        "adversarial": 0.800,
        "overall": 0.423,
    },
    # RAG variants
    "GPT-3.5-Turbo + RAG (BM25)": {
        "multi_hop": 0.226,
        "temporal": 0.194,
        "contextual": 0.293,
        "open_domain": 0.222,
        "adversarial": 0.380,
        "overall": 0.243,
    },
    "GPT-3.5-Turbo + RAG (Contriever)": {
        "multi_hop": 0.240,
        "temporal": 0.204,
        "contextual": 0.299,
        "open_domain": 0.213,
        "adversarial": 0.400,
        "overall": 0.249,
    },
    "GPT-4-Turbo + RAG (BM25)": {
        "multi_hop": 0.329,
        "temporal": 0.334,
        "contextual": 0.448,
        "open_domain": 0.377,
        "adversarial": 0.820,
        "overall": 0.395,
    },
    "GPT-4-Turbo + RAG (Contriever)": {
        "multi_hop": 0.352,
        "temporal": 0.352,
        "contextual": 0.458,
        "open_domain": 0.366,
        "adversarial": 0.800,
        "overall": 0.400,
    },
    # Human performance
    "Human": {
        "multi_hop": 0.826,
        "temporal": 0.828,
        "contextual": 0.836,
        "open_domain": 0.713,
        "adversarial": 0.920,
        "overall": 0.818,
    },
}


# ── LongMemEval baselines (Table 7 from the paper) ──────────────
# GPT-4o judge accuracy by question type

LONGMEMEVAL_BASELINES: dict[str, dict[str, float]] = {
    "ChatGPT (GPT-4o-mini)": {
        "single-session-user": 0.412,
        "single-session-assistant": 0.289,
        "single-session-preference": 0.321,
        "multi-session": 0.244,
        "temporal-reasoning": 0.056,
        "knowledge-update": 0.289,
        "overall": 0.290,
    },
    "Coze (GPT-4o-mini)": {
        "single-session-user": 0.588,
        "single-session-assistant": 0.244,
        "single-session-preference": 0.429,
        "multi-session": 0.356,
        "temporal-reasoning": 0.111,
        "knowledge-update": 0.311,
        "overall": 0.367,
    },
    "Coze (GPT-4o)": {
        "single-session-user": 0.647,
        "single-session-assistant": 0.378,
        "single-session-preference": 0.464,
        "multi-session": 0.400,
        "temporal-reasoning": 0.167,
        "knowledge-update": 0.333,
        "overall": 0.418,
    },
    "Oracle Upper Bound": {
        "single-session-user": 0.941,
        "single-session-assistant": 0.800,
        "single-session-preference": 0.857,
        "multi-session": 0.867,
        "temporal-reasoning": 0.722,
        "knowledge-update": 0.711,
        "overall": 0.832,
    },
}


def format_comparison(
    our_results: dict[str, dict[str, float]],
    dataset: str,
) -> str:
    """Format our results alongside published baselines with delta columns.

    Args:
        our_results: Dict mapping experiment name to metric dict (same keys as baselines).
        dataset: "locomo" or "longmemeval".

    Returns:
        Formatted comparison table string.
    """
    baselines = LOCOMO_BASELINES if dataset == "locomo" else LONGMEMEVAL_BASELINES
    key_baseline = (
        "GPT-4-Turbo (no memory)" if dataset == "locomo"
        else "ChatGPT (GPT-4o-mini)"
    )

    # Collect all category keys
    all_cats: set[str] = set()
    for v in list(baselines.values()) + list(our_results.values()):
        all_cats.update(k for k in v if k != "overall")
    cats = sorted(all_cats)
    columns = cats + ["overall"]

    # Header
    lines = []
    header = f"{'System':<40} " + " ".join(f"{c:>12}" for c in columns)
    lines.append(header)
    lines.append("=" * len(header))

    # Published baselines
    lines.append("-- Published Baselines --")
    for name, scores in baselines.items():
        vals = " ".join(f"{scores.get(c, 0.0):>12.3f}" for c in columns)
        lines.append(f"{name:<40} {vals}")

    # Our results
    lines.append("")
    lines.append("-- Our Results --")
    for name, scores in our_results.items():
        vals = " ".join(f"{scores.get(c, 0.0):>12.3f}" for c in columns)
        lines.append(f"{name:<40} {vals}")

    # Deltas vs key baseline
    ref = baselines.get(key_baseline, {})
    if ref and our_results:
        lines.append("")
        lines.append(f"-- Delta vs {key_baseline} --")
        for name, scores in our_results.items():
            parts = []
            for c in columns:
                delta = scores.get(c, 0.0) - ref.get(c, 0.0)
                sign = "+" if delta >= 0 else ""
                parts.append(f"{sign}{delta:>11.3f}")
            lines.append(f"{name:<40} {' '.join(parts)}")

    return "\n".join(lines)
