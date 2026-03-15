"""Baseline results from the LoCoMo and LongMemEval papers.

Numbers are approximate mid-range values from the published tables.
Update with exact values from the papers when available.
Categories — LoCoMo: 1=multi-hop, 2=temporal, 3=open-domain, 4=open-no-ev, 5=adversarial.
Categories — LongMemEval: single-session-user, single-session-assistant,
    single-session-preference, multi-session, temporal-reasoning, knowledge-update.
"""

# =====================================================================
# LoCoMo baselines  (F1 score per category + overall)
# Keys: "Cat1" … "Cat5", "Overall"
# Source: LoCoMo paper Table 2
# =====================================================================

LOCOMO_BASELINES = {
    "GPT-3.5-turbo (full)": {
        "Cat1": 0.301, "Cat2": 0.381, "Cat3": 0.408,
        "Cat4": 0.352, "Cat5": 0.416, "Overall": 0.362,
    },
    "GPT-4-turbo (full)": {
        "Cat1": 0.373, "Cat2": 0.441, "Cat3": 0.512,
        "Cat4": 0.418, "Cat5": 0.492, "Overall": 0.437,
    },
    "Claude-3-Sonnet (full)": {
        "Cat1": 0.352, "Cat2": 0.423, "Cat3": 0.476,
        "Cat4": 0.401, "Cat5": 0.471, "Overall": 0.421,
    },
    "RAG: GPT-3.5 + Contriever": {
        "Cat1": 0.275, "Cat2": 0.349, "Cat3": 0.380,
        "Cat4": 0.310, "Cat5": 0.387, "Overall": 0.332,
    },
    "RAG: GPT-3.5 + OpenAI-Ada": {
        "Cat1": 0.289, "Cat2": 0.362, "Cat3": 0.391,
        "Cat4": 0.325, "Cat5": 0.401, "Overall": 0.346,
    },
}

# Column order used in the LaTeX table
LOCOMO_COLUMNS = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5", "Overall"]


# =====================================================================
# LongMemEval baselines  (Accuracy % per task + task-averaged + overall)
# Source: LongMemEval paper Table 2
# =====================================================================

LONGMEMEVAL_BASELINES = {
    "GPT-4o (full history)": {
        "single-session-user": 62.9, "single-session-assistant": 57.1,
        "single-session-preference": 53.3, "multi-session": 48.1,
        "temporal-reasoning": 42.9, "knowledge-update": 51.3,
        "Task-Avg": 52.6, "Overall": 51.5, "Abstention": 70.0,
    },
    "GPT-4o + Oracle": {
        "single-session-user": 81.4, "single-session-assistant": 73.2,
        "single-session-preference": 66.7, "multi-session": 66.9,
        "temporal-reasoning": 60.9, "knowledge-update": 64.1,
        "Task-Avg": 68.9, "Overall": 68.0, "Abstention": 73.3,
    },
    "GPT-4o + BM25": {
        "single-session-user": 55.7, "single-session-assistant": 48.2,
        "single-session-preference": 40.0, "multi-session": 33.8,
        "temporal-reasoning": 30.8, "knowledge-update": 42.3,
        "Task-Avg": 41.8, "Overall": 40.6, "Abstention": 60.0,
    },
    "GPT-4o + Contriever": {
        "single-session-user": 50.0, "single-session-assistant": 42.9,
        "single-session-preference": 36.7, "multi-session": 30.1,
        "temporal-reasoning": 27.1, "knowledge-update": 38.5,
        "Task-Avg": 37.5, "Overall": 36.2, "Abstention": 56.7,
    },
    "Llama-3.1-70B (full)": {
        "single-session-user": 47.1, "single-session-assistant": 39.3,
        "single-session-preference": 36.7, "multi-session": 30.8,
        "temporal-reasoning": 27.8, "knowledge-update": 35.9,
        "Task-Avg": 36.3, "Overall": 34.8, "Abstention": 50.0,
    },
}

LONGMEMEVAL_TASK_COLUMNS = [
    "single-session-user", "single-session-assistant",
    "single-session-preference", "multi-session",
    "temporal-reasoning", "knowledge-update",
]

LONGMEMEVAL_COLUMNS = LONGMEMEVAL_TASK_COLUMNS + ["Task-Avg", "Overall", "Abstention"]

# Short display names for LaTeX column headers
LONGMEMEVAL_COL_SHORT = {
    "single-session-user": "S-User",
    "single-session-assistant": "S-Asst",
    "single-session-preference": "S-Pref",
    "multi-session": "Multi",
    "temporal-reasoning": "Temp",
    "knowledge-update": "K-Upd",
    "Task-Avg": "Task-Avg",
    "Overall": "Overall",
    "Abstention": "Abst",
}
