from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
EXT_DIR = PROJECT_ROOT / "ext"

# ---------- Benchmark data paths ----------
LOCOMO_DATA = EXT_DIR / "locomo" / "data" / "locomo10.json"
LONGMEMEVAL_ORACLE = EXT_DIR / "LongMemEval" / "data" / "longmemeval_oracle.json"
LONGMEMEVAL_S = EXT_DIR / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"

# ---------- Cache / output ----------
CACHE_DIR = SRC_DIR / "experiments" / ".cache"
RESULTS_DIR = SRC_DIR / "experiments" / "results"

# ---------- LLM ----------
DEFAULT_MODEL = "llama3.2:latest"
DEFAULT_PROVIDER = "ollama"

# ---------- Graph builder ----------
# Max dialogue turns to feed into a single extraction call
EXTRACTION_BATCH_TURNS = 30

# ---------- Forgetting policy presets ----------
FORGET_PRESETS = {
    "none": {"threshold": 0.0, "max_nodes": None},
    "mild": {"threshold": 0.25, "max_nodes": None},
    "default": {"threshold": 0.35, "max_nodes": None},
    "aggressive": {"threshold": 0.50, "max_nodes": 200},
}

# ---------- Retrieval ----------
DEFAULT_TOP_K = 10
