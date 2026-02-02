# backend/config.py

from pathlib import Path

# ---- Project ----
PROJECT_NAME = "miRAssist"
VERSION = "0.1"

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

EVIDENCE_PATH = PROCESSED_DIR / "evidence_interactions.parquet"

# ---- LLM ----
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_PROMPT_TOKENS = 6500

# ---- Retrieval defaults ----
DEFAULT_K = 50
DEFAULT_MIN_SUPPORT = 1
