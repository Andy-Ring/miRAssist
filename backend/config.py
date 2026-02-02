# backend/config.py

import os
from pathlib import Path

# ---- Project ----
PROJECT_NAME = "miRAssist"
VERSION = "0.1"

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Allow override for Colab / remote environments
EVIDENCE_PATH = Path(
    os.getenv("MIRASSIST_EVIDENCE_PATH", str(PROCESSED_DIR / "evidence_interactions.parquet"))
)

# ---- LLM ----
MODEL_NAME = os.getenv("MIRASSIST_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
MAX_PROMPT_TOKENS = int(os.getenv("MIRASSIST_MAX_PROMPT_TOKENS", "6500"))

# ---- Retrieval defaults ----
DEFAULT_K = int(os.getenv("MIRASSIST_DEFAULT_K", "50"))
DEFAULT_MIN_SUPPORT = int(os.getenv("MIRASSIST_DEFAULT_MIN_SUPPORT", "1"))
