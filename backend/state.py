# backend/state.py

from typing import Dict, Any

# query_id -> status/result
JOBS: Dict[str, Dict[str, Any]] = {}
