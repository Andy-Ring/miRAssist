# backend/app.py

import uuid
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from backend import config
from backend.state import JOBS
from backend.planner import run_planner
from backend.retrieval import retrieve_from_queryspec
from backend.prompting import build_prompt_bundle
from backend.synthesizer import run_synthesizer

app = FastAPI(title="miRAssist API")

# ---- Load evidence once ----
EVIDENCE = pd.read_parquet(config.EVIDENCE_PATH)


# ---- Request / Response models ----
class QueryRequest(BaseModel):
    question: str
    novel: bool = False
    k: int = config.DEFAULT_K
    min_support: int = config.DEFAULT_MIN_SUPPORT
    require_binding: bool = False
    require_expression: bool = False


@app.post("/query")
def submit_query(req: QueryRequest):
    query_id = f"miRAssist_{uuid.uuid4().hex[:10]}"
    JOBS[query_id] = {"status": "running"}

    thread = threading.Thread(
        target=_run_query_pipeline,
        args=(query_id, req),
        daemon=True,
    )
    thread.start()

    return {"query_id": query_id, "status": "running"}


@app.get("/result/{query_id}")
def get_result(query_id: str):
    if query_id not in JOBS:
        raise HTTPException(status_code=404, detail="Query not found")
    return JOBS[query_id]


# ---- Internal pipeline ----
def _run_query_pipeline(query_id: str, req: QueryRequest):
    try:
        # 1) Planner
        queryspec = run_planner(req.question)

        # Defensive defaults (planner should already do this, but keep API robust)
        queryspec.setdefault("filters", {})

        # Enforce UI overrides
        queryspec["novel"] = bool(req.novel)
        queryspec["k"] = int(req.k)
        queryspec["filters"]["min_support"] = int(req.min_support)
        queryspec["filters"]["require_binding_evidence"] = bool(req.require_binding)
        queryspec["filters"]["require_expression"] = bool(req.require_expression)

        # 2) Retrieval (returns shortlist df + inferred direction)
        shortlist_df, direction = retrieve_from_queryspec(EVIDENCE, queryspec)
        queryspec["mode"] = direction  # ensure prompting uses the real direction

        # 3) Prompt bundle
        bundle = build_prompt_bundle(
            queryspec=queryspec,
            shortlist=shortlist_df,
            max_prompt_tokens=config.MAX_PROMPT_TOKENS,
        )

        # 4) Synthesizer
        answer = run_synthesizer(bundle, model=config.MODEL_NAME)

        JOBS[query_id] = {
            "status": "complete",
            "queryspec": queryspec,
            "shortlist": shortlist_df.to_dict(orient="records"),
            "answer": answer,
            "model": f"{config.PROJECT_NAME} v{config.VERSION}",
        }

    except Exception as e:
        JOBS[query_id] = {"status": "error", "error": str(e)}
