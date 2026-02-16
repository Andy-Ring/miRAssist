from __future__ import annotations

import json
import threading
import time
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.jobstore import JobStore
from backend.planner import run_planner
from backend.prompting import build_prompt_bundle
from backend.retrieval import retrieve_from_queryspec
from backend.llm_backend import generate_answer


APP = FastAPI(title="miRAssist Backend", version="0.1.0")
STORE = JobStore()


# --------
# API Models
# --------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    novel: bool = True
    k: int = 25
    min_support: int = 2
    require_binding_evidence: bool = False
    require_expression: bool = False
    # Optional override for pathway integration.
    # - None/"auto": let the planner decide
    # - "boost": soft preference toward genes in matched pathways
    # - "filter": hard filter to genes in matched pathways
    pathway_mode: Optional[str] = None


class QueryResponse(BaseModel):
    query_id: str
    status: str


# --------
# Helpers
# --------
def _run_job(query_id: str, req: QueryRequest) -> None:
    """
    Background job:
      1) planner -> QuerySpec JSON
      2) retrieval -> shortlist DataFrame
      3) prompting -> LLM bundle
      4) llm_backend -> answer text
    """
    try:
        STORE.set_status(query_id, "running")

        qs = run_planner(req.question)

        # Apply user overrides (these are *preferences*; planner can still do entity extraction)
        qs["novel"] = bool(req.novel)
        qs["k"] = int(req.k)
        qs.setdefault("filters", {})
        qs["filters"]["min_support"] = int(req.min_support)
        qs["filters"]["require_binding_evidence"] = bool(req.require_binding_evidence)
        qs["filters"]["require_expression"] = bool(req.require_expression)

        # Optional API override: pathway filter vs boost.
        # This does NOT invent pathway keywords; it only changes how they are applied.
        if req.pathway_mode is not None:
            pm = str(req.pathway_mode).strip().lower()
            if pm == "auto" or pm == "":
                pass
            elif pm in {"boost", "filter"}:
                qs.setdefault("pathway_filter", {})
                qs["pathway_filter"]["enabled"] = True
                qs["pathway_filter"]["mode"] = pm
            else:
                raise ValueError("pathway_mode must be one of: auto, boost, filter")

        # Load evidence once per job
        ev = pd.read_parquet("data/processed/evidence_pairs_tcga.parquet")

        shortlist_df, direction = retrieve_from_queryspec(ev, qs)

        bundle = build_prompt_bundle(
            queryspec=qs,
            shortlist=shortlist_df,
            direction=direction,
        )

        answer = generate_answer(bundle)

        STORE.write_result(query_id, {
            "queryspec": qs,
            "direction": direction,
            "bundle": bundle,
            "answer": answer,
        })
        STORE.set_status(query_id, "completed")

    except Exception as e:
        STORE.set_status(query_id, "failed", error=str(e))


# --------
# Routes
# --------
@APP.get("/healthz")
def healthz():
    return {"ok": True}


@APP.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    query_id = STORE.new_id()
    STORE.create_job(query_id, meta={"question": req.question})

    th = threading.Thread(target=_run_job, args=(query_id, req), daemon=True)
    th.start()

    return QueryResponse(query_id=query_id, status="queued")


@APP.get("/result/{query_id}")
def result(query_id: str):
    job = STORE.get_job(query_id)
    if not job:
        raise HTTPException(status_code=404, detail="query_id not found")
    return job
