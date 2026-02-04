from __future__ import annotations

import argparse
import traceback
import time
import pandas as pd

from backend.jobstore import write_job
from backend.planner import run_planner
from backend.retrieval import retrieve_from_queryspec
from backend.prompting import build_prompt_bundle
from backend.synthesizer import run_synthesizer
from backend.config import EVIDENCE_PATH

def load_evidence():
    return pd.read_parquet(EVIDENCE_PATH)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_id", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--novel", action="store_true")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--min_support", type=int, default=2)
    ap.add_argument("--require_binding_evidence", action="store_true")
    ap.add_argument("--require_expression", action="store_true")
    args = ap.parse_args()

    qid = args.query_id
    write_job(qid, {"status": "running", "started_at": time.time()})

    try:
        qs = run_planner(args.question)
        qs["novel"] = bool(args.novel)
        qs["k"] = int(args.k)
        qs.setdefault("filters", {})
        qs["filters"]["min_support"] = int(args.min_support)
        qs["filters"]["require_binding_evidence"] = bool(args.require_binding_evidence)
        qs["filters"]["require_expression"] = bool(args.require_expression)

        ev = load_evidence()
        shortlist_df, direction = retrieve_from_queryspec(ev, qs)

        bundle = build_prompt_bundle(queryspec=qs, shortlist=shortlist_df)
        answer_text = run_synthesizer(bundle)
        if not isinstance(answer_text, str):
            answer_text = json.dumps(answer_text, ensure_ascii=False, indent=2)

        write_job(qid, {
            "status": "complete",
            "queryspec": qs,
            "direction": direction,
            "shortlist": shortlist_df.to_dict(orient="records"),
            "answer": {"raw_text": answer_text},
            "finished_at": time.time(),
        })

    except Exception as e:
        write_job(qid, {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "finished_at": time.time(),
        })

if __name__ == "__main__":
    main()
