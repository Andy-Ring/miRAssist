from __future__ import annotations

import os
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.cards import cards_from_dataframe  # noqa: E402
from evaluation.utils import build_eval_queryspec, make_blinded_evidence, run_eval_query  # noqa: E402


def main() -> None:
    os.environ["MIRASSIST_EVAL_MODE"] = "1"
    evidence = pd.DataFrame(
        [
            {
                "mirna_name": "hsa-miR-1-3p",
                "gene_symbol": "GENE1",
                "support_count": 3,
                "support_targetscan": 1,
                "support_mirdb": 1,
                "support_encori": 1,
                "mirtarbase_pos": 1,
                "label_mirtarbase": 1,
                "mirdb_best_score": 88.0,
                "ts_best_contextpp": -0.45,
                "clip_exp_sum": 10.0,
            },
            {
                "mirna_name": "hsa-miR-1-3p",
                "gene_symbol": "GENE2",
                "support_count": 1,
                "support_targetscan": 1,
                "mirdb_best_score": 70.0,
                "ts_best_contextpp": -0.20,
            },
        ]
    )
    blinded, _, _ = make_blinded_evidence(evidence)
    queryspec = build_eval_queryspec(
        query_id="smoke_query",
        mode="mirna_to_targets",
        mirna="hsa-miR-1-3p",
        gene=None,
        k=10,
        min_support=1,
        novel=False,
    )
    evidence_path = Path("mock_blinded.parquet")
    with patch("evaluation.utils.load_evidence", return_value=blinded):
        result = run_eval_query(queryspec=queryspec, evidence_path=evidence_path, disable_synthesis=True)
    assert not result.shortlist.empty
    cards = cards_from_dataframe(result.shortlist)
    rendered = " ".join(card.get("evidence", "") for card in cards)
    assert "miRTarBase" not in rendered
    print("smoke_eval_ranking: OK")


if __name__ == "__main__":
    main()
