from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import collect_rankings_from_json, json_dump


def _write_eval_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        json_dir = tmp / "json"
        outdir = tmp / "collected"
        json_dir.mkdir(parents=True, exist_ok=True)

        labels = pd.DataFrame(
            [
                {
                    "mirna_name": "hsa-miR-210-3p",
                    "gene_symbol": "KCMF1",
                    "transcript_id": "ENST00000300001",
                    "mirna_name_normalized": "mir-210-3p",
                    "gene_symbol_normalized": "KCMF1",
                    "mirtarbase_pos": 1,
                    "label_mirtarbase": 1,
                }
            ]
        )
        labels_path = tmp / "heldout_mirtarbase_labels.csv"
        labels.to_csv(labels_path, index=False)

        _write_eval_json(
            json_dir / "eval_00000.json",
            {
                "query_id": "eval_00000",
                "queryspec": {"mirna": "hsa-miR-210-3p", "gene": None},
                "retrieval_diagnostics": {},
                "shortlist": [
                    {
                        "mirna_name": "hsa-miR-210-3p",
                        "gene_symbol": "KCMF1",
                        "transcript_id": "ENST00000300001",
                        "mirna_name_normalized": "mir-210-3p",
                        "gene_symbol_normalized": "KCMF1",
                        "retrieval_score": 0.9,
                        "mirtarbase_pos": 0,
                        "label_mirtarbase": 0,
                    }
                ],
            },
        )
        _write_eval_json(
            json_dir / "eval_00001.json",
            {
                "query_id": "eval_00001",
                "queryspec": {"mirna": "hsa-miR-210-3p", "gene": None},
                "retrieval_diagnostics": {},
                "shortlist": [
                    {
                        "mirna_name": "hsa-miR-210-3p",
                        "gene_symbol": "KCMF1",
                        "transcript_id": "ENST00000300001",
                        "mirna_name_normalized": "mir-210-3p",
                        "gene_symbol_normalized": "KCMF1",
                        "retrieval_score": 0.8,
                        "mirtarbase_pos": 0,
                        "label_mirtarbase": 0,
                        "mirtarbase_pos_x": 0,
                        "label_mirtarbase_x": 0,
                        "mirtarbase_pos_y": 1,
                        "label_mirtarbase_y": 1,
                        "heldout_mirtarbase_pos": 0,
                        "heldout_label_mirtarbase": 0,
                    }
                ],
            },
        )

        rankings, query_summary, diagnostics = collect_rankings_from_json(json_dir, labels)
        outdir.mkdir(parents=True, exist_ok=True)
        rankings.to_csv(outdir / "rankings_long.csv", index=False)
        query_summary.to_csv(outdir / "query_summary.csv", index=False)
        json_dump(outdir / "label_join_diagnostics.json", diagnostics)

        assert int(rankings["is_positive"].sum()) > 0
        assert int(rankings["heldout_mirtarbase_pos"].sum()) > 0
        assert int(query_summary["n_positives_retrieved"].sum()) > 0
        assert query_summary["best_positive_rank"].notna().any()
        assert (outdir / "label_join_diagnostics.json").exists()
        assert diagnostics["n_positive_ranked_rows"] > 0
        assert diagnostics["n_queries_with_positive_retrieved"] > 0
        assert "blinded_mirtarbase_pos" in rankings.columns
        assert "blinded_label_mirtarbase" in rankings.columns
        assert int(rankings.loc[rankings["query_id"] == "eval_00001", "is_positive"].max()) == 1
        assert int(rankings.loc[rankings["query_id"] == "eval_00001", "heldout_mirtarbase_pos"].max()) == 1
        assert "mirtarbase_pos_x" in diagnostics["dropped_stale_label_columns"]
        assert "label_mirtarbase_y" in diagnostics["dropped_stale_label_columns"]
        assert diagnostics["heldout_positive_sum_after_join"] > 0

    print("smoke_collect_eval_rankings_label_join: OK")


if __name__ == "__main__":
    main()
