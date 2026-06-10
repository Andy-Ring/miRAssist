from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import collect_rankings_from_json, json_dump  # noqa: E402


def main() -> None:
    labels = pd.DataFrame(
        [
            {
                "mirna_name": "hsa-miR-1-3p",
                "gene_symbol": "GENE1",
                "mirna_name_norm": "mir-1-3p",
                "gene_symbol_norm": "GENE1",
                "mirtarbase_pos_label": 1,
            },
            {
                "mirna_name": "hsa-miR-1-3p",
                "gene_symbol": "GENE2",
                "mirna_name_norm": "mir-1-3p",
                "gene_symbol_norm": "GENE2",
                "mirtarbase_pos_label": 0,
            },
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        json_dump(
            tmp / "eval_00000.json",
            {
                "query_id": "eval_00000",
                "queryspec": {"mirna": "hsa-miR-1-3p", "gene": None},
                "retrieval_diagnostics": {},
                "shortlist": [
                    {"mirna_name": "hsa-miR-1-3p", "gene_symbol": "GENE2", "retrieval_score": 0.8},
                    {"mirna_name": "hsa-miR-1-3p", "gene_symbol": "GENE1", "retrieval_score": 0.7},
                ],
            },
        )
        rankings, summary, collection_summary = collect_rankings_from_json(tmp, labels)
        assert int(rankings["is_positive"].sum()) == 1
        assert int(summary["n_ranked"].iloc[0]) == 2
        assert "mirtarbase_pos_label" in collection_summary["label_columns_used"]

        labels_missing = labels[["mirna_name", "gene_symbol", "mirna_name_norm", "gene_symbol_norm"]].copy()
        rankings_missing, _, missing_summary = collect_rankings_from_json(tmp, labels_missing)
        assert int(rankings_missing["is_positive"].sum()) == 0
        assert missing_summary["warnings"]
    print("smoke_eval_collect: OK")


if __name__ == "__main__":
    main()
