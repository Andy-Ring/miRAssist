from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import make_blinded_evidence  # noqa: E402


def main() -> None:
    evidence = pd.DataFrame(
        [
            {
                "mirna_name": "hsa-miR-1-3p",
                "gene_symbol": "GENE1",
                "mirtarbase_pos": 1,
                "label_mirtarbase": 1,
                "mirtarbase_pmids": "12345",
                "support_targetscan": 1,
                "support_mirdb": 0,
                "support_encori": 1,
            }
        ]
    )
    blinded, heldout, audit = make_blinded_evidence(evidence)
    assert "mirtarbase_pmids" not in blinded.columns
    assert int(blinded["mirtarbase_pos"].sum()) == 0
    assert int(heldout["mirtarbase_pos"].sum()) == 1
    assert int(blinded["support_count"].iloc[0]) == 2
    print("smoke_eval_blinding: OK")
    print(audit)


if __name__ == "__main__":
    main()
