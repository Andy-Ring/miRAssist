from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import make_blinded_evidence, normalize_mirna_name  # noqa: E402


def main() -> None:
    assert normalize_mirna_name("hsa-miR-21-5p") == "mir-21-5p"
    assert normalize_mirna_name("Hsa-miR-210-3p") == "mir-210-3p"
    assert normalize_mirna_name("MicroRNA-3907") == "mir-3907"

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

    repeated = pd.DataFrame(
        {
            "mirna_name": (["hsa-miR-210-3p", "miRNA-210", "MicroRNA-3907", "mir-21-5p"] * 25000),
            "gene_symbol": (["GENE1", "GENE2", "GENE3", "GENE4"] * 25000),
            "mirtarbase_pos": ([1, 0, 0, 1] * 25000),
            "label_mirtarbase": ([1, 0, 0, 1] * 25000),
            "support_targetscan": ([1, 0, 1, 1] * 25000),
            "support_mirdb": ([0, 1, 0, 1] * 25000),
            "support_encori": ([1, 1, 0, 0] * 25000),
        }
    )
    perf_started = time.perf_counter()
    blinded_large, heldout_large, audit_large = make_blinded_evidence(repeated)
    elapsed = time.perf_counter() - perf_started
    assert len(blinded_large) == 100000
    assert len(heldout_large) == 100000
    assert set(heldout_large["mirna_name_normalized"].unique()) == {"mir-210-3p", "mir-210", "mir-3907", "mir-21-5p"}
    assert audit_large["n_unique_mirna_names"] == 4
    assert elapsed < 15.0, f"make_blinded_evidence took too long: {elapsed:.2f}s"
    print("smoke_eval_blinding: OK")
    print(audit)
    print({"perf_seconds": elapsed, "large_audit": audit_large})


if __name__ == "__main__":
    main()
