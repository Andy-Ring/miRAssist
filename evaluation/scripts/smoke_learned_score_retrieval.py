from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config import get_learned_score_column, get_use_learned_score
from backend.retrieval import apply_learned_score_ranking


def _set_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _assert_flag_parsing() -> None:
    cases = {
        None: True,
        "1": True,
        "true": True,
        "yes": True,
        "on": True,
        "0": False,
        "false": False,
        "no": False,
        "off": False,
    }
    for raw_value, expected in cases.items():
        _set_env("MIRASSIST_USE_LEARNED_SCORE", raw_value)
        actual = get_use_learned_score()
        assert actual is expected, "Flag parsing mismatch for {!r}: expected {}, got {}".format(raw_value, expected, actual)

    _set_env("MIRASSIST_LEARNED_SCORE_COLUMN", None)
    assert get_learned_score_column() == "learned_score_xgb_raw_v1"
    _set_env("MIRASSIST_LEARNED_SCORE_COLUMN", "custom_score")
    assert get_learned_score_column() == "custom_score"


def _assert_ranking_behavior() -> None:
    df = pd.DataFrame(
        [
            {
                "gene_symbol": "GENE_A",
                "retrieval_score": 0.60,
                "support_count": 3,
                "mirdb_best_score": 80,
                "ts_context_strength": 0.42,
                "learned_score_xgb_raw_v1": 0.40,
            },
            {
                "gene_symbol": "GENE_B",
                "retrieval_score": 0.80,
                "support_count": 2,
                "mirdb_best_score": 70,
                "ts_context_strength": 0.20,
                "learned_score_xgb_raw_v1": None,
            },
            {
                "gene_symbol": "GENE_C",
                "retrieval_score": 0.50,
                "support_count": 1,
                "mirdb_best_score": 65,
                "ts_context_strength": 0.10,
                "learned_score_xgb_raw_v1": 0.90,
            },
        ]
    )

    ranked_df, diagnostics = apply_learned_score_ranking(
        df,
        learned_score_column="learned_score_xgb_raw_v1",
        enabled=True,
    )
    assert ranked_df["gene_symbol"].tolist() == ["GENE_C", "GENE_B", "GENE_A"]
    assert ranked_df["retrieval_rank_score"].round(4).tolist() == [0.9, 0.8, 0.4]
    assert diagnostics["learned_score_enabled"] is True
    assert diagnostics["learned_score_present_count"] == 2
    assert diagnostics["learned_score_missing_count"] == 1

    manual_df, manual_diagnostics = apply_learned_score_ranking(
        df,
        learned_score_column="learned_score_xgb_raw_v1",
        enabled=False,
    )
    assert manual_df["gene_symbol"].tolist() == ["GENE_B", "GENE_A", "GENE_C"]
    assert manual_diagnostics["retrieval_ranking_mode"] == "manual"

    missing_df, missing_diagnostics = apply_learned_score_ranking(
        df.drop(columns=["learned_score_xgb_raw_v1"]),
        learned_score_column="learned_score_xgb_raw_v1",
        enabled=True,
    )
    assert missing_df["gene_symbol"].tolist() == ["GENE_B", "GENE_A", "GENE_C"]
    assert missing_diagnostics["learned_score_enabled"] is False
    assert missing_diagnostics.get("warnings")


def main() -> None:
    _assert_flag_parsing()
    _assert_ranking_behavior()
    print("smoke_learned_score_retrieval: ok")


if __name__ == "__main__":
    main()
