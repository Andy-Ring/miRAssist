from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

import numpy as np
import pandas as pd


def _install_fake_matplotlib() -> None:
    if "matplotlib.pyplot" in sys.modules:
        return
    fake_pyplot = types.SimpleNamespace(
        figure=lambda *args, **kwargs: None,
        bar=lambda *args, **kwargs: None,
        xticks=lambda *args, **kwargs: None,
        title=lambda *args, **kwargs: None,
        ylabel=lambda *args, **kwargs: None,
        tight_layout=lambda *args, **kwargs: None,
        savefig=lambda *args, **kwargs: None,
        close=lambda *args, **kwargs: None,
    )
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_pyplot
    sys.modules["matplotlib"] = fake_matplotlib
    sys.modules["matplotlib.pyplot"] = fake_pyplot


def _load_ablation_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "evaluation" / "scripts" / "07_ablation_comparison.py"
    spec = importlib.util.spec_from_file_location("eval_ablation", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _install_fake_matplotlib()
    ablation = _load_ablation_module()

    rankings = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "mirna": "hsa-miR-1-3p",
                "gene_symbol": "GENEA",
                "rank": 1,
                "is_positive": 1,
                "retrieval_score": 0.90,
                "retrieval_support": 4,
                "retrieval_ts_contrib": 0.30,
                "retrieval_clip_contrib": 0.15,
                "retrieval_mirdb_contrib": 0.20,
                "retrieval_tcga_contrib": 0.10,
                "retrieval_seed_contrib": 0.22,
                "retrieval_rnahybrid_contrib": 0.18,
                "retrieval_local_au_contrib": 0.08,
                "retrieval_structure_contrib": 0.48,
                "retrieval_structure_in_score": 0,
            },
            {
                "query_id": "q1",
                "mirna": "hsa-miR-1-3p",
                "gene_symbol": "GENEB",
                "rank": 2,
                "is_positive": 0,
                "retrieval_score": 0.80,
                "retrieval_support": 3,
                "retrieval_ts_contrib": 0.25,
                "retrieval_clip_contrib": 0.10,
                "retrieval_mirdb_contrib": 0.18,
                "retrieval_tcga_contrib": 0.05,
                "retrieval_seed_contrib": 0.15,
                "retrieval_rnahybrid_contrib": 0.11,
                "retrieval_local_au_contrib": 0.04,
                "retrieval_structure_contrib": 0.30,
                "retrieval_structure_in_score": 0,
            },
            {
                "query_id": "q2",
                "mirna": "hsa-miR-2-5p",
                "gene_symbol": "GENEC",
                "rank": 1,
                "is_positive": 0,
                "retrieval_score": 0.55,
                "retrieval_support": 2,
                "retrieval_ts_contrib": 0.10,
                "retrieval_clip_contrib": 0.05,
                "retrieval_mirdb_contrib": 0.12,
                "retrieval_tcga_contrib": 0.08,
                "retrieval_seed_contrib": 0.10,
                "retrieval_rnahybrid_contrib": 0.05,
                "retrieval_local_au_contrib": 0.02,
                "retrieval_structure_contrib": 0.17,
                "retrieval_structure_in_score": 0,
            },
            {
                "query_id": "q2",
                "mirna": "hsa-miR-2-5p",
                "gene_symbol": "GENED",
                "rank": 2,
                "is_positive": 1,
                "retrieval_score": 0.50,
                "retrieval_support": 1,
                "retrieval_ts_contrib": 0.20,
                "retrieval_clip_contrib": 0.02,
                "retrieval_mirdb_contrib": 0.05,
                "retrieval_tcga_contrib": np.nan,
                "retrieval_seed_contrib": 0.24,
                "retrieval_rnahybrid_contrib": 0.14,
                "retrieval_local_au_contrib": 0.10,
                "retrieval_structure_contrib": 0.48,
                "retrieval_structure_in_score": 0,
            },
        ]
    )

    score_modes, warnings_payload = ablation._available_score_modes(rankings)
    assert "full" in score_modes
    assert "support_only" in score_modes
    assert "targetscan_only" in score_modes
    assert "clip_only" in score_modes
    assert "mirdb_only" in score_modes
    assert "tcga_only" in score_modes
    assert "no_targetscan" in score_modes
    assert "seed_only" in score_modes
    assert "rnahybrid_only" in score_modes
    assert "local_au_only" in score_modes
    assert "structure_only" in score_modes
    assert "no_structure" in score_modes
    assert "mirdb_targetscan_only" in score_modes
    assert "mirdb_targetscan_structure" in score_modes
    assert "no_pathway" not in score_modes
    assert any("not included in retrieval_score" in note for note in warnings_payload["notes"])
    assert "no_pathway" in warnings_payload["skipped_modes"]

    reranked_full = ablation.rerank_within_query(rankings.assign(ablation_score=score_modes["full"]), "ablation_score")
    q1_full = reranked_full[reranked_full["query_id"] == "q1"].sort_values("ablation_rank")
    assert q1_full.iloc[0]["gene_symbol"] == "GENEA"
    assert int(q1_full.iloc[0]["ablation_rank"]) == 1

    reranked_ts = ablation.rerank_within_query(
        rankings.assign(ablation_score=score_modes["targetscan_only"]),
        "ablation_score",
    )
    q2_ts = reranked_ts[reranked_ts["query_id"] == "q2"].sort_values("ablation_rank")
    assert q2_ts.iloc[0]["gene_symbol"] == "GENED"

    reranked_structure = ablation.rerank_within_query(
        rankings.assign(ablation_score=score_modes["structure_only"]),
        "ablation_score",
    )
    q2_structure = reranked_structure[reranked_structure["query_id"] == "q2"].sort_values("ablation_rank")
    assert q2_structure.iloc[0]["gene_symbol"] == "GENED"

    reranked_mts = ablation.rerank_within_query(
        rankings.assign(ablation_score=score_modes["mirdb_targetscan_structure"]),
        "ablation_score",
    )
    q1_mts = reranked_mts[reranked_mts["query_id"] == "q1"].sort_values("ablation_rank")
    assert q1_mts.iloc[0]["gene_symbol"] == "GENEA"

    metrics = ablation._compute_metrics_from_vector(np.array([1, 0]), [1, 2], total_positives=1)
    assert metrics["best_positive_rank"] == 1.0
    assert metrics["reciprocal_rank"] == 1.0
    assert metrics["positive_count_at_1"] == 1.0
    assert metrics["recall_at_2"] == 1.0
    assert metrics["precision_at_2"] == 0.5

    print("smoke_eval_ablation: OK")


if __name__ == "__main__":
    main()
