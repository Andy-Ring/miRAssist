from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\andym\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        source_path = tmp / "source.csv"
        rnaplfold_path = tmp / "rnaplfold.csv"
        output_path = tmp / "clean.csv"
        report_path = tmp / "report.txt"

        source_df = pd.DataFrame(
            [
                {
                    "mirna_name": "hsa-miR-210-3p",
                    "gene_symbol": "FGFRL1",
                    "mirna_name_norm": "mir-210-3p",
                    "gene_symbol_norm": "FGFRL1",
                    "transcript_id": "ENST000001",
                    "mirtarbase_pos": 1,
                    "label_mirtarbase": 1,
                    "ts_best_contextpp": -0.32,
                    "ts_best_percentile": 78,
                    "clip_exp_sum": 12,
                    "clip_exp_max": 5,
                    "n_clip_sites": 3,
                    "has_seed_features": 1,
                    "best_seed_rank": 4,
                    "n_sites_6mer": 0,
                    "n_sites_7mer_a1": 1,
                    "n_sites_7mer_m8": 0,
                    "n_sites_8mer": 1,
                    "n_total_sites": 2,
                    "best_seed_class": "8mer",
                    "has_rnahybrid": 1,
                    "n_rnahybrid_sites": 2,
                    "best_mfe": -24.0,
                    "mfe_strength": 24.0,
                    "mean_top3_mfe": -19.0,
                    "mean_top3_mfe_strength": 19.0,
                    "best_8mer_mfe": -22.0,
                    "best_7mer_m8_mfe": -20.0,
                    "best_site_start_by_mfe": 15,
                    "best_site_end_by_mfe": 22,
                    "BRCA_spearman_rho": -0.31,
                    "BRCA_repression_evidence": 1,
                    "PRAD_spearman_rho": -0.10,
                    "PRAD_repression_evidence": 0,
                    "COAD_spearman_rho": 0.02,
                    "COAD_repression_evidence": 0,
                    "score_combined_placeholder": 99,
                },
                {
                    "mirna_name": "miRNA-21",
                    "gene_symbol": "PTEN",
                    "mirna_name_norm": "mir-21",
                    "gene_symbol_norm": "PTEN",
                    "transcript_id": "ENST000002",
                    "ts_best_contextpp": -0.10,
                    "ts_best_percentile": 44,
                    "clip_exp_sum": 0,
                    "clip_exp_max": 0,
                    "n_clip_sites": 0,
                    "has_seed_features": 1,
                    "best_seed_rank": 2,
                    "n_sites_6mer": 1,
                    "n_sites_7mer_a1": 0,
                    "n_sites_7mer_m8": 0,
                    "n_sites_8mer": 0,
                    "n_total_sites": 1,
                    "best_seed_class": "6mer",
                    "has_rnahybrid": 0,
                    "n_rnahybrid_sites": 0,
                    "best_mfe": -15.0,
                    "mfe_strength": 15.0,
                    "BRCA_spearman_rho": 0.05,
                    "BRCA_repression_evidence": 0,
                    "PRAD_spearman_rho": -0.22,
                    "PRAD_repression_evidence": 1,
                    "COAD_spearman_rho": -0.18,
                    "COAD_repression_evidence": 1,
                },
            ]
        )
        source_df.to_csv(source_path, index=False)

        rnaplfold_df = pd.DataFrame(
            [
                {
                    "mirna_name_normalized": "mir-210-3p",
                    "gene_symbol_normalized": "FGFRL1",
                    "transcript_id": "ENST000001",
                    "rnaplfold_seed_unpaired_prob": 0.45,
                    "rnaplfold_site_unpaired_prob": 0.41,
                    "rnaplfold_flank_unpaired_prob": 0.38,
                    "rnaplfold_seed_accessibility_score": 0.45,
                    "rnaplfold_site_accessibility_score": 0.41,
                    "rnaplfold_window_length": 80,
                    "rnaplfold_region_start": 15,
                    "rnaplfold_region_end": 22,
                },
                {
                    "mirna_name_normalized": "mir-210-3p",
                    "gene_symbol_normalized": "FGFRL1",
                    "transcript_id": "ENST000001",
                    "rnaplfold_seed_unpaired_prob": 0.62,
                    "rnaplfold_site_unpaired_prob": 0.57,
                    "rnaplfold_flank_unpaired_prob": 0.49,
                    "rnaplfold_seed_accessibility_score": 0.62,
                    "rnaplfold_site_accessibility_score": 0.57,
                    "rnaplfold_window_length": 80,
                    "rnaplfold_region_start": 40,
                    "rnaplfold_region_end": 47,
                },
            ]
        )
        rnaplfold_df.to_csv(rnaplfold_path, index=False)

        cmd = [
            str(PYTHON),
            str(REPO_ROOT / "scripts" / "build_clean_evidence_db.py"),
            "--input-path",
            str(source_path),
            "--rnaplfold-features",
            str(rnaplfold_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--limit",
            "2",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Smoke build failed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

        clean = pd.read_csv(output_path)
        report = report_path.read_text(encoding="utf-8")
        assert "mirtarbase_pos" not in clean.columns
        assert "score_combined_placeholder" not in clean.columns
        assert "rnaplfold_best_seed_unpaired_prob" in clean.columns
        assert "rnaplfold_mean_seed_unpaired_prob" in clean.columns
        assert "rnaplfold_n_sites_scored" in clean.columns
        assert float(clean.loc[0, "rnaplfold_best_seed_unpaired_prob"]) == 0.62
        assert abs(float(clean.loc[0, "rnaplfold_mean_seed_unpaired_prob"]) - 0.535) < 1e-9
        assert int(clean.loc[0, "rnaplfold_n_sites_scored"]) == 2
        assert int(clean.loc[0, "rnaplfold_n_accessible_sites"]) == 1
        assert "clip_any_support" in clean.columns
        assert "tcga_n_supported_contexts" in clean.columns
        assert "has_seed_evidence" in clean.columns
        assert "has_rnaplfold_evidence" in clean.columns
        assert "Unit of analysis: transcript-level miRNA-target candidate" in report
        assert "Leakage-column validation: PASS" in report
        assert "Confirmation that no miRTarBase evidence columns are present: PASS" in report
        assert "Rows: 2" in report
        print("smoke_build_clean_evidence_db: OK")


if __name__ == "__main__":
    main()
