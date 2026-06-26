# Clean Evidence Schema

This branch adds a separate clean evidence build path for later blinded evaluation. It does not change the current app or production retrieval behavior.

## Retained Evidence Categories

1. Sequence complementarity
2. Thermodynamic stability
3. Sequence conservation
4. Target site accessibility
5. Functional binding evidence
6. Functional repression

## Current Source Tables Inspected

The repo currently points to these evidence-building assets:

- `data/processed/mirassist_backend_features.parquet`
- `data/processed/evidence_interactions.parquet`
- `mirassist_evidence_pairs_full.csv`
- `mirassist_evidence_pairs_test.csv`
- `scripts/export_evidence_for_supabase.py`
- `scripts/export_evidence_full_for_supabase.py`
- `backend/retrieval.py`
- `evaluation/utils.py`

The clean build script logs the discovered input files and source columns at runtime so the actual environment-specific schema is captured in the validation report.

## Unit Of Analysis

The current source evidence is treated as `transcript-level miRNA-target candidate` when `transcript_id` is present. That is the current expected output level of `scripts/build_clean_evidence_db.py`.

RNAplfold output is naturally site-level. In the current clean build path, site-level RNAplfold rows are aggregated back to transcript-level rows using transcript-compatible join keys such as:

- `evidence_row_id` when available
- otherwise `mirna_name_normalized`, `gene_symbol_normalized`, and `transcript_id`

The build script logs the inferred final unit of analysis at runtime and the validation report repeats it explicitly.

## Clean Output Columns

### Identifiers

- `evidence_row_id`
- `mirna_name`
- `mirna_name_normalized`
- `gene_symbol`
- `gene_symbol_normalized`
- `transcript_id`
- `site_id`
- `chrom`
- `start`
- `end`
- `strand`
- `site_sequence`
- `window_sequence`

### Sequence Complementarity

- `seed_match_type`
- `is_8mer`
- `is_7mer_m8`
- `is_7mer_a1`
- `is_6mer`
- `seed_pairing_score`
- `n_seed_sites`
- `best_seed_site_type`

Source mapping:

- `best_seed_class` -> `seed_match_type`, `best_seed_site_type`
- `best_seed_rank` -> `seed_pairing_score`
- `n_total_sites` -> `n_seed_sites`
- `n_sites_8mer` / `n_sites_7mer_m8` / `n_sites_7mer_a1` / `n_sites_6mer` -> boolean site-type flags

Directionality:

- `seed_pairing_score`: source-dependent ordinal score, higher currently means stronger canonical pairing in the existing repo
- `n_seed_sites`: higher is stronger support
- `is_*`: boolean support

### Thermodynamic Stability / RNAhybrid

- `rnahybrid_mfe`
- `rnahybrid_mfe_best_site`
- `rnahybrid_site_start`
- `rnahybrid_site_end`
- `rnahybrid_alignment`
- `rnahybrid_seed_mfe`
- `rnahybrid_strength`

Source mapping:

- `best_mfe` -> `rnahybrid_mfe`, `rnahybrid_mfe_best_site`
- `best_site_start_by_mfe` / `best_site_end_by_mfe` -> RNAhybrid best-site coordinates
- `best_8mer_mfe`, `best_7mer_m8_mfe` -> closest available seed-site MFE proxy for `rnahybrid_seed_mfe`
- `rnahybrid_alignment` -> carried through if present
- `rnahybrid_strength` is derived as `-1 * rnahybrid_mfe`

Directionality:

- `rnahybrid_mfe`: lower or more negative means stronger predicted hybridization
- `rnahybrid_strength`: higher means stronger predicted hybridization

### Sequence Conservation / TargetScan

- `targetscan_context_score`
- `targetscan_context_score_percentile`
- `targetscan_aggregate_context_score`
- `targetscan_conserved_site`
- `targetscan_pct`
- `targetscan_branch_length_score`

Source mapping:

- `ts_best_contextpp` -> `targetscan_context_score`
- `ts_best_percentile` -> `targetscan_context_score_percentile`
- `ts_weighted_context_score` -> `targetscan_aggregate_context_score` if available
- `targetscan_conserved_site` -> carried through if available
- `targetscan_pct` -> carried through if available
- `targetscan_branch_length_score` -> carried through if available

Directionality:

- `targetscan_context_score`: lower or more negative means stronger TargetScan support
- `targetscan_context_score_percentile`: higher means stronger relative support
- `targetscan_aggregate_context_score`: source-dependent; preserve the raw TargetScan aggregate semantics from the input while avoiding leakage-flagged naming

### Target Site Accessibility / RNAplfold

Current clean-table output columns for the transcript-level build:

- `rnaplfold_best_seed_unpaired_prob`
- `rnaplfold_mean_seed_unpaired_prob`
- `rnaplfold_best_site_unpaired_prob`
- `rnaplfold_mean_site_unpaired_prob`
- `rnaplfold_best_flank_unpaired_prob`
- `rnaplfold_mean_flank_unpaired_prob`
- `rnaplfold_n_sites_scored`
- `rnaplfold_n_accessible_sites`

Source mapping:

- `scripts/calc_rnaplfold_accessibility.py` produces site-level RNAplfold columns:
  - `rnaplfold_seed_unpaired_prob`
  - `rnaplfold_site_unpaired_prob`
  - `rnaplfold_flank_unpaired_prob`
  - `rnaplfold_seed_accessibility_score`
  - `rnaplfold_site_accessibility_score`
  - `rnaplfold_window_length`
  - `rnaplfold_region_start`
  - `rnaplfold_region_end`
- The clean builder then aggregates those site-level rows back to transcript-level summaries.
- `best_*` summary columns use the maximum unpaired probability across scored sites.
- `mean_*` summary columns use the arithmetic mean unpaired probability across scored sites.
- `rnaplfold_n_accessible_sites` counts the number of scored sites with `rnaplfold_seed_unpaired_prob >= 0.5` by default.

Directionality:

- Higher unpaired probability means more accessible and therefore stronger accessibility support.
- In the current script, `rnaplfold_seed_accessibility_score` and `rnaplfold_site_accessibility_score` are documented aliases for arithmetic-mean unpaired probabilities over those regions.

Note:

- Existing `best_local_au` / `best_local_au_by_mfe` columns are not treated as the final clean RNAplfold schema. The new branch keeps the builder ready for explicit RNAplfold-derived accessibility features.
- If a future source table becomes truly site-level, the builder can preserve site-level RNAplfold rows directly, but the current branch is documented and validated as transcript-level.

### Functional Binding Evidence / CLIP

- `clip_any_support`
- `clip_max_score`
- `clip_n_experiments`
- `clip_n_cell_lines`
- `clip_source`
- `encori_clip_score`

Source mapping:

- `support_encori`, `n_clip_sites`, `clip_exp_sum`, `clip_exp_max` -> `clip_any_support`
- `clip_exp_max` -> `clip_max_score`
- `clip_exp_sum` -> `clip_n_experiments` and `encori_clip_score`
- `clip_n_cell_lines` -> carried through if available
- `clip_source` is set to `ENCORI` when CLIP support is present

Directionality:

- `clip_any_support`: boolean support
- `clip_max_score`, `encori_clip_score`, `clip_n_experiments`, `clip_n_cell_lines`: higher means stronger or broader support

### Functional Repression / TCGA

- `BRCA_spearman_rho`
- `BRCA_repression_evidence`
- `BRCA_anticorrelated`
- `BRCA_support_tcga`
- `PRAD_spearman_rho`
- `PRAD_repression_evidence`
- `PRAD_anticorrelated`
- `PRAD_support_tcga`
- `COAD_spearman_rho`
- `COAD_repression_evidence`
- `COAD_anticorrelated`
- `COAD_support_tcga`
- `tcga_any_anticorrelated`
- `tcga_n_supported_contexts`
- `tcga_best_repression_evidence`
- `tcga_mean_spearman_rho`

Source mapping:

- Existing BRCA/PRAD/COAD `*_spearman_rho`
- Existing BRCA/PRAD/COAD `*_repression_evidence`
- Existing BRCA/PRAD/COAD `*_anticorrelated` or derived from negative rho when missing
- Existing BRCA/PRAD/COAD `*_support_tcga` or derived from anticorrelation/repression flags when missing
- Pan-cancer summary columns are derived only from BRCA, PRAD, and COAD

Directionality:

- `*_spearman_rho`: lower or more negative means stronger repression-consistent signal
- `*_repression_evidence`, `*_anticorrelated`, `*_support_tcga`: boolean support
- `tcga_n_supported_contexts`: higher means broader context support

## Evidence-Family Presence Flags

The clean table now includes:

- `has_seed_evidence`
- `has_rnahybrid_evidence`
- `has_targetscan_evidence`
- `has_rnaplfold_evidence`
- `has_clip_evidence`
- `has_tcga_evidence`

These are deterministic booleans derived from non-missing or nonzero evidence within each family. The validation report also summarizes:

- row counts and percentages for each family
- the number of rows with 0, 1, 2, 3, 4, 5, or 6 evidence families present

## Intentionally Removed Columns

The clean builder excludes:

- miRTarBase evidence and label fields such as `mirtarbase_pos`, `label_mirtarbase`, `mirtarbase_pmids`, and related auxiliary columns
- old manual placeholders such as `score_evidence_placeholder`, `score_structure_placeholder`, and `score_combined_placeholder`
- app/debug fields not needed for a clean evidence table
- duplicated support summaries when the underlying raw family columns are already kept
- model-generated or explanation-only fields

The validation report now explicitly fails leakage-column validation if any clean output column name contains:

- `mirtarbase`
- `validated`
- `label`
- `manual`
- `weighted`
- `old_score`
- `ground_truth`
- `heldout`

## Important Evaluation Note

miRTarBase is excluded from the clean evidence table on purpose. It is reserved for later blinded evaluation labels and should not be used as input evidence in this rebuild.
