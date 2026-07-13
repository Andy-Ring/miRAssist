# miRAssist output columns

Each object in `candidates` describes one ranked (miRNA, gene) pair. Use these values
verbatim in your answer. Do not compute new percentiles or invent values.

## Identity & ranking
- `rank` - 1-based position in the returned list.
- `mirna_name`, `gene_symbol` - the interaction pair.
- `retrieval_rank_score` / `learned_score_used` - the value the pair was ranked by. This is
  the learned XGBoost score when present, otherwise the manual `retrieval_score` fallback.
- `score_column_used` - which column supplied the rank score for this row
  (`learned_score_xgb_raw_v1` = XGBoost model; `retrieval_score` = manual fallback).
- `_learned_score_missing` - 1 if this row fell back to the manual score.
- `retrieval_score` - the deterministic manual composite score (support + TargetScan + CLIP
  + miRDB + TCGA + optional structure), always present.

## Evidence breadth
- `support_count` - number of distinct evidence categories supporting the pair.
- `evidence_family_count` - count of evidence families present.
- `overall_evidence_support_percentile` - breadth percentile vs. the full database.

## Evidence-type flags / signals
- `mirtarbase_pos` (1/0) - curated functional support in miRTarBase (a validated prior).
- `support_targetscan`, `support_mirdb`, `support_encori`, `support_rnahybrid` (1/0).
- `mirdb_best_score` - miRDB target prediction score (0-100).
- `ts_context_strength` (higher = stronger) / `ts_best_contextpp` (raw, more negative =
  stronger) - TargetScan context++ score.
- `clip_exp_sum` - summed CLIP/ENCORI binding experiment support.
- `best_seed_class` - strongest seed match type (8mer > 7mer-m8 > 7mer-a1 > 6mer).
- `n_total_sites` - number of predicted target sites.

## Per-family support percentiles (0-100, vs. full database)
Interpret with labels: >=95 exceptional, >=90 very high, >=75 high, >=50 above average,
>=25 typical, <25 low, missing = not available.
- `sequence_complementarity_support_percentile`
- `thermodynamic_stability_support_percentile`
- `sequence_conservation_support_percentile`
- `target_site_accessibility_support_percentile`
- `functional_binding_support_percentile`
- `functional_repression_support_percentile`

## Cancer (TCGA) context - only when `--tcga` is set (BRCA, COAD, PRAD)
- `{TCGA}_spearman_rho` - miRNA-gene expression correlation; negative = anticorrelated
  (consistent with repression).
- `{TCGA}_anticorrelated` (1/0), `{TCGA}_repression_evidence` (1/0) - context repression
  support in that cohort.
- `{TCGA}_pair_expressed` (1/0) - both partners expressed in the cohort.

## Pathway grounding
- `pathway_selected_gene` (1/0) - gene is a member of a matched pathway for the requested
  phenotype. This is the ONLY sanctioned gene-pathway claim.
- `pathway_selected_names` - the specific matched pathway names for that gene.

## Top-level fields (outside `candidates`)
- `ranking.ranking_mode` - e.g. `learned:learned_score_xgb_raw_v1` or `manual`.
- `pathway_selection` - which pathways/genes were used and the inferred target role.
- `arm_interpretation_note` - which mature arm was assumed when none was specified.
- `warnings`, `no_candidates_explanation` - surface these to the user when present.
