# Variant A / RF v1 production backend

## Active production contract

- Schema/version: `mirassist_evidence_variant_a_rf_v1`
- Candidate universe: Variant A
- Candidate definition: canonical 3′ UTR seed site **and** at least one of TargetScan support, miRNA-specific CLIP support, or significant TCGA anticorrelation
- Rows: 280,917
- Retained miRTarBase known positives: 2,583
- Model: `mirassist_rf_variant_a_v1`
- Persisted score: `mirassist_model_score`
- Canonical runtime score: `mirassist_score`
- Frozen global rank: `mirassist_score_rank_within_mirna`
- Frozen within-miRNA percentile: `mirassist_score_percentile_within_mirna`
- Filtered/display rank: `mirassist_filtered_rank` (runtime/export only)

The **miRAssist score** is a relative prioritization score within the
evidence-supported Variant A candidate universe. It is not a probability that an
interaction is biologically true. The technical definition is: raw uncalibrated
random-forest positive-class vote fraction used solely as a relative prioritization
score; no biological probability interpretation.

## Production paths

- Active local table: `data/processed/mirassist_evidence_variant_a_rf_v1.parquet`
- Approved immutable source: `outputs/sequence_defined_candidates/variant_a_rf_v1_scored_release/tables/variant_a_rf_v1_scored_evidence_table.parquet`
- Approved model: `outputs/sequence_defined_candidates/variant_a_rf_v1_scored_release/model/mirassist_rf_variant_a_v1.joblib`
- Hosted table: `public.mirassist_evidence_variant_a_rf_v1`
- Optional active view: `public.mirassist_evidence_active`

Both active/source Parquet files must have SHA-256
`2fc1b25af55c22c7e44e4587ac586942c0b5f3eb47afe0e36ccf5cab0512a9ee`.
The model SHA-256 is
`c765ff90ef841d05e976f8948318cd644f60bbd94c1e3466eca197f35dceeb94`.

## Schema

The scored schema has exactly 130 columns. Columns 1–126 are byte/value-preserved
from the frozen premodel evidence table. The four appended fields are:

| Field | Type | Meaning |
|---|---|---|
| `mirassist_model_score` | double precision / float64 | Approved RF v1 relative prioritization score |
| `mirassist_model_version` | text | `mirassist_rf_variant_a_v1` on every row |
| `mirassist_score_rank_within_mirna` | bigint / int64 | Global deterministic rank in the complete Variant A list for that miRNA |
| `mirassist_score_percentile_within_mirna` | double precision / float64 | Rank-derived within-miRNA percentile, 1.0 is first |

`mirassist_xgboost_score` remains one of the original 126 fields and remains missing
in the RF release. It must never receive RF scores.

The complete generated schema is retained in the approved release at
`outputs/sequence_defined_candidates/variant_a_rf_v1_scored_release/schema/variant_a_rf_v1_scored_schema.csv`.

## Score loading and ranking

The production loader applies strict precedence:

1. Use populated `mirassist_model_score` with nonmissing `mirassist_model_version`.
2. Otherwise use `mirassist_xgboost_score` for an explicit legacy table.
3. Reject incomplete mixed model/legacy coverage.
4. If both fields are populated, compare overlap and raise on disagreement greater than `1e-12`; an agreeing overlap still reports a warning and uses the model score.

Default ordering is descending canonical score, descending
`overall_evidence_support_percentile`, descending `evidence_family_count`, then the
frozen global rank/stable biological identifiers. miRTarBase labels, train/test
partition, and XGBoost scores do not rank RF rows.

Filters never overwrite the frozen global rank. Results receive a separate contiguous
`mirassist_filtered_rank` after novelty, pathway, cancer/evidence, and support filters.

## Novel and empty-result behavior

Novel mode excludes candidates aligned to the retained miRTarBase known-positive set.
The RF score does not change. Remaining candidates must not be called definitively novel
or experimentally unvalidated.

An empty Variant A result uses:

> No candidates met the current evidence-supported Variant A eligibility criteria. This does not establish that the miRNA has no biological targets.

No Variant D or sequence-master fallback is permitted.

## Hosted database migration

1. Apply `supabase/migrations/20260810_variant_a_rf_v1.sql` to create the new table,
   fields, indexes, and metadata row without changing the legacy table.
2. Load the approved 130-column release into the empty versioned table.
3. Run `scripts/validate_variant_a_rf_v1_production.py` against the source and perform
   database row/schema/key/score/rank comparisons before cutover.
4. Apply `supabase/migrations/20260810_variant_a_rf_v1_cutover.sql` and configure
   `EVIDENCE_TABLE=mirassist_evidence_variant_a_rf_v1` (or the active view).
5. Run `scripts/smoke_variant_a_production.py` against the deployed table/export.

Database score precision is `double precision`, matching the frozen float64 values.
Indexes cover miRNA+score ordering, gene lookup, novel-mode labels, and common evidence/
cancer filters.

## Rollback

The legacy table, legacy local Parquet/CSV, and legacy model assets are retained.

Local rollback:

```bash
export EVIDENCE_BACKEND=parquet
export MIRASSIST_EVIDENCE=/scratch/ar58064/miRAssist/outputs/production_migration/20260810T152626Z/rollback/legacy_data/mirassist_evidence_pairs.parquet
export MIRASSIST_LEARNED_SCORE_COLUMN=mirassist_xgboost_score
```

Hosted rollback: apply
`supabase/migrations/20260810_variant_a_rf_v1_rollback.sql`, then point
`EVIDENCE_TABLE` to `mirassist_evidence_pairs` or `mirassist_evidence_active`.
Do not drop the staged RF table during rollback.

Rollback manifest:
`outputs/production_migration/20260810T152626Z/rollback/rollback_manifest.json`.

## Scientific limitations

- Evaluation is positive-unlabeled; miRTarBase supplies known positives, not confirmed negatives.
- The RF score is not a biological probability.
- Variant A is evidence-conditioned, not an exhaustive biological target universe.
- TargetScan/CLIP/TCGA eligibility indicators contribute materially to model performance.
- RF v1 does not support Variant D.
- MANE Select may omit isoform-specific interactions.
- TCGA anticorrelation is indirect evidence.
- CLIP does not necessarily prove direct targeting in every assay context.
