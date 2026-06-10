# Evaluation Workflow

This directory adds an evaluation-only workflow for blinded ranking experiments. It does not change the default Posit app path.

## Scientific goal

The evaluation hides all miRTarBase evidence from retrieval, scoring, cards, and prompts, then asks whether miRAssist can still rank experimentally confirmed positives near the top of the candidate list.

## Guardrails

- Set `MIRASSIST_EVAL_MODE=1` for evaluation runs.
- Set `MIRASSIST_DISABLE_SYNTHESIS=1` for deterministic rank-only runs.
- Use a blinded local parquet file, not the production Supabase evidence store.
- Held-out miRTarBase labels are joined back only during collection and metric computation.

## 1. Create blinded evidence

```bash
python evaluation/scripts/00_make_blinded_evidence.py \
  --evidence /path/to/evidence_pairs_tcga.parquet \
  --outdir evaluation/data
```

Outputs:

- `evaluation/data/evidence_blinded_no_mirtarbase.parquet`
- `evaluation/data/heldout_mirtarbase_labels.parquet`
- `evaluation/data/blinding_audit.json`

The blinding step:

- drops miRTarBase-specific auxiliary columns
- neutralizes `mirtarbase_pos` / `label_mirtarbase` to zero if they must remain for compatibility
- recomputes `support_count` without miRTarBase evidence
- asserts that no non-neutralized miRTarBase-like columns remain

## 2. Create the query manifest

```bash
python evaluation/scripts/01_make_eval_queries.py \
  --blinded-evidence evaluation/data/evidence_blinded_no_mirtarbase.parquet \
  --labels evaluation/data/heldout_mirtarbase_labels.parquet \
  --out evaluation/data/eval_queries.csv \
  --k 1000 \
  --min-positive-count 1
```

This creates one `mirna_to_targets` query per miRNA with at least one held-out positive.

Important assumptions:

- The manifest uses `mirna_name` values as they appear in the evidence table.
- Default question format is `What genes are regulated by {mirna}?`
- Primary evaluation is no-pathway-filter, no-cancer-context, `novel=false`

## 3. Run one deterministic query locally

```bash
set MIRASSIST_EVAL_MODE=1
set MIRASSIST_DISABLE_SYNTHESIS=1
set EVIDENCE_BACKEND=parquet
set JOBSTORE_BACKEND=filesystem
set MIRASSIST_EVIDENCE=%CD%\\evaluation\\data\\evidence_blinded_no_mirtarbase.parquet

python evaluation/scripts/02_run_eval_query.py \
  --manifest evaluation/data/eval_queries.csv \
  --index 0 \
  --evidence evaluation/data/evidence_blinded_no_mirtarbase.parquet \
  --outdir evaluation/runs/local_eval/json
```

By default this builds `QuerySpec` directly and skips planner/synthesizer variability.

## 4. Submit the SLURM array

```bash
N=$(($(wc -l < evaluation/data/eval_queries.csv)-2))
sbatch --array=0-${N} evaluation/slurm/run_eval_array.sbatch
```

Indexing note:

- `SLURM_ARRAY_TASK_ID=0` maps to the first data row in the CSV
- the shell example subtracts 2 because `wc -l` includes the header row

Edit the variables at the top of the SLURM script before submission:

- `CONDA_ENV_NAME`
- `RUN_NAME`
- `BLINDED_EVIDENCE`
- `MANIFEST`
- `RUN_ROOT`

## 5. Collect rankings

```bash
python evaluation/scripts/03_collect_eval_rankings.py \
  --json-dir evaluation/runs/local_eval/json \
  --labels evaluation/data/heldout_mirtarbase_labels.parquet \
  --outdir evaluation/runs/local_eval/tables
```

Outputs:

- `rankings_long.parquet/csv`
- `query_summary.parquet/csv`

## 6. Compute metrics

```bash
python evaluation/scripts/04_compute_eval_metrics.py \
  --rankings-long evaluation/runs/local_eval/tables/rankings_long.parquet \
  --query-summary evaluation/runs/local_eval/tables/query_summary.parquet \
  --outdir evaluation/runs/local_eval/metrics
```

Outputs include:

- `metrics_summary.json`
- `metrics_by_query.csv/parquet`
- `recall_at_k.csv`
- `precision_at_k.csv`
- `topk_enrichment.csv`

## 7. Make plots

```bash
python evaluation/scripts/05_make_eval_plots.py \
  --metrics-dir evaluation/runs/local_eval/metrics \
  --tables-dir evaluation/runs/local_eval/tables \
  --outdir evaluation/runs/local_eval/figures
```

Plots:

- `recall_at_k.png`
- `precision_at_k.png`
- `histogram_best_positive_rank.png`
- `score_distribution_positive_vs_other.png`
- `topK_enrichment.png`

## Random baseline positive recovery

This step compares observed held-out positive recovery against random rankings within each miRNA/query candidate set.

```bash
python evaluation/scripts/06_random_baseline.py \
  --rankings evaluation/runs/local_eval/tables/rankings_long.parquet \
  --query-summary evaluation/runs/local_eval/tables/query_summary.parquet \
  --outdir evaluation/runs/local_eval/baseline_random \
  --n-permutations 1000 \
  --seed 2026
```

Outputs:

- `random_baseline_summary.json`
- `observed_vs_random_recall_at_k.csv`
- `observed_vs_random_precision_at_k.csv`
- `observed_vs_random_mrr.csv`
- `observed_query_metrics.parquet`
- `random_query_metrics.parquet`
- `observed_vs_random_recall_at_k.png`
- `observed_vs_random_precision_at_k.png`
- `observed_vs_random_mrr.png`

## Single-source and ablation comparison

This step reranks each query post hoc with individual evidence-source scores and leave-one-source-out variants, without rerunning retrieval.

```bash
python evaluation/scripts/07_ablation_comparison.py \
  --rankings evaluation/runs/<RUN_NAME>/tables/rankings_long.parquet \
  --query-summary evaluation/runs/<RUN_NAME>/tables/query_summary.parquet \
  --outdir evaluation/runs/<RUN_NAME>/ablation
```

Interpretation:

- `full` = original blinded miRAssist score
- `*_only` modes test individual evidence sources
- `no_*` modes test leave-one-source-out behavior
- structure-aware ablation requires structure contribution columns in `rankings_long`

## Structure-aware ablation

To evaluate seed/site and structure-aware evidence explicitly, rerun the evaluation export so `rankings_long` includes:

- `retrieval_seed_contrib`
- `retrieval_rnahybrid_contrib`
- `retrieval_local_au_contrib`
- `retrieval_structure_contrib`

These columns are exported diagnostically by default. If you also want structure to be part of the integrated `retrieval_score` for leave-one-out comparisons such as `no_structure`, rerun the evaluation with:

```bash
set MIRASSIST_USE_STRUCTURE_IN_SCORE=1
```

Then rerun:

```bash
python evaluation/scripts/02_run_eval_query.py ...
python evaluation/scripts/03_collect_eval_rankings.py ...
python evaluation/scripts/07_ablation_comparison.py \
  --rankings evaluation/runs/<RUN_NAME>/tables/rankings_long.parquet \
  --query-summary evaluation/runs/<RUN_NAME>/tables/query_summary.parquet \
  --outdir evaluation/runs/<RUN_NAME>/ablation
```

## Learned non-miRTarBase ranking score

This step trains learned ranking scores using only non-miRTarBase features. miRTarBase labels are used only as `y` labels during training and evaluation, and all miRTarBase-like evidence columns are excluded from the model feature matrix.

Design notes:

- the default split is grouped by miRNA to reduce leakage
- logistic regression and XGBoost are compared against component baselines such as miRDB, TargetScan, CLIP, structure, and combined scores
- the goal is positive recovery and enrichment near the top of each query-specific ranking, not true-negative classification

Example command:

```bash
python evaluation/scripts/08_train_learned_ranker.py \
  --rankings evaluation/runs/<RUN_NAME>/tables/rankings_long.parquet \
  --query-summary evaluation/runs/<RUN_NAME>/tables/query_summary.parquet \
  --outdir evaluation/runs/<RUN_NAME>/learned_ranker \
  --models logistic,xgboost \
  --split-mode group_by_mirna \
  --test-size 0.2 \
  --cv-folds 5 \
  --seed 2026
```

## Deploying learned scores to Supabase

To move a learned non-miRTarBase score into production retrieval:

1. Train and save a production artifact. miRTarBase labels are used only as training/evaluation labels, never as model features.

```bash
python evaluation/scripts/08_train_learned_ranker.py \
  --rankings evaluation/runs/<RUN_NAME>/tables/rankings_long.parquet \
  --query-summary evaluation/runs/<RUN_NAME>/tables/query_summary.parquet \
  --outdir evaluation/runs/<RUN_NAME>/learned_ranker \
  --models xgboost \
  --feature-set raw \
  --include-missingness-indicators true \
  --save-model-artifact evaluation/runs/<RUN_NAME>/artifacts/xgb_raw_v1.joblib \
  --model-name xgb_raw_v1
```

2. Score the production evidence table export with that artifact.

```bash
python evaluation/scripts/09_score_evidence_with_learned_model.py \
  --evidence /path/to/evidence_pairs_tcga.parquet \
  --model-artifact evaluation/runs/<RUN_NAME>/artifacts/xgb_raw_v1.joblib \
  --out evaluation/runs/<RUN_NAME>/scored_evidence.parquet \
  --score-column learned_score_xgb_raw_v1
```

3. Export a compact update file for Supabase.

```bash
python evaluation/scripts/10_export_learned_scores_for_supabase.py \
  --scored-evidence evaluation/runs/<RUN_NAME>/scored_evidence.parquet \
  --out-csv evaluation/runs/<RUN_NAME>/supabase_learned_scores.csv
```

4. Apply the migration in `supabase/migrations/add_learned_scores.sql`, upload/update the learned-score columns in `mirassist_evidence_pairs`, and enable:

```bash
set MIRASSIST_USE_LEARNED_SCORE=1
set MIRASSIST_LEARNED_SCORE_COLUMN=learned_score_xgb_raw_v1
```

In production retrieval, miRAssist will prefer the configured learned-score column when it is present, fall back row-by-row to the manual `retrieval_score` when the learned score is null, and fall back globally to manual ranking with a warning if the configured learned-score column is absent.

## Smoke checks

```bash
python evaluation/scripts/smoke_eval_blinding.py
python evaluation/scripts/smoke_eval_ranking.py
python evaluation/scripts/smoke_eval_collect.py
python evaluation/scripts/smoke_eval_ablation.py
python evaluation/scripts/smoke_eval_learned_ranker.py
python evaluation/scripts/smoke_score_evidence_with_learned_model.py
```

## Evidence-column assumptions

The current blinded support-count recomputation treats these as non-miRTarBase evidence families when available:

- `support_targetscan`
- `support_mirdb`
- `support_encori`
- `support_rnahybrid`
- seed/site fields such as `has_seed_features`, `n_total_sites`, `n_sites_*`, `best_seed_class`
- structure fields such as `has_rnahybrid`, `n_rnahybrid_sites`, `best_mfe`, `mfe_strength`
- TCGA evidence fields such as `*_support_tcga`, `*_anticorrelated`, `*_repression_evidence`, and negative `*_spearman_rho`

If your evidence schema differs, adjust `evaluation/utils.py` before large runs.
