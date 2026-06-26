# Clean Evidence Evaluation

This evaluation path uses the fixed clean evidence input:

- `data/processed/mirassist_clean_evidence.parquet`

The clean evidence table is treated as a validated transcript-level evaluation input. It is not rebuilt inside these evaluation scripts.

## Core Principles

- miRTarBase is used only for label construction.
- miRTarBase is never used as an evidence feature.
- Feature columns are checked and the run fails if any feature name contains:
  - `mirtarbase`
  - `validated`
  - `label`
  - `manual`
  - `weighted`
  - `old_score`
  - `ground_truth`
  - `heldout`
- The evaluation compares only the six intended evidence families:
  1. Sequence complementarity / seed regions
  2. Thermodynamic stability / RNAhybrid
  3. Sequence conservation / TargetScan
  4. Target-site accessibility / RNAplfold
  5. Functional binding / CLIP-seq
  6. Functional repression / TCGA BRCA, PRAD, COAD

## Backend Models Compared

- Logistic Regression
- XGBoost
- SVM
- MLP
- Naive Bayes

## Split Strategy

The default split strategy groups by `mirna_name_normalized`, which treats each miRNA-to-target ranking problem as a grouped query and reduces train/test leakage across identical miRNA contexts.

## Metrics

Primary ranking metrics:

- Recall at 1
- Recall at 3
- Recall at 5
- Recall at 10
- Precision at 1
- Precision at 3
- Precision at 5
- Precision at 10

Secondary classification metrics:

- AUROC
- PR-AUC

This clean-evidence evaluation intentionally does **not** use:

- MRR
- random-baseline enrichment

## Scripts

- `scripts/evaluation/run_backend_model_comparison.py`
- `scripts/evaluation/run_evidence_family_ablation.py`
- `scripts/evaluation/run_external_model_comparison.py`
- `scripts/evaluation/plot_evaluation_results.py`
- `scripts/evaluation/eval_utils.py`

## Reproducible Workflow

### 1. Backend model comparison

```bash
python scripts/evaluation/run_backend_model_comparison.py \
  --evidence data/processed/mirassist_clean_evidence.parquet \
  --labels evaluation/data/heldout_mirtarbase_labels.parquet
```

### 2. Leave-one-evidence-family-out validation

```bash
python scripts/evaluation/run_evidence_family_ablation.py \
  --evidence data/processed/mirassist_clean_evidence.parquet \
  --labels evaluation/data/heldout_mirtarbase_labels.parquet
```

### 3. External and individual evidence-family comparison

```bash
python scripts/evaluation/run_external_model_comparison.py \
  --evidence data/processed/mirassist_clean_evidence.parquet \
  --labels evaluation/data/heldout_mirtarbase_labels.parquet \
  --best-model-metadata evaluation/clean_evidence_eval/models/best_backend_model_metadata.json \
  --external-root evaluation/clean_evidence_eval/external_models
```

### 4. Regenerate publication-ready figures

```bash
python scripts/evaluation/plot_evaluation_results.py
```

## Step 3: External And Individual Evidence-Family Comparison

Step 3 evaluates only the Step 1 test split and compares:

- the selected final miRAssist backend model
- six single-family clean-evidence scores
- TargetScan from the clean evidence table
- row-aligned published-model files for miRDB, miRanda, RNA22, and DIANA-MicroT

Held-out labels are attached by `evidence_row_id == eval_row_id` when those row IDs are available, with normalized-key fallback only if row IDs are missing.

### Score Direction Choices

- Final miRAssist model: higher predicted probability means stronger interaction support.
- Sequence complementarity: stronger seed classes and larger site-support values increase the standardized score.
- Thermodynamic stability: more negative RNAhybrid MFE values are sign-flipped so higher is stronger.
- Sequence conservation / TargetScan: if raw context score is used, it is sign-flipped because more negative is stronger; percentile-style columns are used directly when chosen.
- Target-site accessibility / RNAplfold: higher unpaired probability means stronger accessibility support.
- Functional binding / CLIP: higher CLIP score or broader experiment support means stronger support.
- Functional repression / TCGA: more negative repression-consistent Spearman rho values are sign-flipped so higher is stronger.
- External aligned models: missing predictions are treated as score `0` and do not cause rows to be dropped.

### External Inputs

- `evaluation/clean_evidence_eval/external_models/mirdb/parsed/mirdb_scores_aligned_to_evidence.csv.gz`
- `evaluation/clean_evidence_eval/external_models/miranda/parsed/miranda_scores_aligned_to_evidence.csv.gz`
- `evaluation/clean_evidence_eval/external_models/rna22/parsed/rna22_scores_aligned_to_evidence.csv.gz`
- `evaluation/clean_evidence_eval/external_models/diana_microt/parsed/diana_microt_scores_aligned_to_evidence.csv.gz`

### Step 3 Outputs

- `evaluation/clean_evidence_eval/results/external_model_comparison_metrics.csv`
- `evaluation/clean_evidence_eval/results/external_model_score_status.csv`
- `evaluation/clean_evidence_eval/results/external_model_comparison_long.csv`
- `evaluation/clean_evidence_eval/figures/external_model_comparison_recall_at_10.png`
- `evaluation/clean_evidence_eval/figures/external_model_comparison_precision_at_10.png`
- `evaluation/clean_evidence_eval/figures/external_model_comparison_pr_auc.png`
- `evaluation/clean_evidence_eval/figures/external_model_comparison_auroc.png`
- `evaluation/clean_evidence_eval/figures/evidence_family_ablation_pr_curve.png`
- `evaluation/clean_evidence_eval/figures/evidence_family_ablation_roc_curve.png`

### External Model Curve Plots

These publication-style plots include only:

- the final miRAssist model
- miRDB
- TargetScan
- DIANA-MicroT
- miRanda
- RNA22

They do not include the evidence-family-only scores.

Inputs:

- `data/processed/mirassist_clean_evidence.parquet`
- `evaluation/data/heldout_mirtarbase_labels.parquet`
- `evaluation/clean_evidence_eval/models/best_backend_model_metadata.json`
- `evaluation/clean_evidence_eval/models/best_backend_model.pkl`
- `evaluation/clean_evidence_eval/results/external_model_comparison_metrics.csv`
- the row-aligned external score files under `evaluation/clean_evidence_eval/external_models/`

Score columns and direction choices:

- miRAssist: score from `best_backend_model.pkl`, higher predicted probability is stronger
- miRDB: `mirdb_score`, higher is stronger
- TargetScan: `-targetscan_context_score`, because more negative raw context score is stronger
- DIANA-MicroT: `diana_microt_score`, higher is stronger
- miRanda: `miranda_best_score`, higher is stronger
- RNA22: `rna22_best_energy_strength`, higher is stronger

Run:

```bash
python scripts/evaluation/plot_external_model_curves.py \
  --evidence data/processed/mirassist_clean_evidence.parquet \
  --labels evaluation/data/heldout_mirtarbase_labels.parquet \
  --best-model-metadata evaluation/clean_evidence_eval/models/best_backend_model_metadata.json \
  --best-model-pickle evaluation/clean_evidence_eval/models/best_backend_model.pkl \
  --external-root evaluation/clean_evidence_eval/external_models \
  --step3-metrics evaluation/clean_evidence_eval/results/external_model_comparison_metrics.csv \
  --output-root evaluation/clean_evidence_eval
```

Outputs:

- `evaluation/clean_evidence_eval/figures/external_models_roc_curve.png`
- `evaluation/clean_evidence_eval/figures/external_models_precision_recall_curve.png`
- `evaluation/clean_evidence_eval/figures/external_models_recall_at_5.png`
- `evaluation/clean_evidence_eval/results/external_model_curves_summary.csv`

### miRDB-Style RNA-seq Validation

This standalone validation asks whether higher prediction scores correspond to stronger observed mRNA repression after miRNA overexpression, following the style of the miRDB RNA-seq evaluation.

Input files:

- `evaluation/data/miRDB_RNAseq_data.xlsx`
- `data/processed/mirassist_clean_evidence.parquet`
- `evaluation/clean_evidence_eval/models/best_backend_model_metadata.json`
- `evaluation/clean_evidence_eval/models/best_backend_model.pkl`
- the Step 3 aligned external score files under `evaluation/clean_evidence_eval/external_models/`

Score direction:

- miRAssist: `mirassist_xgboost_score`, higher means stronger predicted repression
- TargetScan: `targetscan_score = -targetscan_context_score`
- miRDB: `mirdb_score`, higher is stronger
- DIANA-MicroT: `diana_microt_score`, higher is stronger
- miRanda: `miranda_best_score`, higher is stronger
- RNA22: `rna22_best_energy_strength`, higher is stronger

Run:

```bash
python scripts/evaluation/run_mirdb_rnaseq_validation.py \
  --rnaseq evaluation/data/miRDB_RNAseq_data.xlsx \
  --evidence data/processed/mirassist_clean_evidence.parquet \
  --best-model-metadata evaluation/clean_evidence_eval/models/best_backend_model_metadata.json \
  --best-model-pickle evaluation/clean_evidence_eval/models/best_backend_model.pkl \
  --external-root evaluation/clean_evidence_eval/external_models \
  --output-root evaluation/clean_evidence_eval
```

Main outputs:

- `evaluation/clean_evidence_eval/results/mirdb_rnaseq_expression_changes.csv.gz`
- `evaluation/clean_evidence_eval/results/mirdb_rnaseq_model_scores_gene_level.csv.gz`
- `evaluation/clean_evidence_eval/results/mirdb_rnaseq_validation_summary.csv`
- `evaluation/clean_evidence_eval/results/mirdb_rnaseq_validation_matched_pairs.csv.gz`
- `evaluation/clean_evidence_eval/results/mirdb_rnaseq_validation_native_prediction_summary.csv`
- `evaluation/clean_evidence_eval/figures/mirdb_rnaseq_spearman_repression_strength.png`
- `evaluation/clean_evidence_eval/figures/mirdb_rnaseq_topk_mean_repression.png`
- `evaluation/clean_evidence_eval/figures/mirdb_rnaseq_topk_mean_log2fc.png`

## Outputs

Results:

- `evaluation/clean_evidence_eval/results/backend_model_comparison_metrics.csv`
- `evaluation/clean_evidence_eval/results/backend_model_comparison_predictions.parquet`
- `evaluation/clean_evidence_eval/results/best_backend_model_summary.txt`
- `evaluation/clean_evidence_eval/results/evidence_family_ablation_metrics.csv`
- `evaluation/clean_evidence_eval/results/evidence_family_ablation_predictions.parquet`
- `evaluation/clean_evidence_eval/results/external_model_comparison_metrics.csv`
- `evaluation/clean_evidence_eval/results/external_model_score_status.csv`
- `evaluation/clean_evidence_eval/results/external_model_comparison_long.csv`

Figures are saved as both PNG and PDF under:

- `evaluation/clean_evidence_eval/figures/`
