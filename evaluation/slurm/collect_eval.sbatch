#!/bin/bash

#SBATCH --job-name=mirassist-eval-collect
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=evaluation/runs/%x_%j.out
#SBATCH --error=evaluation/runs/%x_%j.err

set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-miRAssist}"
RUN_NAME="${RUN_NAME:-blinded_eval}"
RUN_ROOT="${RUN_ROOT:-evaluation/runs/${RUN_NAME}}"
LABELS="${LABELS:-evaluation/data/heldout_mirtarbase_labels.parquet}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

python evaluation/scripts/03_collect_eval_rankings.py \
  --json-dir "${RUN_ROOT}/json" \
  --labels "${LABELS}" \
  --outdir "${RUN_ROOT}/tables"

python evaluation/scripts/04_compute_eval_metrics.py \
  --rankings-long "${RUN_ROOT}/tables/rankings_long.parquet" \
  --query-summary "${RUN_ROOT}/tables/query_summary.parquet" \
  --outdir "${RUN_ROOT}/metrics"

python evaluation/scripts/05_make_eval_plots.py \
  --metrics-dir "${RUN_ROOT}/metrics" \
  --tables-dir "${RUN_ROOT}/tables" \
  --outdir "${RUN_ROOT}/figures"
