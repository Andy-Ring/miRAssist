#!/bin/bash
#SBATCH --job-name=mirassist_clean_eval_step3
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate miRAssist

python scripts/evaluation/run_external_model_comparison.py \
  --evidence data/processed/mirassist_clean_evidence.parquet \
  --labels evaluation/data/heldout_mirtarbase_labels.parquet \
  --best-model-metadata evaluation/clean_evidence_eval/models/best_backend_model_metadata.json \
  --external-root evaluation/clean_evidence_eval/external_models \
  --output-root evaluation/clean_evidence_eval
