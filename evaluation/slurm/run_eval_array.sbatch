#!/bin/bash
# sbatch --array=0-N evaluation/slurm/run_eval_array.sbatch

#SBATCH --job-name=mirassist-eval
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=evaluation/runs/%x_%A_%a.out
#SBATCH --error=evaluation/runs/%x_%A_%a.err

set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-miRAssist}"
RUN_NAME="${RUN_NAME:-blinded_eval}"
MANIFEST="${MANIFEST:-evaluation/data/eval_queries.csv}"
BLINDED_EVIDENCE="${BLINDED_EVIDENCE:-evaluation/data/evidence_blinded_no_mirtarbase.parquet}"
RUN_ROOT="${RUN_ROOT:-evaluation/runs/${RUN_NAME}}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

mkdir -p "${RUN_ROOT}/json" "${RUN_ROOT}/logs"

export MIRASSIST_EVAL_MODE=1
export MIRASSIST_DISABLE_SYNTHESIS=1
export MIRASSIST_USE_MIRTARBASE_EVIDENCE=0
export EVIDENCE_BACKEND=parquet
export JOBSTORE_BACKEND=filesystem
export MIRASSIST_EVIDENCE="${BLINDED_EVIDENCE}"
export MIRASSIST_JOB_DIR="${RUN_ROOT}/jobs"

python evaluation/scripts/02_run_eval_query.py \
  --manifest "${MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --evidence "${BLINDED_EVIDENCE}" \
  --outdir "${RUN_ROOT}/json"
