#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/ar58064/miRAssist}"
RUN_NAME="${RUN_NAME:-paper_learned_ranker_v3}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/evaluation/runs/${RUN_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST_MAX_MIRNAS="${MANIFEST_MAX_MIRNAS:-}"
RANDOM_BASELINE_N_PERMUTATIONS="${RANDOM_BASELINE_N_PERMUTATIONS:-1000}"
LEARNED_RANKER_CV_FOLDS="${LEARNED_RANKER_CV_FOLDS:-5}"
LEARNED_RANKER_TEST_SIZE="${LEARNED_RANKER_TEST_SIZE:-0.2}"
RESUME=0

if [[ "${1:-}" == "--resume" ]]; then
  RESUME=1
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--resume]"
  exit 1
fi

LOG_DIR="${RUN_ROOT}/logs"
TABLES_DIR="${RUN_ROOT}/tables"
FIGURES_DIR="${RUN_ROOT}/figures"
PAPER_FIGURES_DIR="${RUN_ROOT}/paper_figures"
MODELS_DIR="${RUN_ROOT}/models"
REPORTS_DIR="${RUN_ROOT}/reports"
JSON_DIR="${TABLES_DIR}/json"
BLINDED_DIR="${TABLES_DIR}/blinded"
COLLECTED_DIR="${TABLES_DIR}/collected"
BASELINE_METRICS_DIR="${REPORTS_DIR}/baseline_metrics"
RANDOM_BASELINE_DIR="${REPORTS_DIR}/random_baseline"
ABLATION_DIR="${REPORTS_DIR}/ablation_comparison"
XGB_TRUE_DIR="${REPORTS_DIR}/learned_ranker_xgboost_raw_missing_true"
XGB_FALSE_DIR="${REPORTS_DIR}/learned_ranker_xgboost_raw_missing_false"
MODEL_MATRIX_DIR="${REPORTS_DIR}/learned_ranker_model_matrix"

mkdir -p \
  "${LOG_DIR}" \
  "${TABLES_DIR}" \
  "${FIGURES_DIR}" \
  "${PAPER_FIGURES_DIR}" \
  "${MODELS_DIR}" \
  "${REPORTS_DIR}" \
  "${JSON_DIR}" \
  "${BLINDED_DIR}" \
  "${COLLECTED_DIR}" \
  "${BASELINE_METRICS_DIR}" \
  "${RANDOM_BASELINE_DIR}" \
  "${ABLATION_DIR}" \
  "${XGB_TRUE_DIR}" \
  "${XGB_FALSE_DIR}" \
  "${MODEL_MATRIX_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

fail_with_log() {
  local log_file="$1"
  echo "[$(timestamp)] Step failed. Tail of ${log_file}:"
  tail -n 80 "${log_file}" || true
  exit 1
}

run_logged_step() {
  local step_name="$1"
  local log_file="$2"
  shift 2
  echo "[$(timestamp)] ${step_name}"
  if "$@" >"${log_file}" 2>&1; then
    echo "[$(timestamp)] ${step_name} completed"
  else
    fail_with_log "${log_file}"
  fi
}

export MIRASSIST_EVAL_MODE=1
export MIRASSIST_USE_MIRTARBASE_EVIDENCE=0
export MIRASSIST_USE_LEARNED_SCORE=0
export EVIDENCE_BACKEND=parquet
export MIRASSIST_DISABLE_SYNTHESIS=1
export JOBSTORE_BACKEND=filesystem

SOURCE_EVIDENCE="${SOURCE_EVIDENCE:-}"
if [[ -z "${SOURCE_EVIDENCE}" ]]; then
  if [[ -f "${REPO_ROOT}/data/processed/evidence_interactions.parquet" ]]; then
    SOURCE_EVIDENCE="${REPO_ROOT}/data/processed/evidence_interactions.parquet"
  elif [[ -f "${REPO_ROOT}/data/processed/evidence_pairs_tcga.parquet" ]]; then
    SOURCE_EVIDENCE="${REPO_ROOT}/data/processed/evidence_pairs_tcga.parquet"
  else
    echo "Could not locate source evidence under ${REPO_ROOT}/data/processed."
    exit 1
  fi
fi

BLINDED_EVIDENCE="${BLINDED_DIR}/evidence_blinded_no_mirtarbase.parquet"
HELDOUT_LABELS="${BLINDED_DIR}/heldout_mirtarbase_labels.parquet"
MANIFEST_PATH="${TABLES_DIR}/eval_queries.csv"

run_logged_step \
  "Step 1: blinded evidence" \
  "${LOG_DIR}/step01_make_blinded_evidence.log" \
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/00_make_blinded_evidence.py" \
    --evidence "${SOURCE_EVIDENCE}" \
    --outdir "${BLINDED_DIR}"

export MIRASSIST_EVIDENCE="${BLINDED_EVIDENCE}"

MANIFEST_ARGS=(
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/01_make_eval_queries.py"
  --blinded-evidence "${BLINDED_EVIDENCE}"
  --labels "${HELDOUT_LABELS}"
  --out "${MANIFEST_PATH}"
)
if [[ -n "${MANIFEST_MAX_MIRNAS}" ]]; then
  MANIFEST_ARGS+=(--max-mirnas "${MANIFEST_MAX_MIRNAS}")
fi

run_logged_step \
  "Step 2: evaluation manifest" \
  "${LOG_DIR}/step02_make_eval_queries.log" \
  "${MANIFEST_ARGS[@]}"

MANIFEST_ROWS="$("${PYTHON_BIN}" -c "import pandas as pd, sys; p=sys.argv[1]; df=pd.read_parquet(p) if p.endswith('.parquet') else pd.read_csv(p); print(len(df))" "${MANIFEST_PATH}")"
echo "[$(timestamp)] Step 3: running ${MANIFEST_ROWS} manifest rows"

for (( idx=0; idx<MANIFEST_ROWS; idx++ )); do
  query_id="$(printf "eval_%05d" "${idx}")"
  query_log="${LOG_DIR}/step03_${query_id}.log"
  query_json="${JSON_DIR}/${query_id}.json"
  if [[ "${RESUME}" == "1" && -f "${query_json}" ]]; then
    echo "[$(timestamp)] Step 3: skipping ${query_id} because --resume is enabled"
    continue
  fi
  if ! "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/02_run_eval_query.py" \
      --manifest "${MANIFEST_PATH}" \
      --evidence "${BLINDED_EVIDENCE}" \
      --outdir "${JSON_DIR}" \
      --index "${idx}" >"${query_log}" 2>&1; then
    fail_with_log "${query_log}"
  fi
done

run_logged_step \
  "Step 4: collect rankings" \
  "${LOG_DIR}/step04_collect_eval_rankings.log" \
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/03_collect_eval_rankings.py" \
    --json-dir "${JSON_DIR}" \
    --labels "${HELDOUT_LABELS}" \
    --outdir "${COLLECTED_DIR}"

run_logged_step \
  "Step 5: baseline metrics" \
  "${LOG_DIR}/step05_compute_eval_metrics.log" \
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/04_compute_eval_metrics.py" \
    --rankings-long "${COLLECTED_DIR}/rankings_long.parquet" \
    --query-summary "${COLLECTED_DIR}/query_summary.parquet" \
    --outdir "${BASELINE_METRICS_DIR}"

run_logged_step \
  "Step 6: random baseline" \
  "${LOG_DIR}/step06_random_baseline.log" \
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/06_random_baseline.py" \
    --rankings "${COLLECTED_DIR}/rankings_long.parquet" \
    --query-summary "${COLLECTED_DIR}/query_summary.parquet" \
    --outdir "${RANDOM_BASELINE_DIR}" \
    --n-permutations "${RANDOM_BASELINE_N_PERMUTATIONS}"

run_logged_step \
  "Step 7: ablation comparison" \
  "${LOG_DIR}/step07_ablation_comparison.log" \
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/07_ablation_comparison.py" \
    --rankings "${COLLECTED_DIR}/rankings_long.parquet" \
    --query-summary "${COLLECTED_DIR}/query_summary.parquet" \
    --outdir "${ABLATION_DIR}" \
    --include-random-baseline "${RANDOM_BASELINE_DIR}"

run_logged_step \
  "Step 8: train xgboost raw with missingness" \
  "${LOG_DIR}/step08_train_xgb_raw_missing_true.log" \
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/08_train_learned_ranker.py" \
    --rankings "${COLLECTED_DIR}/rankings_long.parquet" \
    --query-summary "${COLLECTED_DIR}/query_summary.parquet" \
    --outdir "${XGB_TRUE_DIR}" \
    --models xgboost \
    --feature-set raw \
    --test-size "${LEARNED_RANKER_TEST_SIZE}" \
    --cv-folds "${LEARNED_RANKER_CV_FOLDS}" \
    --include-missingness-indicators true \
    --model-name xgb_raw_missing_true_v1 \
    --save-model-artifact "${MODELS_DIR}/xgb_raw_missing_true_v1.joblib"

run_logged_step \
  "Step 9: train xgboost raw without missingness" \
  "${LOG_DIR}/step09_train_xgb_raw_missing_false.log" \
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/08_train_learned_ranker.py" \
    --rankings "${COLLECTED_DIR}/rankings_long.parquet" \
    --query-summary "${COLLECTED_DIR}/query_summary.parquet" \
    --outdir "${XGB_FALSE_DIR}" \
    --models xgboost \
    --feature-set raw \
    --test-size "${LEARNED_RANKER_TEST_SIZE}" \
    --cv-folds "${LEARNED_RANKER_CV_FOLDS}" \
    --include-missingness-indicators false \
    --model-name xgb_raw_missing_false_v1 \
    --save-model-artifact "${MODELS_DIR}/xgb_raw_missing_false_v1.joblib"

echo "model,feature_set,missingness,outdir" > "${MODEL_MATRIX_DIR}/model_matrix_runs.csv"
for model in logistic xgboost; do
  for feature_set in raw components all; do
    for missingness in true false; do
      matrix_outdir="${MODEL_MATRIX_DIR}/${model}_${feature_set}_missing_${missingness}"
      mkdir -p "${matrix_outdir}"
      matrix_log="${LOG_DIR}/step10_${model}_${feature_set}_missing_${missingness}.log"
      if [[ "${RESUME}" == "1" && -f "${matrix_outdir}/learned_ranker_metrics_summary.csv" ]]; then
        echo "[$(timestamp)] Step 10: skipping ${model}/${feature_set}/missing=${missingness} because --resume is enabled"
      else
        if ! "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/08_train_learned_ranker.py" \
            --rankings "${COLLECTED_DIR}/rankings_long.parquet" \
            --query-summary "${COLLECTED_DIR}/query_summary.parquet" \
            --outdir "${matrix_outdir}" \
            --models "${model}" \
            --feature-set "${feature_set}" \
            --test-size "${LEARNED_RANKER_TEST_SIZE}" \
            --cv-folds "${LEARNED_RANKER_CV_FOLDS}" \
            --include-missingness-indicators "${missingness}" >"${matrix_log}" 2>&1; then
          fail_with_log "${matrix_log}"
        fi
      fi
      echo "${model},${feature_set},${missingness},${matrix_outdir}" >> "${MODEL_MATRIX_DIR}/model_matrix_runs.csv"
    done
  done
done

run_logged_step \
  "Step 10b: aggregate model matrix summaries" \
  "${LOG_DIR}/step10b_model_matrix_aggregate.log" \
  "${PYTHON_BIN}" -c "from pathlib import Path; import pandas as pd, sys; root=Path(sys.argv[1]); frames=[pd.read_csv(path).assign(summary_path=str(path)) for path in sorted(root.rglob('learned_ranker_metrics_summary.csv'))]; (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_csv(root / 'model_matrix_summary.csv', index=False)" \
  "${MODEL_MATRIX_DIR}"

run_logged_step \
  "Step 11: collect paper figures" \
  "${LOG_DIR}/step11_collect_paper_figures.log" \
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/10_collect_paper_figures.py" \
    --run-root "${RUN_ROOT}"

run_logged_step \
  "Step 12: paper results summary" \
  "${LOG_DIR}/step12_make_paper_results_summary.log" \
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/11_make_paper_results_summary.py" \
    --run-root "${RUN_ROOT}"

run_logged_step \
  "Step 13: validate paper evaluation run" \
  "${LOG_DIR}/step13_validate_paper_eval_run.log" \
  "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/scripts/09_validate_paper_eval_run.py" \
    --run-root "${RUN_ROOT}"

echo "[$(timestamp)] Paper learned-ranker evaluation completed at ${RUN_ROOT}"
