# miRAssist

miRAssist now runs as a direct Streamlit app that keeps planner, retrieval, and ranking in-process.

## Deploying miRAssist on Posit Connect Cloud

Recommended first deployment architecture:

1. Posit Connect Cloud hosts the Streamlit app only.
2. Supabase/Postgres stores persistent jobs and optionally evidence data.
3. OpenAI-hosted models run the planner and synthesizer.
4. No Google Cloud Run backend is required for the first deployment.
5. No separate FastAPI backend is required.

Required Posit environment variables:

```bash
MIRASSIST_LLM_BACKEND=openai
OPENAI_API_KEY=...
MIRASSIST_PLANNER_MODEL=gpt-5.4-nano
MIRASSIST_SYNTH_MODEL=gpt-5.4-mini
MIRASSIST_SYNTH_MAX_TOKENS=2500
MIRASSIST_DEFAULT_K=10
MIRASSIST_DEFAULT_RESULT_COUNT=5
JOBSTORE_BACKEND=postgres
DATABASE_URL=...
EVIDENCE_BACKEND=postgres
EVIDENCE_TABLE=mirassist_evidence_pairs
```

Posit should run the app with:

```bash
streamlit run app.py
```

The repo also includes a local [uvicorn shim](C:\Users\andym\OneDrive - University of Georgia\Documents\miRAssist\uvicorn\__init__.py), which forces Streamlit onto its legacy websocket backend before the server boots. This is a deployment guard for Posit Connect Cloud, where the newer `uvicorn` websocket backend was returning `403 Forbidden` during the `/_stcore/stream` upgrade and leaving the app stuck on the loading skeleton.

Preferred Posit entrypoint: `app.py`.

Do not set the primary file to `frontend/app.py` unless necessary. If Posit is already configured to launch `frontend/app.py`, the app now bootstraps the repository root onto `sys.path` so sibling imports like `backend.config` still work.

The repo includes a root [requirements.txt](C:\Users\andym\OneDrive - University of Georgia\Documents\miRAssist\requirements.txt) for the hosted Streamlit path and intentionally avoids heavy local inference dependencies like `torch`.

## Local direct mode

This is the same execution path Posit should use:

```bash
JOBSTORE_BACKEND=filesystem
EVIDENCE_BACKEND=parquet
MIRASSIST_EVIDENCE=/path/to/evidence_pairs_tcga.parquet
MIRASSIST_LLM_BACKEND=openai
OPENAI_API_KEY=...
streamlit run app.py
```

In direct mode, Streamlit runs the workflow in-process via `backend.worker.run_query_job` and still writes job state through the shared jobstore layer.

The default UI is intentionally standalone:

- No backend connection box is shown in the app
- The sidebar includes `How to use miRAssist` and `About evidence` help sections
- Normal users see the ranked candidate chart plus `Planner output (QuerySpec)` and `Evidence shortlist`

## Using OpenAI-hosted models

miRAssist can use separate OpenAI-hosted models for planner and synthesizer. This is the recommended model backend for current deployments.

Example:

```bash
export MIRASSIST_LLM_BACKEND=openai
export OPENAI_API_KEY="..."
export MIRASSIST_PLANNER_MODEL="gpt-5.4-nano"
export MIRASSIST_SYNTH_MODEL="gpt-5.4-mini"
```

Notes:

- The planner defaults to the cheaper `gpt-5.4-nano`.
- The synthesizer defaults to the stronger `gpt-5.4-mini`.
- The synthesizer output budget defaults to `MIRASSIST_SYNTH_MAX_TOKENS=2500`.
- To switch synthesis later without code changes, set `MIRASSIST_SYNTH_MODEL=gpt-5.5`.
- Optional overrides include `MIRASSIST_OPENAI_BASE_URL`, `MIRASSIST_OPENAI_TIMEOUT`, `MIRASSIST_OPENAI_TEMPERATURE_PLANNER`, and `MIRASSIST_OPENAI_TEMPERATURE_SYNTH`.

## Evidence backend

Parquet remains the default evidence source for development:

```bash
EVIDENCE_BACKEND=parquet
MIRASSIST_EVIDENCE=/path/to/evidence_pairs_tcga.parquet
```

Postgres evidence mode is available with:

```bash
EVIDENCE_BACKEND=postgres
EVIDENCE_TABLE=mirassist_evidence_pairs
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME
```

If the evidence table has learned non-miRTarBase ranking columns populated, retrieval can prefer them with:

```bash
MIRASSIST_USE_LEARNED_SCORE=1
MIRASSIST_LEARNED_SCORE_COLUMN=learned_score_xgb_raw_v1
```

Rows with missing learned scores automatically fall back to the manual `retrieval_score`.

The Postgres evidence path is implemented as a direct table read, but the table still needs to contain the columns expected by the current retrieval logic. If the table is missing or unreadable, miRAssist now fails with a clear error mentioning `EVIDENCE_TABLE` and `DATABASE_URL`.

## Using learned XGBoost ranking in production

If `mirassist_evidence_pairs` includes precomputed learned-score columns, production retrieval can rank with the learned score while keeping the manual `retrieval_score` for fallback and debugging.

Required Supabase columns:

- `learned_score_xgb_raw_v1` or another configured learned-score column
- Optional metadata columns such as `learned_score_xgb_raw_nomissing_v1`, `learned_score_model_version`, `learned_score_feature_set`, and `learned_score_updated_at`

Recommended environment variables:

```bash
MIRASSIST_USE_LEARNED_SCORE=1
MIRASSIST_LEARNED_SCORE_COLUMN=learned_score_xgb_raw_v1
MIRASSIST_DB_CANDIDATE_LIMIT=1000
MIRASSIST_DEBUG_MAX_ROWS=100
```

Ranking behavior:

- If the configured learned-score column is present and non-null for a row, that value is used as the primary ranking score.
- If the learned score is missing for a row, miRAssist falls back to the manual `retrieval_score` for that row.
- If the configured learned-score column is absent entirely, miRAssist logs a warning and falls back to manual ranking without failing startup.
- In `EVIDENCE_BACKEND=postgres` mode, miRAssist fetches only a bounded candidate pool from Supabase and does not load the full evidence parquet or any XGBoost model artifact at app runtime.

## Evidence interpretation and feature percentiles

miRAssist now computes evidence feature percentiles across the full evidence table before synthesis. These annotations are added to the retrieved shortlist and stored in the job payload, so the LLM receives backend-computed evidence labels instead of inventing them.

- Percentiles are computed against the full evidence database, not just the current shortlist.
- Higher-is-better transformed features are used where possible, such as `ts_context_strength` instead of raw `ts_best_contextpp`, and `mfe_strength` instead of raw `best_mfe`.
- Percentile labels are deterministic:
  - `>= 95`: `exceptional`
  - `>= 90`: `very high`
  - `>= 75`: `high`
  - `>= 50`: `above average`
  - `>= 25`: `typical`
  - `< 25`: `low`
- Missing values remain `not available`.

The evidence cards now distinguish different evidence types:

- miRTarBase: curated prior functional support
- miRDB and TargetScan: computational prediction support
- CLIP / ENCORI: binding-associated support
- seed/site architecture: canonical site support
- RNAhybrid and MFE-derived features: structure-compatible support
- TCGA anticorrelation: context-specific repression support
- pathway filtering: deterministic membership in selected pathways only

The LLM is instructed to use these backend-provided values and labels exactly as given. It must not calculate percentiles, invent pathway membership, or add unsupported gene-phenotype claims.

Evidence breadth versus strength:

- `Evidence support count` measures breadth across distinct evidence categories.
- `Overall priority` considers both breadth and the strength of the underlying values.
- A candidate with fewer categories can still rank higher if those categories are strong.
- A candidate with more categories can still be exploratory if most values are weak or typical.

## Grounded pathway filtering

miRAssist now treats pathway and phenotype context as grounded database filters rather than LLM-generated gene annotations.

- The planner extracts phenotype and pathway intent only.
- A deterministic pathway resolver matches those terms against `data/processed/pathways/pathways.parquet`.
- Gene restriction is then applied using `data/processed/pathways/gene_to_pathways.parquet`.
- The LLM is not allowed to invent gene-to-phenotype or gene-to-pathway connections.
- Pathway behavior is filter-only.

If a user mentions apoptosis, proliferation, EMT, invasion, or another pathway or phenotype, miRAssist restricts candidates to genes in matching pathways before scoring.

## Supabase / Postgres job storage

Use Postgres-backed jobs with:

```bash
JOBSTORE_BACKEND=postgres
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME
```

The backend auto-creates a `mirassist_jobs` table with:

- `query_id text primary key`
- `payload jsonb not null`
- `status text`
- `stage text`
- `updated_at timestamp`
- `created_at timestamp`

If `JOBSTORE_BACKEND=postgres` is set without `DATABASE_URL`, the app falls back to filesystem storage for development.

By default, miRAssist uses `k=10` as the candidate pool passed to synthesis, then prints the top 5 ranked results from that pool.

- `k` is the number of evidence cards/candidates returned to the synthesizer after backend filtering and scoring.
- The final printed result count defaults to 5.
- Users can increase the candidate pool with the `Candidate pool size (k)` control.

## Environment templates

You can start from:

```bash
cp .env.example .env
```

Do not use `.env` for production secrets on Posit Connect Cloud. Set them through Posit environment variables and secrets instead.

## Smoke checks

Basic direct-mode import smoke:

```bash
python scripts/smoke_test_direct_mode_imports.py
```

Import-path smoke:

```bash
python scripts/smoke_test_import_paths.py
```

OpenAI backend smoke:

```bash
python scripts/smoke_test_openai_backend.py
```

Supabase/Postgres jobstore smoke:

```bash
python scripts/smoke_test_jobstore_postgres.py
```

Existing backend/API smoke coverage is still available in:

```bash
python -m unittest tests.test_smoke
```

Focused evidence interpretation smoke coverage is available in:

```bash
python -m unittest tests.test_evidence_features
```

## Remaining TODO

- Confirm the Supabase evidence table schema matches the columns expected by the current retrieval logic.
- Move long-running production execution to a queue/worker system later if you outgrow in-process direct execution.
