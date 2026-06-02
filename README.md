# miRAssist

miRAssist supports two deployment styles while preserving the same planner, retrieval, prompt-bundle, and synthesizer workflow:

- Direct Streamlit mode for Posit Connect Cloud
- Optional FastAPI/API mode for a separated backend later

## Deploying miRAssist on Posit Connect Cloud

Recommended first deployment architecture:

1. Posit Connect Cloud hosts the Streamlit app only.
2. Supabase/Postgres stores persistent jobs and optionally evidence data.
3. OpenAI-hosted models run the planner and synthesizer.
4. No Google Cloud Run backend is required for the first deployment.
5. FastAPI/API mode remains available as an optional future architecture.

Required Posit environment variables:

```bash
MIRASSIST_APP_MODE=direct
MIRASSIST_LLM_BACKEND=openai
OPENAI_API_KEY=...
MIRASSIST_PLANNER_MODEL=gpt-5.4-nano
MIRASSIST_SYNTH_MODEL=gpt-5.4-mini
JOBSTORE_BACKEND=postgres
DATABASE_URL=...
EVIDENCE_BACKEND=postgres
EVIDENCE_TABLE=mirassist_evidence_pairs
```

Posit should run the app with:

```bash
streamlit run app.py
```

The repo includes a root [requirements.txt](C:\Users\andym\OneDrive - University of Georgia\Documents\miRAssist\requirements.txt) for the hosted Streamlit path and intentionally avoids heavy local inference dependencies like `torch`.

## Local direct mode

This is the same execution path Posit should use:

```bash
MIRASSIST_APP_MODE=direct
JOBSTORE_BACKEND=filesystem
EVIDENCE_BACKEND=parquet
MIRASSIST_EVIDENCE=/path/to/evidence_pairs_tcga.parquet
MIRASSIST_LLM_BACKEND=openai
OPENAI_API_KEY=...
streamlit run app.py
```

In direct mode, Streamlit runs the workflow in-process via `backend.worker.run_query_job` and still writes job state through the shared jobstore layer.

## Optional API mode

API mode preserves the current split frontend/backend flow:

- `POST /query`
- `GET /status/{query_id}`
- `GET /result/{query_id}`

Run it locally with:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --host 0.0.0.0 --port 7861
MIRASSIST_APP_MODE=api
BACKEND_URL=http://127.0.0.1:7861
streamlit run app.py
```

## Using OpenAI-hosted models

miRAssist can use separate OpenAI-hosted models for planner and synthesizer.

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

The Postgres evidence path is implemented as a direct table read, but the table still needs to contain the columns expected by the current retrieval logic. If the table is missing or unreadable, miRAssist now fails with a clear error mentioning `EVIDENCE_TABLE` and `DATABASE_URL`.

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

## App diagnostics

In direct mode, the Streamlit sidebar includes an app diagnostics panel showing:

- `app_mode`
- `llm_backend`
- `planner_model`
- `synth_model`
- `openai_configured`
- `jobstore_backend`
- `evidence_backend`
- `database_configured`
- whether `BACKEND_URL` is being used

## Environment templates

You can start from:

```bash
cp .env.example .env
```

Do not use `.env` for production secrets on Posit Connect Cloud. Set them through Posit environment variables and secrets instead.

## Optional Cloud Run / FastAPI backend

Cloud Run and the FastAPI backend remain in the repository as optional alternatives if you later want a separated backend service.

Artifacts kept for that path:

- [backend/app.py](C:\Users\andym\OneDrive - University of Georgia\Documents\miRAssist\backend\app.py)
- [Dockerfile](C:\Users\andym\OneDrive - University of Georgia\Documents\miRAssist\Dockerfile)
- [scripts/deploy_cloud_run.sh](C:\Users\andym\OneDrive - University of Georgia\Documents\miRAssist\scripts\deploy_cloud_run.sh)

That path is no longer required for the first Posit deployment.

## Smoke checks

Basic direct-mode import smoke:

```bash
python scripts/smoke_test_direct_mode_imports.py
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

## Remaining TODO

- Confirm the Supabase evidence table schema matches the columns expected by the current retrieval logic.
- Move long-running production execution to a queue/worker system later if you outgrow in-process direct execution.
- Keep the optional Cloud Run/API path only if you later want a split architecture.
