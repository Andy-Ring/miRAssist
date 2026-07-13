<div align="center">

# miRAssist

**Directed, evidence-grounded miRNA–target interaction predictions.**

miRAssist ranks candidate microRNA–mRNA (miRNA–target) interactions with a learned
XGBoost model on top of a curated, multi-source evidence database, and explains each
result in plain language. Use it as a hosted **Streamlit web app** or install it as a
**Claude skill** and query it in natural language from inside Claude.

</div>

---

## Contents

- [What it is](#what-it-is)
- [How it works](#how-it-works)
- [Evidence sources](#evidence-sources)
- [The Supabase database](#the-supabase-database)
- [Option A — Streamlit web app](#option-a--streamlit-web-app)
- [Option B — Claude skill](#option-b--claude-skill)
- [Configuration reference](#configuration-reference)
- [Repository layout](#repository-layout)
- [Citation & license](#citation--license)

---

## What it is

miRAssist answers two core questions:

- **miRNA → targets:** *"Which genes does miR-21 target in breast cancer?"*
- **gene → miRNAs:** *"Which miRNAs regulate PTEN?"*

For each query it retrieves candidate interactions from an evidence database, ranks them
with a learned XGBoost model (with a transparent fallback to a manual composite score),
and returns a short, grounded explanation of *why* each candidate ranks where it does.
Queries can be filtered by cancer type (TCGA cohort) and by phenotype/pathway context
such as apoptosis, proliferation, EMT, invasion, or migration.

Every reported score, percentile, and pathway membership comes from the database — the
language layer is instructed never to invent numbers or gene–phenotype relationships.

## How it works

```
Question ─► Planner ─► Retrieval ─► XGBoost ranking ─► Synthesizer ─► Grounded answer
            (LLM)      (Supabase)    (learned score)     (LLM)
```

1. **Planner** converts a natural-language question into a structured `QuerySpec`
   (entity, direction, cancer, phenotype/pathway intent, filters).
2. **Retrieval** pulls a bounded candidate pool from the evidence table, normalizes miRNA
   names and mature arms, and applies grounded pathway filtering.
3. **Ranking** orders candidates by the precomputed learned XGBoost score
   (`learned_score_xgb_raw_v1`), falling back to the manual `retrieval_score` per row when
   the learned score is missing.
4. **Synthesizer** writes the explanation using only backend-computed values and labels.

In the **Streamlit app**, the planner and synthesizer run on OpenAI-hosted models. In the
**Claude skill**, Claude itself performs the planner and synthesizer roles, so no OpenAI
key is required — only read access to the database.

## Evidence sources

Each candidate integrates, where available:

| Evidence family | Source | Signal |
|---|---|---|
| Curated functional support | miRTarBase | validated interactions |
| Computational prediction | TargetScan, miRDB | seed/context prediction |
| Binding | CLIP / ENCORI | crosslinking support |
| Site architecture | seed / site type | 6mer → 8mer canonical sites |
| Thermodynamics | RNAhybrid, MFE | duplex stability |
| Accessibility | RNAplfold | target-site openness |
| Context repression | TCGA (BRCA, COAD, PRAD) | expression anticorrelation |

Feature percentiles are computed against the full database and reported with fixed labels
(`≥95` exceptional, `≥90` very high, `≥75` high, `≥50` above average, `≥25` typical,
`<25` low).

---

## The Supabase database

Both the app and the skill read candidate evidence from a hosted **Supabase (Postgres)**
database. The connection string (Supabase transaction pooler) is:

```
postgresql://postgres.vkbkeernvifbefwmkaae:[YOUR-PASSWORD]@aws-1-us-west-2.pooler.supabase.com:6543/postgres
```

- **Evidence table:** `mirassist_evidence_pairs`
- **Password:** replace `[YOUR-PASSWORD]` with the database password. In the Supabase
  dashboard, go to **Project Settings → Database → Connection string** (or reset the
  password under **Database → Reset database password**). Use a **read-only** role for
  distribution wherever possible.

Set it as an environment variable:

```bash
export DATABASE_URL="postgresql://postgres.vkbkeernvifbefwmkaae:YOUR_ACTUAL_PASSWORD@aws-1-us-west-2.pooler.supabase.com:6543/postgres"
export EVIDENCE_BACKEND=postgres
export EVIDENCE_TABLE=mirassist_evidence_pairs
```

> **Security note:** never commit a real password. Keep it in `.env` (already
> git-ignored) or your host's secret manager. Share only the pooler path above with the
> `[YOUR-PASSWORD]` placeholder intact.

---

## Option A — Streamlit web app

The Streamlit app provides the full interactive UI: ranked-candidate chart, planner output
(`QuerySpec`), and the evidence shortlist.

### 1. Install

```bash
git clone https://github.com/<your-org>/miRAssist.git
cd miRAssist
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

Copy the template and fill in your secrets:

```bash
cp .env.example .env
```

Then edit `.env` so it contains, at minimum:

```bash
MIRASSIST_LLM_BACKEND=openai
OPENAI_API_KEY=sk-...                     # your OpenAI key
MIRASSIST_PLANNER_MODEL=gpt-5.4-nano
MIRASSIST_SYNTH_MODEL=gpt-5.4-mini

EVIDENCE_BACKEND=postgres
EVIDENCE_TABLE=mirassist_evidence_pairs
DATABASE_URL=postgresql://postgres.vkbkeernvifbefwmkaae:YOUR_ACTUAL_PASSWORD@aws-1-us-west-2.pooler.supabase.com:6543/postgres

JOBSTORE_BACKEND=postgres                 # or filesystem for local dev
MIRASSIST_USE_LEARNED_SCORE=1
MIRASSIST_LEARNED_SCORE_COLUMN=learned_score_xgb_raw_v1
```

### 3. Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints and enter a question such as
*"What does miR-21 target in breast cancer?"*

### Deploying (Posit Connect Cloud)

The app runs cleanly on Posit Connect Cloud with `app.py` as the entry point. Set the
environment variables above through Posit's environment/secrets UI rather than committing
`.env`. The bundled `uvicorn` shim pins Streamlit to its legacy websocket backend to avoid
`403` upgrade failures on Posit's newer builds. `pandas`, `sqlalchemy`, and `psycopg` are
included; heavy local-inference dependencies (e.g. `torch`) are intentionally excluded.

---

## Option B — Claude skill

The Claude skill wraps the same deterministic retrieval + XGBoost ranking core, but Claude
plays the planner and synthesizer roles — so **no OpenAI key is needed**. On first use it
downloads a one-time evidence snapshot from this repo's GitHub Releases and caches it
locally, so **end users configure nothing** and it works in sandboxed environments where
the live database host isn't reachable. Skill assets live in
[`mirassist-skill/`](mirassist-skill/), and a pre-packaged installer is provided as
`mirassist.skill`.

### 1. Install the skill

Download **`mirassist.skill`** from this repository (or the Releases page), open it in
Claude (desktop app or web), and click **Save skill**.

Alternatively, clone the repo and zip the folder yourself:

```bash
cd mirassist-skill && zip -r ../mirassist.skill . -x '*.pyc' -x '*__pycache__*'
```

### 2. Install dependencies

The only end-user step is the light runtime (no OpenAI, no torch, no database password):

```bash
pip install -r mirassist-skill/requirements.txt   # pandas, numpy, pyarrow, requests
```

The evidence snapshot downloads automatically on first query — see
[Publishing your own copy](#publishing-your-own-copy) below if you are forking the project.

### 3. Use it

Ask Claude a miRNA-target question in natural language, e.g.
*"Which genes does miR-21 target to promote apoptosis in breast cancer?"* Claude builds the
query, runs the retrieval core, and returns a ranked, evidence-grounded answer.

Under the hood the skill calls a headless CLI you can also run directly:

```bash
python mirassist-skill/scripts/retrieve.py \
  --mirna "hsa-miR-21-5p" --tcga BRCA \
  --phenotype apoptosis --observed-change promoted --perturbation overexpression \
  --result-count 5 --pretty
```

See [`mirassist-skill/SKILL.md`](mirassist-skill/SKILL.md) for the full query grammar and
[`mirassist-skill/reference/`](mirassist-skill/reference/) for the QuerySpec schema and
output-column reference.

### Publishing your own copy

If you fork miRAssist, publish the evidence snapshot once so the skill can fetch it:

1. **Export the Supabase evidence table to parquet**, keeping the learned-score
   (`learned_score_xgb_raw_v1`) and precomputed `*_percentile` columns:

   ```python
   import pandas as pd, sqlalchemy as sa
   e = sa.create_engine("postgresql+psycopg://USER:PASSWORD@HOST:5432/postgres")
   pd.read_sql("select * from public.mirassist_evidence_pairs", e) \
     .to_parquet("mirassist_evidence_pairs.parquet", index=False)
   ```

2. **Attach the parquet to a public GitHub Release** of your fork.

3. **Set the asset URL** in
   [`mirassist-skill/scripts/skill_settings.json`](mirassist-skill/scripts/skill_settings.json):

   ```json
   { "evidence_parquet_url": "https://github.com/<you>/miRAssist/releases/download/v1.0-evidence/mirassist_evidence_pairs.parquet" }
   ```

The snapshot is read-only public data — no keys required for end users. When the data
changes, upload a new release and update the URL; caches re-download automatically.
Retrieval reads only the rows for the queried miRNA/gene (parquet predicate pushdown), so
the full table is never loaded into memory.

**Optional live-database modes.** `EVIDENCE_BACKEND=parquet` + `MIRASSIST_EVIDENCE=/path.parquet`
runs fully offline against a local file. `EVIDENCE_BACKEND=rest` queries Supabase live via
its REST API with a publishable `anon` key (`supabase_url` + `supabase_anon_key` in
skill_settings.json, plus a read-only RLS `SELECT` policy) — but only where your environment
allowlists the Supabase host.

---

## Configuration reference

| Variable | Purpose | Typical value |
|---|---|---|
| `EVIDENCE_BACKEND` | Evidence source | `github` (skill default) / `rest` / `postgres` (app) / `parquet` (dev) |
| `MIRASSIST_EVIDENCE_URL` | GitHub Release snapshot URL (skill default mode) | `https://github.com/.../mirassist_evidence_pairs.parquet` |
| `MIRASSIST_EVIDENCE` | Local parquet path (offline mode) | `/path/to/evidence.parquet` |
| `MIRASSIST_SUPABASE_URL` | Supabase project URL (skill REST mode) | `https://vkbkeernvifbefwmkaae.supabase.co` |
| `MIRASSIST_SUPABASE_ANON_KEY` | Publishable anon key (skill REST mode) | `sb_publishable_...` |
| `DATABASE_URL` | Supabase Postgres connection (app) | pooler string above |
| `EVIDENCE_TABLE` | Evidence table name | `mirassist_evidence_pairs` |
| `MIRASSIST_USE_LEARNED_SCORE` | Enable XGBoost ranking | `1` |
| `MIRASSIST_LEARNED_SCORE_COLUMN` | Learned-score column | `learned_score_xgb_raw_v1` |
| `MIRASSIST_DB_CANDIDATE_LIMIT` | Bounded candidate pool size | `1000` |
| `MIRASSIST_DEFAULT_K` | Candidate pool passed to synthesis | `10` |
| `MIRASSIST_DEFAULT_RESULT_COUNT` | Ranked results shown | `5` |
| `JOBSTORE_BACKEND` | Job persistence (app only) | `postgres` / `filesystem` |
| `MIRASSIST_LLM_BACKEND` | LLM backend (app only) | `openai` |
| `OPENAI_API_KEY` | OpenAI key (app only) | `sk-...` |
| `MIRASSIST_PLANNER_MODEL` | Planner model (app only) | `gpt-5.4-nano` |
| `MIRASSIST_SYNTH_MODEL` | Synthesizer model (app only) | `gpt-5.4-mini` |

For local development without Supabase, set `EVIDENCE_BACKEND=parquet` and
`MIRASSIST_EVIDENCE=/path/to/evidence_interactions.parquet`.

---

## Repository layout

```
miRAssist/
├── app.py                     # Streamlit entry point
├── frontend/                  # Streamlit UI
├── backend/                   # planner, retrieval, ranking, pathways, synthesizer
├── data/processed/            # evidence + pathway data (parquet)
├── evaluation/                # benchmarking & paper-figure pipeline
├── mirassist-skill/           # Claude skill (SKILL.md, retrieval CLI, bundled core)
├── mirassist.skill            # packaged, installable skill
├── requirements.txt           # Streamlit-app dependencies
└── .env.example               # configuration template
```

---

## Citation & license

Released under the [MIT License](LICENSE) © 2026 Andrew Ring.

If you use miRAssist in your research, please cite this repository.
