<div align="center">

# miRAssist

**Directed, evidence-grounded miRNA–target interaction predictions.**

miRAssist ranks candidate microRNA–mRNA (miRNA–target) interactions with a learned
XGBoost model on top of a curated, multi-source evidence database, and explains each
result in plain language.

**▶ Try it now (no install):** hosted web app on Posit Connect Cloud →
**https://andy-ring-mirassist.share.connect.posit.cloud/**

Or install it as a **Claude skill** and ask in natural language from inside Claude.

</div>

---

## Contents

- [What it is](#what-it-is)
- [How it works](#how-it-works)
- [Evidence sources](#evidence-sources)
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
                       (evidence DB)  (learned score)
```

1. **Planner** converts a natural-language question into a structured `QuerySpec`
   (entity, direction, cancer, phenotype/pathway intent, filters).
2. **Retrieval** pulls a bounded candidate pool for the queried miRNA/gene, normalizes
   miRNA names and mature arms, and applies grounded pathway filtering.
3. **Ranking** orders candidates by the precomputed learned XGBoost score
   (`learned_score_xgb_raw_v1`), falling back to the manual `retrieval_score` per row when
   the learned score is missing.
4. **Synthesizer** writes the explanation using only backend-computed values and labels.

The two deployments differ only in *who plays the planner/synthesizer* and *where the
evidence lives*:

| | Planner & synthesizer | Evidence source |
|---|---|---|
| **Streamlit app** | OpenAI-hosted models | Supabase (Postgres), live |
| **Claude skill** | Claude itself (no OpenAI key) | GitHub-hosted parquet snapshot, cached locally |

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

## Option A — Streamlit web app

The interactive UI shows the ranked-candidate chart, the planner output (`QuerySpec`), and
the evidence shortlist.

### Use the hosted app (no setup)

The app is live on Posit Connect Cloud — just open it and start asking questions:

**https://andy-ring-mirassist.share.connect.posit.cloud/**

### Run it yourself

```bash
git clone https://github.com/Andy-Ring/miRAssist.git
cd miRAssist
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your OpenAI key and the Supabase evidence database:

```bash
MIRASSIST_LLM_BACKEND=openai
OPENAI_API_KEY=sk-...                     # your OpenAI key
MIRASSIST_PLANNER_MODEL=gpt-5.4-nano
MIRASSIST_SYNTH_MODEL=gpt-5.4-mini

EVIDENCE_BACKEND=postgres
EVIDENCE_TABLE=mirassist_evidence_pairs
DATABASE_URL=postgresql://postgres.vkbkeernvifbefwmkaae:YOUR_PASSWORD@aws-1-us-west-2.pooler.supabase.com:6543/postgres
MIRASSIST_USE_LEARNED_SCORE=1
```

Get the database password from the Supabase dashboard (**Project Settings → Database**).
Never commit a real password — keep it in `.env` (already git-ignored) or your host's
secret manager. Then run:

```bash
streamlit run app.py
```

**Deploying on Posit Connect Cloud:** use `app.py` as the entry point and set the
environment variables above through Posit's environment/secrets UI rather than committing
`.env`. The bundled `uvicorn` shim pins Streamlit to its legacy websocket backend to avoid
`403` upgrade failures on Posit's newer builds; heavy local-inference dependencies (e.g.
`torch`) are intentionally excluded.

---

## Option B — Claude skill

The Claude skill wraps the same deterministic retrieval + XGBoost ranking core, but Claude
plays the planner and synthesizer roles — so **no OpenAI key and no database password are
needed**. On first use it downloads a one-time evidence snapshot from this repo's GitHub
Releases and caches it locally, so **end users configure nothing**. Skill assets live in
[`mirassist-skill/`](mirassist-skill/), with a pre-packaged installer at `mirassist.skill`.

### 1. Install the skill

Download **[`mirassist.skill`](mirassist.skill)** from this repository, open it in Claude
(desktop app or web), and click **Save skill**.

### 2. Install dependencies

```bash
pip install -r mirassist-skill/requirements.txt   # pandas, numpy, pyarrow, requests
```

### 3. Use it

Ask Claude a miRNA-target question in natural language, e.g.
*"Which miRNAs regulate PTEN?"* or *"Which genes does miR-21 target to promote apoptosis in
breast cancer?"* Claude builds the query, runs the retrieval core, and returns a ranked,
evidence-grounded answer.

The first query downloads the evidence snapshot (~106 MB) from the GitHub Release and
caches it; subsequent queries are fast. Under the hood the skill calls a headless CLI you
can also run directly:

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

1. **Export the evidence table to parquet** with the bundled helper (guarantees the
   learned-score and percentile columns retrieval needs):

   ```bash
   python mirassist-skill/scripts/export_snapshot.py --from-supabase \
     --database-url "postgresql://USER:PASSWORD@HOST:5432/postgres" \
     --out mirassist_evidence_pairs.parquet
   ```

   It prints `OK: all key retrieval columns are present` (or warns what's missing). To
   convert an existing CSV instead, use `--from-csv path.csv`.

2. **Attach the parquet to a public GitHub Release** of your fork.

3. **Set the asset URL** in
   [`mirassist-skill/scripts/skill_settings.json`](mirassist-skill/scripts/skill_settings.json):

   ```json
   { "evidence_parquet_url": "https://github.com/Andy-Ring/miRAssist/releases/download/v0.0.1/mirassist_evidence_pairs.parquet" }
   ```

The snapshot is read-only public data — no keys required for end users. When the data
changes, upload a new release and update the URL; caches re-download automatically.
Retrieval reads only the rows for the queried miRNA/gene (parquet predicate pushdown), so
the full table is never loaded into memory.

> **Note on sandboxed environments.** GitHub *release-asset* downloads must be reachable
> from wherever the skill runs. If a locked-down environment blocks them, the CLI returns a
> clear "failed to download" error; alternatives are `EVIDENCE_BACKEND=parquet` +
> `MIRASSIST_EVIDENCE=/local.parquet` (fully offline) or `EVIDENCE_BACKEND=rest` against
> Supabase (needs the Supabase host allowlisted).

---

## Configuration reference

| Variable | Purpose | Typical value |
|---|---|---|
| `EVIDENCE_BACKEND` | Evidence source | `github` (skill default) / `postgres` (app) / `parquet` (offline) / `rest` |
| `MIRASSIST_EVIDENCE_URL` | GitHub Release snapshot URL (skill default) | `https://github.com/Andy-Ring/miRAssist/releases/download/v0.0.1/mirassist_evidence_pairs.parquet` |
| `MIRASSIST_EVIDENCE` | Local parquet path (offline mode) | `/path/to/evidence.parquet` |
| `EVIDENCE_TABLE` | Evidence table name | `mirassist_evidence_pairs` |
| `MIRASSIST_USE_LEARNED_SCORE` | Enable XGBoost ranking | `1` |
| `MIRASSIST_LEARNED_SCORE_COLUMN` | Learned-score column | `learned_score_xgb_raw_v1` |
| `MIRASSIST_DEFAULT_RESULT_COUNT` | Ranked results shown | `5` |
| `DATABASE_URL` | Supabase Postgres connection (app) | `postgresql://…@…pooler.supabase.com:6543/postgres` |
| `OPENAI_API_KEY` | OpenAI key (app only) | `sk-...` |
| `MIRASSIST_PLANNER_MODEL` / `MIRASSIST_SYNTH_MODEL` | LLM models (app only) | `gpt-5.4-nano` / `gpt-5.4-mini` |

---

## Repository layout

```
miRAssist/
├── app.py                     # Streamlit entry point (hosted on Posit)
├── frontend/                  # Streamlit UI
├── backend/                   # planner, retrieval, XGBoost ranking, pathways, synthesizer
├── data/processed/            # evidence + pathway data
├── evaluation/                # benchmarking & paper-figure pipeline
├── mirassist-skill/           # Claude skill (SKILL.md, retrieval CLI, snapshot loader,
│                              #   export_snapshot.py, bundled core)
├── mirassist.skill            # packaged, installable skill
├── requirements.txt           # Streamlit-app dependencies
└── .env.example               # configuration template
```

---

## Citation & license

Released under the [MIT License](LICENSE) © 2026 Andrew Ring.

If you use miRAssist in your research, please cite this repository.
