<div align="center">

<img src="frontend/assets/miRAssist_logo.png" alt="miRAssist" width="440">

### A context-aware, evidence-integration framework for interpretable miRNA–target prioritization

miRAssist prioritizes candidate microRNA–target interactions (MTIs) by integrating six
families of experimental and computational evidence, ranking them with a backend XGBoost
model, and interpreting the results through a large language model (LLM) planner and
synthesis layer — so each ranked candidate comes with the evidence behind it.

**▶ Try it now (no install):** hosted web app on Posit Connect Cloud →
**https://andy-ring-mirassist.share.connect.posit.cloud/**

Or install it as a **Claude skill** and ask in natural language from inside Claude.

</div>

---

## Contents

- [What it is](#what-it-is)
- [How it works](#how-it-works)
- [Evidence families](#evidence-families)
- [Option A — Streamlit web app](#option-a--streamlit-web-app)
- [Option B — Claude skill](#option-b--claude-skill)
- [Configuration reference](#configuration-reference)
- [Repository layout](#repository-layout)
- [Citation & license](#citation--license)

---

## What it is

Many MTI prediction tools return a score or ranked list without making the supporting
evidence easy to interpret, which makes it hard to tell *why* a target was prioritized or
whether the evidence is relevant to a specific biological question. miRAssist addresses
this: it ranks candidate MTIs *and* explains the evidence supporting each one, in the
biological context the user cares about.

It answers two core questions:

- **miRNA → targets:** *"Which genes does miR-21 target in breast cancer?"*
- **gene → miRNAs:** *"Which miRNAs regulate PTEN?"*

Under the hood, a transcript-level candidate database integrates six evidence families
across 577,118 potential MTIs. A backend XGBoost model — trained to prioritize interactions
resembling experimentally confirmed positives from miRTarBase — ranks the candidates, and
outperforms individual external prediction tools (TargetScan, miRDB, DIANA-microT, miRanda,
RNA22) at recovering held-out miRTarBase positives (AUROC 0.835 on the test set). A large
language model (LLM) planner and synthesis layer then lets users ask natural-language
questions and receive ranked candidates with evidence-grounded explanations, including
context-aware filtering by biological pathway (MSigDB) and cancer type (TCGA).

Every reported score, percentile, and pathway membership comes from the database — the
language layer is instructed never to invent numbers or gene–phenotype relationships.

## How it works

<div align="center">

<img src="frontend/assets/Figure5.png" alt="miRAssist framework overview" width="760">

</div>

1. **Planner** — an LLM converts a natural-language question into a structured database
   query (entity, direction, cancer context, pathway/phenotype intent, filters).
2. **Retrieval & ranking** — miRAssist pulls candidate MTIs for the query and ranks them by
   the backend XGBoost score (with a transparent fallback to a manual composite score when
   a learned score is unavailable).
3. **Context-aware filtering** — when a question names a pathway, the search is restricted
   to genes in matching MSigDB gene sets; when it names a cancer type, functional-repression
   evidence is filtered to that TCGA cohort (BRCA, COAD, PRAD).
4. **Synthesis** — an LLM writes an evidence-grounded explanation using only backend-computed
   values and labels.

The two deployments differ only in *who plays the planner and synthesis layer*:

| | Planner & synthesizer | Evidence source |
|---|---|---|
| **Streamlit app** | OpenAI-hosted models | GitHub-hosted parquet snapshot (Supabase no longer required) |
| **Claude skill** | Claude itself (no OpenAI key) | GitHub-hosted parquet snapshot, cached locally |

Both read the same evidence snapshot published on GitHub Releases, so **neither requires a
live database**. (Live Supabase Postgres/REST remains available as an option — see the
configuration reference.)

## Evidence families

miRAssist integrates six families of evidence spanning sequence, structure, and functional
measurements:

| Evidence family | Source | Captures |
|---|---|---|
| Sequence complementarity | miRNA & mRNA 3′UTR sequences | 8mer / 7mer-m8 / 7mer-A1 / 6mer seed matches |
| Sequence conservation | TargetScan | cross-species conservation of target sites |
| Thermodynamic stability | RNAhybrid | free energy of the miRNA–mRNA duplex |
| Target-site accessibility | RNAplfold | local unpaired probability of the target site |
| Functional binding | CLIP-seq (ENCORI) | measured miRNA–mRNA binding |
| Functional repression | TCGA (BRCA, COAD, PRAD) | miRNA–mRNA expression anticorrelation |

Experimentally confirmed interactions from **miRTarBase** are held out as ground-truth
labels for evaluation and are excluded from the model's input features.

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

The app needs only an OpenAI key — evidence is pulled from the GitHub snapshot and jobs are
stored on the local filesystem, so **no database is required**.

```bash
git clone https://github.com/Andy-Ring/miRAssist.git
cd miRAssist
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your OpenAI key (the evidence and job-storage defaults already work):

```bash
MIRASSIST_LLM_BACKEND=openai
OPENAI_API_KEY=sk-...
MIRASSIST_PLANNER_MODEL=gpt-5.4-nano
MIRASSIST_SYNTH_MODEL=gpt-5.4-mini

EVIDENCE_BACKEND=github     # downloads the evidence snapshot from GitHub Releases
JOBSTORE_BACKEND=filesystem # no database
```

Then run:

```bash
streamlit run app.py
```

On first query the app downloads the evidence snapshot (~106 MB) and caches it under
`~/.cache/mirassist`; later queries reuse the cache.

**Deploying on Posit Connect Cloud:** use `app.py` as the entry point and set `OPENAI_API_KEY`
through Posit's environment/secrets UI. The evidence snapshot downloads at runtime, so the
deployment needs outbound access to GitHub release assets and a writable cache dir (set
`MIRASSIST_CACHE_DIR` if the home directory isn't writable). The bundled `uvicorn` shim pins
Streamlit to its legacy websocket backend to avoid `403` upgrade failures on Posit's newer
builds; heavy local-inference dependencies (e.g. `torch`) are intentionally excluded.

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

If you fork miRAssist, publish the evidence snapshot once so both the app and skill can
fetch it:

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

3. **Point both entry points at the asset URL:** set `evidence_parquet_url` in
   [`mirassist-skill/scripts/skill_settings.json`](mirassist-skill/scripts/skill_settings.json)
   (skill) and `DEFAULT_EVIDENCE_URL` in [`backend/config.py`](backend/config.py) — or the
   `MIRASSIST_EVIDENCE_URL` env var — for the app.

The snapshot is read-only public data — no keys required for end users. When the data
changes, upload a new release and update the URL; caches re-download automatically.
Retrieval reads only the rows for the queried miRNA/gene (parquet predicate pushdown), so
the full table is never loaded into memory.

> **Note on sandboxed environments.** GitHub *release-asset* downloads must be reachable
> from wherever the code runs. If a locked-down environment blocks them, the tool returns a
> clear "failed to download" error; alternatives are `EVIDENCE_BACKEND=parquet` +
> `MIRASSIST_EVIDENCE=/local.parquet` (fully offline) or a live Supabase connection.

---

## Configuration reference

| Variable | Purpose | Typical value |
|---|---|---|
| `EVIDENCE_BACKEND` | Evidence source | `github` (default) / `parquet` (offline) / `postgres` / `rest` |
| `MIRASSIST_EVIDENCE_URL` | Snapshot URL (github mode) | `https://github.com/Andy-Ring/miRAssist/releases/download/v0.0.1/mirassist_evidence_pairs.parquet` |
| `MIRASSIST_CACHE_DIR` | Snapshot cache dir | `~/.cache/mirassist` |
| `MIRASSIST_EVIDENCE` | Local parquet path (offline mode) | `/path/to/evidence.parquet` |
| `JOBSTORE_BACKEND` | Job persistence (app) | `filesystem` / `postgres` |
| `MIRASSIST_USE_LEARNED_SCORE` | Enable XGBoost ranking | `1` |
| `MIRASSIST_LEARNED_SCORE_COLUMN` | Learned-score column | `learned_score_xgb_raw_v1` |
| `MIRASSIST_DEFAULT_RESULT_COUNT` | Ranked results shown | `5` |
| `OPENAI_API_KEY` | OpenAI key (app only) | `sk-...` |
| `MIRASSIST_PLANNER_MODEL` / `MIRASSIST_SYNTH_MODEL` | LLM models (app only) | `gpt-5.4-nano` / `gpt-5.4-mini` |
| `DATABASE_URL` | Live Supabase Postgres (optional/legacy) | `postgresql://…pooler.supabase.com:6543/postgres` |

---

## Repository layout

```
miRAssist/
├── app.py                     # Streamlit entry point (hosted on Posit)
├── frontend/                  # Streamlit UI
├── backend/                   # planner, retrieval, XGBoost ranking, pathways, synthesizer,
│                              #   snapshot loader (evidence_bootstrap, parquet_snapshot)
├── data/processed/            # pathway data
├── evaluation/                # benchmarking & paper-figure pipeline
├── mirassist-skill/           # Claude skill (SKILL.md, retrieval CLI, export_snapshot.py)
├── mirassist.skill            # packaged, installable skill
├── requirements.txt           # Streamlit-app dependencies
└── .env.example               # configuration template
```

---

## Citation & license

If you use miRAssist in your research, please cite:

> Ring A, Xi Y. *miRAssist: a context-aware, evidence-integration framework for
> interpretable miRNA-target prioritization.*

- **Database:** Zenodo — https://doi.org/10.5281/zenodo.21072247
- **Code:** https://github.com/Andy-Ring/miRAssist
- **App:** https://andy-ring-mirassist.share.connect.posit.cloud/

Released under the [MIT License](LICENSE) © 2026 Andrew Ring.
