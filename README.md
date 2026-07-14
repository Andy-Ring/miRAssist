<div align="center">

<img src="frontend/assets/miRAssist_logo.png" alt="miRAssist" width="440">

### A context-aware, evidence-integration framework for interpretable miRNA–target prioritization

miRAssist prioritizes candidate microRNA–target interactions (MTIs) by integrating six
families of experimental and computational evidence, ranking them with a backend XGBoost
model, and interpreting the results through a large language model (LLM) planner and
synthesis layer.

**Try it now (no install required):** hosted web app on Posit Connect Cloud →
**https://andy-ring-mirassist.share.connect.posit.cloud/**

Or install it as a **Claude skill** and integrate miRAssist into your Claude workflows.

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
| **Streamlit app** | OpenAI-hosted models | GitHub-hosted CSV snapshot (Supabase no longer required) |
| **Claude skill** | Claude itself (no OpenAI key) | GitHub-hosted CSV snapshot, cached locally |

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

### Run it locally

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

`mirassist_evidence_pairs.csv.gz` can be downloaded from the most recent GitHub release and stored at data/processed. If the file isn't present, the app instead
downloads the snapshot from the GitHub Release and caches it under
`~/.cache/mirassist`.

---

## Option B — Claude skill

The Claude skill wraps the same deterministic retrieval + XGBoost ranking core, but Claude
plays the planner and synthesizer roles. **No API keys are required**. On first use it downloads a one-time evidence snapshot from this repo's GitHub
Releases and caches it locally, so **end users configure nothing**. Skill assets live in
[`mirassist-skill/`](mirassist-skill/), with a pre-packaged installer at `mirassist.skill`.

### Install & use

1. **Download** [`mirassist_skill.zip`](https://github.com/Andy-Ring/miRAssist/releases)
   from the latest release.
2. In Claude, turn on **code execution** (Settings → Capabilities), then add the skill via
   **Settings → Skills → `+` → Create skill** and select the zip.
3. Start a new chat and ask a miRNA-target question in natural language, e.g.
   *"Which miRNAs regulate PTEN?"* or *"Which genes does miR-21 target to promote apoptosis
   in breast cancer?"*

New to Claude skills? A step-by-step walkthrough is in
[**INSTALL_SKILL.md**](INSTALL_SKILL.md).

The first query downloads the evidence snapshot (~106 MB) and caches it; later queries are
fast. Under the hood the skill runs a headless CLI you can also call directly:

```bash
python mirassist-skill/scripts/retrieve.py \
  --mirna "hsa-miR-21-5p" --tcga BRCA \
  --phenotype apoptosis --observed-change promoted --perturbation overexpression \
  --result-count 5 --pretty
```

See [`mirassist-skill/SKILL.md`](mirassist-skill/SKILL.md) for the full query grammar and
[`mirassist-skill/reference/`](mirassist-skill/reference/) for the QuerySpec schema and
output-column reference.

## Configuration reference

| Variable | Purpose | Typical value |
|---|---|---|
| `EVIDENCE_BACKEND` | Evidence source | `github` (default) / `parquet` (offline) / `postgres` / `rest` |
| `MIRASSIST_EVIDENCE_URL` | Snapshot URL (github mode) | `https://github.com/Andy-Ring/miRAssist/releases/download/v0.0.1/mirassist_evidence_pairs.csv.gz` |
| `MIRASSIST_CACHE_DIR` | Snapshot cache dir | `~/.cache/mirassist` |
| `MIRASSIST_EVIDENCE` | Local parquet path (offline mode) | `/path/to/evidence.parquet` |
| `JOBSTORE_BACKEND` | Job persistence (app) | `filesystem` / `postgres` |
| `MIRASSIST_USE_LEARNED_SCORE` | Enable XGBoost ranking | `1` |
| `MIRASSIST_LEARNED_SCORE_COLUMN` | Learned-score column | `learned_score_xgb_raw_v1` |
| `MIRASSIST_DEFAULT_RESULT_COUNT` | Ranked results shown | `5` |
| `OPENAI_API_KEY` | OpenAI key (app only) | `sk-...` |
| `MIRASSIST_PLANNER_MODEL` / `MIRASSIST_SYNTH_MODEL` | LLM models (app only) | `gpt-5.4-nano` / `gpt-5.4-mini` |

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
> interpretable miRNA-target prioritization.* In preparation 

- **Code:** https://github.com/Andy-Ring/miRAssist
- **App:** https://andy-ring-mirassist.share.connect.posit.cloud/

Released under the [MIT License](LICENSE) © 2026 Andrew Ring.
