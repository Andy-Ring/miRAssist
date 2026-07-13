---
name: mirassist
description: Predict and rank miRNA-target (miRNA-mRNA) interactions from grounded experimental and computational evidence. Use when a researcher asks which genes a microRNA targets, which miRNAs regulate a gene, or wants ranked, evidence-backed miRNA-target candidates - optionally filtered by cancer type (TCGA) or a phenotype/pathway such as apoptosis, proliferation, EMT, invasion, or migration. Triggers on mentions of miRNA/microRNA/miR-*, targets, seed sites, TargetScan/miRDB/ENCORI/CLIP, miRTarBase, or TCGA anticorrelation.
---

# miRAssist

miRAssist answers miRNA-target interaction questions by retrieving candidates from a
curated evidence database and ranking them with a learned XGBoost model. The database
integrates miRTarBase (curated functional support), TargetScan and miRDB (computational
prediction), CLIP/ENCORI (binding), seed/site architecture, RNAhybrid/MFE thermodynamics,
RNAplfold accessibility, and TCGA expression anticorrelation, with grounded pathway
membership for phenotype filtering.

**Your role.** This skill exposes only the deterministic scientific core as a CLI
(`scripts/retrieve.py`). No LLM runs inside it. You act as the two intelligent layers:

1. **Planner** - turn the researcher's question into CLI arguments.
2. **Synthesizer** - turn the CLI's JSON output into a grounded, plain-language answer.

The single hard rule: **use only the values, labels, and pathway memberships the CLI
returns. Never invent scores, percentiles, pathway/phenotype links, or gene functions
that are not in the output.**

---

## Setup

**For end users: nothing to configure.** On first use the skill downloads a one-time
evidence snapshot from the project's GitHub Releases (github.com is on the Cowork network
allowlist) and caches it locally; later queries reuse the cache. Install the light
dependencies once:

```bash
pip install -r requirements.txt      # pandas, numpy, pyarrow, requests
```

Then just ask miRNA-target questions in natural language.

### One-time setup for the skill author/publisher

If you are forking or re-publishing miRAssist, publish the evidence snapshot once:

1. **Export the Supabase evidence table to parquet**, including the learned-score
   (`learned_score_xgb_raw_v1`) and precomputed `*_percentile` columns:

   ```python
   import pandas as pd, sqlalchemy as sa
   e = sa.create_engine("postgresql+psycopg://USER:PASSWORD@HOST:5432/postgres")
   pd.read_sql("select * from public.mirassist_evidence_pairs", e) \
     .to_parquet("mirassist_evidence_pairs.parquet", index=False)
   ```

2. **Attach it to a public GitHub Release** of your miRAssist repo.

3. **Put the asset URL** in `scripts/skill_settings.json` under `evidence_parquet_url`:

   ```json
   { "evidence_parquet_url": "https://github.com/<you>/miRAssist/releases/download/v1.0-evidence/mirassist_evidence_pairs.parquet" }
   ```

The snapshot is read-only public data - no keys required for end users. When the data
changes, upload a new release (bump the tag) and update the URL; the cache re-downloads
automatically. Retrieval reads only the rows for the queried miRNA/gene (via parquet
predicate pushdown), so the full table is never loaded into memory.

### Other backends (optional)

- **Offline / development:** `EVIDENCE_BACKEND=parquet` + `MIRASSIST_EVIDENCE=/path.parquet`.
- **Live Supabase REST** (only if your environment allowlists the Supabase host):
  `EVIDENCE_BACKEND=rest`, with `supabase_url` + `supabase_anon_key` in skill_settings.json
  and a read-only RLS `SELECT` policy on the table.

If the snapshot cannot be fetched, the CLI returns a clear JSON error - relay it rather
than guessing.

---

## Step 1 - Plan: build the query

Read the researcher's question and decide the arguments. Do not ask clarifying questions
for details the CLI already defaults sensibly (e.g. mature arm, result count).

**Direction / mode**
- `mirna_to_targets` - the question centers on a miRNA ("what does miR-21 target?").
- `gene_to_mirnas` - the question centers on a gene ("which miRNAs regulate PTEN?").

**Entity**
- `--mirna` accepts any form: `hsa-miR-21-5p`, `miR-21`, `microRNA-21`. If no mature arm
  (`-3p`/`-5p`) is given, the tool defaults to `-5p` and returns an `arm_interpretation_note`
  you should pass along to the user. Only add an arm if the user specified one.
- `--gene` takes a gene symbol (uppercased automatically), e.g. `PTEN`, `PDCD4`.

**Cancer context** (optional): `--tcga BRCA|COAD|PRAD` (or `--cancer-name "breast cancer"`,
which the planner maps to a TCGA code). Only three cohorts have TCGA anticorrelation data:
BRCA, COAD, PRAD.

**Phenotype / pathway context** (optional but powerful). If the user mentions a phenotype
(apoptosis, proliferation, EMT, invasion, migration, energy metabolism), pass `--phenotype`.
If they describe an experiment, also pass the direction so the tool can infer whether the
relevant targets are positive or negative regulators:
- `--observed-change promoted|suppressed|increased|decreased|associated`
- `--perturbation overexpression|knockdown|inhibition|unknown`

The logic (miRNAs usually repress their targets): with **overexpression**, a *promoted/
increased* phenotype implies the direct targets are **negative regulators** of it; a
*suppressed/decreased* phenotype implies **positive regulators**. Knockdown/inhibition is
treated as ambiguous. When phenotype context is present, the tool restricts candidates to
genes in matching pathways (grounded, filter-only) before scoring.

**Other flags**
- `--novel` - only novel/unvalidated targets (excludes miRTarBase-positive pairs). Use when
  the user asks for "new", "exploratory", or "unvalidated" targets.
- `--min-support N` - minimum number of distinct evidence categories (default 1). Raise to 2
  if the user asks for "high-confidence" / "multiple lines of evidence".
- `--require-binding` - require TargetScan/ENCORI/miRDB binding evidence.
- `--require-expression` - require paired TCGA expression (needs `--tcga`).
- `--result-count N` - final results to return (default 5). `--k N` - candidate pool size
  (default 10).

## Step 2 - Retrieve: run the CLI

```bash
python3 scripts/retrieve.py --mirna "hsa-miR-21-5p" --tcga BRCA \
  --phenotype apoptosis --observed-change promoted --perturbation overexpression \
  --result-count 5 --pretty
```

For full control you can instead pass a complete QuerySpec JSON (schema in
`reference/queryspec_schema.md`): `python3 scripts/retrieve.py --queryspec-json spec.json`.

The command prints one JSON object: `query` (the resolved spec), `pathway_selection`
(which pathways/genes were used), `ranking` (which score column ranked the results),
`arm_interpretation_note`, `warnings`, and `candidates` (the ranked list).

## Step 3 - Synthesize: write the grounded answer

Read the JSON and write a concise, researcher-facing answer. Rules:

- **Ground everything.** Report scores, `support_count`, percentiles, and pathway
  membership exactly as returned. Do not compute your own percentiles or invent
  gene-phenotype or gene-pathway links. `pathway_selected_gene: 1` (with names in
  `pathway_selected_names`) is the only sanctioned pathway claim.
- **Explain the ranking basis.** State whether results were ranked by the learned XGBoost
  score (`ranking.ranking_mode` like `learned:learned_score_xgb_raw_v1`) or fell back to
  the manual retrieval score (`learned:retrieval_score` / `manual`), and note if some rows
  used the fallback (`learned_score_missing_count`).
- **Distinguish breadth from strength.** `support_count` / `evidence_family_count` measure
  how many independent evidence categories support a pair (breadth); the rank score
  reflects overall priority. A pair with fewer categories can still outrank one with more
  if its signals are strong. Say so when relevant.
- **Name the evidence types** behind a candidate using the flags present: miRTarBase
  (`mirtarbase_pos`) = curated functional support; TargetScan/miRDB = computational
  prediction; CLIP/ENCORI (`clip_exp_sum`) = binding; seed/site architecture; RNAhybrid/MFE
  = thermodynamic; TCGA `*_anticorrelated`/`*_repression_evidence` = context-specific
  repression in that cancer.
- **Interpret percentiles with the standard labels** (computed against the full database):
  >=95 exceptional, >=90 very high, >=75 high, >=50 above average, >=25 typical, <25 low,
  missing = not available.
- **Pass along notes and caveats.** Surface `arm_interpretation_note` (which arm was
  assumed), any `warnings`, and `no_candidates_explanation` when the list is empty.
- **Do not overstate.** These are ranked *candidates* from integrated evidence, not proven
  interactions. Recommend experimental validation (e.g. luciferase reporter, Ago-CLIP,
  Western blot after mimic/inhibitor) where appropriate.

A good answer typically leads with the top candidates and why they rank highly (evidence
types + strength), notes the ranking basis and arm assumption, and ends with any caveats.

---

## Reference
- `reference/queryspec_schema.md` - full QuerySpec JSON schema for `--queryspec-json`.
- `reference/evidence_columns.md` - meaning of the evidence/score columns in the output.
