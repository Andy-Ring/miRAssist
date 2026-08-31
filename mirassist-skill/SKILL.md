---
name: mirassist
description: Retrieve and explain ranked, evidence-grounded miRNA–target candidates. Use when a researcher asks which genes a miRNA targets, which miRNAs regulate a gene, or requests candidates filtered by TCGA cancer context, phenotype, pathway, novelty, or evidence support.
---

# miRAssist 1.0

Use the bundled deterministic CLI to retrieve candidates from the production evidence
snapshot. You provide the two language layers:

1. **Planner** — translate the researcher’s question into CLI arguments.
2. **Synthesizer** — explain the returned JSON without adding unsupported biology.

The CLI integrates sequence complementarity, RNAhybrid thermodynamics, TargetScan
conservation, RNAplfold accessibility, CLIP/ENCORI binding, and TCGA repression evidence.
It ranks candidates with the persisted production random-forest score.

The hard rule is: **use only values, labels, candidate identities, and pathway memberships
returned by the CLI.** Never invent scores, percentiles, mechanisms, validation status,
gene functions, or gene–phenotype relationships.

## Setup

End users configure nothing. Install the lightweight Python dependencies once:

~~~bash
pip install -r requirements.txt
~~~

On first use, the CLI downloads the production evidence snapshot from the latest
miRAssist GitHub release and caches it. Later queries reuse the cache. If the download
fails, report the returned error instead of guessing.

## Plan the query

Choose the direction and entity:

- `mirna_to_targets`: the question centers on a miRNA.
- `gene_to_mirnas`: the question centers on a gene.
- `--mirna` accepts forms such as `hsa-miR-21-5p`, `miR-21`, or
  `microRNA-21`. Do not add a mature arm the researcher did not specify; the backend
  applies its default and returns an interpretation note.
- `--gene` accepts a gene symbol such as `PTEN` or `PDCD4`.

Optional context:

- `--tcga BRCA|COAD|PRAD`, or `--cancer-name` with breast, colorectal, or prostate
  cancer. Do not imply that other cohorts have TCGA repression evidence.
- `--phenotype` for apoptosis, proliferation, EMT, invasion, migration, or another
  explicit biological process.
- `--observed-change` and `--perturbation` when the question describes an experiment.
  The backend applies the usual miRNA-repression assumption and labels any inferred target
  role as an interpretation, never proof.

Filters:

- `--novel`: exclude candidates aligned to the retained miRTarBase known-positive set.
  Use for “new,” “exploratory,” or “unvalidated” requests.
- `--min-support N`: require at least N evidence families. Use 2 when the researcher
  explicitly asks for multiple independent lines of evidence.
- `--require-binding`: require sequence or binding evidence.
- `--require-expression`: require paired TCGA expression evidence; pair with `--tcga`.
- `--result-count N`: number of results to return; default 5.
- `--k N`: retrieval shortlist size; default 10.

Do not ask a clarifying question when backend defaults resolve the omission safely.
Do ask when the entity or intended query direction cannot be determined.

## Retrieve

Run the CLI from the skill directory:

~~~bash
python3 scripts/retrieve.py \
  --mirna "hsa-miR-21-5p" \
  --tcga BRCA \
  --phenotype apoptosis \
  --observed-change promoted \
  --perturbation overexpression \
  --result-count 5 \
  --pretty
~~~

For complete control, pass a QuerySpec JSON:

~~~bash
python3 scripts/retrieve.py --queryspec-json spec.json --pretty
~~~

The result contains:

- `query`: resolved direction, entity, context, and filters.
- `pathway_selection`: deterministically selected pathways and genes.
- `ranking`: evidence backend and persisted score diagnostics.
- `arm_interpretation_note`, `warnings`, and `no_candidates_explanation`.
- `candidates`: the ordered evidence-grounded shortlist.

The QuerySpec schema is in [reference/queryspec_schema.md](reference/queryspec_schema.md).

## Synthesize

Write a concise researcher-facing answer in the exact candidate order returned.

For each candidate:

- identify the miRNA and gene;
- report the miRAssist score and rank exactly as returned;
- distinguish evidence breadth (`support_count` or `evidence_family_count`) from
  evidence strength;
- name only evidence families supported by returned fields;
- report pathway membership only when `pathway_selected_gene` is true and use only
  `pathway_selected_names`;
- surface relevant cancer-specific TCGA values only for the requested cohort;
- pass along arm notes, warnings, missing evidence, and empty-result explanations.

Explain that candidates are ranked by the production miRAssist score when
`ranking.score_column_used` is `mirassist_model_score`. If the CLI reports a fallback,
name the returned fallback mode and do not present it as the production model score.

Interpret percentiles using these fixed labels:

- 95 or higher: exceptional
- 90–94.9: very high
- 75–89.9: high
- 50–74.9: above average
- 25–49.9: typical
- below 25: low
- missing: not available

Scientific boundaries:

- The miRAssist score is relative prioritization, not a biological probability.
- Sequence and TargetScan support are computational, not experimental validation.
- CLIP supports binding but does not by itself prove context-specific direct targeting.
- TCGA anticorrelation is consistent with repression but does not prove direct repression.
- Absence of a returned evidence family is missing evidence, not evidence against an
  interaction.
- Do not use external literature unless the researcher separately asks for a literature
  search; keep those claims distinct from the retrieved miRAssist evidence.
- Recommend experimental follow-up where appropriate, but do not claim confirmation.

Evidence field definitions are in
[reference/evidence_columns.md](reference/evidence_columns.md).
