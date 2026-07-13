# QuerySpec schema (for `--queryspec-json`)

Most queries are best expressed with individual CLI flags (see SKILL.md). For full control
you can write a QuerySpec JSON file and pass `--queryspec-json spec.json`. The CLI runs it
through the same validator the app uses, so missing fields are filled with safe defaults;
you only need to provide the keys you care about.

```json
{
  "original_question": "Which genes does miR-21 target to promote apoptosis in breast cancer?",
  "mode": "mirna_to_targets",
  "mirna": "hsa-miR-21-5p",
  "gene": null,
  "cancer": { "name": "breast cancer", "tcga": "BRCA" },
  "phenotype_context": {
    "phenotype": "apoptosis",
    "observed_change": "promoted",
    "miRNA_perturbation": "overexpression",
    "raw_phrase": "overexpression of miR-21 promoted apoptosis",
    "direction": null
  },
  "phenotype_keywords": ["apoptosis"],
  "pathway_keywords": [],
  "novel": false,
  "k": 10,
  "result_count": 5,
  "filters": {
    "min_support": 1,
    "require_binding_evidence": false,
    "require_expression": false
  }
}
```

## Field notes
- `mode` - `mirna_to_targets` or `gene_to_mirnas`.
- `mirna` / `gene` - provide the one that matches `mode`. Any miRNA form is accepted; if no
  `-3p`/`-5p` arm is given, `-5p` is assumed and a note is returned.
- `cancer.tcga` - one of `BRCA`, `COAD`, `PRAD` (the cohorts with TCGA data). `cancer.name`
  free text is mapped to a TCGA code when recognized.
- `phenotype_context` - the tool derives target-role inference and directional pathway terms
  from `phenotype` + `observed_change` + `miRNA_perturbation`. Leave fields null if unknown.
- `novel` - true restricts to non-miRTarBase (unvalidated) pairs.
- `k` - candidate pool size carried through retrieval. `result_count` - number of ranked
  results returned (default 5).
- `filters.min_support` - minimum distinct evidence categories (raise to 2 for
  high-confidence). `require_binding_evidence`, `require_expression` - optional hard gates.

Fields not listed here (target_role_inference, pathway_selection_request, pathway_filter,
etc.) are computed automatically by the validator - you do not need to supply them.
