from __future__ import annotations


def get_normal_debug_sections() -> list[str]:
    return [
        "Planner output (QuerySpec)",
        "Evidence shortlist",
    ]


def get_how_to_use_markdown() -> str:
    return """
miRAssist prioritizes candidate miRNA-target interactions using six major evidence families: sequence complementarity, thermodynamic stability, sequence conservation, target site accessibility, functional binding, and functional repression.

Example query types:
- miRNA to targets: `What genes are regulated by hsa-miR-210-3p?`
- Gene to miRNAs: `I am studying ISCU. What miRNAs may regulate it?`
- Cancer context: `I am studying miRNA-210 in breast cancer cells. What genes might it regulate?`
- Pathway or phenotype context: `I think miRNA-210 is involved in energy metabolism. What genes might it regulate?`
- Phenotype direction: `I overexpressed miR-21 in breast cancer cells and saw increased apoptosis. What genes might it be targeting that could explain this?`

How pathway filtering works:
- If your prompt mentions a pathway or phenotype, miRAssist uses strict pathway filtering.
- Candidate genes are restricted to genes in matching pathway sets before scoring.
- The LLM does not invent gene-pathway membership.

How cancer context works:
- `breast cancer` maps to TCGA `BRCA`
- `colon` or `colorectal cancer` maps to TCGA `COAD`
- `prostate cancer` maps to TCGA `PRAD`
- If included, a cancer-type specific anticorrelation feature will be added to the query.

How to interpret results:
- miRAssist retrieves and ranks candidate interactions from the evidence database, then returns the top candidates as a chart.
- Evidence support count = number of distinct evidence families, not number of raw features.
- Evidence support count measures breadth, not strength.
- Percentiles show whether a feature or family signal is unusually strong across the database.
- More negative RNAhybrid MFE means stronger predicted binding.
- More negative TargetScan context score means stronger sequence-conservation support.
- More negative TCGA Spearman rho means stronger repression-consistent anticorrelation.
- Overall priority considers both evidence breadth and evidence strength.

Novel mode:
- Novel mode excludes known curated interactions when that filter is available in the backend.
""".strip()


def get_about_evidence_markdown() -> str:
    return """
- `Sequence complementarity`: seed match type, seed pairing score, and site-count support
- `Thermodynamic stability`: RNAhybrid minimum free energy support; more negative is stronger
- `Sequence conservation`: TargetScan context and conservation support; more negative context scores are stronger
- `Target site accessibility`: RNAplfold accessibility support; higher unpaired probabilities are stronger
- `Functional binding`: CLIP and ENCORI support for physical binding evidence
- `Functional repression`: TCGA anticorrelation and repression-consistent context support
- `Pathway filter`: deterministic pathway membership used as a strict filter
""".strip()
