from __future__ import annotations


def should_show_api_connection_controls(app_mode: str) -> bool:
    return str(app_mode or "").strip().lower() == "api"


def get_normal_debug_sections() -> list[str]:
    return [
        "Planner output (QuerySpec)",
        "Evidence shortlist",
    ]


def get_how_to_use_markdown() -> str:
    return """
miRAssist prioritizes candidate miRNA-target interactions using curated evidence, published prediction models, CLIP/binding evidence, seed/site features, structure-aware features, TCGA context, and pathway filtering.

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
- Evidence support count = number of distinct evidence categories, not number of raw features.
- Evidence support count measures breadth, not strength.
- Percentiles show whether a feature is unusually high across the database.
- miRTarBase is curated prior evidence.
- Overall priority considers both evidence breadth and evidence strength.
- By default, miRAssist returns the top 5 ranked candidates unless you request more.

Novel mode:
- Novel mode excludes known miRTarBase functional interactions from the ranked list.
""".strip()


def get_about_evidence_markdown() -> str:
    return """
- `miRTarBase`: curated prior functional evidence
- `miRDB`: expression-based published model; high score = more likely interaction
- `TargetScan`: sequence-based published model; lower score = more likely interaction
- `CLIP`: binding-associated support
- `Seed/site`: canonical site architecture support
- `RNAhybrid/structure`: structure-compatible binding support
- `TCGA`: cancer-context repression support
- `Pathway filter`: deterministic pathway membership used as a strict filter
""".strip()
