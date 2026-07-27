# Bundled pathway database

miRAssist bundles gene sets from four collections so phenotype filtering can prefer
specific biological processes and curated pathways over broad contextual signatures:

1. GO Biological Process (`GO_Biological_Process_2025`, Enrichr export)
2. Reactome (`Reactome_Pathways_2024`, Enrichr export)
3. WikiPathways Human (`WikiPathways_2024_Human`, Enrichr export)
4. MSigDB Hallmark (the 50 sets previously bundled by miRAssist)

`pathways.csv.gz` / `pathways.parquet` contain pathway metadata and JSON-encoded gene
lists. `gene_to_pathways.csv.gz` / `gene_to_pathways.parquet` contain the normalized
gene-to-pathway mapping used by downstream filtering.

Rebuild the tables with pinned local GMT files:

```bash
python scripts/build_pathway_database.py \
  --go-gmt /path/to/GO_Biological_Process_2025.gmt \
  --reactome-gmt /path/to/Reactome_Pathways_2024.gmt \
  --wikipathways-gmt /path/to/WikiPathways_2024_Human.gmt
```

If the GMT arguments are omitted, the builder downloads those named public libraries
from the Enrichr gene-set library endpoint. The script also refreshes the pathway
Parquet files bundled with `mirassist-skill`.

The collection and library release labels are stored on every pathway row. Reactome
exports do not include stable pathway identifiers in this GMT representation, so the
builder assigns a deterministic `REACTOME:<sha1>` identifier from the pathway name.
