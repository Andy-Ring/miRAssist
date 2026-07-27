#!/usr/bin/env python3
"""Build the bundled miRAssist pathway lookup tables.

The input libraries are pinned by name so a refresh is explicit and auditable.
Local GMT paths may be supplied for offline/reproducible builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd


ENRICHR_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary"
LIBRARIES = {
    "GO Biological Process": "GO_Biological_Process_2025",
    "Reactome": "Reactome_Pathways_2024",
    "WikiPathways": "WikiPathways_2024_Human",
}
COLLECTION_ORDER = {
    "GO Biological Process": 0,
    "Reactome": 1,
    "WikiPathways": 2,
    "Hallmark": 3,
}


def _download_library(library_name: str, destination: Path) -> Path:
    query = urllib.parse.urlencode({"mode": "text", "libraryName": library_name})
    request = urllib.request.Request(
        f"{ENRICHR_URL}?{query}",
        headers={"User-Agent": "miRAssist-pathway-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())
    return destination


def _read_gmt(path: Path) -> Iterable[tuple[str, list[str]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_number}: expected GMT name, description, and genes")
            name = fields[0].strip()
            genes = sorted({gene.strip().upper() for gene in fields[2:] if gene.strip()})
            if name and genes:
                yield name, genes


def _stable_id(collection: str, raw_name: str) -> tuple[str, str]:
    if collection == "GO Biological Process":
        match = re.search(r"\((GO:\d+)\)\s*$", raw_name, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"GO term lacks a GO identifier: {raw_name}")
        return match.group(1).upper(), raw_name[: match.start()].strip()
    if collection == "WikiPathways":
        match = re.search(r"\b(WP\d+)\s*$", raw_name, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"WikiPathways term lacks a WP identifier: {raw_name}")
        return match.group(1).upper(), raw_name[: match.start()].strip()
    digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:16]
    return f"REACTOME:{digest}", raw_name.strip()


def _library_records(collection: str, library_name: str, path: Path) -> list[dict]:
    records = []
    for raw_name, genes in _read_gmt(path):
        pathway_id, pathway_name = _stable_id(collection, raw_name)
        records.append(
            {
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
                # Enrichr GMT exports do not contain ontology definitions. Keep
                # this empty so a repeated pathway name is not mistaken for an
                # independent exact-description match.
                "description": "",
                "collection": collection,
                "source": library_name,
                "n_genes": len(genes),
                "genes": json.dumps(genes, separators=(",", ":")),
            }
        )
    return records


def _hallmark_records(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    records = []
    for row in frame.to_dict(orient="records"):
        pathway_id = str(row["pathway_id"]).strip()
        pathway_name = str(row["pathway_name"]).strip()
        genes_value = row.get("genes", "[]")
        genes = json.loads(genes_value) if isinstance(genes_value, str) else list(genes_value)
        genes = sorted({str(gene).strip().upper() for gene in genes if str(gene).strip()})
        source = str(row.get("source") or row.get("pathway_desc") or "MSigDB Hallmark").strip()
        core_name = re.sub(r"^HALLMARK_", "", pathway_name).replace("_", " ").lower()
        records.append(
            {
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
                "description": f"Hallmark gene set for {core_name}.",
                "collection": "Hallmark",
                "source": source,
                "n_genes": len(genes),
                "genes": json.dumps(genes, separators=(",", ":")),
            }
        )
    return records


def _write_tables(records: list[dict], output_dir: Path, skill_output_dir: Path | None) -> None:
    pathways = pd.DataFrame(records)
    pathways["_collection_order"] = pathways["collection"].map(COLLECTION_ORDER)
    pathways = pathways.sort_values(
        ["_collection_order", "pathway_name", "pathway_id"], kind="stable"
    ).drop(columns=["_collection_order"]).reset_index(drop=True)
    pathways = pathways.drop_duplicates(["pathway_id", "pathway_name"], keep="first")

    gene_rows = []
    for row in pathways.itertuples(index=False):
        for gene in json.loads(row.genes):
            gene_rows.append(
                {
                    "gene_symbol": gene,
                    "pathway_id": row.pathway_id,
                    "pathway_name": row.pathway_name,
                    "collection": row.collection,
                }
            )
    gene_to_pathways = pd.DataFrame(gene_rows).drop_duplicates(
        ["gene_symbol", "pathway_id", "pathway_name"]
    )
    gene_to_pathways = gene_to_pathways.sort_values(
        ["gene_symbol", "pathway_name", "pathway_id"], kind="stable"
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pathways.to_csv(
        output_dir / "pathways.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    gene_to_pathways.to_csv(
        output_dir / "gene_to_pathways.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    pathways.to_parquet(output_dir / "pathways.parquet", index=False)
    gene_to_pathways.to_parquet(output_dir / "gene_to_pathways.parquet", index=False)

    if skill_output_dir is not None:
        skill_output_dir.mkdir(parents=True, exist_ok=True)
        pathways.to_csv(
            skill_output_dir / "pathways.csv.gz",
            index=False,
            compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        )
        gene_to_pathways.to_csv(
            skill_output_dir / "gene_to_pathways.csv.gz",
            index=False,
            compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        )
        pathways.to_parquet(skill_output_dir / "pathways.parquet", index=False)
        gene_to_pathways.to_parquet(
            skill_output_dir / "gene_to_pathways.parquet", index=False
        )

    print(
        f"Wrote {len(pathways):,} pathways and {len(gene_to_pathways):,} "
        f"gene-pathway memberships; collections={pathways['collection'].value_counts().to_dict()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hallmark-csv",
        type=Path,
        default=Path("data/processed/pathways/pathways.csv.gz"),
    )
    parser.add_argument("--go-gmt", type=Path)
    parser.add_argument("--reactome-gmt", type=Path)
    parser.add_argument("--wikipathways-gmt", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/pathways")
    )
    parser.add_argument(
        "--skill-output-dir",
        type=Path,
        default=Path("mirassist-skill/scripts/data/processed/pathways"),
    )
    args = parser.parse_args()

    supplied_paths = {
        "GO Biological Process": args.go_gmt,
        "Reactome": args.reactome_gmt,
        "WikiPathways": args.wikipathways_gmt,
    }
    with tempfile.TemporaryDirectory(prefix="mirassist-pathways-") as temporary:
        temporary_dir = Path(temporary)
        records = _hallmark_records(args.hallmark_csv)
        for collection, library_name in LIBRARIES.items():
            gmt_path = supplied_paths[collection]
            if gmt_path is None:
                gmt_path = _download_library(
                    library_name, temporary_dir / f"{library_name}.gmt"
                )
            records.extend(_library_records(collection, library_name, gmt_path))
        _write_tables(records, args.output_dir, args.skill_output_dir)


if __name__ == "__main__":
    main()
