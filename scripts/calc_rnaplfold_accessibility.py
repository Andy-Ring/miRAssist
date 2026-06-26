from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Calculate RNAplfold accessibility features for candidate target windows.")
    ap.add_argument("--sites", required=True, help="Input candidate site table (.csv or .parquet).")
    ap.add_argument("--sequence-column", required=True, help="Column containing the target window sequence.")
    ap.add_argument("--site-start-column", required=True, help="1-based site start position within the sequence window.")
    ap.add_argument("--site-end-column", required=True, help="1-based site end position within the sequence window.")
    ap.add_argument("--output", required=True, help="Output RNAplfold feature table (.csv or .parquet).")
    ap.add_argument("--cache-dir", required=True, help="Directory to cache per-sequence RNAplfold outputs.")
    ap.add_argument("--window-size", type=int, default=80, help="RNAplfold -W window size.")
    ap.add_argument("--max-base-pair-span", type=int, default=40, help="RNAplfold -L maximum base pair span.")
    ap.add_argument("--unpaired-length", type=int, default=8, help="RNAplfold -u unpaired stretch length.")
    ap.add_argument("--flank-size", type=int, default=10, help="Flank size used for the broader accessibility region.")
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke testing.")
    return ap.parse_args()


def _log(message: str) -> None:
    print(f"[calc_rnaplfold_accessibility] {message}")


def _read_table(path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=limit)
    if suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read parquet file {path}. Install pyarrow or fastparquet in the active environment."
            ) from exc
        return df.head(limit).copy() if limit is not None else df
    raise ValueError(f"Unsupported input format for {path}. Expected .csv or .parquet.")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format for {path}. Use .csv or .parquet.")


def _require_rnaplfold() -> str:
    binary = shutil.which("RNAplfold")
    if not binary:
        raise RuntimeError(
            "RNAplfold was not found on PATH. Install ViennaRNA so the RNAplfold binary is available in the active conda environment."
        )
    version_cmd = [binary, "--version"]
    proc = subprocess.run(version_cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "RNAplfold is installed but '--version' failed. Check the ViennaRNA installation in the active environment."
        )
    _log(f"Detected RNAplfold: {proc.stdout.strip() or proc.stderr.strip()}")
    return binary


def _sequence_hash(sequence: str, window_size: int, max_span: int, unpaired_length: int) -> str:
    payload = f"{sequence}|W={window_size}|L={max_span}|u={unpaired_length}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class SequenceAccessibility:
    sequence_hash: str
    sequence_length: int
    lunp_path: Path


def _run_rnaplfold_for_sequence(
    binary: str,
    sequence: str,
    cache_dir: Path,
    *,
    window_size: int,
    max_span: int,
    unpaired_length: int,
) -> SequenceAccessibility:
    cache_dir.mkdir(parents=True, exist_ok=True)
    seq_hash = _sequence_hash(sequence, window_size, max_span, unpaired_length)
    seq_dir = cache_dir / seq_hash
    lunp_path = seq_dir / "plfold_lunp"
    if lunp_path.exists():
        return SequenceAccessibility(seq_hash, len(sequence), lunp_path)

    seq_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = seq_dir / "input.fa"
    fasta_path.write_text(f">seq\n{sequence}\n", encoding="utf-8")

    cmd = [
        binary,
        "-W",
        str(window_size),
        "-L",
        str(max_span),
        "-u",
        str(unpaired_length),
    ]
    proc = subprocess.run(
        cmd,
        cwd=seq_dir,
        input=fasta_path.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "RNAplfold failed for a sequence window. "
            f"Command: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    if not lunp_path.exists():
        raise RuntimeError(
            f"RNAplfold completed but {lunp_path} was not created. Check ViennaRNA output in {seq_dir}."
        )
    (seq_dir / "rnaplfold_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (seq_dir / "rnaplfold_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    return SequenceAccessibility(seq_hash, len(sequence), lunp_path)


def _parse_lunp_table(path: Path) -> pd.DataFrame:
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if not parts[0].isdigit():
                continue
            rows.append([float(part) for part in parts])
    if not rows:
        raise RuntimeError(f"Could not parse any numeric rows from RNAplfold output: {path}")
    table = pd.DataFrame(rows)
    table = table.rename(columns={0: "position"})
    return table


def _region_mean_probability(lunp: pd.DataFrame, start: int, end: int, unpaired_length: int) -> float:
    if start > end:
        return float("nan")
    value_column = unpaired_length
    if value_column not in lunp.columns:
        return float("nan")
    region = lunp[(lunp["position"] >= start) & (lunp["position"] <= end)]
    if region.empty:
        return float("nan")
    return float(pd.to_numeric(region[value_column], errors="coerce").mean())


def _clip_region(start: int, end: int, sequence_length: int) -> Tuple[int, int]:
    start = max(1, int(start))
    end = min(sequence_length, int(end))
    return start, max(start, end)


def _feature_row_from_lunp(
    lunp: pd.DataFrame,
    sequence_length: int,
    *,
    site_start: int,
    site_end: int,
    unpaired_length: int,
    flank_size: int,
) -> Dict[str, float]:
    site_start, site_end = _clip_region(site_start, site_end, sequence_length)
    seed_end = min(site_end, site_start + max(unpaired_length - 1, 0))
    flank_start, flank_end = _clip_region(site_start - flank_size, site_end + flank_size, sequence_length)

    seed_prob = _region_mean_probability(lunp, site_start, seed_end, unpaired_length)
    site_prob = _region_mean_probability(lunp, site_start, site_end, unpaired_length)
    flank_prob = _region_mean_probability(lunp, flank_start, flank_end, unpaired_length)

    return {
        "rnaplfold_seed_unpaired_prob": seed_prob,
        "rnaplfold_site_unpaired_prob": site_prob,
        "rnaplfold_flank_unpaired_prob": flank_prob,
        "rnaplfold_seed_accessibility_score": seed_prob,
        "rnaplfold_site_accessibility_score": site_prob,
        "rnaplfold_window_length": float(sequence_length),
        "rnaplfold_region_start": float(site_start),
        "rnaplfold_region_end": float(site_end),
    }


def main() -> None:
    args = parse_args()
    binary = _require_rnaplfold()
    sites_path = Path(args.sites).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_path = Path(args.output).resolve()

    sites_df = _read_table(sites_path, limit=args.limit)
    for required in (args.sequence_column, args.site_start_column, args.site_end_column):
        if required not in sites_df.columns:
            raise RuntimeError(f"Required column '{required}' was not found in {sites_path}.")

    id_columns = [
        column
        for column in (
            "evidence_row_id",
            "eval_row_id",
            "site_id",
            "mirna_name",
            "mirna_name_normalized",
            "gene_symbol",
            "gene_symbol_normalized",
            "transcript_id",
        )
        if column in sites_df.columns
    ]
    if not id_columns:
        raise RuntimeError(
            "The sites table must contain at least one identifier column such as evidence_row_id, site_id, or mirna/gene/transcript keys."
        )

    feature_rows: List[Dict[str, object]] = []
    for idx, row in sites_df.iterrows():
        sequence = str(row[args.sequence_column]).strip().upper().replace("T", "U")
        if not sequence:
            continue
        site_start = int(pd.to_numeric(row[args.site_start_column], errors="coerce"))
        site_end = int(pd.to_numeric(row[args.site_end_column], errors="coerce"))
        seq_access = _run_rnaplfold_for_sequence(
            binary,
            sequence,
            cache_dir,
            window_size=args.window_size,
            max_span=args.max_base_pair_span,
            unpaired_length=args.unpaired_length,
        )
        lunp = _parse_lunp_table(seq_access.lunp_path)
        features = _feature_row_from_lunp(
            lunp,
            seq_access.sequence_length,
            site_start=site_start,
            site_end=site_end,
            unpaired_length=args.unpaired_length,
            flank_size=args.flank_size,
        )
        out_row: Dict[str, object] = {column: row[column] for column in id_columns}
        out_row["sequence_hash"] = seq_access.sequence_hash
        out_row.update(features)
        feature_rows.append(out_row)
        if idx % 100 == 0:
            _log(f"Processed {idx + 1} rows")

    features_df = pd.DataFrame(feature_rows)
    metadata = {
        "input_path": str(sites_path),
        "output_path": str(output_path),
        "cache_dir": str(cache_dir),
        "unit_of_analysis": "site-level RNAplfold accessibility rows",
        "rows_in": int(len(sites_df)),
        "rows_out": int(len(features_df)),
        "window_size": int(args.window_size),
        "max_base_pair_span": int(args.max_base_pair_span),
        "unpaired_length": int(args.unpaired_length),
        "flank_size": int(args.flank_size),
        "probability_summary_formula": "arithmetic mean unpaired probability across each region",
        "directionality": "higher unpaired probability means more accessible / stronger accessibility support",
    }
    _write_table(features_df, output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _log(f"Wrote RNAplfold features to: {output_path}")
    _log(f"Wrote RNAplfold metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
