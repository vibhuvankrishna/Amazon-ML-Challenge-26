"""TSV I/O helpers — always tab-separated."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]


def read_source(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)
    missing = [c for c in SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[SOURCE_COLS].copy()


def read_ground_truth(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)
    missing = [c for c in GT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[GT_COLS].copy()


def parse_id_list(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    # Deduplicate while preserving order
    seen, out = set(), []
    for tok in value.split(","):
        tok = tok.strip()
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def format_id_list(ids: list[str] | set[str]) -> str:
    if isinstance(ids, set):
        ids = sorted(ids)
    return ",".join(ids)


def write_matching_results(path: str | Path, rows: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = rows[["source1_entity_id", "matched_entity_ids"]].copy()
    out.to_csv(path, sep="\t", index=False)


def write_candidate_pairs(path: str | Path, rows: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = rows[["source1_entity_id", "candidate_entity_ids"]].copy()
    out.to_csv(path, sep="\t", index=False)


def iter_chunks(path: str | Path, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Stream a large TSV in chunks (memory-friendly on laptops)."""
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize
    ):
        yield chunk[SOURCE_COLS] if set(SOURCE_COLS).issubset(chunk.columns) else chunk
