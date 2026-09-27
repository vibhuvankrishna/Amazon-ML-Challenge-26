"""Inverted-index blocking — build once, query many (full-scale friendly)."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .normalize import blocking_keys


def build_index(
    df: pd.DataFrame,
    id_col: str = "entity_id",
    max_per_key: int = 250,
) -> dict[str, list[str]]:
    """Map blocking_key -> capped list of entity_ids."""
    raw: dict[str, list[str]] = defaultdict(list)
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="index", leave=False):
        eid = getattr(row, id_col)
        for key in blocking_keys(row.business_name, row.business_address, row.country):
            bucket = raw[key]
            if len(bucket) < max_per_key:
                bucket.append(eid)
    return dict(raw)


def records_by_id(df: pd.DataFrame) -> dict[str, tuple[str, str, str]]:
    """entity_id -> (name, address, country) for fast pair expansion."""
    out: dict[str, tuple[str, str, str]] = {}
    for row in df.itertuples(index=False):
        out[row.entity_id] = (row.business_name, row.business_address, row.country)
    return out


def generate_candidates_from_index(
    s1: pd.DataFrame,
    index: dict[str, list[str]],
    max_candidates: int = 80,
) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    for row in tqdm(s1.itertuples(index=False), total=len(s1), desc="block", leave=False):
        scores: dict[str, int] = defaultdict(int)
        for key in blocking_keys(row.business_name, row.business_address, row.country):
            for cid in index.get(key, ()):
                scores[cid] += 1
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        # Always take top-K by key-hit score (precision comes from the classifier)
        chosen = [cid for cid, _ in ranked[:max_candidates]]
        candidates[row.entity_id] = chosen
    return candidates


def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    max_per_key: int = 250,
    max_candidates: int = 80,
    index: Optional[dict[str, list[str]]] = None,
) -> dict[str, list[str]]:
    if index is None:
        s23 = pd.concat([s2, s3], ignore_index=True)
        index = build_index(s23, max_per_key=max_per_key)
    return generate_candidates_from_index(s1, index, max_candidates=max_candidates)


def candidates_to_frame(candidates: dict[str, list[str]]) -> pd.DataFrame:
    from .io_utils import format_id_list

    rows = [
        {"source1_entity_id": sid, "candidate_entity_ids": format_id_list(cids)}
        for sid, cids in candidates.items()
    ]
    return pd.DataFrame(rows)


def expand_pairs(
    candidates: dict[str, list[str]],
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    s1_rec: Optional[dict[str, tuple[str, str, str]]] = None,
    s23_rec: Optional[dict[str, tuple[str, str, str]]] = None,
) -> pd.DataFrame:
    """Materialize candidate pairs for feature extraction."""
    if s1_rec is None:
        s1_rec = records_by_id(s1)
    if s23_rec is None:
        s23_rec = records_by_id(s23)

    records = []
    for sid, cids in candidates.items():
        a = s1_rec.get(sid)
        if a is None:
            continue
        for cid in cids:
            b = s23_rec.get(cid)
            if b is None:
                continue
            records.append(
                {
                    "source1_entity_id": sid,
                    "candidate_entity_id": cid,
                    "name1": a[0],
                    "addr1": a[1],
                    "country1": a[2],
                    "name2": b[0],
                    "addr2": b[1],
                    "country2": b[2],
                }
            )
    return pd.DataFrame.from_records(records)


def blocking_recall(
    candidates: dict[str, list[str]],
    gt_map: dict[str, set[str]],
) -> dict[str, float]:
    """Fraction of true match IDs that appear in the candidate set."""
    total_true = 0
    found = 0
    entities_with_true = 0
    entities_full = 0
    for sid, true in gt_map.items():
        if not true:
            continue
        entities_with_true += 1
        cand = set(candidates.get(sid, []))
        total_true += len(true)
        hit = len(true & cand)
        found += hit
        if hit == len(true):
            entities_full += 1
    return {
        "pair_recall": (found / total_true) if total_true else 1.0,
        "entity_full_recall": (entities_full / entities_with_true) if entities_with_true else 1.0,
        "n_true_pairs": float(total_true),
        "n_found": float(found),
        "avg_candidates": (
            sum(len(v) for v in candidates.values()) / max(len(candidates), 1)
        ),
    }
