"""Pairwise string similarity features — CPU only (rapidfuzz + tokens)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from tqdm import tqdm

from .normalize import (
    address_tokens,
    extract_digits,
    name_tokens,
    normalize_address,
    normalize_name,
)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def pair_features_row(
    name1: str,
    addr1: str,
    country1: str,
    name2: str,
    addr2: str,
    country2: str,
) -> dict[str, float]:
    n1, n2 = normalize_name(name1), normalize_name(name2)
    a1, a2 = normalize_address(addr1), normalize_address(addr2)
    nt1, nt2 = set(name_tokens(name1)), set(name_tokens(name2))
    at1, at2 = set(address_tokens(addr1)), set(address_tokens(addr2))
    d1, d2 = extract_digits(addr1), extract_digits(addr2)

    return {
        "country_match": 1.0 if (country1 or "").upper() == (country2 or "").upper() else 0.0,
        "name_jw": JaroWinkler.normalized_similarity(n1, n2) if n1 or n2 else 0.0,
        "name_ratio": fuzz.ratio(n1, n2) / 100.0,
        "name_partial": fuzz.partial_ratio(n1, n2) / 100.0,
        "name_token_set": fuzz.token_set_ratio(n1, n2) / 100.0,
        "name_token_sort": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_jaccard": _jaccard(nt1, nt2),
        "name_lev": Levenshtein.normalized_similarity(n1, n2) if n1 or n2 else 0.0,
        "addr_ratio": fuzz.ratio(a1, a2) / 100.0 if (a1 or a2) else 0.0,
        "addr_token_set": fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 or a2) else 0.0,
        "addr_jaccard": _jaccard(at1, at2),
        "digit_jaccard": _jaccard(d1, d2),
        "name_len_ratio": (
            min(len(n1), len(n2)) / max(len(n1), len(n2)) if n1 and n2 else 0.0
        ),
        "both_addr_empty": 1.0 if (not a1 and not a2) else 0.0,
        "one_addr_empty": 1.0 if (bool(a1) != bool(a2)) else 0.0,
    }


FEATURE_NAMES = list(pair_features_row("", "", "", "", "", "").keys())


def build_feature_matrix(pairs: pd.DataFrame, show_progress: bool = True) -> np.ndarray:
    if pairs.empty:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    n = len(pairs)
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    it = pairs.itertuples(index=False)
    if show_progress and n > 20_000:
        it = tqdm(it, total=n, desc="features", leave=False)
    for i, r in enumerate(it):
        f = pair_features_row(r.name1, r.addr1, r.country1, r.name2, r.addr2, r.country2)
        for j, k in enumerate(FEATURE_NAMES):
            X[i, j] = f[k]
    return X
