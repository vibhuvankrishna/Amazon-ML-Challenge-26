"""Unit checks for raise-score changes (no full dataset required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))

from ber.normalize import normalize_address
from rank_match import _consonant_skeleton, decide_matches, extra_key_pairs


def _vec(
    name_s=0.9,
    short=0.0,
    house=1.0,
    pin=1.0,
    loc=1.0,
    empty=0.0,
    conflict=0.0,
    nblk=2.0,
    pos=1.0,
):
    v = np.zeros(30, dtype=np.float32)
    v[0] = name_s
    v[11] = short
    v[18] = house
    v[19] = pin
    v[20] = loc
    v[23] = empty
    v[24] = nblk
    v[25] = pos
    v[29] = conflict
    return v


def test_france_abbrev() -> None:
    assert "street" in normalize_address("12 Rue de Rivoli")
    assert "avenue" in normalize_address("5 Av. Victor Hugo")
    assert "boulevard" in normalize_address("10 Bd Haussmann")
    assert "cedex" in normalize_address("75001 Paris Cedex")
    print("france_abbrev OK")


def test_typo_keys() -> None:
    keys = {k for _, k in extra_key_pairs("Blue Ocean Shipping", "12 MG Road 411001 Pune", "India")}
    kinds = {k.split("|")[1] for k in keys}
    assert "rp" in kinds, kinds  # rare token + pin
    assert "ph" in kinds or "th" in kinds, kinds
    assert "sh" in kinds or "s" in kinds, kinds
    assert _consonant_skeleton("sharma").startswith("s")
    a = _consonant_skeleton("sharma")
    b = _consonant_skeleton("sarma")
    assert a and b and a[0] == b[0]
    print("typo_keys OK", sorted(kinds))


def test_sibling_accept() -> None:
    rule = {
        "t": 0.72,
        "t_empty": 0.82,
        "t_extra": 0.72,
        "t_short": 0.82,
        "t_conflict": 0.72,
        "margin": 0.12,
    }
    # Top is strong; second has lower proba but same house+pin and decent name.
    vectors = [
        _vec(name_s=0.92, house=1, pin=1, loc=1),
        _vec(name_s=0.70, house=1, pin=1, loc=0, pos=2),
        _vec(name_s=0.40, house=1, pin=0, loc=0, pos=3),  # weak name — reject
    ]
    ids = ["S2-a", "S3-b", "S2-c"]
    # Second would be dropped by margin alone (0.90 - 0.55 > 0.12)
    probas = [0.90, 0.55, 0.50]
    kept = decide_matches(vectors, ids, probas, rule)
    assert "S2-a" in kept
    assert "S3-b" in kept, kept  # sibling
    assert "S2-c" not in kept, kept  # weak name
    print("sibling_accept OK", kept)


def test_weak_name_same_house_not_sibling_alone() -> None:
    rule = {
        "t": 0.72,
        "t_empty": 0.82,
        "t_extra": 0.72,
        "t_short": 0.82,
        "t_conflict": 0.72,
        "margin": 0.12,
    }
    # No strong anchor accepted — only a below-threshold same-house pair.
    vectors = [_vec(name_s=0.50, house=1, pin=1)]
    kept = decide_matches(vectors, ["S2-x"], [0.50], rule)
    assert kept == [], kept
    print("weak_name_guard OK")


if __name__ == "__main__":
    test_france_abbrev()
    test_typo_keys()
    test_sibling_accept()
    test_weak_name_same_house_not_sibling_alone()
    print("ALL PASS")
