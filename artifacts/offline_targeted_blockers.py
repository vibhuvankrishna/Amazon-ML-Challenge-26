"""Holdout-only test of two blockers. Does not touch the scorer or the test set."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "code/business_entity_resolution/src")

from ber.metrics import score_entity
from ber.normalize import (
    address_tokens,
    name_tokens,
    normalize_address,
    normalize_name,
    significant_name_tokens,
)
from rank_match import _load_gt_sample, _scan_ids, block_keys, candidates_for

csv.field_size_limit(10_000_000)
MAX_DF = 40

CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "क़": "q",
    "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}
VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऋ": "ri",
}
MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri",
    "ॅ": "e", "ॉ": "o",
}
VIRAMA = "्"
ANUSVARA = {"ं": "n", "ँ": "n", "ः": "h"}
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DIGIT_RUNS = re.compile(r"\d+")


def transliterate(text: str) -> str:
    """Deterministic Devanagari-to-Latin. Latin characters are kept."""
    out = []
    chars = list(text or "")
    i = 0
    n = len(chars)
    while i < n:
        ch = chars[i]
        if ch in VOWELS:
            out.append(VOWELS[ch])
            i += 1
            continue
        if ch in CONSONANTS:
            base = CONSONANTS[ch]
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:
                out.append(base)
                i += 2
                continue
            if nxt in MATRAS:
                out.append(base + MATRAS[nxt])
                i += 2
                continue
            out.append(base + "a")
            i += 1
            continue
        if ch in ANUSVARA:
            out.append(ANUSVARA[ch])
            i += 1
            continue
        if ch == VIRAMA:
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def fold_digits(text: str) -> str:
    return (text or "").translate(DEV_DIGITS)


def _runs(address: str) -> list[str]:
    """Digit groups in left-to-right order. Not the unordered set used elsewhere."""
    return _DIGIT_RUNS.findall(fold_digits(address or ""))


def _country(country: str) -> str:
    return (country or "").strip().upper() or "UNK"


def _house(address: str) -> str:
    ds = [d for d in _runs(address) if 1 <= len(d) <= 6]
    return ds[0] if ds else ""


def _pin(address: str) -> str:
    for d in _runs(address):
        if len(d) >= 5:
            return d
    return ""


def _loc(address: str) -> str:
    ts = [t for t in address_tokens(address or "") if len(t) >= 6 and not t.isdigit()]
    return max(ts, key=len) if ts else ""


def name_house_keys(name: str, address: str, country: str) -> list[str]:
    nn = normalize_name(name)
    house = _house(address)
    if len(nn) >= 8 and house:
        return [f"{_country(country)}|nh|{nn}_{house}"]
    return []


def translit_keys(name: str, address: str, country: str) -> list[str]:
    """Original name+house, plus the same key on the transliterated name."""
    keys = name_house_keys(name, address, country)
    latin = transliterate(name)
    if latin != (name or ""):
        keys.extend(name_house_keys(latin, address, country))
    return keys


def short_keys(name: str, address: str, country: str) -> list[str]:
    house = _house(address)
    if not house:
        return []
    c = _country(country)
    toks = [t for t in (significant_name_tokens(name) or name_tokens(name)) if len(t) >= 5]
    seen = set()
    out = []
    for t in toks:
        key = f"{c}|th|{t}_{house}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def non_latin(text: str) -> bool:
    return any(ord(ch) > 127 for ch in (text or ""))


def categorize(a, b) -> str:
    if (a[2] or "").strip().upper() != (b[2] or "").strip().upper():
        return "country_mismatch"
    if non_latin(a[0]) or non_latin(b[0]):
        return "transliteration"
    nn1, nn2 = normalize_name(a[0]), normalize_name(b[0])
    aa1, aa2 = normalize_address(a[1]), normalize_address(b[1])
    name_ratio = fuzz.ratio(nn1, nn2) / 100 if nn1 and nn2 else 0.0
    name_sort = fuzz.token_sort_ratio(nn1, nn2) / 100 if nn1 and nn2 else 0.0
    addr_sort = fuzz.token_sort_ratio(aa1, aa2) / 100 if aa1 and aa2 else -1.0
    lev = Levenshtein.normalized_similarity(nn1, nn2) if nn1 and nn2 else 0.0
    raw1, raw2 = (a[0] or "").lower(), (b[0] or "").lower()
    if ".com" in raw1 or ".com" in raw2 or raw1.endswith(".c0m") or raw2.endswith(".c0m"):
        if name_sort < 0.75:
            return "trade_or_domain_name"
    if nn1 and nn1 == nn2:
        return "exact_name_blocked_as_too_common"
    if name_sort >= 0.92 and name_ratio < 0.85:
        return "word_order"
    t1, t2 = set(nn1.split()), set(nn2.split())
    if t1 and t2 and (t1 <= t2 or t2 <= t1) and t1 != t2:
        return "abbreviation_or_shortened_name"
    if name_ratio >= 0.82 or lev >= 0.82:
        return "name_typo"
    if not aa1 or not aa2:
        return "missing_address"
    p1, p2 = _pin(a[1]), _pin(b[1])
    if p1 and p2 and p1 != p2:
        return "pin_mismatch"
    if bool(p1) != bool(p2) and name_sort >= 0.7:
        return "pin_missing"
    h1, h2 = _house(a[1]), _house(b[1])
    if h1 and h2 and h1 != h2 and name_sort >= 0.7:
        return "house_number_difference"
    if addr_sort >= 0.45 and name_sort < 0.6:
        return "different_name_similar_address"
    l1, l2 = _loc(a[1]), _loc(b[1])
    if l1 and l2 and l1 != l2 and name_sort >= 0.6:
        return "locality_variation"
    if max(len(t1), len(t2)) <= 1:
        return "short_or_common_name"
    if addr_sort >= 0.35:
        return "address_typo_or_reorder"
    return "other"


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0
    i = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[i]


def summarize(pairs) -> dict:
    bt = bf = 0
    oracle = []
    cc = []
    for true, cands in pairs:
        cset = set(cands)
        bt += len(true)
        bf += len(true & cset)
        oracle.append(score_entity(true & cset, true))
        cc.append(len(cset))
    return {
        "pair_recall": round(bf / max(bt, 1), 4),
        "avg": round(sum(cc) / len(cc), 2),
        "median": statistics.median(cc),
        "p95": pct(cc, 95),
        "p99": pct(cc, 99),
        "max": max(cc),
        "zero_pct": round(sum(1 for x in cc if x == 0) / len(cc), 4),
        "oracle_f05": round(sum(oracle) / len(oracle), 4),
        "retrieved": bf,
        "true_links": bt,
    }


def main() -> None:
    gt = _load_gt_sample("dataset/train/train_ground_truth.tsv", 400)
    s1 = _scan_ids("dataset/train/train_source1.tsv", {sid for sid, _ in gt})
    records = [(sid, *s1[sid]) for sid, _ in gt if sid in s1]
    conn = sqlite3.connect("artifacts/train_blocks_v2.sqlite")
    current = candidates_for(conn, records, 12)
    conn.close()
    packed = []
    for sid, true in gt:
        if sid not in s1:
            continue
        packed.append((sid, true, current.get(sid, []), s1[sid]))
    val = packed[int(len(packed) * 0.7) :]
    print(f"val_s1={len(val)}", flush=True)

    watch_t, watch_s = set(), set()
    val_t, val_s = {}, {}
    for sid, true, cands, a in val:
        tk = translit_keys(*a)
        sk = short_keys(*a)
        val_t[sid] = tk
        val_s[sid] = sk
        watch_t.update(tk)
        watch_s.update(sk)
    print(f"watch translit={len(watch_t)} short={len(watch_s)}", flush=True)

    post_t, post_s = defaultdict(list), defaultdict(list)
    dead_t, dead_s = set(), set()
    texts = {}
    need = set()
    for _, true, _, _ in val:
        need.update(true)

    n = 0
    for path in ("dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"):
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                eid = row["entity_id"]
                name = row.get("business_name") or ""
                addr = row.get("business_address") or ""
                country = row.get("country") or ""
                if eid in need and eid not in texts:
                    texts[eid] = (name, addr, country)
                for key in translit_keys(name, addr, country):
                    if key not in watch_t or key in dead_t:
                        continue
                    bucket = post_t[key]
                    bucket.append(eid)
                    if len(bucket) > MAX_DF:
                        dead_t.add(key)
                        del post_t[key]
                for key in short_keys(name, addr, country):
                    if key not in watch_s or key in dead_s:
                        continue
                    bucket = post_s[key]
                    bucket.append(eid)
                    if len(bucket) > MAX_DF:
                        dead_s.add(key)
                        del post_s[key]
                n += 1
                if n % 2_000_000 == 0:
                    print(f"scanned {n:,}", flush=True)
    print(f"scanned {n:,}", flush=True)

    def retrieve(keys, post):
        seen = []
        got = set()
        for key in keys:
            for eid in post.get(key, ()):
                if eid not in got:
                    got.add(eid)
                    seen.append(eid)
        return seen

    cur_pairs, tr_pairs, sh_pairs, un_pairs = [], [], [], []
    groups = {
        "transliteration": [],
        "abbreviation_or_shortened_name": [],
        "name_typo": [],
    }
    for sid, true, cands, a in val:
        tr = retrieve(val_t[sid], post_t)
        sh = retrieve(val_s[sid], post_s)
        union = list(dict.fromkeys(list(cands) + tr + sh))
        cur_pairs.append((true, cands))
        tr_pairs.append((true, tr))
        sh_pairs.append((true, sh))
        un_pairs.append((true, union))
        cset = set(cands)
        for mid in true:
            if mid in cset:
                continue
            b = texts.get(mid)
            if not b:
                continue
            lab = categorize(a, b)
            if lab in groups:
                groups[lab].append((mid, set(tr), set(sh), set(union)))

    def recovered(pairs, which):
        # which: 0 translit, 1 short, 2 union
        return sum(1 for mid, tr, sh, un in pairs if mid in (tr, sh, un)[which])

    report = {
        "current": summarize(cur_pairs),
        "transliteration": summarize(tr_pairs),
        "short_name_house": summarize(sh_pairs),
        "combined": summarize(un_pairs),
        "targets": {
            lab: {
                "n": len(pairs),
                "recovered_by_transliteration": recovered(pairs, 0),
                "recovered_by_short": recovered(pairs, 1),
                "recovered_by_combined": recovered(pairs, 2),
                "still_missing_after_combined": len(pairs) - recovered(pairs, 2),
            }
            for lab, pairs in groups.items()
        },
        "dead_translit_keys": len(dead_t),
        "dead_short_keys": len(dead_s),
        "max_df": MAX_DF,
    }
    Path("artifacts/targeted_blockers.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("WROTE artifacts/targeted_blockers.json", flush=True)


if __name__ == "__main__":
    main()
