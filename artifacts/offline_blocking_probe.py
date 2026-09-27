"""Offline holdout probe. Does not write test predictions or change the scorer."""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "code/business_entity_resolution/src")

from ber.metrics import score_entity
from ber.normalize import (
    address_tokens,
    extract_digits,
    name_tokens,
    normalize_address,
    normalize_name,
    significant_name_tokens,
)
from rank_match import _load_gt_sample, _scan_ids, block_keys, candidates_for

MAX_DF = 40


def _c(country: str) -> str:
    return (country or "").strip().upper() or "UNK"


def _toks(name: str) -> list[str]:
    return [t for t in (significant_name_tokens(name) or name_tokens(name)) if len(t) >= 3]


def _tri(token: str) -> list[str]:
    if len(token) < 6:
        return []
    return [token[i : i + 3] for i in range(len(token) - 2)]


def _house(address: str) -> str:
    ds = [d for d in extract_digits(address or "") if 1 <= len(d) <= 6]
    return ds[0] if ds else ""


def _pin(address: str) -> str:
    for d in extract_digits(address or ""):
        if len(d) >= 5:
            return d
    return ""


def _loc(address: str) -> str:
    ts = [t for t in address_tokens(address or "") if len(t) >= 6 and not t.isdigit()]
    return max(ts, key=len) if ts else ""


def keys_name_tri(name, address, country):
    c = _c(country)
    out = []
    for t in _toks(name):
        if len(t) >= 7:
            out.extend(f"{c}|t|{g}" for g in _tri(t)[:4])
    return out


def keys_sorted(name, address, country):
    toks = _toks(name)
    if len(toks) < 2:
        return []
    return [f"{_c(country)}|s|{'|'.join(sorted(toks))}"]


def keys_name_house(name, address, country):
    nn = normalize_name(name)
    h = _house(address)
    if len(nn) >= 8 and h:
        return [f"{_c(country)}|nh|{nn}_{h}"]
    return []


def keys_house_loc(name, address, country):
    h, loc = _house(address), _loc(address)
    if h and loc:
        return [f"{_c(country)}|hl|{h}_{loc}"]
    return []


def keys_addr_tri(name, address, country):
    loc = _loc(address)
    if len(loc) < 8:
        return []
    c = _c(country)
    return [f"{c}|a|{g}" for g in _tri(loc)[:4]]


def keys_rare_token(name, address, country):
    c = _c(country)
    return [f"{c}|r|{t}" for t in _toks(name) if len(t) >= 6]


STRATEGIES = {
    "name_trigram": keys_name_tri,
    "sorted_all_tokens": keys_sorted,
    "name_plus_house": keys_name_house,
    "house_plus_locality": keys_house_loc,
    "address_trigram": keys_addr_tri,
    "single_rare_token": keys_rare_token,
}


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[i]


def non_latin(s: str) -> bool:
    return any(ord(ch) > 127 for ch in (s or ""))


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


def stats_for(rows, cand_of) -> dict:
    bt = bf = 0
    oracle = []
    cc = []
    for true, cands in rows:
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
        "max": max(cc) if cc else 0,
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

    watch = {name: set() for name in STRATEGIES}
    val_keys = {}
    for sid, true, cands, a in val:
        per = {}
        for name, fn in STRATEGIES.items():
            ks = fn(*a)
            per[name] = ks
            watch[name].update(ks)
        val_keys[sid] = per
    print({k: len(v) for k, v in watch.items()}, flush=True)

    postings = {name: defaultdict(list) for name in STRATEGIES}
    dead = {name: set() for name in STRATEGIES}
    texts = {}
    need = set()
    for _, true, _, _ in val:
        need.update(true)
    n = 0
    for path in (
        "dataset/train/train_source2.tsv",
        "dataset/train/train_source3.tsv",
    ):
        import csv

        csv.field_size_limit(10_000_000)
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row["entity_id"]
                name = row.get("business_name") or ""
                addr = row.get("business_address") or ""
                country = row.get("country") or ""
                if eid in need and eid not in texts:
                    texts[eid] = (name, addr, country)
                for sname, fn in STRATEGIES.items():
                    w = watch[sname]
                    if not w:
                        continue
                    for key in fn(name, addr, country):
                        if key not in w or key in dead[sname]:
                            continue
                        bucket = postings[sname][key]
                        bucket.append(eid)
                        if len(bucket) > MAX_DF:
                            dead[sname].add(key)
                            del postings[sname][key]
                n += 1
                if n % 2_000_000 == 0:
                    print(f"scanned {n:,}", flush=True)
    print(f"scanned {n:,} texts={len(texts):,}", flush=True)

    def retrieve(sid, sname):
        found = []
        seen = set()
        for key in val_keys[sid][sname]:
            for eid in postings[sname].get(key, ()):
                if eid not in seen:
                    seen.add(eid)
                    found.append(eid)
        return found

    strategy_rows = {}
    for sname in STRATEGIES:
        pairs = []
        for sid, true, _, _ in val:
            pairs.append((true, retrieve(sid, sname)))
        strategy_rows[sname] = stats_for(pairs, None)
        print(sname, json.dumps(strategy_rows[sname]), flush=True)

    # unions of the specific (non-single-token) strategies, then top-hit trim
    use = ["name_trigram", "sorted_all_tokens", "name_plus_house", "house_plus_locality", "address_trigram"]
    union_sets = []
    for sid, true, _, _ in val:
        scores = Counter()
        for sname in use:
            for eid in retrieve(sid, sname):
                scores[eid] += 1
        union_sets.append((true, scores))

    def trimmed(k):
        pairs = []
        for true, scores in union_sets:
            ranked = [eid for eid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]
            pairs.append((true, ranked))
        return stats_for(pairs, None)

    combined = {"union_all": trimmed(10_000), "union_top12": trimmed(12), "union_top20": trimmed(20)}
    for name, st in combined.items():
        print(name, json.dumps(st), flush=True)

    current_pairs = [(true, cands) for _, true, cands, _ in val]
    current_stats = stats_for(current_pairs, None)
    print("current_v2", json.dumps(current_stats), flush=True)

    # categorize misses vs current v2 candidates
    reasons = Counter()
    key_fail = Counter()
    examples = defaultdict(list)
    missed = 0
    true_n = 0
    for sid, true, cands, a in val:
        cset = set(cands)
        kinds_a = {k.split("|", 2)[1] for k in block_keys(*a) if k.count("|") >= 2}
        for mid in true:
            true_n += 1
            if mid in cset:
                continue
            missed += 1
            b = texts.get(mid)
            if not b:
                reasons["missing_text"] += 1
                continue
            kinds_b = {k.split("|", 2)[1] for k in block_keys(*b) if k.count("|") >= 2}
            shared = kinds_a & kinds_b
            if shared:
                key_fail["shared_kind_" + ",".join(sorted(shared))] += 1
            else:
                key_fail["no_shared_kind"] += 1
            lab = categorize(a, b)
            reasons[lab] += 1
            if len(examples[lab]) < 2:
                examples[lab].append(
                    {
                        "n1": a[0][:55],
                        "n2": b[0][:55],
                        "a1": (a[1] or "")[:45],
                        "a2": (b[1] or "")[:45],
                        "shared_key_kinds": sorted(shared),
                    }
                )

    report = {
        "current_v2": current_stats,
        "strategies": strategy_rows,
        "combined": combined,
        "missed": missed,
        "true_links": true_n,
        "reasons": dict(reasons),
        "key_fail_top": key_fail.most_common(12),
        "examples": examples,
        "max_df": MAX_DF,
    }
    Path("artifacts/blocking_offline.json").write_text(json.dumps(report, indent=2, default=list))
    print("WROTE artifacts/blocking_offline.json", flush=True)


if __name__ == "__main__":
    main()
