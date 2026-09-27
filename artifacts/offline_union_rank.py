"""Holdout union of the useful blockers, then a candidate-budget sweep.

Does not write test predictions, retrain the classifier, or change the holdout.
Character trigrams are omitted: their measured recall was about zero.
"""

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

from ber.metrics import f05, score_entity
from ber.normalize import (
    address_tokens,
    name_tokens,
    normalize_address,
    normalize_name,
    significant_name_tokens,
)
from rank_match import ADDR_STOP, _load_gt_sample, block_keys, pair_signals

csv.field_size_limit(10_000_000)
MAX_DF = 40
KS = (8, 12, 16, 20, 24)

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
_RUNS = re.compile(r"\d+")


def transliterate(text: str) -> str:
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


def _has_deva(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in (text or ""))


def _country(country: str) -> str:
    return (country or "").strip().upper() or "UNK"


def _runs(address: str) -> list[str]:
    return _RUNS.findall((address or "").translate(DEV_DIGITS))


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


def _name_toks(name: str) -> list[str]:
    return [t for t in (significant_name_tokens(name) or name_tokens(name)) if len(t) >= 3]


def keys_short(name, address, country):
    house = _house(address)
    if not house:
        return []
    c = _country(country)
    seen = set()
    out = []
    for t in _name_toks(name):
        if len(t) < 5:
            continue
        key = f"{c}|th|{t}_{house}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def keys_name_house(name, address, country):
    nn = normalize_name(name)
    house = _house(address)
    if len(nn) >= 8 and house:
        return [f"{_country(country)}|nhx|{nn}_{house}"]
    return []


def keys_house_loc(name, address, country):
    house, loc = _house(address), _loc(address)
    if house and loc:
        return [f"{_country(country)}|hl|{house}_{loc}"]
    return []


def keys_sorted(name, address, country):
    toks = _name_toks(name)
    if len(toks) < 2:
        return []
    return [f"{_country(country)}|s|{'|'.join(sorted(toks))}"]


def keys_address(name, address, country):
    c = _country(country)
    at = [
        t
        for t in address_tokens(address or "")
        if len(t) >= 5 and t not in ADDR_STOP and not t.isdigit()
    ]
    keys = []
    if len(at) >= 2:
        long = sorted(at, key=len, reverse=True)[:2]
        x, y = sorted(long)
        keys.append(f"{c}|apx|{x}_{y}")
    if at:
        longest = max(at, key=len)
        if len(longest) >= 8:
            keys.append(f"{c}|alx|{longest}")
        house = _house(address)
        if house and len(longest) >= 6:
            keys.append(f"{c}|hnx|{house}_{longest}")
    return keys


def keys_script(name, address, country):
    """Name+house on the Devanagari-to-Latin form only. The original name is its own blocker."""
    if not _has_deva(name):
        return []
    latin = transliterate(name)
    if normalize_name(latin) == normalize_name(name):
        return []
    return keys_name_house(latin, address, country)


GENERATORS = {
    "short_name_house": keys_short,
    "name_house": keys_name_house,
    "house_locality": keys_house_loc,
    "sorted_name": keys_sorted,
    "address": keys_address,
    "other_script": keys_script,
}


def non_latin(text: str) -> bool:
    return any(ord(ch) > 127 for ch in (text or ""))


def categorize(a, b) -> str:
    if _country(a[2]) != _country(b[2]):
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
        return "exact_name"
    if name_sort >= 0.92 and name_ratio < 0.85:
        return "word_order"
    t1, t2 = set(nn1.split()), set(nn2.split())
    if t1 and t2 and (t1 <= t2 or t2 <= t1) and t1 != t2:
        return "abbreviation_or_shortened_name"
    if name_ratio >= 0.82 or lev >= 0.82:
        return "name_typo"
    if not aa1 or not aa2:
        return "missing_address"
    if addr_sort >= 0.45 and name_sort < 0.6:
        return "different_name_similar_address"
    if addr_sort >= 0.35:
        return "address_typo_or_reorder"
    return "other"


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0
    i = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[i]


def current_pool(conn, records) -> dict[str, list[str]]:
    """Every id the current index returns, before the production top-12 cut."""
    key_of = {}
    all_keys = set()
    for eid, name, addr, country in records:
        ks = block_keys(name, addr, country)
        key_of[eid] = ks
        all_keys.update(ks)
    postings = defaultdict(list)
    keys = list(all_keys)
    for i in range(0, len(keys), 400):
        part = keys[i : i + 400]
        q = ",".join("?" * len(part))
        for k, cid in conn.execute(f"SELECT k, eid FROM kv WHERE k IN ({q})", part):
            postings[k].append(cid)
    out = {}
    for eid, *_ in records:
        seen = set()
        got = []
        for k in key_of[eid]:
            for cid in postings.get(k, ()):
                if cid not in seen:
                    seen.add(cid)
                    got.append(cid)
        out[eid] = got
    return out


def _near(query_toks, cand_toks, shared) -> int:
    hits = 0
    for t in query_toks:
        if t in shared or len(t) < 5:
            continue
        for u in cand_toks:
            if u in shared or abs(len(t) - len(u)) > 2:
                continue
            if fuzz.ratio(t, u) >= 84:
                hits += 1
                break
    return hits


def rank_score(query, cand, n_blockers: int) -> int:
    """Fixed weights. Not fit on this holdout."""
    qn, qa, qc = query
    cn, ca, cc = cand
    qt = _name_toks(qn)
    ct = _name_toks(cn)
    qs, cs = set(qt), set(ct)
    shared = qs & cs
    rare = sum(1 for t in shared if len(t) >= 6)
    nn1, nn2 = normalize_name(qn), normalize_name(cn)
    exact_name = 1 if nn1 and nn1 == nn2 else 0
    aq = [t for t in address_tokens(qa or "") if len(t) >= 4 and not t.isdigit()]
    ac = [t for t in address_tokens(ca or "") if len(t) >= 4 and not t.isdigit()]
    addr_shared = len(set(aq) & set(ac))
    house = 1 if (_house(qa) and _house(qa) == _house(ca)) else 0
    pin = 1 if (_pin(qa) and _pin(qa) == _pin(ca)) else 0
    loc_q, loc_c = _loc(qa), _loc(ca)
    loc = 1 if loc_q and loc_q == loc_c else 0
    country = 1 if _country(qc) == _country(cc) else 0
    return (
        4 * len(shared)
        + 3 * _near(qt, ct, shared)
        + 2 * rare
        + 6 * exact_name
        + 5 * house
        + 4 * pin
        + 3 * loc
        + 2 * addr_shared
        + 3 * n_blockers
        + country
    )


def pool_stats(rows) -> dict:
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
        "retrieved": bf,
        "true_links": bt,
        "avg": round(sum(cc) / len(cc), 2),
        "median": statistics.median(cc),
        "p95": pct(cc, 95),
        "p99": pct(cc, 99),
        "max": max(cc) if cc else 0,
        "zero_pct": round(sum(1 for x in cc if x == 0) / max(len(cc), 1), 4),
        "oracle_f05": round(sum(oracle) / max(len(oracle), 1), 4),
    }


def bucket_recall(ranked, labels, k) -> dict:
    out = {}
    for lab, pairs in labels.items():
        if not pairs:
            out[lab] = {"n": 0, "retrieved": 0, "recall": 0.0}
            continue
        hit = sum(1 for i, mid in pairs if mid in ranked[i][:k])
        out[lab] = {"n": len(pairs), "retrieved": hit, "recall": round(hit / len(pairs), 4)}
    return out


def choose_k(table) -> int:
    """Smallest budget at or above 70% recall whose average list stays <= 20.
    If several qualify, keep the highest oracle. If none do, keep the best oracle.
    """
    eligible = [r for r in table if r["pair_recall"] >= 0.70 and r["avg"] <= 20]
    pool = eligible or table
    pool = sorted(pool, key=lambda r: (r["oracle_f05"], r["pair_recall"], -r["k"]))
    return pool[-1]["k"]


def classify(model, threshold, val, ranked, texts, k):
    import numpy as np

    scores = []
    tp = fp = fn = 0
    single_n = single_ok = 0
    matched_p = []
    matched_r = []
    for i, (sid, true, query) in enumerate(val):
        cands = []
        for cid in ranked[i][:k]:
            b = texts.get(cid)
            if not b:
                continue
            sig = pair_signals(query[0], query[1], query[2], b[0], b[1], b[2])
            if sig is None:
                continue
            cands.append((cid, sig))
        if cands:
            X = np.asarray(
                [(n, a, max(s, -1.0), 1.0 if a < 0 else 0.0) for _, (n, a, s) in cands],
                dtype=np.float32,
            )
            proba = model.predict_proba(X)[:, 1]
            pred = {cid for (cid, _), p in zip(cands, proba) if p >= threshold}
        else:
            pred = set()
        scores.append(score_entity(pred, true))
        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)
        if not true:
            single_n += 1
            single_ok += int(not pred)
        else:
            matched_p.append((len(pred & true) / len(pred)) if pred else 0.0)
            matched_r.append(len(pred & true) / len(true))
    micro_p = tp / max(tp + fp, 1)
    micro_r = tp / max(tp + fn, 1)
    return {
        "k": k,
        "threshold": threshold,
        "macro_f05": round(sum(scores) / len(scores), 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f05": round(f05(micro_p, micro_r), 4),
        "macro_precision_on_matched": round(sum(matched_p) / max(len(matched_p), 1), 4),
        "macro_recall_on_matched": round(sum(matched_r) / max(len(matched_r), 1), 4),
        "false_matches": fp,
        "missed_matches": fn,
        "true_positives": tp,
        "singleton_accuracy": round(single_ok / max(single_n, 1), 4),
        "singletons": single_n,
        "singletons_correct": single_ok,
    }


def main() -> None:
    gt = _load_gt_sample("dataset/train/train_ground_truth.tsv", 400)
    wanted = {sid for sid, _ in gt}
    s1 = {}
    with open("dataset/train/train_source1.tsv", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in wanted:
                s1[row["entity_id"]] = (
                    row.get("business_name") or "",
                    row.get("business_address") or "",
                    row.get("country") or "",
                )
    packed = [(sid, true, s1[sid]) for sid, true in gt if sid in s1]
    val = packed[int(len(packed) * 0.7) :]
    records = [(sid, *query) for sid, _, query in val]
    print(f"val_s1={len(val)}", flush=True)

    conn = sqlite3.connect("artifacts/train_blocks_v2.sqlite")
    current = current_pool(conn, records)
    conn.close()
    cur_rows = [(true, current[sid]) for sid, true, _ in val]
    # Production comparison is the old weight order, top 12, inside this same pool.
    from rank_match import candidates_for

    conn = sqlite3.connect("artifacts/train_blocks_v2.sqlite")
    current_top = candidates_for(conn, records, 12)
    conn.close()
    base_rows = [(true, current_top.get(sid, [])) for sid, true, _ in val]
    base = pool_stats(base_rows)
    print("current_top12", json.dumps(base), flush=True)

    watch = set()
    val_keys = {}
    src_counts = defaultdict(int)
    for sid, _, query in val:
        per = defaultdict(set)
        for sname, fn in GENERATORS.items():
            for key in fn(*query):
                per[key].add(sname)
                watch.add(key)
                src_counts[sname] += 1
        val_keys[sid] = per
    print(dict(src_counts), "watch", len(watch), flush=True)

    post = defaultdict(list)
    dead = set()
    texts = {}
    need = set()
    for sid, true, _ in val:
        need.update(true)
        need.update(current[sid])
        need.update(current_top.get(sid, []))
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
                emitted = set()
                for fn in GENERATORS.values():
                    for key in fn(name, addr, country):
                        if key in emitted or key not in watch or key in dead:
                            continue
                        emitted.add(key)
                        bucket = post[key]
                        bucket.append(eid)
                        if eid not in texts:
                            texts[eid] = (name, addr, country)
                        if len(bucket) > MAX_DF:
                            dead.add(key)
                            del post[key]
                n += 1
                if n % 2_000_000 == 0:
                    print(f"scanned {n:,}", flush=True)
    print(f"scanned {n:,} texts={len(texts):,} dead_keys={len(dead):,}", flush=True)

    ranked = []
    union_rows = []
    multi = 0
    true_multi = 0
    labels = {
        "abbreviation_or_shortened_name": [],
        "name_typo": [],
        "transliteration": [],
    }
    for i, (sid, true, query) in enumerate(val):
        sources = defaultdict(set)
        for cid in current[sid]:
            sources[cid].add("current")
        for key, srcs in val_keys[sid].items():
            for cid in post.get(key, ()):
                sources[cid].update(srcs)
        scored = []
        for cid, src in sources.items():
            b = texts.get(cid)
            if b is None:
                score = len(src)
            else:
                score = rank_score(query, b, len(src))
            scored.append((-score, -len(src), cid))
            if len(src) >= 2:
                multi += 1
        scored.sort()
        order = [cid for _, _, cid in scored]
        ranked.append(order)
        union_rows.append((true, order))
        true_multi += sum(1 for mid in true if len(sources.get(mid, ())) >= 2)
        for mid in true:
            b = texts.get(mid)
            if not b:
                continue
            lab = categorize(query, b)
            if lab in labels:
                labels[lab].append((i, mid))

    untrimmed = pool_stats(union_rows)
    print("union_untrimmed", json.dumps(untrimmed), flush=True)
    print(
        f"candidates_with_2plus_blockers={multi} true_links_with_2plus_blockers={true_multi}",
        flush=True,
    )

    table = []
    for k in KS:
        trimmed = [(true, cands[:k]) for true, cands in union_rows]
        st = pool_stats(trimmed)
        st["k"] = k
        st["vs_baseline_links"] = st["retrieved"] - base["retrieved"]
        st["vs_baseline_recall"] = round(st["pair_recall"] - base["pair_recall"], 4)
        st["buckets"] = bucket_recall(ranked, labels, k)
        table.append(st)
        print(f"K={k}", json.dumps(st), flush=True)

    best_k = choose_k(table)
    print(f"chosen_k={best_k}", flush=True)

    import joblib

    blob = joblib.load("artifacts/rank_model.joblib")
    model, threshold = blob["model"], float(blob["threshold"])
    print(f"frozen_threshold={threshold}", flush=True)
    # Baseline: same frozen classifier on the production top-12 list.
    base_ranked = [current_top.get(sid, []) for sid, _, _ in val]
    baseline_clf = classify(model, threshold, val, base_ranked, texts, 12)
    chosen_clf = classify(model, threshold, val, ranked, texts, best_k)
    print("classifier_current_top12", json.dumps(baseline_clf), flush=True)
    print("classifier_chosen", json.dumps(chosen_clf), flush=True)

    report = {
        "holdout": "stride 400, last 30%",
        "max_df": MAX_DF,
        "excluded": ["name_trigram", "address_trigram", "single_rare_token"],
        "rank_weights": {
            "shared_name_tokens": 4,
            "near_name_token": 3,
            "rare_shared_token_len_ge_6": 2,
            "exact_normalized_name": 6,
            "house_match": 5,
            "pin_match": 4,
            "locality_match": 3,
            "shared_address_tokens": 2,
            "each_blocker": 3,
            "country_match": 1,
        },
        "current_top12": base,
        "union_untrimmed": untrimmed,
        "multi_blocker_candidates": multi,
        "true_links_found_by_2plus_blockers": true_multi,
        "bucket_sizes": {lab: len(pairs) for lab, pairs in labels.items()},
        "by_k": table,
        "chosen_k": best_k,
        "classifier_current_top12": baseline_clf,
        "classifier_chosen": chosen_clf,
        "logged_baseline_val_f05": 0.7022,
    }
    Path("artifacts/union_rank.json").write_text(json.dumps(report, indent=2))
    print("WROTE artifacts/union_rank.json", flush=True)


if __name__ == "__main__":
    main()
