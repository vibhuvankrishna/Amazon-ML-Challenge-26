"""Stride sample of test pairs. Read-only. Compares features and key-hit rates to production lists."""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path("/Users/m.harshithakrishna/Desktop/student_resource")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
csv.field_size_limit(10_000_000)

from rank_match import (  # noqa: E402
    block_keys,
    cut_ranked,
    decide_matches,
    decision_features,
    pools_from_index,
)

NAMES = [
    "name_s", "name_jaccard", "name_contain", "n_shared_name", "rare_shared",
    "name_ratio", "name_set", "name_lev", "exact_name", "name_len_ratio",
    "short_shared", "short_query", "addr_sort", "addr_set", "addr_jaccard",
    "addr_ratio", "addr_contain", "exact_addr", "house", "pin", "loc",
    "house_conflict", "pin_conflict", "addr_empty", "n_blockers", "block_pos",
    "name_and_house", "name_and_pin", "weak_name_strong_addr", "name_agree_addr_conflict",
]
STRIDE = 866  # about 2,000 of 1,732,544


def kind_of(key: str) -> str:
    parts = key.split("|")
    return parts[1] if len(parts) >= 3 else "?"


def pct(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return {}
    return {
        "n": int(len(arr)),
        "min": round(float(arr.min()), 4),
        "p50": round(float(np.median(arr)), 4),
        "mean": round(float(arr.mean()), 4),
        "p90": round(float(np.quantile(arr, 0.90)), 4),
        "p95": round(float(np.quantile(arr, 0.95)), 4),
        "max": round(float(arr.max()), 4),
    }


def main():
    sample = []
    s1 = ROOT / "dataset/test/test_source1.tsv"
    match = ROOT / "output_v4/matching_results.tsv"
    cand = ROOT / "output_v4/candidate_pairs.tsv"
    with s1.open(newline="") as a, match.open(newline="") as b, cand.open(newline="") as c:
        ra, rb, rc = (csv.DictReader(f, delimiter="\t") for f in (a, b, c))
        for i, (row, mr, cr) in enumerate(zip(ra, rb, rc)):
            if i % STRIDE:
                continue
            sample.append(
                {
                    "eid": row["entity_id"],
                    "name": row.get("business_name") or "",
                    "addr": row.get("business_address") or "",
                    "country": (row.get("country") or "").strip().upper() or "UNK",
                    "pred": [x for x in (mr.get("matched_entity_ids") or "").split(",") if x],
                    "file_cands": [x for x in (cr.get("candidate_entity_ids") or "").split(",") if x],
                }
            )
    print("sample", len(sample), flush=True)
    blob = joblib.load(ROOT / "artifacts/rank_model_hgb_v2.joblib")
    model, rule = blob["model"], blob["rule"]
    block = sqlite3.connect(f"file:{ROOT / 'artifacts/test_blocks_union.sqlite'}?mode=ro", uri=True)
    text = sqlite3.connect(f"file:{ROOT / 'artifacts/test_s23.sqlite'}?mode=ro", uri=True)
    key_hit = Counter()
    key_miss = Counter()
    pool_sizes = []
    list_match = 0
    vectors = []
    accepted = []
    best_rows = []
    max_ps = []
    all_ps = []
    by_country = Counter()
    country_zero = Counter()
    country_n = Counter()
    multi_blocker_pools = 0
    for start in range(0, len(sample), 200):
        chunk = sample[start : start + 200]
        records = [(r["eid"], r["name"], r["addr"], r["country"]) for r in chunk]
        pools = pools_from_index(block, records)
        need = list({cid for sources in pools.values() for cid in sources})
        rec = {}
        for i in range(0, len(need), 800):
            part = need[i : i + 800]
            if not part:
                continue
            q = ",".join("?" * len(part))
            for eid, name, addr, country in text.execute(
                f"SELECT entity_id, name, addr, country FROM entities WHERE entity_id IN ({q})",
                part,
            ):
                rec[eid] = (name or "", addr or "", country or "")
        # key hit rates from this chunk's own keys
        keys = []
        owner = []
        for r in chunk:
            ks = block_keys(r["name"], r["addr"], r["country"])
            for k in ks:
                keys.append(k)
                owner.append(k)
        found = set()
        for i in range(0, len(keys), 400):
            part = list(dict.fromkeys(keys[i : i + 400]))
            q = ",".join("?" * len(part))
            for (k,) in block.execute(f"SELECT DISTINCT k FROM kv WHERE k IN ({q})", part):
                found.add(k)
        for r in chunk:
            ks = block_keys(r["name"], r["addr"], r["country"])
            for k in ks:
                kind = kind_of(k)
                if k in found:
                    key_hit[kind] += 1
                else:
                    key_miss[kind] += 1
            sources = pools.get(r["eid"], {})
            pool_sizes.append(len(sources))
            if any(len(v) >= 2 for v in sources.values()):
                multi_blocker_pools += 1
            cids = cut_ranked((r["name"], r["addr"], r["country"]), sources, rec, 12)
            if cids == r["file_cands"]:
                list_match += 1
            country_n[r["country"]] += 1
            if not cids:
                country_zero[r["country"]] += 1
                max_ps.append(0.0)
                continue
            vecs, ids = [], []
            query = (r["name"], r["addr"], r["country"])
            for pos, cid in enumerate(cids, start=1):
                other = rec.get(cid)
                if not other:
                    continue
                vecs.append(decision_features(query, other, len(sources.get(cid, ())), pos))
                ids.append(cid)
            if not vecs:
                max_ps.append(0.0)
                continue
            X = np.vstack(vecs)
            proba = model.predict_proba(X)[:, 1]
            kept = decide_matches(vecs, ids, proba, rule)
            vectors.append(X)
            if kept:
                sel = [vecs[i] for i, cid in enumerate(ids) if cid in kept]
                accepted.append(np.vstack(sel))
            j = int(np.argmax(proba))
            best_rows.append(vecs[j])
            max_ps.append(float(proba[j]))
            all_ps.extend(float(p) for p in proba)
            by_country[r["country"]] += 1
        print("scored", min(start + 200, len(sample)), flush=True)
    block.close()
    text.close()
    V = np.vstack(vectors) if vectors else np.zeros((1, 30))
    A = np.vstack(accepted) if accepted else np.zeros((1, 30))
    B = np.vstack(best_rows) if best_rows else np.zeros((1, 30))
    feats = []
    for i, name in enumerate(NAMES):
        feats.append(
            {
                "feature": name,
                "cand_mean": round(float(V[:, i].mean()), 4),
                "cand_p50": round(float(np.median(V[:, i])), 4),
                "accepted_mean": round(float(A[:, i].mean()), 4) if accepted else None,
                "best_mean": round(float(B[:, i].mean()), 4) if best_rows else None,
            }
        )
    kinds = sorted(set(key_hit) | set(key_miss))
    kind_rows = []
    for kind in kinds:
        h, m = key_hit[kind], key_miss[kind]
        kind_rows.append({"kind": kind, "hit": h, "miss": m, "hit_rate": round(h / max(h + m, 1), 4)})
    kind_rows.sort(key=lambda r: -(r["hit"] + r["miss"]))
    payload = {
        "stride": STRIDE,
        "n": len(sample),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "candidate_list_exact_match": list_match,
        "candidate_list_match_rate": round(list_match / max(len(sample), 1), 4),
        "pool_before_k": pct(pool_sizes),
        "pool_gt_12": round(sum(1 for n in pool_sizes if n > 12) / max(len(pool_sizes), 1), 4),
        "pool_eq_0": round(sum(1 for n in pool_sizes if n == 0) / max(len(pool_sizes), 1), 4),
        "entities_with_a_multi_blocker_candidate": round(multi_blocker_pools / max(len(sample), 1), 4),
        "features": feats,
        "max_proba": pct(max_ps),
        "all_proba": pct(all_ps),
        "key_kinds": kind_rows,
        "sample_countries": [
            {
                "country": cc,
                "n": country_n[cc],
                "zero_pool": country_zero[cc],
                "zero_pool_rate": round(country_zero[cc] / country_n[cc], 4),
            }
            for cc, _ in country_n.most_common(15)
        ],
    }
    path = ROOT / "artifacts/analysis/gap_test_sample.json"
    path.write_text(json.dumps(payload))
    print("list_match", payload["candidate_list_match_rate"], "pool", payload["pool_before_k"], flush=True)
    print("max_p", payload["max_proba"], flush=True)
    print("kinds", kind_rows, flush=True)
    print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
