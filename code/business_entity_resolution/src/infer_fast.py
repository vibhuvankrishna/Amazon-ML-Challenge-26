"""FAST full-test inference (single-process, slim index, cheap scores).

Designed to finish in ~1–2 hours on a laptop using existing
artifacts/test_s23.sqlite. No AWS.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ber.io_utils import format_id_list, read_source, write_candidate_pairs, write_matching_results
from ber.normalize import extract_digits, name_tokens, normalize_name, significant_name_tokens


def fast_keys(name: str, address: str, country: str) -> set[str]:
    """Only selective keys so the index stays small and scoring stays in RAM."""
    c = (country or "").strip().upper() or "UNK"
    toks = significant_name_tokens(name) or name_tokens(name)
    keys: set[str] = set()
    if len(toks) >= 2:
        keys.add(f"{c}|n2|{toks[0]}_{toks[1]}")
    elif toks:
        keys.add(f"{c}|n1|{toks[0]}")
    for d in extract_digits(address or ""):
        if len(d) >= 5:
            keys.add(f"{c}|pin|{d}")
            break
    return keys


def pair_score(n1: str, a1: str, c1: str, n2: str, a2: str, c2: str) -> float:
    """Precision-first score. Weak name overlap is rejected even if a blocking key hit."""
    if (c1 or "").upper() != (c2 or "").upper():
        return 0.0
    nn1, nn2 = normalize_name(n1), normalize_name(n2)
    if not nn1 or not nn2:
        return 0.0
    ratio = fuzz.ratio(nn1, nn2) / 100.0
    partial = fuzz.partial_ratio(nn1, nn2) / 100.0
    token_set = fuzz.token_set_ratio(nn1, nn2) / 100.0
    jw = JaroWinkler.normalized_similarity(nn1, nn2)
    t1, t2 = set(nn1.split()), set(nn2.split())
    jac = (len(t1 & t2) / len(t1 | t2)) if (t1 or t2) else 0.0
    d1, d2 = extract_digits(a1), extract_digits(a2)
    if d1 and d2:
        dig = len(d1 & d2) / len(d1 | d2)
    else:
        dig = 0.0
    # Different businesses that share only a first token stay below the cutoff.
    if max(ratio, token_set) < 0.72 and dig < 0.5:
        return 0.0
    score = 0.35 * ratio + 0.25 * token_set + 0.20 * jw + 0.10 * partial + 0.10 * jac
    if dig >= 0.5 and ratio >= 0.75:
        score = max(score, 0.93)
    elif d1 and d2 and dig == 0.0 and ratio < 0.92:
        # Same-looking name, conflicting house/PIN digits → likely a different site.
        score *= 0.75
    return float(score)


def build_index(db_path: Path, max_per_key: int = 8) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    conn = sqlite3.connect(str(db_path))
    total = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    print(f"FAST index build ({total:,} entities)…", flush=True)
    t0 = time.time()
    cur = conn.execute("SELECT entity_id, name, addr, country FROM entities")
    n = 0
    while True:
        rows = cur.fetchmany(150_000)
        if not rows:
            break
        for eid, name, addr, country in rows:
            for key in fast_keys(name or "", addr or "", country or ""):
                bucket = index[key]
                if len(bucket) < max_per_key:
                    bucket.append(eid)
        n += len(rows)
        if n % 1_000_000 == 0:
            print(
                f"  {n:,}/{total:,}  keys={len(index):,}  [{time.time()-t0:.0f}s]",
                flush=True,
            )
    conn.close()
    print(f"Index ready: keys={len(index):,} in {time.time()-t0:.0f}s", flush=True)
    return dict(index)


def fetch_many(conn: sqlite3.Connection, ids: list[str]) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    for i in range(0, len(ids), 800):
        part = ids[i : i + 800]
        q = ",".join("?" * len(part))
        for eid, name, addr, country in conn.execute(
            f"SELECT entity_id, name, addr, country FROM entities WHERE entity_id IN ({q})",
            part,
        ):
            out[eid] = (name or "", addr or "", country or "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--db", default="artifacts/test_s23.sqlite")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--chunk-s1", type=int, default=5000)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Need {db_path} (built earlier).")

    index = build_index(db_path)
    s1 = read_source(
        Path(args.test_dir) / "test_source1.tsv",
        nrows=(args.sample if args.sample > 0 else None),
    )
    print(
        f"Scoring S1={len(s1):,}  thr={args.threshold}  max_cands={args.max_candidates}",
        flush=True,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    match_path = out / "matching_results.tsv"
    cand_path = out / "candidate_pairs.tsv"
    for p in (match_path, cand_path):
        if p.exists():
            p.unlink()

    conn = sqlite3.connect(str(db_path))
    wrote = False
    n = len(s1)
    t0 = time.time()
    nonempty = 0

    for start in tqdm(range(0, n, args.chunk_s1), desc="fast-infer"):
        part = s1.iloc[start : start + args.chunk_s1]
        # candidates
        cand_map: dict[str, list[str]] = {}
        need: set[str] = set()
        for r in part.itertuples(index=False):
            scores: dict[str, int] = defaultdict(int)
            for key in fast_keys(r.business_name, r.business_address, r.country):
                for cid in index.get(key, ()):
                    scores[cid] += 1
            ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
            # Final ranking prefers a SMALL candidate set per S1.
            # Keep multi-key hits first; only fill with single-key hits if needed.
            multi = [c for c, sc in ranked if sc >= 2]
            if len(multi) >= args.max_candidates:
                cids = multi[: args.max_candidates]
            else:
                cids = [c for c, _ in ranked[: args.max_candidates]]
            cand_map[r.entity_id] = cids
            need.update(cids)

        rec = fetch_many(conn, list(need))
        match_ids_list = []
        cand_ids_list = []
        s1_ids = []
        for r in part.itertuples(index=False):
            s1_ids.append(r.entity_id)
            cids = cand_map[r.entity_id]
            cand_ids_list.append(format_id_list(cids))
            kept = []
            for cid in cids:
                b = rec.get(cid)
                if not b:
                    continue
                sc = pair_score(
                    r.business_name, r.business_address, r.country, b[0], b[1], b[2]
                )
                if sc >= args.threshold:
                    kept.append((sc, cid))
            kept.sort(reverse=True)
            mids = [c for _, c in kept]
            if mids:
                nonempty += 1
            match_ids_list.append(format_id_list(mids))

        match_df = pd.DataFrame(
            {"source1_entity_id": s1_ids, "matched_entity_ids": match_ids_list}
        )
        cand_df = pd.DataFrame(
            {"source1_entity_id": s1_ids, "candidate_entity_ids": cand_ids_list}
        )
        match_df.to_csv(match_path, sep="\t", index=False, mode="a", header=not wrote)
        cand_df.to_csv(cand_path, sep="\t", index=False, mode="a", header=not wrote)
        wrote = True

        done = start + len(part)
        if done % 50_000 < args.chunk_s1:
            rate = done / max(time.time() - t0, 1)
            eta = (n - done) / max(rate, 1e-6)
            print(
                f"  progress {done:,}/{n:,}  {rate:.0f} S1/s  ETA {eta/3600:.1f}h  nonempty={nonempty:,}",
                flush=True,
            )

    conn.close()
    elapsed = time.time() - t0
    print(
        f"DONE in {elapsed/60:.1f} min  rows={n:,}  nonempty={nonempty:,}\n"
        f"  {match_path}\n  {cand_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
