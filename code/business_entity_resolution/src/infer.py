"""Run inference with SQLite-backed S2/S3 lookup — survives limited RAM."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ber.blocking import candidates_to_frame, generate_candidates_from_index
from ber.features import build_feature_matrix
from ber.io_utils import format_id_list, iter_chunks, read_source
from ber.normalize import blocking_keys


def build_sqlite_and_index(
    paths: list[Path],
    db_path: Path,
    max_per_key: int = 300,
    chunksize: int = 150_000,
) -> dict[str, list[str]]:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute(
        "CREATE TABLE entities (entity_id TEXT PRIMARY KEY, name TEXT, addr TEXT, country TEXT)"
    )
    index: dict[str, list[str]] = defaultdict(list)
    insert = conn.executemany
    for path in paths:
        print(f"Indexing {path.name} → sqlite + memory keys…", flush=True)
        batch = []
        for chunk in tqdm(iter_chunks(path, chunksize=chunksize), desc=path.name):
            for row in chunk.itertuples(index=False):
                eid = row.entity_id
                batch.append((eid, row.business_name, row.business_address, row.country))
                for key in blocking_keys(row.business_name, row.business_address, row.country):
                    bucket = index[key]
                    if len(bucket) < max_per_key:
                        bucket.append(eid)
            if len(batch) >= 50_000:
                insert("INSERT OR REPLACE INTO entities VALUES (?,?,?,?)", batch)
                batch.clear()
        if batch:
            insert("INSERT OR REPLACE INTO entities VALUES (?,?,?,?)", batch)
        conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eid ON entities(entity_id)")
    conn.commit()
    conn.close()
    print(f"SQLite ready at {db_path}  index_keys={len(index):,}", flush=True)
    return dict(index)


def fetch_records(conn: sqlite3.Connection, ids: list[str]) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    # chunk IN queries
    for i in range(0, len(ids), 900):
        part = ids[i : i + 900]
        q = ",".join("?" * len(part))
        for eid, name, addr, country in conn.execute(
            f"SELECT entity_id, name, addr, country FROM entities WHERE entity_id IN ({q})",
            part,
        ):
            out[eid] = (name, addr, country)
    return out


def rebuild_index_from_sqlite(
    db_path: Path,
    max_per_key: int = 300,
    batch: int = 100_000,
) -> dict[str, list[str]]:
    """Rebuild blocking index by scanning existing SQLite (no TSV re-read)."""
    from collections import defaultdict

    index: dict[str, list[str]] = defaultdict(list)
    conn = sqlite3.connect(str(db_path))
    total = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    print(f"Rebuilding index from sqlite ({total:,} entities)…", flush=True)
    cur = conn.execute("SELECT entity_id, name, addr, country FROM entities")
    n = 0
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        for eid, name, addr, country in rows:
            for key in blocking_keys(name or "", addr or "", country or ""):
                bucket = index[key]
                if len(bucket) < max_per_key:
                    bucket.append(eid)
        n += len(rows)
        if n % 500_000 == 0 or n >= total:
            print(f"  indexed {n:,}/{total:,}", flush=True)
    conn.close()
    return dict(index)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", default="dataset/test")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--db", default="artifacts/test_s23.sqlite")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--max-candidates", type=int, default=80)
    p.add_argument("--max-per-key", type=int, default=300)
    p.add_argument("--chunk-s1", type=int, default=4000)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument(
        "--reuse-db",
        action="store_true",
        help="Reuse artifacts/test_s23.sqlite; rebuild index from it (skip TSV ingest)",
    )
    p.add_argument("--skip-index-cache", action="store_true", help="Do not joblib-dump the index")
    args = p.parse_args()

    bundle = joblib.load(Path(args.artifacts) / "model.joblib")
    model = bundle["model"]
    threshold = args.threshold if args.threshold is not None else float(bundle["threshold"])
    print(f"threshold={threshold:.3f} max_cands={args.max_candidates}", flush=True)

    test_dir = Path(args.test_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    match_path = out / "matching_results.tsv"
    cand_path = out / "candidate_pairs.tsv"
    for path in (match_path, cand_path):
        if path.exists():
            path.unlink()

    nrows = args.sample if args.sample > 0 else None
    s1 = read_source(test_dir / "test_source1.tsv", nrows=nrows)
    print(f"S1={len(s1):,}", flush=True)

    db_path = Path(args.db)
    index_path = Path(args.artifacts) / "block_index.joblib"

    if args.reuse_db and db_path.exists():
        # Prefer rebuilding from sqlite — joblib cache may be corrupt if a prior run OOM'd
        index = rebuild_index_from_sqlite(db_path, max_per_key=args.max_per_key)
        print(f"Index keys={len(index):,}", flush=True)
    else:
        index = build_sqlite_and_index(
            [test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"],
            db_path,
            max_per_key=args.max_per_key,
        )
        if args.skip_index_cache:
            print("Skipping index cache dump (RAM-safe)", flush=True)
        else:
            print("Dumping block index to disk (can use a lot of RAM)…", flush=True)
            try:
                joblib.dump(index, index_path)
                print(f"Cached index → {index_path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"WARN: could not cache index ({exc}); continuing", flush=True)

    conn = sqlite3.connect(str(db_path))
    wrote_h = False
    n = len(s1)
    for start in tqdm(range(0, n, args.chunk_s1), desc="infer"):
        part = s1.iloc[start : start + args.chunk_s1]
        s1_rec = {
            r.entity_id: (r.business_name, r.business_address, r.country)
            for r in part.itertuples(index=False)
        }
        candidates = generate_candidates_from_index(part, index, max_candidates=args.max_candidates)
        cand_df = candidates_to_frame(candidates)

        # fetch only needed S2/S3 rows
        need_ids = list({cid for cids in candidates.values() for cid in cids})
        s23_rec = fetch_records(conn, need_ids)

        rows = []
        for sid, cids in candidates.items():
            a = s1_rec[sid]
            for cid in cids:
                b = s23_rec.get(cid)
                if not b:
                    continue
                rows.append(
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
        pairs = pd.DataFrame.from_records(rows)
        match_map = {sid: [] for sid in part["entity_id"]}
        if not pairs.empty:
            X = build_feature_matrix(pairs, show_progress=False)
            proba = model.predict_proba(X)[:, 1]
            pairs = pairs.assign(proba=proba)
            kept = pairs[pairs["proba"] >= threshold]
            for sid, group in kept.groupby("source1_entity_id"):
                ordered = group.sort_values("proba", ascending=False)["candidate_entity_id"]
                match_map[sid] = ordered.tolist()

        match_df = pd.DataFrame(
            {
                "source1_entity_id": list(match_map.keys()),
                "matched_entity_ids": [format_id_list(v) for v in match_map.values()],
            }
        )
        match_df.to_csv(match_path, sep="\t", index=False, mode="a", header=not wrote_h)
        cand_df.to_csv(cand_path, sep="\t", index=False, mode="a", header=not wrote_h)
        wrote_h = True

    conn.close()
    print(f"Wrote {match_path}", flush=True)
    print(f"Wrote {cand_path}", flush=True)


if __name__ == "__main__":
    main()
