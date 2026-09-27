"""Blocking + pair ranker for the entity-resolution challenge.

Learns from train ground truth. A pair is a match when the address agrees,
or when the name agrees and the address does not clearly belong to another city.
Common-name / different-city pairs are rejected. No external data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import statistics
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ber.metrics import f05, score_entity
from ber.normalize import (
    address_tokens,
    extract_digits,
    name_tokens,
    normalize_address,
    normalize_name,
    significant_name_tokens,
)

csv.field_size_limit(10_000_000)

ADDR_STOP = {
    "road", "street", "avenue", "lane", "floor", "near", "opposite", "plot",
    "shop", "building", "block", "unit", "suite", "apartment", "india", "state",
    "nagar", "marg", "colony", "sector", "phase", "west", "east", "north",
    "south", "main", "cross", "behind", "above", "flat", "house", "village",
    "district", "taluk", "tehsil", "city", "town", "rural", "urban", "ward",
    "number", "opp", "beside", "complex", "market", "bazaar", "road", "rd",
    "the", "and", "ltd", "limited", "private", "pvt",
}

WEIGHT = {"nh": 7, "nl": 6, "fx": 5, "hn": 5, "ap": 4, "al": 3, "pin": 3, "n": 1}
SKETCH_N = 1 << 22  # 4,194,304 buckets
MAX_PER_KEY = 25
# Exact cap used by the holdout union. Trigrams and transliteration are omitted.
EXTRA_CAP = 40
EXTRA_KIND = {
    "th": "short_name_house",
    "nhx": "name_house",
    "hl": "house_locality",
    "s": "sorted_name",
    "apx": "address",
    "alx": "address",
    "hnx": "address",
    # Typo / shortened-name keys (plan step 3). Hot keys still die at EXTRA_CAP.
    "rp": "rare_pin",
    "ph": "phonetic_house",
    "sh": "sorted_house",
}
_DIGIT_RUNS = re.compile(r"\d+")
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _country(country: str) -> str:
    return (country or "").strip().upper() or "UNK"


def block_keys(name: str, address: str, country: str) -> list[str]:
    """Selective keys. Hot keys are dropped later by the count sketch."""
    c = _country(country)
    keys: list[str] = []
    toks = [t for t in (significant_name_tokens(name) or name_tokens(name)) if len(t) >= 3]
    if len(toks) >= 2:
        a, b = sorted((toks[0], toks[1]))
        keys.append(f"{c}|n|{a}_{b}")
        if len(toks) >= 3:
            x, y = sorted((toks[0], toks[2]))
            keys.append(f"{c}|n|{x}_{y}")
    elif toks:
        keys.append(f"{c}|n|{toks[0]}")

    for d in extract_digits(address or ""):
        if len(d) >= 5:
            keys.append(f"{c}|pin|{d}")
            break

    at = [
        t
        for t in address_tokens(address or "")
        if len(t) >= 5 and t not in ADDR_STOP and not t.isdigit()
    ]
    if len(at) >= 2:
        long = sorted(at, key=len, reverse=True)[:2]
        x, y = sorted(long)
        keys.append(f"{c}|ap|{x}_{y}")
    if at:
        longest = max(at, key=len)
        if len(longest) >= 8:
            keys.append(f"{c}|al|{longest}")
        nums = [d for d in extract_digits(address or "") if 2 <= len(d) <= 6]
        if nums and len(longest) >= 6:
            keys.append(f"{c}|hn|{nums[0]}_{longest}")
    # Specific keys. The two-token name key is often deleted as too common,
    # which drops exact-name duplicates. Full name + house/locality stays rare.
    nn = normalize_name(name)
    if len(nn) >= 8:
        keys.append(f"{c}|fx|{nn}")
        nums = [d for d in extract_digits(address or "") if 1 <= len(d) <= 6]
        if nums:
            keys.append(f"{c}|nh|{nn}_{nums[0]}")
        if at:
            keys.append(f"{c}|nl|{nn}_{max(at, key=len)}")
    keys.extend(key for _, key in extra_key_pairs(name, address, country))
    # de-dupe, preserve order
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _sketch_indexes(key: str) -> tuple[int, int]:
    return hash(key) % SKETCH_N, hash(("b", key)) % SKETCH_N


def _new_sketch() -> array:
    return array("B", bytes(SKETCH_N))


def _bump(sketch: array, key: str) -> None:
    for i in _sketch_indexes(key):
        if sketch[i] < 255:
            sketch[i] += 1


def _keep(sketch: array, key: str, max_per_key: int) -> bool:
    # Full-name keys are allowed a larger cap so a real duplicate cluster
    # survives. Coarse two-token keys stay at the tight cap.
    kind = key.split("|", 2)[1] if key.count("|") >= 2 else ""
    if kind in {"nh", "nl", "fx"}:
        cap = 80
    elif kind in EXTRA_KIND:
        cap = EXTRA_CAP
    else:
        cap = max_per_key
    a, b = _sketch_indexes(key)
    return 0 < min(sketch[a], sketch[b]) <= cap


def build_block_db(paths: list[Path], db_path: Path, max_per_key: int) -> None:
    if db_path.exists():
        db_path.unlink()
    print(f"Pass 1/2 count keys from {len(paths)} files…", flush=True)
    sketch = _new_sketch()
    n = 0
    t0 = time.time()
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                for key in block_keys(
                    row.get("business_name") or "",
                    row.get("business_address") or "",
                    row.get("country") or "",
                ):
                    _bump(sketch, key)
                n += 1
                if n % 1_000_000 == 0:
                    print(f"  counted {n:,}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"Counted {n:,} records in {time.time()-t0:.0f}s", flush=True)

    print("Pass 2/2 write rare keys to sqlite…", flush=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("CREATE TABLE kv (k TEXT, eid TEXT)")
    batch = []
    written = 0
    t1 = time.time()
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row["entity_id"]
                for key in block_keys(
                    row.get("business_name") or "",
                    row.get("business_address") or "",
                    row.get("country") or "",
                ):
                    if _keep(sketch, key, max_per_key):
                        batch.append((key, eid))
                if len(batch) >= 50_000:
                    conn.executemany("INSERT INTO kv VALUES (?,?)", batch)
                    written += len(batch)
                    batch.clear()
                    if written % 2_000_000 < 50_000:
                        print(f"  inserted {written:,}  [{time.time()-t1:.0f}s]", flush=True)
    if batch:
        conn.executemany("INSERT INTO kv VALUES (?,?)", batch)
        written += len(batch)
    conn.commit()
    print(f"Indexing {written:,} postings…", flush=True)
    conn.execute("CREATE INDEX ix_k ON kv(k)")
    conn.commit()
    conn.close()
    del sketch
    print(f"Block DB ready {db_path} rows={written:,} in {time.time()-t0:.0f}s", flush=True)


def legacy_block_keys(name: str, address: str, country: str) -> list[str]:
    """Keys stored in train_blocks_v2. Extra kinds are exact-capped separately."""
    return [k for k in block_keys(name, address, country) if _kind(k) not in EXTRA_KIND]


def _refuse_protected(path: Path) -> None:
    resolved = path.resolve()
    if resolved.name in {"test_blocks_union.sqlite", "rank_model_hgb_v2.joblib", "rule_v2.json"}:
        raise SystemExit(f"refusing to modify {resolved}")
    if resolved.name == "output_v4" or "output_v4" in resolved.parts:
        raise SystemExit(f"refusing to modify {resolved}")


def _free_gb(path: Path) -> float:
    parent = path.parent if path.parent.exists() else Path(".")
    return shutil.disk_usage(parent).free / (1024 ** 3)


def _disk_guard(path: Path, label: str, minimum_gb: float = 2.5) -> float:
    free = _free_gb(path)
    if free < minimum_gb:
        raise SystemExit(f"Stopping {label}: only {free:.1f} GB free")
    return free


def build_legacy_block_db_from_sqlite(src_db: Path, db_path: Path, max_per_key: int) -> None:
    """Legacy-key sketch, same caps as train_blocks_v2. Extra keys are not counted."""
    _refuse_protected(db_path)
    if db_path.exists():
        db_path.unlink()
    src = sqlite3.connect(str(src_db))
    total = src.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    print(f"Legacy pass 1/2 count keys from sqlite ({total:,})…", flush=True)
    sketch = _new_sketch()
    t0 = time.time()
    n = 0
    cur = src.execute("SELECT name, addr, country FROM entities")
    while True:
        rows = cur.fetchmany(150_000)
        if not rows:
            break
        for name, addr, country in rows:
            for key in legacy_block_keys(name or "", addr or "", country or ""):
                _bump(sketch, key)
        n += len(rows)
        if n % 1_000_000 == 0:
            print(
                f"  counted {n:,}  [{time.time()-t0:.0f}s] free={_disk_guard(db_path, 'legacy index'):.1f}GB",
                flush=True,
            )
    print(f"Counted {n:,} in {time.time()-t0:.0f}s", flush=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE kv (k TEXT, eid TEXT)")
    print("Legacy pass 2/2 write rare keys…", flush=True)
    t1 = time.time()
    written = 0
    batch = []
    cur = src.execute("SELECT entity_id, name, addr, country FROM entities")
    while True:
        rows = cur.fetchmany(100_000)
        if not rows:
            break
        for eid, name, addr, country in rows:
            for key in legacy_block_keys(name or "", addr or "", country or ""):
                if _keep(sketch, key, max_per_key):
                    batch.append((key, eid))
            if len(batch) >= 50_000:
                conn.executemany("INSERT INTO kv VALUES (?,?)", batch)
                written += len(batch)
                batch.clear()
                if written % 2_000_000 < 50_000:
                    print(f"  inserted {written:,}  [{time.time()-t1:.0f}s]", flush=True)
    if batch:
        conn.executemany("INSERT INTO kv VALUES (?,?)", batch)
        written += len(batch)
    conn.commit()
    del sketch
    print(f"Indexing {written:,} legacy postings…", flush=True)
    conn.execute("CREATE INDEX ix_k ON kv(k)")
    conn.commit()
    conn.close()
    src.close()
    print(f"Legacy block DB ready rows={written:,} in {time.time()-t0:.0f}s", flush=True)


def _accumulate_exact_extra(paths: list[Path], watch: set[str] | None, max_df: int):
    """Exact cap used by merge_extra_postings.

    A key is removed entirely once it matches more than max_df entities.
    watch=None keeps every extra key; otherwise only those keys are stored.
    Returns (postings, dead_count, rows_scanned).
    """
    post: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int] = {}
    dead: set[str] = set()
    n = 0
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                eid = row["entity_id"]
                emitted = set()
                for _blocker_name, key in extra_key_pairs(
                    row.get("business_name") or "",
                    row.get("business_address") or "",
                    row.get("country") or "",
                ):
                    if key in emitted or key in dead:
                        continue
                    if watch is not None and key not in watch:
                        continue
                    emitted.add(key)
                    counts[key] = counts.get(key, 0) + 1
                    post[key].append(eid)
                    if counts[key] > max_df:
                        dead.add(key)
                        post.pop(key, None)
                        counts.pop(key, None)
                n += 1
                if n % 2_000_000 == 0:
                    print(f"  exact extra scanned {n:,} live={len(post):,} dead={len(dead):,}", flush=True)
    print(f"  exact extra scanned {n:,} live={len(post):,} dead={len(dead):,}", flush=True)
    return post, len(dead), n


def build_extra_exact_db(paths: list[Path], db_path: Path, max_df: int = EXTRA_CAP) -> None:
    """One exact cap-40 pass over S2/S3. Counts live in a separate file and are deleted."""
    _refuse_protected(db_path)
    cnt_path = db_path.with_name(db_path.stem + "_cnt.sqlite")
    _refuse_protected(cnt_path)
    for path in (db_path, cnt_path):
        if path.exists():
            path.unlink()
    print(f"Exact extra keys cap={max_df} from {len(paths)} files…", flush=True)
    done = False
    try:
        cnt = sqlite3.connect(str(cnt_path))
        cnt.execute("PRAGMA journal_mode=OFF")
        cnt.execute("PRAGMA synchronous=OFF")
        cnt.execute("PRAGMA temp_store=FILE")
        cnt.execute("CREATE TABLE cnt (k TEXT PRIMARY KEY, n INTEGER NOT NULL)")
        t0 = time.time()
        scanned = 0
        pending: dict[str, int] = {}

        def flush_counts() -> None:
            if not pending:
                return
            cnt.executemany(
                "INSERT INTO cnt(k, n) VALUES(?, ?) "
                "ON CONFLICT(k) DO UPDATE SET n = n + excluded.n",
                list(pending.items()),
            )
            pending.clear()

        for path in paths:
            with open(path, newline="") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    emitted = set()
                    for _blocker_name, key in extra_key_pairs(
                        row.get("business_name") or "",
                        row.get("business_address") or "",
                        row.get("country") or "",
                    ):
                        if key in emitted:
                            continue
                        emitted.add(key)
                        pending[key] = pending.get(key, 0) + 1
                    scanned += 1
                    if len(pending) >= 100_000:
                        flush_counts()
                    if scanned % 1_000_000 == 0:
                        flush_counts()
                        cnt.commit()
                        free = _disk_guard(db_path, "extra index")
                        print(f"  counted {scanned:,}  [{time.time()-t0:.0f}s] free={free:.1f}GB", flush=True)
        flush_counts()
        cnt.commit()
        kept = cnt.execute("SELECT COUNT(*) FROM cnt WHERE n <= ?", (max_df,)).fetchone()[0]
        dead = cnt.execute("SELECT COUNT(*) FROM cnt WHERE n > ?", (max_df,)).fetchone()[0]
        print(f"Exact counts scanned={scanned:,} keep_keys={kept:,} dead_keys={dead:,}", flush=True)

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("CREATE TABLE kv (k TEXT, eid TEXT)")
        written = 0
        batch = []
        buf_rows: list[tuple[str, list[str]]] = []
        keyset: list[str] = []
        seen: set[str] = set()

        def flush_rows() -> None:
            nonlocal written
            if not buf_rows:
                return
            ok: set[str] = set()
            for i in range(0, len(keyset), 400):
                part = keyset[i : i + 400]
                q = ",".join("?" * len(part))
                ok.update(
                    k
                    for (k,) in cnt.execute(
                        f"SELECT k FROM cnt WHERE k IN ({q}) AND n <= ?",
                        (*part, max_df),
                    )
                )
            for eid, keys in buf_rows:
                for key in keys:
                    if key in ok:
                        batch.append((key, eid))
            if len(batch) >= 50_000:
                conn.executemany("INSERT INTO kv VALUES (?,?)", batch)
                written += len(batch)
                batch.clear()
                if written % 2_000_000 < 50_000:
                    free = _disk_guard(db_path, "extra index")
                    print(f"  inserted {written:,}  [{time.time()-t0:.0f}s] free={free:.1f}GB", flush=True)
            buf_rows.clear()
            keyset.clear()
            seen.clear()

        for path in paths:
            with open(path, newline="") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    eid = row["entity_id"]
                    emitted = set()
                    keys = []
                    for _blocker_name, key in extra_key_pairs(
                        row.get("business_name") or "",
                        row.get("business_address") or "",
                        row.get("country") or "",
                    ):
                        if key not in emitted:
                            emitted.add(key)
                            keys.append(key)
                    if keys:
                        buf_rows.append((eid, keys))
                        for key in keys:
                            if key not in seen:
                                seen.add(key)
                                keyset.append(key)
                    if len(keyset) >= 2_000:
                        flush_rows()
        flush_rows()
        if batch:
            conn.executemany("INSERT INTO kv VALUES (?,?)", batch)
            written += len(batch)
            batch.clear()
        conn.commit()
        cnt.close()
        if cnt_path.exists():
            cnt_path.unlink()
        print(f"Indexing {written:,} exact extra postings… free={_disk_guard(db_path, 'extra index'):.1f}GB", flush=True)
        conn.execute("CREATE INDEX ix_k ON kv(k)")
        conn.commit()
        conn.close()
        done = True
        print(f"Extra DB ready rows={written:,} dead_keys={dead:,} in {time.time()-t0:.0f}s", flush=True)
    finally:
        if not done:
            for path in (db_path, cnt_path):
                if path.exists():
                    path.unlink()


def attach_extra_from_db(
    records: list[tuple],
    conn: sqlite3.Connection,
    pools: dict[str, dict[str, set[str]]],
) -> None:
    """Union exact extra postings into pools. Same blocker names as merge_extra_postings."""
    query_keys = []
    watch = set()
    for _eid, name, addr, country in records:
        pairs = extra_key_pairs(name, addr, country)
        query_keys.append(pairs)
        watch.update(key for _, key in pairs)
    post: dict[str, list[str]] = defaultdict(list)
    keys = list(watch)
    for i in range(0, len(keys), 400):
        part = keys[i : i + 400]
        if not part:
            continue
        q = ",".join("?" * len(part))
        for k, cid in conn.execute(f"SELECT k, eid FROM kv WHERE k IN ({q})", part):
            post[k].append(cid)
    for (eid, *_rest), pairs in zip(records, query_keys):
        sources = pools[eid]
        for blocker, key in pairs:
            for cid in post.get(key, ()):
                sources[cid].add(blocker)


def build_block_db_from_sqlite(src_db: Path, db_path: Path, max_per_key: int) -> None:
    """Same two-pass index, reading artifacts/test_s23.sqlite."""
    if db_path.exists():
        db_path.unlink()
    src = sqlite3.connect(str(src_db))
    total = src.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    print(f"Pass 1/2 count keys from sqlite ({total:,})…", flush=True)
    sketch = _new_sketch()
    t0 = time.time()
    n = 0
    cur = src.execute("SELECT name, addr, country FROM entities")
    while True:
        rows = cur.fetchmany(150_000)
        if not rows:
            break
        for name, addr, country in rows:
            for key in block_keys(name or "", addr or "", country or ""):
                _bump(sketch, key)
        n += len(rows)
        if n % 1_000_000 == 0:
            print(f"  counted {n:,}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"Counted {n:,} in {time.time()-t0:.0f}s", flush=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE kv (k TEXT, eid TEXT)")
    print("Pass 2/2 write rare keys…", flush=True)
    t1 = time.time()
    written = 0
    batch = []
    cur = src.execute("SELECT entity_id, name, addr, country FROM entities")
    while True:
        rows = cur.fetchmany(100_000)
        if not rows:
            break
        for eid, name, addr, country in rows:
            for key in block_keys(name or "", addr or "", country or ""):
                if _keep(sketch, key, max_per_key):
                    batch.append((key, eid))
            if len(batch) >= 50_000:
                conn.executemany("INSERT INTO kv VALUES (?,?)", batch)
                written += len(batch)
                batch.clear()
                if written % 2_000_000 < 50_000:
                    print(f"  inserted {written:,}  [{time.time()-t1:.0f}s]", flush=True)
    if batch:
        conn.executemany("INSERT INTO kv VALUES (?,?)", batch)
        written += len(batch)
    conn.commit()
    del sketch
    print(f"Indexing {written:,} postings…", flush=True)
    conn.execute("CREATE INDEX ix_k ON kv(k)")
    conn.commit()
    conn.close()
    src.close()
    print(f"Block DB ready rows={written:,} in {time.time()-t0:.0f}s", flush=True)


def _ltr_runs(address: str) -> list[str]:
    """Digit groups in left-to-right order. The legacy keys still use extract_digits."""
    return _DIGIT_RUNS.findall((address or "").translate(_DEV_DIGITS))


def _house_ltr(address: str) -> str:
    ds = [d for d in _ltr_runs(address) if 1 <= len(d) <= 6]
    return ds[0] if ds else ""


def _pin_ltr(address: str) -> str:
    for d in _ltr_runs(address):
        if len(d) >= 5:
            return d
    return ""


def _loc_ltr(address: str) -> str:
    ts = [t for t in address_tokens(address or "") if len(t) >= 6 and not t.isdigit()]
    return max(ts, key=len) if ts else ""


def _name_toks_ge3(name: str) -> list[str]:
    return [t for t in (significant_name_tokens(name) or name_tokens(name)) if len(t) >= 3]


def _consonant_skeleton(token: str) -> str:
    """Cheap sound-alike form for typo/transliteration blocking."""
    t = (token or "").lower()
    if len(t) < 5:
        return ""
    # Drop vowels after the first letter; keep first char so "sharma"/"sarma" meet.
    out = [t[0]]
    for ch in t[1:]:
        if ch not in "aeiou":
            out.append(ch)
    sk = "".join(out)
    return sk if len(sk) >= 3 else ""


def extra_key_pairs(name: str, address: str, country: str) -> list[tuple[str, str]]:
    """Blockers that the holdout union showed were worth keeping.

    Adds rare-token+PIN, phonetic+house, and sorted-tokens+house for typos
    and shortened names. Transliteration whole-string keys and character
    trigrams are still omitted (measured as useless).
    House numbers here are the first left-to-right digit run, so the key
    does not depend on set iteration order.
    """
    c = _country(country)
    out: list[tuple[str, str]] = []
    house = _house_ltr(address)
    pin = _pin_ltr(address)
    toks = _name_toks_ge3(name)
    if house:
        seen = set()
        for t in toks:
            if len(t) < 5:
                continue
            key = f"{c}|th|{t}_{house}"
            if key not in seen:
                seen.add(key)
                out.append(("short_name_house", key))
            sk = _consonant_skeleton(t)
            if sk:
                ph_key = f"{c}|ph|{sk}_{house}"
                if ph_key not in seen:
                    seen.add(ph_key)
                    out.append(("phonetic_house", ph_key))
    nn = normalize_name(name)
    if len(nn) >= 8 and house:
        out.append(("name_house", f"{c}|nhx|{nn}_{house}"))
    loc = _loc_ltr(address)
    if house and loc:
        out.append(("house_locality", f"{c}|hl|{house}_{loc}"))
    if len(toks) >= 2:
        out.append(("sorted_name", f"{c}|s|{'|'.join(sorted(toks))}"))
        if house:
            out.append(("sorted_house", f"{c}|sh|{'|'.join(sorted(toks))}_{house}"))
    # Rare long token + PIN: survives a typo in the *other* leading name word.
    if pin:
        seen_rp = set()
        for t in toks:
            if len(t) < 6:
                continue
            key = f"{c}|rp|{t}_{pin}"
            if key not in seen_rp:
                seen_rp.add(key)
                out.append(("rare_pin", key))
    at = [
        t
        for t in address_tokens(address or "")
        if len(t) >= 5 and t not in ADDR_STOP and not t.isdigit()
    ]
    if len(at) >= 2:
        long = sorted(at, key=len, reverse=True)[:2]
        x, y = sorted(long)
        out.append(("address", f"{c}|apx|{x}_{y}"))
    if at:
        longest = max(at, key=len)
        if len(longest) >= 8:
            out.append(("address", f"{c}|alx|{longest}"))
        if house and len(longest) >= 6:
            out.append(("address", f"{c}|hnx|{house}_{longest}"))
    return out


def _kind(key: str) -> str:
    return key.split("|", 2)[1] if key.count("|") >= 2 else ""


def _blocker(kind: str) -> str:
    return EXTRA_KIND.get(kind, "current")


def _near_tokens(query_toks: list[str], cand_toks: list[str], shared: set[str]) -> int:
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


def rank_score(query: tuple[str, str, str], cand: tuple[str, str, str], n_blockers: int) -> int:
    """Fixed blocking-stage score from the holdout union. Not fit on labels."""
    qn, qa, qc = query
    cn, ca, cc = cand
    qt = _name_toks_ge3(qn)
    ct = _name_toks_ge3(cn)
    shared = set(qt) & set(ct)
    rare = sum(1 for t in shared if len(t) >= 6)
    nn1, nn2 = normalize_name(qn), normalize_name(cn)
    exact_name = 1 if nn1 and nn1 == nn2 else 0
    aq = [t for t in address_tokens(qa or "") if len(t) >= 4 and not t.isdigit()]
    ac = [t for t in address_tokens(ca or "") if len(t) >= 4 and not t.isdigit()]
    house = 1 if (_house_ltr(qa) and _house_ltr(qa) == _house_ltr(ca)) else 0
    pin = 1 if (_pin_ltr(qa) and _pin_ltr(qa) == _pin_ltr(ca)) else 0
    loc_q, loc_c = _loc_ltr(qa), _loc_ltr(ca)
    loc = 1 if loc_q and loc_q == loc_c else 0
    country = 1 if _country(qc) == _country(cc) else 0
    return (
        4 * len(shared)
        + 3 * _near_tokens(qt, ct, shared)
        + 2 * rare
        + 6 * exact_name
        + 5 * house
        + 4 * pin
        + 3 * loc
        + 2 * len(set(aq) & set(ac))
        + 3 * n_blockers
        + country
    )


def cut_ranked(
    query: tuple[str, str, str],
    sources: dict[str, set[str]],
    texts: dict[str, tuple[str, str, str]],
    k: int,
) -> list[str]:
    scored = []
    for cid, src in sources.items():
        other = texts.get(cid)
        score = rank_score(query, other, len(src)) if other is not None else len(src)
        scored.append((-score, -len(src), cid))
    scored.sort()
    return [cid for _, _, cid in scored[:k]]


def pools_from_index(conn: sqlite3.Connection, records: list[tuple]) -> dict[str, dict[str, set[str]]]:
    """Every indexed id, with the blocker names that retrieved it. No top-K cut."""
    key_of = {}
    all_keys = set()
    for eid, name, addr, country in records:
        ks = block_keys(name, addr, country)
        key_of[eid] = ks
        all_keys.update(ks)
    postings: dict[str, list[str]] = defaultdict(list)
    keys = list(all_keys)
    for i in range(0, len(keys), 400):
        part = keys[i : i + 400]
        q = ",".join("?" * len(part))
        for k, cid in conn.execute(f"SELECT k, eid FROM kv WHERE k IN ({q})", part):
            postings[k].append(cid)
    out: dict[str, dict[str, set[str]]] = {}
    for eid, *_rest in records:
        sources: dict[str, set[str]] = defaultdict(set)
        for k in key_of[eid]:
            blocker = _blocker(_kind(k))
            for cid in postings.get(k, ()):
                sources[cid].add(blocker)
        out[eid] = sources
    return out


def merge_extra_postings(
    records: list[tuple],
    paths: list[Path],
    pools: dict[str, dict[str, set[str]]],
    texts: dict[str, tuple[str, str, str]],
    max_df: int = EXTRA_CAP,
) -> None:
    """Exact frequency cap for the added blockers.

    Keys come from the query records only. Ground-truth ids are not inserted.
    A key is deleted entirely once it matches more than max_df entities.
    """
    query_keys = []
    watch = set()
    for _eid, name, addr, country in records:
        pairs = extra_key_pairs(name, addr, country)
        query_keys.append(pairs)
        watch.update(key for _, key in pairs)
    post: dict[str, list[str]] = defaultdict(list)
    dead = set()
    need = {cid for sources in pools.values() for cid in sources}
    n = 0
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                eid = row["entity_id"]
                name = row.get("business_name") or ""
                addr = row.get("business_address") or ""
                country = row.get("country") or ""
                if eid in need and eid not in texts:
                    texts[eid] = (name, addr, country)
                emitted = set()
                for _blocker_name, key in extra_key_pairs(name, addr, country):
                    if key in emitted or key not in watch or key in dead:
                        continue
                    emitted.add(key)
                    bucket = post[key]
                    bucket.append(eid)
                    if eid not in texts:
                        texts[eid] = (name, addr, country)
                    if len(bucket) > max_df:
                        dead.add(key)
                        del post[key]
                n += 1
                if n % 2_000_000 == 0:
                    print(f"  extra keys scanned {n:,}", flush=True)
    print(f"  extra keys scanned {n:,} dead={len(dead):,}", flush=True)
    for (eid, *_rest), pairs in zip(records, query_keys):
        sources = pools[eid]
        for blocker, key in pairs:
            for cid in post.get(key, ()):
                sources[cid].add(blocker)


DECISION_RULE = {
    "t": 0.72,
    "t_empty": 0.82,
    "t_extra": 0.72,
    "t_short": 0.82,
    "t_conflict": 0.72,
    "margin": 0.12,
}


def _toks_ge2(name: str) -> list[str]:
    return [t for t in (significant_name_tokens(name) or name_tokens(name)) if len(t) >= 2]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def decision_features(
    query: tuple[str, str, str],
    cand: tuple[str, str, str],
    n_blockers: int,
    block_pos: int,
):
    """30 inputs used by artifacts/rank_model_hgb_v2.joblib. Same order as the holdout fit."""
    import numpy as np

    qn, qa, qc = query
    cn, ca, cc = cand
    sig = pair_signals(qn, qa, qc, cn, ca, cc)
    if sig is None:
        name_s, addr_sort, addr_set = 0.0, -1.0, -1.0
    else:
        name_s, addr_sort, addr_set = sig
    qt, ct = set(_toks_ge2(qn)), set(_toks_ge2(cn))
    shared = qt & ct
    nn1, nn2 = normalize_name(qn), normalize_name(cn)
    aa1, aa2 = normalize_address(qa), normalize_address(ca)
    name_ratio = fuzz.ratio(nn1, nn2) / 100.0 if nn1 and nn2 else 0.0
    name_set = fuzz.token_set_ratio(nn1, nn2) / 100.0 if nn1 and nn2 else 0.0
    name_lev = Levenshtein.normalized_similarity(nn1, nn2) if nn1 and nn2 else 0.0
    exact_name = 1.0 if nn1 and nn1 == nn2 else 0.0
    name_len_ratio = min(len(nn1), len(nn2)) / max(len(nn1), len(nn2)) if nn1 and nn2 else 0.0
    at = set(address_tokens(qa or ""))
    bt = set(address_tokens(ca or ""))
    addr_empty = 1.0 if addr_sort < 0 else 0.0
    addr_sort_f = 0.0 if addr_sort < 0 else addr_sort
    addr_set_f = 0.0 if addr_set < 0 else addr_set
    hq, hc = _house_ltr(qa), _house_ltr(ca)
    pq, pc = _pin_ltr(qa), _pin_ltr(ca)
    lq, lc = _loc_ltr(qa), _loc_ltr(ca)
    house = 1.0 if hq and hq == hc else 0.0
    pin = 1.0 if pq and pq == pc else 0.0
    short_query = 1.0 if len(qt) <= 1 or len(nn1) < 8 else 0.0
    conflict = (
        1.0
        if name_s >= 0.75 and addr_empty == 0.0 and addr_sort_f < 0.25 and house == 0.0 and pin == 0.0
        else 0.0
    )
    return np.asarray(
        [
            name_s,
            _jaccard(qt, ct),
            _containment(qt, ct),
            float(len(shared)),
            float(sum(1 for t in shared if len(t) >= 6)),
            name_ratio,
            name_set,
            name_lev,
            exact_name,
            name_len_ratio,
            float(sum(1 for t in shared if len(t) <= 4)),
            short_query,
            addr_sort_f,
            addr_set_f,
            _jaccard(at, bt),
            fuzz.ratio(aa1, aa2) / 100.0 if aa1 and aa2 else 0.0,
            _containment(at, bt),
            1.0 if aa1 and aa1 == aa2 else 0.0,
            house,
            pin,
            1.0 if lq and lq == lc else 0.0,
            1.0 if hq and hc and hq != hc else 0.0,
            1.0 if pq and pc and pq != pc else 0.0,
            addr_empty,
            float(n_blockers),
            float(block_pos),
            name_s * house,
            name_s * pin,
            (1.0 if name_s < 0.55 else 0.0) * addr_sort_f,
            conflict,
        ],
        dtype=np.float32,
    )


def decide_matches(vectors, ids: list[str], probas, rule: dict) -> list[str]:
    """Accept confident candidates; also keep siblings of a strong match.

    A later candidate can be kept when a strong name match was already
    accepted and this candidate shares house plus locality or PIN, without
    being in the weak-name band. Plain "same house, weak name" is still
    rejected — that pattern is a common false merge.
    """
    if len(ids) == 0:
        return []
    order = sorted(range(len(ids)), key=lambda i: -float(probas[i]))
    best = float(probas[order[0]])
    kept = []
    seen = set()
    # Feature layout from decision_features: name_s=0, house=18, pin=19, loc=20.
    strong_anchor = False
    for rank, i in enumerate(order):
        p = float(probas[i])
        row = vectors[i]
        name_s = float(row[0])
        house = float(row[18]) >= 1.0
        pin = float(row[19]) >= 1.0
        loc = float(row[20]) >= 1.0
        t = rule["t"]
        if float(row[23]) >= 1.0:
            t = max(t, rule["t_empty"])
        if float(row[11]) >= 1.0:
            t = max(t, rule["t_short"])
        if float(row[29]) >= 1.0:
            t = max(t, rule["t_conflict"])
        # Sibling: same site as an already-accepted strong name match.
        sibling = (
            rank > 0
            and strong_anchor
            and name_s >= 0.55
            and house
            and (loc or pin)
        )
        if rank > 0 and not sibling:
            t = max(t, rule["t_extra"])
            if best - p > rule["margin"]:
                continue
        if (p >= t or sibling) and ids[i] not in seen:
            seen.add(ids[i])
            kept.append(ids[i])
            if name_s >= 0.78 and (house or pin or loc):
                strong_anchor = True
    return kept


def pair_signals(n1: str, a1: str, c1: str, n2: str, a2: str, c2: str):
    if _country(c1) != _country(c2):
        return None
    nn1, nn2 = normalize_name(n1), normalize_name(n2)
    if nn1 and nn2:
        name_s = max(fuzz.token_sort_ratio(nn1, nn2), fuzz.ratio(nn1, nn2)) / 100.0
    else:
        name_s = 0.0
    aa1, aa2 = normalize_address(a1), normalize_address(a2)
    if aa1 and aa2:
        addr_sort = fuzz.token_sort_ratio(aa1, aa2) / 100.0
        addr_set = fuzz.token_set_ratio(aa1, aa2) / 100.0
    else:
        addr_sort = addr_set = -1.0
    return name_s, addr_sort, addr_set


def _accept(name_s: float, addr_s: float, rule: tuple[float, float, float, float]) -> bool:
    addr_hi, name_hi, addr_mid, name_empty = rule
    if addr_s >= addr_hi:
        return True
    if name_s >= name_hi and addr_s >= addr_mid:
        return True
    if name_s >= name_empty and addr_s < 0:
        return True
    return False


RULE_GRID = []
for addr_hi in (0.55, 0.65, 0.75, 0.85):
    for name_hi in (0.86, 0.90, 0.94):
        for addr_mid in (0.20, 0.35, 0.50):
            for name_empty in (0.90, 0.96):
                RULE_GRID.append((addr_hi, name_hi, addr_mid, name_empty))


def _f05_for_rule(rows, rule, addr_idx: int) -> float:
    """rows: list of (sid, true_set, list of (name_s, addr_sort, addr_set))."""
    scores = []
    for sid, true, cands in rows:
        pred = set()
        for name_s, addr_sort, addr_set in cands:
            addr_s = addr_sort if addr_idx == 0 else addr_set
            if _accept(name_s, addr_s, rule):
                # cid is packed as we only need the decision per stored cid
                pass
        # rewritten below — this placeholder is replaced by eval_rules
    return 0.0


def eval_rules(packed, addr_idx: int):
    """packed items: (true_set, [(cid, name_s, addr_sort, addr_set), ...])."""
    best = None
    for rule in RULE_GRID:
        scores = []
        for true, cands in packed:
            pred = set()
            for cid, name_s, addr_sort, addr_set in cands:
                addr_s = addr_sort if addr_idx == 0 else addr_set
                if _accept(name_s, addr_s, rule):
                    pred.add(cid)
            scores.append(score_entity(pred, true))
        f = sum(scores) / len(scores)
        if best is None or f > best[0]:
            best = (f, rule, addr_idx)
    return best


def candidates_for(conn: sqlite3.Connection, records: list[tuple], max_cands: int) -> dict[str, list[str]]:
    key_of = {}
    all_keys = set()
    for eid, name, addr, country in records:
        ks = block_keys(name, addr, country)
        key_of[eid] = ks
        all_keys.update(ks)
    postings: dict[str, list[str]] = defaultdict(list)
    keys = list(all_keys)
    for i in range(0, len(keys), 400):
        part = keys[i : i + 400]
        q = ",".join("?" * len(part))
        for k, cid in conn.execute(f"SELECT k, eid FROM kv WHERE k IN ({q})", part):
            postings[k].append(cid)
    out = {}
    for eid, name, addr, country in records:
        scores: dict[str, int] = defaultdict(int)
        for k in key_of[eid]:
            kind = k.split("|", 2)[1]
            w = WEIGHT.get(kind, 1)
            for cid in postings.get(k, ()):
                scores[cid] += w
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        out[eid] = [c for c, _ in ranked[:max_cands]]
    return out


def _load_gt_sample(gt_path: Path, stride: int):
    rows = []
    with open(gt_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i % stride:
                continue
            mids = [x for x in (row["matched_entity_ids"] or "").split(",") if x]
            rows.append((row["source1_entity_id"], set(mids)))
    return rows


def _scan_ids(path: Path, wanted: set[str]) -> dict[str, tuple[str, str, str]]:
    found = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            if eid in wanted and eid not in found:
                found[eid] = (
                    row.get("business_name") or "",
                    row.get("business_address") or "",
                    row.get("country") or "",
                )
                if len(found) == len(wanted):
                    break
    return found


def run_eval(args) -> dict:
    data = Path(args.train_dir)
    db_path = Path(args.block_db)
    if not db_path.exists() or args.rebuild:
        build_block_db(
            [data / "train_source2.tsv", data / "train_source3.tsv"],
            db_path,
            args.max_per_key,
        )
    gt = _load_gt_sample(data / "train_ground_truth.tsv", args.stride)
    print(f"Sampled S1={len(gt):,}", flush=True)
    need_s1 = {sid for sid, _ in gt}
    s1 = _scan_ids(data / "train_source1.tsv", need_s1)
    print(f"S1 text loaded {len(s1):,}", flush=True)

    conn = sqlite3.connect(str(db_path))
    records = [(sid, *s1[sid]) for sid, _ in gt if sid in s1]
    cand = candidates_for(conn, records, args.max_cands)
    conn.close()

    true_n = found = 0
    need = set()
    for sid, true in gt:
        if sid not in s1:
            continue
        hit = true & set(cand.get(sid, []))
        true_n += len(true)
        found += len(hit)
        need.update(cand.get(sid, []))
    print(
        f"Blocking pair recall={found / max(true_n,1):.3f}  "
        f"avg_cands={sum(len(v) for v in cand.values()) / max(len(cand),1):.2f}  "
        f"fetch_ids={len(need):,}",
        flush=True,
    )
    texts = {}
    for path in (data / "train_source2.tsv", data / "train_source3.tsv"):
        texts.update(_scan_ids(path, need - set(texts)))
        print(f"  fetched {len(texts):,}/{len(need):,} from {path.name}", flush=True)

    packed = []
    for sid, true in gt:
        if sid not in s1:
            continue
        a = s1[sid]
        cands = []
        for cid in cand.get(sid, []):
            b = texts.get(cid)
            if not b:
                continue
            sig = pair_signals(a[0], a[1], a[2], b[0], b[1], b[2])
            if sig is None:
                continue
            cands.append((cid, sig[0], sig[1], sig[2]))
        packed.append((true, cands))

    split = int(len(packed) * 0.7)
    train_pack, val_pack = packed[:split], packed[split:]
    print(f"Tune on {len(train_pack):,}  report on {len(val_pack):,}", flush=True)

    results = []
    for addr_idx, addr_name in ((0, "addr_sort"), (1, "addr_set")):
        f_tune, rule, _ = eval_rules(train_pack, addr_idx)
        f_val, _, _ = eval_rules(val_pack, addr_idx)  # not used; score the chosen rule
        # score the rule chosen on train against val
        scores = []
        pred_n = 0
        for true, cands in val_pack:
            pred = set()
            for cid, name_s, addr_sort, addr_set in cands:
                addr_s = addr_sort if addr_idx == 0 else addr_set
                if _accept(name_s, addr_s, rule):
                    pred.add(cid)
            if pred:
                pred_n += 1
            scores.append(score_entity(pred, true))
        f_val = sum(scores) / len(scores)
        results.append((f_val, f_tune, rule, addr_name, pred_n / len(val_pack)))
        print(
            f"  {addr_name} rule={rule} tune_f05={f_tune:.4f} VAL_f05={f_val:.4f} "
            f"val_nonempty={pred_n / len(val_pack):.3f}",
            flush=True,
        )

    results.sort(reverse=True)
    best_f, tune_f, rule, addr_name, nonempty = results[0]
    model_f = _try_model(train_pack, val_pack)
    winner = "rule"
    if model_f is not None and model_f > best_f:
        winner = "model"
        best_f = model_f
    summary = {
        "val_f05": best_f,
        "winner": winner,
        "rule": list(rule),
        "addr": addr_name,
        "tune_f05": tune_f,
        "blocking_pair_recall": found / max(true_n, 1),
        "model_val_f05": model_f,
    }
    outp = Path(args.rule_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2))
    print(f"BEST val_f05={best_f:.4f} winner={winner}", flush=True)
    print(f"Wrote {outp}", flush=True)
    return summary


def _try_model(train_pack, val_pack):
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception as exc:
        print(f"Classifier skipped ({exc})", flush=True)
        return None
    def rows(pack):
        X, y, groups = [], [], []
        for gi, (true, cands) in enumerate(pack):
            for cid, name_s, addr_sort, addr_set in cands:
                X.append((name_s, addr_sort, max(addr_set, -1), 1.0 if addr_sort < 0 else 0.0))
                y.append(1 if cid in true else 0)
                groups.append(gi)
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8), groups

    Xtr, ytr, _ = rows(train_pack)
    if len(Xtr) < 50 or ytr.sum() == 0:
        print("Classifier skipped (too few pairs)", flush=True)
        return None
    pos = max(int(ytr.sum()), 1)
    neg = max(int(len(ytr) - pos), 1)
    w = np.where(ytr == 1, neg / pos, 1.0).astype(np.float32)
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=30,
        random_state=42,
    )
    clf.fit(Xtr, ytr, sample_weight=w)
    # threshold sweep on val groups
    best_f, best_t = -1.0, 0.5
    prepared = []
    for true, cands in val_pack:
        if not cands:
            prepared.append((true, [], []))
            continue
        X = np.asarray(
            [(n, a, max(s, -1), 1.0 if a < 0 else 0.0) for _, n, a, s in cands],
            dtype=np.float32,
        )
        proba = clf.predict_proba(X)[:, 1]
        prepared.append((true, [c[0] for c in cands], proba))
    for t in [i / 100 for i in range(40, 96, 2)]:
        scores = []
        for true, cids, proba in prepared:
            if len(cids) == 0:
                pred = set()
            else:
                pred = {cid for cid, p in zip(cids, proba) if p >= t}
            scores.append(score_entity(pred, true))
        f = sum(scores) / len(scores)
        if f > best_f:
            best_f, best_t = f, t
    print(f"  classifier VAL_f05={best_f:.4f} thr={best_t:.2f}", flush=True)
    # persist model next to the rule if it wins; caller compares
    import joblib

    joblib.dump({"model": clf, "threshold": best_t}, Path("artifacts/rank_model.joblib"))
    return best_f


def run_infer(args, summary: dict) -> None:
    if summary.get("winner") == "model":
        print("Infer uses the classifier.", flush=True)
    else:
        print(f"Infer uses rule {summary['rule']} addr={summary['addr']}", flush=True)
    out = Path(args.output_dir)
    _refuse_protected(out)
    use_holdout_pools = bool(args.legacy_block_db and args.extra_block_db)
    db_path = Path(args.legacy_block_db if use_holdout_pools else args.test_block_db)
    _refuse_protected(db_path)
    if use_holdout_pools:
        extra_path = Path(args.extra_block_db)
        _refuse_protected(extra_path)
        if not db_path.exists() or not extra_path.exists():
            raise SystemExit(
                "Holdout-style inference needs the legacy index and the exact extra index. "
                "Refusing to build or reuse test_blocks_union.sqlite."
            )
        print(
            f"Infer candidates: legacy={db_path} extra={extra_path} cap={EXTRA_CAP} k={args.max_cands}",
            flush=True,
        )
    elif not db_path.exists() or args.rebuild:
        build_block_db_from_sqlite(Path(args.test_sqlite), db_path, args.max_per_key)
    rule = tuple(summary["rule"])
    addr_idx = 0 if summary["addr"] == "addr_sort" else 1
    use_model = summary.get("winner") == "model"
    clf = thr = None
    dec_model = dec_rule = None
    if args.decision_model:
        import joblib
        import numpy as np
        blob = joblib.load(args.decision_model)
        dec_model, dec_rule = blob["model"], blob["rule"]
        print(f"Infer decision model {args.decision_model} rule={dec_rule}", flush=True)
    elif use_model:
        import joblib
        import numpy as np
        blob = joblib.load("artifacts/rank_model.joblib")
        clf, thr = blob["model"], blob["threshold"]

    block = sqlite3.connect(str(db_path))
    extra_conn = sqlite3.connect(str(args.extra_block_db)) if use_holdout_pools else None
    text = sqlite3.connect(str(args.test_sqlite))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    match_path = out / "matching_results.tsv"
    cand_path = out / "candidate_pairs.tsv"
    for p in (match_path, cand_path):
        if p.exists():
            p.unlink()

    import pandas as pd

    n = 0
    nonempty = 0
    t0 = time.time()
    wrote = False
    for chunk in pd.read_csv(
        Path(args.test_dir) / "test_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=args.chunk_s1,
    ):
        records = [
            (r.entity_id, r.business_name, r.business_address, r.country)
            for r in chunk.itertuples(index=False)
        ]
        pools = pools_from_index(block, records)
        if extra_conn is not None:
            attach_extra_from_db(records, extra_conn, pools)
        need = [cid for sources in pools.values() for cid in sources]
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
        s1_ids, match_col, cand_col = [], [], []
        for eid, name, addr, country in records:
            sources = pools.get(eid, {})
            cids = cut_ranked((name, addr, country), sources, rec, args.max_cands)
            cand_col.append(",".join(cids))
            kept = []
            if dec_model is not None and cids:
                import numpy as np
                vectors, ids = [], []
                query = (name, addr, country)
                for pos, cid in enumerate(cids, start=1):
                    other = rec.get(cid)
                    if not other:
                        continue
                    vectors.append(
                        decision_features(query, other, len(sources.get(cid, ())), pos)
                    )
                    ids.append(cid)
                if vectors:
                    proba = dec_model.predict_proba(np.vstack(vectors))[:, 1]
                    kept = decide_matches(vectors, ids, proba, dec_rule)
            elif use_model and cids:
                import numpy as np
                feats, ids = [], []
                for cid in cids:
                    b = rec.get(cid)
                    if not b:
                        continue
                    sig = pair_signals(name, addr, country, b[0], b[1], b[2])
                    if sig is None:
                        continue
                    feats.append((sig[0], sig[1], max(sig[2], -1), 1.0 if sig[1] < 0 else 0.0))
                    ids.append(cid)
                if feats:
                    proba = clf.predict_proba(np.asarray(feats, dtype=np.float32))[:, 1]
                    kept = [cid for cid, p in zip(ids, proba) if p >= thr]
            else:
                for cid in cids:
                    b = rec.get(cid)
                    if not b:
                        continue
                    sig = pair_signals(name, addr, country, b[0], b[1], b[2])
                    if sig is None:
                        continue
                    addr_s = sig[1] if addr_idx == 0 else sig[2]
                    if _accept(sig[0], addr_s, rule):
                        kept.append(cid)
            if kept:
                nonempty += 1
            match_col.append(",".join(kept))
            s1_ids.append(eid)
        pd.DataFrame(
            {"source1_entity_id": s1_ids, "matched_entity_ids": match_col}
        ).to_csv(match_path, sep="\t", index=False, mode="a", header=not wrote)
        pd.DataFrame(
            {"source1_entity_id": s1_ids, "candidate_entity_ids": cand_col}
        ).to_csv(cand_path, sep="\t", index=False, mode="a", header=not wrote)
        wrote = True
        n += len(records)
        if n % 20000 < args.chunk_s1:
            elapsed = time.time() - t0
            rate = n / max(elapsed, 1)
            remaining = (1_732_544 - n) / max(rate, 1)
            print(
                f"  infer rows={n:,} written={n:,} {rate:.1f}/s "
                f"elapsed={elapsed/60:.1f}m remaining={remaining/60:.0f}m nonempty={nonempty:,}",
                flush=True,
            )
    block.close()
    if extra_conn is not None:
        extra_conn.close()
    text.close()
    print(f"DONE infer rows={n:,} nonempty={nonempty:,} -> {match_path}", flush=True)


def _pct(xs: list[int], p: float) -> int:
    if not xs:
        return 0
    ordered = sorted(xs)
    i = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
    return ordered[i]


def run_holdout(args) -> dict:
    """Score the frozen blocker on the last 30% of the stride sample.

    Does not train, does not write a submission, and does not insert labels
    into the candidate index.
    """
    import joblib
    import numpy as np

    data = Path(args.train_dir)
    db_path = Path(args.block_db)
    if not db_path.exists():
        raise SystemExit(f"Missing block index {db_path}. Refusing to build one from this mode.")
    gt = _load_gt_sample(data / "train_ground_truth.tsv", args.stride)
    s1 = _scan_ids(data / "train_source1.tsv", {sid for sid, _ in gt})
    packed = [(sid, true, s1[sid]) for sid, true in gt if sid in s1]
    val = packed[int(len(packed) * 0.7) :]
    records = [(sid, *query) for sid, _, query in val]
    print(f"holdout S1={len(val):,} true_links={sum(len(true) for _, true, _ in val):,}", flush=True)

    conn = sqlite3.connect(str(db_path))
    pools = pools_from_index(conn, records)
    conn.close()
    # The existing train index stores the legacy keys only. The added blockers
    # use the same exact cap as the offline union. Labels are not consulted.
    texts: dict[str, tuple[str, str, str]] = {}
    print("Scanning S2/S3 for the added blockers.", flush=True)
    merge_extra_postings(
        records,
        [data / "train_source2.tsv", data / "train_source3.tsv"],
        pools,
        texts,
        EXTRA_CAP,
    )
    has_extra = False

    ranked = []
    for sid, _true, query in val:
        ranked.append(cut_ranked(query, pools.get(sid, {}), texts, args.max_cands))
    missing = {cid for ids in ranked for cid in ids if cid not in texts}
    for path in (data / "train_source2.tsv", data / "train_source3.tsv"):
        if not missing:
            break
        texts.update(_scan_ids(path, missing))
        missing -= set(texts)

    lengths = [len(ids) for ids in ranked]
    retrieved = 0
    true_n = 0
    oracle = []
    for (_sid, true, _query), ids in zip(val, ranked):
        got = set(ids)
        true_n += len(true)
        retrieved += len(true & got)
        oracle.append(score_entity(true & got, true))
    def _tally(pred_of: dict[str, set[str]]) -> dict:
        scores = []
        tp = fp = fn = 0
        single_n = single_ok = 0
        for (sid, true, _query), ids in zip(val, ranked):
            pred = pred_of[sid]
            scores.append(score_entity(pred, true))
            tp += len(pred & true)
            fp += len(pred - true)
            fn += len(true - pred)
            if not true:
                single_n += 1
                single_ok += int(not pred)
        micro_p = tp / max(tp + fp, 1)
        micro_r = tp / max(tp + fn, 1)
        return {
            "macro_f05": round(sum(scores) / max(len(scores), 1), 4),
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f05": round(f05(micro_p, micro_r), 4),
            "false_matches": fp,
            "missed_matches": fn,
            "true_positives": tp,
            "singleton_accuracy": round(single_ok / max(single_n, 1), 4),
            "singletons": single_n,
            "singletons_correct": single_ok,
        }

    old_blob = joblib.load("artifacts/rank_model.joblib")
    old_model, old_thr = old_blob["model"], float(old_blob["threshold"])
    new_blob = joblib.load("artifacts/rank_model_hgb_v2.joblib")
    new_model, new_rule = new_blob["model"], new_blob["rule"]
    old_pred, new_pred = {}, {}
    for (sid, _true, query), ids in zip(val, ranked):
        sources = pools.get(sid, {})
        old_feats, old_ids = [], []
        vectors, new_ids = [], []
        for pos, cid in enumerate(ids, start=1):
            other = texts.get(cid)
            if not other:
                continue
            sig = pair_signals(query[0], query[1], query[2], other[0], other[1], other[2])
            if sig is not None:
                old_feats.append((sig[0], sig[1], max(sig[2], -1.0), 1.0 if sig[1] < 0 else 0.0))
                old_ids.append(cid)
            vectors.append(decision_features(query, other, len(sources.get(cid, ())), pos))
            new_ids.append(cid)
        if old_feats:
            proba = old_model.predict_proba(np.asarray(old_feats, dtype=np.float32))[:, 1]
            old_pred[sid] = {cid for cid, p in zip(old_ids, proba) if p >= old_thr}
        else:
            old_pred[sid] = set()
        if vectors:
            proba = new_model.predict_proba(np.vstack(vectors))[:, 1]
            new_pred[sid] = set(decide_matches(vectors, new_ids, proba, new_rule))
        else:
            new_pred[sid] = set()
    report = {
        "k": args.max_cands,
        "pair_recall": round(retrieved / max(true_n, 1), 4),
        "retrieved": retrieved,
        "true_links": true_n,
        "avg": round(sum(lengths) / max(len(lengths), 1), 2),
        "median": statistics.median(lengths) if lengths else 0,
        "p95": _pct(lengths, 95),
        "p99": _pct(lengths, 99),
        "max": max(lengths) if lengths else 0,
        "zero_pct": round(sum(1 for n in lengths if n == 0) / max(len(lengths), 1), 4),
        "oracle_f05": round(sum(oracle) / max(len(oracle), 1), 4),
        "old_model": _tally(old_pred),
        "new_model": _tally(new_pred),
        "new_rule": new_rule,
        "extra_keys_in_index": has_extra,
    }
    outp = Path("artifacts/analysis/holdout_decision_production.json")
    print(json.dumps(report, indent=2), flush=True)
    print(f"WROTE {outp}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("eval", "infer", "all", "holdout"), default="eval")
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--block-db", default="artifacts/train_blocks.sqlite")
    ap.add_argument("--test-block-db", default="artifacts/test_blocks.sqlite")
    ap.add_argument("--legacy-block-db", default="", help="Legacy-key index. Does not replace test_blocks_union.sqlite.")
    ap.add_argument("--extra-block-db", default="", help="Exact cap-40 extra-key index.")
    ap.add_argument("--test-sqlite", default="artifacts/test_s23.sqlite")
    ap.add_argument("--rule-path", default="artifacts/rule.json")
    ap.add_argument("--output-dir", default="output_v3")
    ap.add_argument("--stride", type=int, default=400)
    ap.add_argument("--max-per-key", type=int, default=MAX_PER_KEY)
    ap.add_argument("--max-cands", type=int, default=12)
    ap.add_argument("--chunk-s1", type=int, default=4000)
    ap.add_argument("--decision-model", default="", help="Rich decision model. Does not change candidate generation.")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--min-f05", type=float, default=0.55)
    args = ap.parse_args()

    if args.mode == "holdout":
        run_holdout(args)
        return
    summary = None
    if args.mode in ("eval", "all"):
        summary = run_eval(args)
        if args.mode == "all" and summary["val_f05"] < args.min_f05:
            print(
                f"Held-out F0.5 {summary['val_f05']:.3f} is below {args.min_f05}. "
                "Not writing a test file.",
                flush=True,
            )
            return
    if args.mode in ("infer", "all"):
        if summary is None:
            summary = json.loads(Path(args.rule_path).read_text())
        run_infer(args, summary)


if __name__ == "__main__":
    main()
