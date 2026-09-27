"""Refresh OneDrive download URLs via instructions printed for the agent browser helper.

This script downloads from a manifest JSON:
[{"path": "dataset/train/foo.tsv", "url": "...", "size": 123}]

Usage:
  python tools/download_onedrive_manifest.py C:\\Users\\vibhu\\amlc26-data\\manifest.json
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\Users\vibhu\amlc26-data")


def download(url: str, dest: Path, expected: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected and dest.stat().st_size == expected:
        print(f"SKIP {dest} already {expected}", flush=True)
        return
    print(f"GET {dest.name} -> {dest} expect={expected}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as out:
        total = 0
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            if total % (64 * 1024 * 1024) < 8 * 1024 * 1024:
                print(f"  {dest.name}: {total/1e6:.0f} MB", flush=True)
    size = dest.stat().st_size
    print(f"DONE {dest} bytes={size}", flush=True)
    if expected and size != expected:
        raise SystemExit(f"size mismatch for {dest}: got {size} expected {expected}")


def main() -> None:
    manifest = Path(sys.argv[1])
    files = json.loads(manifest.read_text(encoding="utf-8"))
    for item in files:
        dest = ROOT / item["path"]
        # repo-relative artifacts path also accepted
        if item["path"].startswith("artifacts/"):
            # store sqlite on C: and junction/copy into repo later
            dest = ROOT / item["path"]
        download(item["url"], dest, item.get("size"))
    print("ALL DOWNLOADS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
