#!/usr/bin/env bash
# Credit-safe smoke test: tiny sample on YOUR LAPTOP (no AWS needed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="code/business_entity_resolution/src:${PYTHONPATH:-}"

echo "== Train on 5k S1 sample (local CPU) =="
python3 code/business_entity_resolution/src/train.py \
  --data-dir dataset/train \
  --artifacts artifacts \
  --sample 5000 \
  --max-candidates 40

echo "== Infer on 2k test S1 sample =="
python3 code/business_entity_resolution/src/infer.py \
  --test-dir dataset/test \
  --artifacts artifacts \
  --output-dir output \
  --sample 2000 \
  --max-candidates 40

echo "== Validate format =="
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test || true

echo "NOTE: sample infer does not cover all test S1 IDs, so validator may FAIL on missing rows."
echo "That is expected for smoke tests. Full run: omit --sample on infer."
