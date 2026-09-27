#!/usr/bin/env bash
# Full local inference — $0 AWS credits. Run overnight if needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="code/business_entity_resolution/src:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"
if [[ -x /tmp/ber_venv/bin/python ]]; then PY=/tmp/ber_venv/bin/python; fi
if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; fi

mkdir -p output artifacts logs
LOG="logs/full_infer_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
echo "Full test infer starting at $(date)" | tee -a "$LOG"

"$PY" code/business_entity_resolution/src/infer.py \
  --sample 0 \
  --max-candidates 120 \
  --chunk-s1 10000 \
  --output-dir output \
  --artifacts artifacts 2>&1 | tee -a "$LOG"

echo "Validating…" | tee -a "$LOG"
"$PY" utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test 2>&1 | tee -a "$LOG"

echo "Done at $(date)" | tee -a "$LOG"
