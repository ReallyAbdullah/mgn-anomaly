#!/usr/bin/env bash
# Phase 2, validation only (P2 is NOT touched): Gate D, protocol-v3 validation records, conformal check, extra seed.
# Usage: scripts/overnight_phase2.sh <shape-training PID>
set -euo pipefail
cd "$(dirname "$0")/.."
PID=${1:?shape training PID}
FREEZE=$(git rev-parse HEAD)
echo "frozen at $FREEZE"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
grep -E "final val" runs/shape_train.log || { echo "shape training did not finish cleanly"; exit 1; }
run() {
  git diff --quiet "$FREEZE" -- src || { echo "src/ changed since $FREEZE, aborting"; exit 1; }
  echo "== $* ($(date +%H:%M))"; caffeinate -i uv run python "$@"
}
# Gate D: run-level scoring on validation 20-49 (clean reference 50-69), then the pre-registered test
run -m mgn.runlevel --start 20 --stop 50 --ref-start 50 --ref-stop 70 --out results/runlevel
run -m mgn.gate_d
# protocol v3 frame-level records on validation 20-99: best baselines (20-49), triage/hybrid fit (50-69), calibration (70-99)
run -m mgn.evaluate --split valid --start 20 --stop 100 --no-clip --out results/valid_v3
# conformal thresholds (calibration 70-99) checked on validation 20-49 (a sanity check, not a P2 claim)
run -m mgn.leadtime --method conformal --split valid --start 20 --stop 50 --out results/conformal_valid
# extra full-scale seed of the Phase 1 velocity model (Step 3), 8 h
run -m mgn.train --name mgn_seed1 --seed 1 --n-train 250 --n-val 10 --hours 8 --iters 260000 --eval-every 10000
echo "== all done ($(date +%H:%M))"
