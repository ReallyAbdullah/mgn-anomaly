#!/usr/bin/env bash
# Runs the frozen protocol once the training process exits. Stops at the first failure.
# Usage: scripts/after_training.sh <training PID>
set -euo pipefail
cd "$(dirname "$0")/.."
PID=${1:?training PID}
FREEZE=$(git rev-parse HEAD)  # the protocol is frozen at this commit; any src/ change during the chain aborts it
echo "protocol frozen at $FREEZE"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
grep -E "val one-step|final|Traceback" runs/mgn_train.log | tail -3
grep -q "final val" runs/mgn_train.log || { echo "training did not finish cleanly"; exit 1; }
run() {
  git diff --quiet "$FREEZE" -- src || { echo "src/ changed since $FREEZE: protocol no longer frozen, aborting"; exit 1; }
  echo "== $* ($(date +%H:%M))"; caffeinate -i uv run python "$@"
}

# 0. archive the 2-sim smoke outputs and pilot records so nobody quotes them
mkdir -p results/smoke_2sim
git mv -k results/auroc_vs_severity.png results/metrics.csv results/metrics.md results/vlm_confusion.png \
  results/vlm_metrics.md results/vlm_results.json results/records_noise1e-3.pkl results/records_nonoise.pkl results/smoke_2sim/
git rm -q --cached --ignore-unmatch results/.DS_Store
git commit -qm "Archive 2-sim smoke outputs and pilot records under results/smoke_2sim

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true

# Gate B: final one-step error vs constant velocity on fixed validation frames (sims 20-49)
run -c "
from mgn.train import device, evaluate, load_model, load_split
dev = device(); m = load_model('runs/mgn/model.pt', dev)
rmse, base = evaluate(m, load_split('valid', 50)[20:], dev)
line = f'Gate B: one-step RMSE {rmse:.3e} vs constant velocity {base:.3e} (ratio {rmse / base:.1f}x), valid sims 20-49, fixed frames'
print(line); open('results/gate_b.txt', 'w').write(line + chr(10))"

# 1. validation (no CLIP: CLIP is excluded from the best-baseline pool, see docs/gate_c.md)
run -m mgn.evaluate --split valid --start 20 --stop 50 --no-clip --out results/valid
# 2. test, scored once (with CLIP)
run -m mgn.evaluate --split test --start 8 --stop 100 --out results/test --best-from results/valid/metrics.csv
# 3. Gate C (pre-registered)
run -m mgn.gate_c
# 4. lead time: thresholds on validation, applied unchanged on test
run -m mgn.leadtime --split valid --start 20 --stop 50 --out results/leadtime
run -m mgn.leadtime --split test --start 8 --stop 100 --thresholds results/leadtime/thresholds.json --out results/leadtime
# 5. VLM explanations and 6. counterfactual pairs
run -m mgn.explain --records results/test/records.pkl
run -m mgn.counterfactual
echo "== all done ($(date +%H:%M))"
