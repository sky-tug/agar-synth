#!/usr/bin/env bash
#
# Levels 10/25/50 LoRA training -- the missing third of decision 3.2 (a separate
# LoRA per level, the leakage lock). Level 100 was trained on 31 Aug.
#
# 1500 steps, exactly like lora_100: the production-checkpoint decision (3.89)
# is still open, and 1500 is the point both candidates share -- if the decision
# lands on 4500 these become the first third of that run rather than waste.
#
# IDEMPOTENT: a level whose adapt_metrics.json already exists is finished and is
# skipped. Interrupted halfway? Run the same command again; it continues with the
# levels that are still missing. `adapt.py train` itself has NO --resume, so a
# level that was cut mid-training restarts from step 0 -- only whole levels are
# resumable, which is why they run one at a time instead of in parallel.
#
# Usage:
#     nohup ./run_loras.sh > lora_levels.log 2>&1 &
#     tail -f lora_levels.log
#
# Expect ~31 min per level on the RTX 4060 (measured on level 100: 0.509 GPU-h
# for 1500 steps), so ~1.5 h for all three.

set -u
cd "$(dirname "$0")"

for L in 10 25 50; do
  if [ -f "src/generate/lora_$L/adapt_metrics.json" ]; then
    echo "=== seviye $L zaten bitmis, atlaniyor ==="
    continue
  fi
  echo "=========== LoRA seviye $L  baslangic $(date +%H:%M:%S) ==========="
  python src/generate/adapt.py train --level "$L" \
    --crops "src/generate/crops_$L" \
    --steps 1500 \
    --source-list "data/processed/lists/train_$L.txt" \
    --out "src/generate/lora_$L"
  echo "--- seviye $L bitti $(date +%H:%M:%S) ---"
done

echo "=========== TAMAMLANDI $(date +%H:%M:%S) ==========="
echo
echo "Kontrol:"
for L in 10 25 50 100; do
  f="src/generate/lora_$L/adapt_metrics.json"
  if [ -f "$f" ]; then
    python - "$f" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"  seviye {m['level']:>3}  {m['n_crops']:>6} kirpma  {m['steps']} adim  "
      f"eval {m['eval_loss_start']:.5f} -> {m['eval_loss_end']:.5f}  "
      f"{m['hours']:.3f} GPU-sa")
PY
  else
    echo "  seviye $L  EKSIK"
  fi
done
