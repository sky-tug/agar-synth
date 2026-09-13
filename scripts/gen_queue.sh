#!/usr/bin/env bash
# =============================================================================
# gen_queue.sh -- production generation, one level at a time        (3.100)
# =============================================================================
#
# Runs the three-stage generation chain for each level that the synthetic arms
# of Phase 6 need:
#
#   layout.py sample  ->  plans_g<L>/     (CPU, seconds)
#   mask.py build     ->  build_g<L>/     (CPU, minutes)
#   inpaint.py run    ->  synth_g<L>/     (GPU, 4-13 hours)
#
# Like run_queue.sh this is IDEMPOTENT and meant to be driven by cron: each
# stage is skipped when its output is already complete, and inpaint.py is
# called with --resume so a generation killed in its twelfth hour continues
# instead of starting over.
#
# IT SHARES run_queue.sh's LOCK ON PURPOSE.
# Both scripts want the same GPU. With one lock they serialise: while training
# runs, every generation tick exits immediately, and generation starts by
# itself when the training queue empties. No scheduling, no ordering flags,
# nothing to remember.
#
# Install alongside the other supervisor:
#   crontab -l 2>/dev/null | { cat; echo "*/15 * * * * $HOME/agar-synth/scripts/gen_queue.sh >> $HOME/agar-synth/logs/cron.log 2>&1"; } | crontab -
#
# HOW THE COUNTS BELOW WERE OBTAINED
# ----------------------------------
# From the arm definitions in scripts/budget.py: every "+S" arm tops its
# training set up to the full-data size (2,987 images), so its synthetic share
# is (1.00 - real share). Per level the largest share any arm needs is
# generated ONCE and the smaller arms take nested subsets of it:
#
#   level 100   S100            1.00        -> 2987 plates   ~8.6 GPU-h
#   level  10   G10+S           0.90        -> 2688          ~7.8
#   level  25   G25+S    0.75 |             -> 4481          ~13.0
#               G25+S_0.5x 0.375| max 1.50
#               G25+S_2x   1.50 |
#   level  50   G50+S           0.50        -> 1494          ~4.3
#                                              ------------------
#                                              11650        ~33.7 GPU-h
#
# Nested subsets are not a shortcut, they are the right design: the amount
# sweep asks what changes when the QUANTITY of synthetic data changes, so
# drawing 1120 / 2240 / 4481 plates from one pool isolates quantity from
# generation-to-generation variation. Generating three independent pools would
# confound the two.
#
# EACH LEVEL USES ITS OWN LoRA -- decisions 3.2 and 3.7, and inpaint.py stops
# if the name does not carry the level. A G25+S arm generated with lora_100
# would carry information from images outside the 25% subset, which is exactly
# the leak the per-level protocol exists to prevent. Production checkpoint is
# the 1500-step LoRA (decision 3.99).
#
# ORDER IS PRIORITY. The window may not be long enough for all four, so the
# levels that decide the paper's claim come first: 100 (does synthetic data
# work at all, on its own?) and 10 (does it help where real data is scarcest?).
# Level 50 is last because "does synthetic help when you already have half the
# data" is the least interesting question of the four.
# =============================================================================

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ROOT="${AGAR_ROOT:-$HOME/agar-synth}"
LOGDIR="$ROOT/logs"
STATUS="$LOGDIR/gen_status.tsv"
LOCK="$ROOT/.queue.lock"          # SHARED with run_queue.sh -- see above
G="src/generate"
SEED="${GEN_SEED:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-2000}"

# level:count, in priority order
LEVELS="100:2987 10:2688 25:4481 50:1494"

cd "$ROOT" 2>/dev/null || { echo "ERROR: $ROOT not found"; exit 1; }
mkdir -p "$LOGDIR"

# WAIT for the lock, do not give up on it instantly.
#
# Both supervisors fire on the same cron minute and run_queue.sh is listed
# first, so it takes the lock even when all it does is scan the queue and find
# every run deferred -- about one second of work. A `flock -n` here would see
# that lock, exit, and the GPU would then sit idle until the next tick, every
# tick, forever: generation would never start at all. Waiting three minutes
# costs nothing and covers the scan; when training is genuinely running the
# wait expires and this tick exits, which is the behaviour we want.
exec 9>"$LOCK" || exit 1
if ! flock -w 180 9; then
    exit 0                        # training holds it for real
fi

say()  { echo "$(date -Is) [gen] $*" | tee -a "$LOGDIR/gen.log"; }
mark() { printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS"; }

# shellcheck disable=SC1091
source src/.venv/bin/activate 2>/dev/null || {
    say "ERROR: could not activate src/.venv"; exit 1; }

avail=$(free -m | awk '/^Mem:/{print $7}')
if [ "$avail" -lt "$MIN_FREE_MB" ]; then
    say "only ${avail} MB available -- exiting, the next tick will retry"
    exit 0
fi

say "=== generation queue start ==="

for entry in $LEVELS; do
    L="${entry%%:*}"
    N="${entry##*:}"
    plans="$G/plans_g$L"
    build="$G/build_g$L"
    synth="$G/synth_g$L"
    lora="$G/lora_$L"
    params="$G/layout_$L.json"
    pool="$G/bg_pool_$L.json"
    log="$LOGDIR/gen_$L.log"

    # --- finished? ----------------------------------------------------------
    have=$(ls "$synth/images" 2>/dev/null | wc -l)
    if [ "$have" -ge "$N" ]; then
        continue
    fi

    say "--- level $L: $have/$N plates present ---"

    for f in "$params" "$pool"; do
        [ -f "$f" ] || { say "level $L: MISSING $f, skipping this level"; mark "L$L" "missing_input"; continue 2; }
    done
    [ -d "$lora" ] || { say "level $L: MISSING $lora, skipping this level"; mark "L$L" "missing_lora"; continue; }

    # --- stage 1: layout plans ---------------------------------------------
    # Seeded, so re-running produces the same plans; still skipped when the
    # count is already there, because there is no reason to redo it.
    n_plans=$(ls "$plans/plans" 2>/dev/null | wc -l)
    if [ "$n_plans" -lt "$N" ]; then
        say "level $L: layout sample ($n_plans/$N)"
        python "$G/layout.py" sample --level "$L" --params "$params" \
            --count "$N" --seed "$SEED" --out "$plans" >> "$log" 2>&1
        n_plans=$(ls "$plans/plans" 2>/dev/null | wc -l)
        if [ "$n_plans" -lt "$N" ]; then
            say "level $L: LAYOUT FAILED -- see $log"; mark "L$L" "layout_fail"; continue
        fi
        mark "L$L" "layout_ok"
    fi

    # --- stage 2: masks and backgrounds -------------------------------------
    # --split-masks is NOT optional: inpaint.py refuses to run without the two
    # separate masks (decision 3.85), because one combined mask cannot say
    # "fill this region, empty that one" and every erase region would come back
    # holding an unlabelled colony.
    n_prov=$(ls "$build/provenance" 2>/dev/null | wc -l)
    if [ "$n_prov" -lt "$N" ]; then
        say "level $L: mask build ($n_prov/$N)"
        python "$G/mask.py" build --level "$L" --plans "$plans" --pool "$pool" \
            --seed "$SEED" --split-masks --out "$build" >> "$log" 2>&1
        n_prov=$(ls "$build/provenance" 2>/dev/null | wc -l)
        if [ "$n_prov" -lt "$N" ]; then
            say "level $L: MASK BUILD FAILED -- see $log"; mark "L$L" "mask_fail"; continue
        fi
        mark "L$L" "mask_ok"
    fi

    # --- stage 3: inpainting (the expensive one) ----------------------------
    # Protocol values are frozen and come from the pilot measurement
    # (tile 512, steps 4 -- decision 3.66): do not tune them here, the
    # substitution curve depends on every arm being generated the same way.
    say "level $L: inpaint run ($have/$N) -- this is the long one"
    python "$G/inpaint.py" run --level "$L" --input "$build" --plans "$plans" \
        --lora "$lora" --tile 512 --steps 4 --seed "$SEED" --resume \
        --out "$synth" >> "$log" 2>&1
    have=$(ls "$synth/images" 2>/dev/null | wc -l)
    if [ "$have" -ge "$N" ]; then
        say "level $L: COMPLETE ($have plates)"
        mark "L$L" "inpaint_complete"
        # Build every arm list this level has just made possible (3.101). Doing
        # it here rather than at the very end means the training queue can start
        # on level 100's arms while level 25 is still generating -- the GPU never
        # waits for work that already exists.
        python scripts/make_arm_lists.py >> "$LOGDIR/arm_lists.log" 2>&1 \
            && say "arm lists rebuilt" || say "make_arm_lists.py FAILED -- see logs/arm_lists.log"
    else
        # Not an error worth stopping for: a kill mid-generation leaves real
        # work on disk and the next tick continues from it. Only report.
        say "level $L: partial ($have/$N) -- the next tick continues"
        mark "L$L" "inpaint_partial_$have"
        exit 0                     # release the GPU, come back in 15 minutes
    fi
done

python scripts/make_arm_lists.py >> "$LOGDIR/arm_lists.log" 2>&1 || true
say "=== generation queue end ==="
