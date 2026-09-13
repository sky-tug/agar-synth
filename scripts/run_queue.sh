#!/usr/bin/env bash
# =============================================================================
# run_queue.sh -- an idempotent, crash-tolerant run queue          (3.99)
# =============================================================================
#
# WHY THIS EXISTS
# ---------------
# On 13 September the B_G25 chain died at 01:40. `evaluate.py` was holding
# 9.35 GB resident, VS Code held another few, and the kernel OOM killer took
# the python process:
#
#   Sep 13 01:40:34 kernel: Out of memory: Killed process 356034 (python)
#                           anon-rss:9347716kB
#
# The chain used `set -e`, so the surviving two seeds were cancelled with it,
# and the queued follow-up chain -- a plain `while kill -0 <pid>; do sleep 60;
# done` waiter -- went down in the same group. The GPU then sat idle for ten
# and a half hours with nobody watching.
#
# The lesson is not "add a retry". It is that an unattended queue must be
# RESTARTABLE BY ANYTHING, including a dumb timer, because whatever kills it
# will not leave a note. So:
#
#   * running this script twice is harmless -- completed runs are skipped,
#     half-finished training is resumed, and a flock makes concurrent
#     instances impossible;
#   * therefore a cron entry every 15 minutes IS the supervisor. Nothing
#     watches the watcher, and nothing needs to;
#   * no `set -e`: one failed run never cancels the rest of the queue;
#   * each step gets a second attempt, because an OOM kill is often transient
#     (whatever else was using the memory may be gone by then);
#   * the queue writes what happened to logs/queue_status.tsv, so on return
#     one file answers "what ran, what failed, and when".
#
# Install the supervisor once:
#   crontab -l 2>/dev/null | { cat; echo "*/15 * * * * $HOME/agar-synth/scripts/run_queue.sh >> $HOME/agar-synth/logs/cron.log 2>&1"; } | crontab -
#
# Cron runs outside the graphical session, so it also survives a logout or a
# session crash -- which a nohup'd background job does not reliably do.
#
# QUEUE FILE FORMAT (default: queue.txt), whitespace separated, '#' comments:
#   <name>      <level>  <seed>  <overlay|->
#   B_G25_s1    25       1       aug_b_classic.yaml
#   M_G25_s0    25       0       aug_m_mosaic_mixup.yaml
#   G100_s0     100      0       -
#
# Order in the file is the order of execution: put the runs that answer the
# question first, because the queue may not finish.
# =============================================================================

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ROOT="${AGAR_ROOT:-$HOME/agar-synth}"
QUEUE="${1:-$ROOT/queue.txt}"
LOGDIR="$ROOT/logs"
STATUS="$LOGDIR/queue_status.tsv"
LOCK="$ROOT/.queue.lock"
MIN_FREE_MB="${MIN_FREE_MB:-2000}"   # do not start a step below this
RETRY_SLEEP="${RETRY_SLEEP:-60}"     # pause between the two attempts

cd "$ROOT" 2>/dev/null || { echo "ERROR: $ROOT not found"; exit 1; }
mkdir -p "$LOGDIR"

# --- one instance at a time -------------------------------------------------
# Without this, the cron supervisor would start a second copy on top of a
# running one and two processes would write to the same runs/<name>/.
exec 9>"$LOCK" || exit 1
if ! flock -n 9; then
    exit 0                                # a copy is already working; fine
fi

say()  { echo "$(date -Is) $*" | tee -a "$LOGDIR/queue.log"; }
mark() { printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS"; }

if [ ! -f "$QUEUE" ]; then
    say "ERROR: queue file not found: $QUEUE"
    exit 1
fi

# shellcheck disable=SC1091
source src/.venv/bin/activate 2>/dev/null || {
    say "ERROR: could not activate src/.venv"; exit 1; }

say "=== queue start ($QUEUE) ==="

# --- memory advice, not a gate ----------------------------------------------
# evaluate.py peaks around 9.4 GB on this dataset (measured 13 Sept). On a
# 15 GB machine that only fits when nothing else large is running, so say so
# loudly in the log rather than discovering it from an OOM line later.
big=$(ps -eo rss=,comm= --sort=-rss | awk 'NR<=5 && $1>1500000 {printf "%s(%.1fGB) ", $2, $1/1048576}')
[ -n "$big" ] && say "NOTE: large processes present: $big-- an OOM kill is likelier"
say "memory: $(free -m | awk '/^Mem:/{print $7" MB available"}'), swap $(free -m | awk '/^Swap:/{print $3"/"$2" MB used"}')"

done_n=0; fail_n=0; skip_n=0

while read -r name level seed overlay _rest; do
    case "$name" in ''|\#*) continue ;; esac
    [ -z "$level" ] || [ -z "$seed" ] && { say "SKIP malformed line: $name"; continue; }
    overlay="${overlay:--}"

    # --- already finished? --------------------------------------------------
    if [ -f "runs/$name/eval_val/summary.json" ]; then
        skip_n=$((skip_n+1))
        continue
    fi

    say "--- $name (level $level, seed $seed, overlay $overlay) ---"

    # Do not start a step into a machine that is already short of memory --
    # that is how 13 September happened. And do not SLEEP through it either:
    # sleeping holds the flock, which would block the cron supervisor for no
    # reason. Release the lock, let the next tick try. Fifteen minutes is
    # nothing against an 84-hour window.
    avail=$(free -m | awk '/^Mem:/{print $7}')
    if [ "$avail" -lt "$MIN_FREE_MB" ]; then
        say "$name: only ${avail} MB available (need ${MIN_FREE_MB}) -- exiting, the next tick will retry"
        mark "$name" "deferred_low_memory"
        exit 0
    fi

    ov=()
    [ "$overlay" != "-" ] && ov=(--overlay "configs/$overlay")

    # --- training -----------------------------------------------------------
    if [ -f "runs/$name/run_metrics.json" ]; then
        say "$name: training already complete"
    else
        for attempt in 1 2; do
            res=()
            # A weights file with no run_metrics.json means training started
            # and did not finish -- resume instead of throwing the epochs away
            # (decision 3.87: resume deviation is 0.0009, below every measured
            # threshold).
            [ -f "runs/$name/weights/last.pt" ] && res=(--resume)
            say "$name: train attempt $attempt ${res[*]}"
            python scripts/train.py --config configs/base.yaml "${ov[@]}" \
                --level "$level" --seed "$seed" --name "$name" "${res[@]}" \
                >> "$LOGDIR/$name.log" 2>&1
            if [ -f "runs/$name/run_metrics.json" ]; then
                say "$name: train ok"
                mark "$name" "train_ok"
                break
            fi
            say "$name: TRAIN FAILED (attempt $attempt) -- see logs/$name.log"
            mark "$name" "train_fail_$attempt"
            sleep "$RETRY_SLEEP"
        done
    fi

    if [ ! -f "runs/$name/run_metrics.json" ]; then
        say "$name: giving up on training, moving to the next run"
        fail_n=$((fail_n+1))
        continue                          # no weights -> nothing to evaluate
    fi

    # --- evaluation ---------------------------------------------------------
    # This is the step that was OOM-killed. It is retried on its own, so a
    # failure here never costs the training hours that came before it.
    for attempt in 1 2; do
        say "$name: eval attempt $attempt"
        python src/eval/evaluate.py --weights "runs/$name/weights/best.pt" \
            --split val --out "runs/$name/eval_val" --tag "$name" \
            >> "$LOGDIR/$name.log" 2>&1
        if [ -f "runs/$name/eval_val/summary.json" ]; then
            say "$name: eval ok"
            mark "$name" "eval_ok"
            done_n=$((done_n+1))
            break
        fi
        say "$name: EVAL FAILED (attempt $attempt) -- see logs/$name.log"
        mark "$name" "eval_fail_$attempt"
        sleep "$RETRY_SLEEP"
    done

    [ -f "runs/$name/eval_val/summary.json" ] || fail_n=$((fail_n+1))

done < "$QUEUE"

# --- collect ----------------------------------------------------------------
# Always, even if runs failed: results/ is the only place the numbers live
# (runs/ is gitignored) and a partial collection is better than none.
say "collecting results"
python scripts/collect.py >> "$LOGDIR/queue.log" 2>&1 \
    && say "collect ok" || say "COLLECT FAILED"

say "=== queue end: $done_n completed, $fail_n failed, $skip_n already done ==="
