#!/usr/bin/env python3
"""
Measures `significant_diff_threshold` from the seed spread of one arm.

WHY THIS EXISTS (Phase 6, section 6 of FAZ6_BASLANGIC):
  `configs/base.yaml` carries `significant_diff_threshold: null` on purpose. Two
  claims of the study are blocked on it:

    - whether the resume deviation (0.0009 on the last epoch) is negligible
    - whether the gap between two arms of the grid means anything at all

  Both need one number: how far apart two runs of the SAME arm land when only
  the seed changes. This script measures it, and it is meant to be the FIRST
  output of the grid -- before any result table.

DEFINITION (decided 4 September 2026, before the numbers were seen):
  sigma  = sample standard deviation (ddof=1) of mAP50-95 across the seeds
  thresh = 2 * sigma

  A two-sample comparison of two 5-seed arms would need about 1.46 * sigma, so
  2 * sigma is deliberately conservative: it refuses to call a difference real
  slightly more often than the statistics demand. It cannot make us over-claim.

  sigma is itself estimated from very few runs, so a chi-square interval on it
  is reported alongside (decision 3.90: a gate number without an error bar
  cannot decide anything). With 3 seeds the 90% interval spans roughly
  0.58*sigma to 4.42*sigma -- the point estimate is not the whole story and the
  paper must say so.

RUNTIME CHECKS (principle 1: a methodological rule is a runtime check):
  The script STOPS -- it does not warn and continue -- when the runs it was
  given are not comparable: different level, different split, a smoke test, or
  a run that was resumed after an interruption. A resumed run cannot be part of
  the measurement that is supposed to judge resume deviation in the first place.
  A differing git_commit is reported loudly but does not stop the run, because
  a commit may touch only bookkeeping; the operator has to look and decide.

Usage:
    python scripts/threshold.py --runs G100_s0 G100_s1 G100_s2
    python scripts/threshold.py --runs G100_s0 G100_s1 G100_s2 --split test
    python scripts/threshold.py --runs G100_s0 G100_s1 G100_s2 --write

Output:
    results/threshold.json     the measurement, goes into git
    (--write) configs/base.yaml: fills in significant_diff_threshold
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRIC = "mAP50-95"
CONF_LEVEL = 0.90

# The threshold is defined on METRIC alone. These are reported next to it
# because the paper argues about small objects and per-species behaviour, and
# the seed spread of a sub-metric is not the seed spread of the primary one.
# A sub-metric is judged by ITS OWN sigma, never by the headline threshold
# (principle 4: what a gate does not measure).
SECONDARY = ["mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
             "mAP_small_minGT", "mAP_medium_minGT", "mAP_large_minGT",
             "AP_S.aureus", "AP_B.subtilis", "AP_P.aeruginosa",
             "AP_E.coli", "AP_C.albicans"]

# Decision 3.96d, fixed in 3.97g. The three size cells above are the COCO
# columns: they average over the size cells of every class, including cells
# with almost no ground truth. In this dataset the LARGE cell of S.aureus
# holds exactly ONE box in val, and it carries a fifth of the large-object
# average -- so mAP_large moved with a single detection and its seed sigma came
# out ~19x the primary metric's. Decision 3.94a added the *_minGT columns,
# which drop cells below `min_gt_for_cell` ground-truth boxes, precisely to
# remove that pathology. This script was reading only the COCO columns and so
# reported the pathology as if it were seed noise.
#
# Both are reported now, on purpose: metrics.py stays COCO-identical (3.94),
# so the COCO column is the comparable-to-the-literature number and the minGT
# column is the one to reason about. A run trained before 3.94a has no minGT
# column; that key is then skipped and listed in `missing_secondary`.
MINGT_PREFERRED = {
    "mAP_small": "mAP_small_minGT",
    "mAP_medium": "mAP_medium_minGT",
    "mAP_large": "mAP_large_minGT",
}

# The deviation measured in decision 3.87 between an interrupted run that was
# resumed and the same run left alone. This script exists partly to judge it.
RESUME_DEVIATION = 0.0009

# chi-square quantiles for df = 1..9 at the 5% and 95% points, so the interval
# works without SciPy. SciPy is used when present.
_CHI2 = {
    1: (0.003932, 3.841459),
    2: (0.102587, 5.991465),
    3: (0.351846, 7.814728),
    4: (0.710721, 9.487729),
    5: (1.145476, 11.070498),
    6: (1.635383, 12.591587),
    7: (2.167350, 14.067140),
    8: (2.732637, 15.507313),
    9: (3.325113, 16.918978),
}


def _chi2_bounds(df: int) -> tuple[float, float]:
    """(lower 5% quantile, upper 95% quantile) of chi-square with `df`."""
    try:
        from scipy.stats import chi2
        a = (1.0 - CONF_LEVEL) / 2.0
        return float(chi2.ppf(a, df)), float(chi2.ppf(1.0 - a, df))
    except ImportError:
        if df not in _CHI2:
            sys.exit(f"ERROR: SciPy missing and df={df} is not in the table "
                     f"(1..9). Install SciPy or use fewer seeds.")
        return _CHI2[df]


def load_run(runs_dir: Path, name: str, split: str) -> dict:
    run_dir = runs_dir / name
    if not run_dir.is_dir():
        sys.exit(f"ERROR: run folder not found: {run_dir}")

    mf = run_dir / "run_metrics.json"
    if not mf.exists():
        sys.exit(f"ERROR: {name}: run_metrics.json is missing. Was it trained "
                 f"with YOLO directly instead of train.py?")
    metrics = json.loads(mf.read_text(encoding="utf-8"))

    sf = run_dir / f"eval_{split}" / "summary.json"
    if not sf.exists():
        conf = "" if split == "val" else " --conf-thr 0.35"
        sys.exit(f"ERROR: {name}: {sf.relative_to(runs_dir)} is missing.\n"
                 f"  Run the evaluation first:\n"
                 f"    python src/eval/evaluate.py \\\n"
                 f"        --weights runs/{name}/weights/best.pt \\\n"
                 f"        --split {split} --out runs/{name}/eval_{split} \\\n"
                 f"        --tag {name}{conf}")
    summary = json.loads(sf.read_text(encoding="utf-8"))

    if METRIC not in summary:
        sys.exit(f"ERROR: {name}: '{METRIC}' is not in summary.json. "
                 f"Did evaluate.py rename the key?")

    return {"run": name, "metrics": metrics, "summary": summary}


def check_comparable(records: list[dict], split: str) -> list[str]:
    """Stops on anything that would make the spread mean something else."""
    notes = []
    fatal = []

    levels = {r["metrics"].get("level") for r in records}
    if len(levels) > 1:
        fatal.append(f"the runs are not from the same level: {sorted(levels)}")

    splits = {r["summary"].get("split") for r in records}
    if splits != {split}:
        fatal.append(f"summary.json split disagrees with --split {split}: {splits}")

    seeds = [r["metrics"].get("seed") for r in records]
    if len(set(seeds)) != len(seeds):
        fatal.append(f"the same seed appears more than once: {seeds}")

    for r in records:
        if r["metrics"].get("smoke_test"):
            fatal.append(f"{r['run']} is marked as a smoke test")
        if r["metrics"].get("resumed"):
            fatal.append(
                f"{r['run']} was resumed after an interruption. The seed spread "
                f"is what judges resume deviation -- it cannot contain one. "
                f"Re-run this seed from scratch.")

    commits = {r["metrics"].get("git_commit") for r in records}
    if len(commits) > 1:
        notes.append(
            f"[!] the runs come from different commits: {sorted(commits)}. "
            f"Check that nothing on the training path changed between them, "
            f"otherwise code variance is being read as seed variance.")

    # conf_thr_count is the COUNTING threshold, re-selected on val by
    # evaluate.py in every run. It moves from seed to seed (0.35 / 0.40 were
    # both seen in Phase 6). It does NOT touch mAP: average precision is
    # integrated over the whole confidence range, so the metric this script
    # measures is unaffected. It DOES decide MAE / RMSE / sMAPE / ME, so the
    # counting numbers of these runs are NOT comparable to each other. Loud
    # note, not a stop -- and the reason decision 3.88 locked 0.35 for the grid.
    confs = {r["summary"].get("conf_thr_count") for r in records}
    if len(confs) > 1:
        notes.append(
            f"[!] the counting threshold selected on val differs between runs: "
            f"{sorted(confs)}. mAP is unaffected (it is integrated over all "
            f"confidences), so the threshold below stands. But the counting "
            f"metrics (MAE / RMSE / sMAPE / ME) of these runs were computed "
            f"with different thresholds and must NOT be compared seed to seed.")

    locked = {r["summary"].get("conf_thr_locked") for r in records
              if r["summary"].get("conf_thr_locked") is not None}
    if len(locked) > 1:
        fatal.append(f"the LOCKED conf_thr differs between runs: {sorted(locked)}")
    if any(r["summary"].get("conf_thr_overridden") for r in records):
        fatal.append("at least one run was evaluated with --override-conf-thr")

    if fatal:
        print("STOPPED -- the runs are not comparable:", file=sys.stderr)
        for f in fatal:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    return notes


def measure(records: list[dict]) -> dict:
    values = [r["summary"][METRIC] for r in records]
    n = len(values)
    if n < 2:
        sys.exit("ERROR: at least 2 seeds are needed for a standard deviation.")

    mean = statistics.fmean(values)
    sigma = statistics.stdev(values)            # ddof = 1
    threshold = 2.0 * sigma

    df = n - 1
    lo_q, hi_q = _chi2_bounds(df)
    # (n-1) s^2 / chi2_high  <=  sigma^2  <=  (n-1) s^2 / chi2_low
    sigma_lo = (df * sigma ** 2 / hi_q) ** 0.5
    sigma_hi = (df * sigma ** 2 / lo_q) ** 0.5

    secondary = {}
    missing_secondary = []
    for key in SECONDARY:
        vals = [r["summary"].get(key) for r in records]
        if any(v is None for v in vals):
            missing_secondary.append(key)
            continue
        s = statistics.stdev(vals)
        entry = {
            "mean": statistics.fmean(vals),
            "sigma": s,
            "own_threshold_2sigma": 2.0 * s,
            "times_primary_sigma": (s / sigma) if sigma > 0 else None,
        }
        # Say out loud which of the two size columns this is, so nobody has to
        # remember (3.96d: this script used to report only the COCO one).
        if key in MINGT_PREFERRED:
            entry["column"] = "COCO (all size cells; see min_gt note)"
            entry["prefer_instead"] = MINGT_PREFERRED[key]
        elif key.endswith("_minGT"):
            entry["column"] = "min_gt_for_cell filtered (3.94a) -- prefer this one"
        secondary[key] = entry

    if missing_secondary:
        print(f"  [!] sub-metrics absent from at least one run, not reported: "
              f"{', '.join(missing_secondary)}")
        if any(k.endswith('_minGT') for k in missing_secondary):
            print("      the *_minGT columns arrived with decision 3.94a; a run "
                  "trained before it has only the COCO size columns.")

    return {
        "metric": METRIC,
        "n_seeds": n,
        "secondary": secondary,
        "missing_secondary": missing_secondary,
        "values": {r["run"]: r["summary"][METRIC] for r in records},
        "mean": mean,
        "spread_max_minus_min": max(values) - min(values),
        "sigma": sigma,
        "sigma_ci90": [sigma_lo, sigma_hi],
        "significant_diff_threshold": threshold,
        "threshold_ci90": [2.0 * sigma_lo, 2.0 * sigma_hi],
        "definition": "2 * sample stdev (ddof=1) of mAP50-95 across seeds",
        "resume_deviation": RESUME_DEVIATION,
        "resume_deviation_below_threshold": RESUME_DEVIATION < threshold,
        "resume_deviation_below_threshold_lower_bound":
            RESUME_DEVIATION < 2.0 * sigma_lo,
    }


def write_into_config(cfg: Path, value: float) -> None:
    text = cfg.read_text(encoding="utf-8")
    pat = re.compile(r"^(\s*significant_diff_threshold:\s*)(\S+)(.*)$", re.M)
    m = pat.search(text)
    if not m:
        sys.exit(f"ERROR: significant_diff_threshold not found in {cfg}")
    if m.group(2) != "null":
        sys.exit(f"ERROR: {cfg} already holds "
                 f"significant_diff_threshold: {m.group(2)}. Refusing to "
                 f"overwrite a frozen value -- edit it by hand and say why in "
                 f"decisions.md.")
    new = pat.sub(rf"\g<1>{value:.6f}"
                  f"    # measured in Phase 6, see results/threshold.json", text)
    cfg.write_text(new, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run names, e.g. G100_s0 G100_s1 G100_s2")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="which evaluation to read (default: val)")
    ap.add_argument("--out", default=str(ROOT / "results" / "threshold.json"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--write", action="store_true",
                    help="fill significant_diff_threshold into the config "
                         "(refuses if it is not null)")
    a = ap.parse_args()

    runs_dir = Path(a.runs_dir).expanduser().resolve()
    records = [load_run(runs_dir, name, a.split) for name in a.runs]
    notes = check_comparable(records, a.split)

    res = measure(records)
    res["split"] = a.split
    res["runs_dir"] = str(runs_dir)
    res["date_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    res["notes"] = notes

    w = 72
    print("=" * w)
    print(f"SIGNIFICANT_DIFF_THRESHOLD   split={a.split}   metric={METRIC}")
    print("=" * w)
    for run, v in res["values"].items():
        print(f"  {run:<24} {v:.5f}")
    print(f"  {'mean':<24} {res['mean']:.5f}")
    print(f"  {'max - min':<24} {res['spread_max_minus_min']:.5f}")
    print("-" * w)
    print(f"  sigma (ddof=1)           {res['sigma']:.5f}")
    print(f"  sigma 90% CI             {res['sigma_ci90'][0]:.5f} .. "
          f"{res['sigma_ci90'][1]:.5f}   (n={res['n_seeds']}, chi-square)")
    print("-" * w)
    print(f"  THRESHOLD = 2 * sigma    {res['significant_diff_threshold']:.5f}")
    print(f"  threshold 90% CI         {res['threshold_ci90'][0]:.5f} .. "
          f"{res['threshold_ci90'][1]:.5f}")
    print("-" * w)
    print(f"  resume deviation (3.87)  {RESUME_DEVIATION:.5f}")
    verdict = ("BELOW the threshold" if res["resume_deviation_below_threshold"]
               else "ABOVE the threshold")
    print(f"    point estimate         {verdict}")
    if res["resume_deviation_below_threshold_lower_bound"]:
        print("    lower CI bound         still below -- the judgement holds "
              "across the interval")
    else:
        print("    lower CI bound         NOT below -- with this few seeds the "
              "judgement is not settled")
    if res.get("secondary"):
        print("-" * w)
        print("  SUB-METRICS -- each has its own spread. The threshold above")
        print("  does NOT apply to them; judge each by its own 2*sigma.")
        print(f"  {'metric':<18}{'mean':>9}{'sigma':>10}{'2*sigma':>10}"
              f"{'x primary':>11}")
        for k, d in res["secondary"].items():
            mult = d["times_primary_sigma"]
            # 3.96d/3.97g: mark the size cells so the two columns are never
            # mistaken for one another at a glance.
            mark = ""
            if k in MINGT_PREFERRED:
                mark = "   <- COCO column, see *_minGT below"
            elif k.endswith("_minGT"):
                mark = "   <- use this one"
            print(f"  {k:<18}{d['mean']:>9.4f}{d['sigma']:>10.5f}"
                  f"{d['own_threshold_2sigma']:>10.5f}"
                  f"{(f'{mult:.1f}x' if mult else '-'):>11}{mark}")
    for n in notes:
        print(f"\n  {n}")
    print("=" * w)

    out = Path(a.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"  written -> {out}")

    if a.write:
        cfg = Path(a.config).expanduser().resolve()
        write_into_config(cfg, res["significant_diff_threshold"])
        print(f"  config  -> {cfg}  (significant_diff_threshold filled in)")
    else:
        print("  config untouched. Add --write to fill it in.")


if __name__ == "__main__":
    main()
