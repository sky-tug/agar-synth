#!/usr/bin/env python3
"""
Collects the run results -- the results table of the paper will come from here.

WHY THIS EXISTS (decision 3.33):
  `runs/` is in .gitignore (and it should be: the weights are gigabytes). But
  under runs/ there are not only weights -- every run's `run_metrics.json`
  (duration, VRAM, GPU-hours, git commit) and its `eval*/summary.json` (mAP,
  per-class AP, counting errors) are in there too. That is, ALL THE RESULTS OF
  THE PAPER were staying outside version control.

  These files are a few KB but they are NOT REPRODUCIBLE: it takes 500+
  GPU-hours. If the laptop dies, or once we move to the BIDB server, they exist
  nowhere.

  `collect.py` scans them all and writes them under `results/`. `results/` DOES
  GO into git. On top of that you had to build the 61-run grid table somehow --
  this solves both needs at once.

Usage:
    python scripts/collect.py                     # runs/ -> results/
    python scripts/collect.py --runs runs --out results
    python scripts/collect.py --include-smoke     # take the smoke tests too

Output:
    results/table.csv        one row per run -- the main table
    results/raw/<run>.json   metrics + evaluation, merged per run
    results/summary.txt      a copy of the summary printed to the screen
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# table column order -- keep it close to the table of the paper
COLUMNS = [
    "run", "level", "seed", "overlay", "split",
    "mAP50-95", "mAP50", "mAP75",
    "mAP_small", "mAP_medium", "mAP_large",
    "AP_S.aureus", "AP_B.subtilis", "AP_P.aeruginosa", "AP_E.coli", "AP_C.albicans",
    "MAE", "RMSE", "sMAPE", "ME", "count_r", "conf_thr_count",
    "n_train", "n_val", "n_images", "n_gt_boxes",
    "imgsz", "batch", "epochs_planned", "epochs_actual",
    "total_min", "sec_per_epoch", "gpu_hours", "peak_vram_gb",
    "images_per_sec", "ms_per_image",
    "gpu_name", "torch", "git_commit", "date_utc", "smoke_test",
]

# evaluate.py is still mid-translation: a handful of summary.json keys have not
# been renamed yet. The table columns are the English names; these aliases let
# the old spellings still be picked up, so summaries written before and after
# the rename both land in the same column.
def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ! could not read: {p}  ({e})", file=sys.stderr)
        return None


def collect(runs: Path, include_smoke: bool):
    records = []
    for run_dir in sorted(d for d in runs.iterdir() if d.is_dir()):
        mf = run_dir / "run_metrics.json"
        metrics = _read_json(mf) if mf.exists() else None

        # eval*/summary.json  -- catch eval, eval_val, eval_test, all of them
        summaries = {}
        for ed in sorted(run_dir.glob("eval*")):
            sf = ed / "summary.json"
            if sf.exists():
                d = _read_json(sf)
                if d:
                    summaries[d.get("split", ed.name)] = d

        if metrics is None and not summaries:
            continue                       # empty folder, skip
        if metrics and metrics.get("smoke_test") and not include_smoke:
            continue

        records.append({"run": run_dir.name, "metrics": metrics,
                        "summaries": summaries})
    return records


def to_rows(records):
    """One row per (run, split) pair. If there is no evaluation, metrics only."""
    rows = []
    for rec in records:
        m = rec["metrics"] or {}
        env = m.get("environment", {})
        base = {
            "run": rec["run"],
            "level": m.get("level"), "seed": m.get("seed"),
            "overlay": Path(m["overlay"]).stem if m.get("overlay") else "base",
            "n_train": m.get("n_train"), "n_val": m.get("n_val"),
            "imgsz": m.get("imgsz"), "batch": m.get("batch"),
            "epochs_planned": m.get("epochs_planned"),
            "epochs_actual": m.get("epochs_actual"),
            "total_min": m.get("total_min"),
            "sec_per_epoch": m.get("sec_per_epoch"),
            "gpu_hours": m.get("gpu_hours"), "peak_vram_gb": m.get("peak_vram_gb"),
            "images_per_sec": m.get("images_per_sec"),
            "gpu_name": env.get("gpu_name"), "torch": env.get("torch"),
            "git_commit": m.get("git_commit"), "date_utc": m.get("date_utc"),
            "smoke_test": m.get("smoke_test"),
        }
        if not rec["summaries"]:
            rows.append({**base, "split": None})
            continue
        for split, s in sorted(rec["summaries"].items()):
            r = dict(base)
            r["split"] = split
            for c in COLUMNS:
                if c in r:
                    continue
                if c in s:
                    r[c] = s[c]
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--include-smoke", action="store_true",
                    help="take the runs marked as smoke tests into the table too")
    a = ap.parse_args()

    runs = Path(a.runs).expanduser().resolve()
    if not runs.is_dir():
        sys.exit(f"ERROR: {runs} does not exist.")
    out = Path(a.out).expanduser().resolve()
    (out / "raw").mkdir(parents=True, exist_ok=True)

    records = collect(runs, a.include_smoke)
    if not records:
        sys.exit(f"ERROR: no metrics/evaluation found under {runs}.\n"
                 f"  (to include the smoke tests use --include-smoke)")

    # --- raw copies: one file per run, goes into git ---
    for rec in records:
        (out / "raw" / f"{rec['run']}.json").write_text(
            json.dumps({"metrics": rec["metrics"], "evaluation": rec["summaries"]},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    # --- table ---
    rows = to_rows(records)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = None
        df = df[COLUMNS]
        df.to_csv(out / "table.csv", index=False)
    except ImportError:
        import csv
        with (out / "table.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        df = None

    # --- summary ---
    L = []
    L.append("=" * 72)
    L.append(f"COLLECTED RESULTS   ({datetime.now(timezone.utc).isoformat(timespec='seconds')})")
    L.append("=" * 72)
    L.append(f"  source            : {runs}")
    L.append(f"  runs              : {len(records)}")
    L.append(f"  table rows        : {len(rows)}  (run x split)")

    measured = [r for r in records if r["metrics"]]
    gpu = sum(r["metrics"].get("gpu_hours") or 0 for r in measured)
    L.append(f"  total GPU-hours   : {gpu:.2f}")
    evaluated = [r for r in records if r["summaries"]]
    L.append(f"  evaluated         : {len(evaluated)} / {len(records)}")

    missing_eval = [r["run"] for r in records if not r["summaries"]]
    if missing_eval:
        L.append(f"  ! runs with no evaluation: {', '.join(missing_eval)}")
    missing_metrics = [r["run"] for r in records if not r["metrics"]]
    if missing_metrics:
        L.append(f"  [!] runs with NO run_metrics.json: {', '.join(missing_metrics)}")
        L.append("     (was it run with YOLO directly instead of train.py? "
                 "the duration/VRAM are lost)")

    # test rows that have the primary metric
    test = [r for r in rows if r.get("split") == "test" and r.get("mAP50-95") is not None]
    if test:
        L.append("")
        L.append("  --- TEST RESULTS (primary metric) ---")
        L.append(f"  {'run':<22} {'mAP50-95':>9} {'small':>8} {'MAE':>7}")
        for r in sorted(test, key=lambda x: -(x.get("mAP50-95") or 0)):
            L.append(f"  {r['run']:<22} {r['mAP50-95']:>9.4f} "
                     f"{(r.get('mAP_small') or float('nan')):>8.4f} "
                     f"{(r.get('MAE') or float('nan')):>7.2f}")
    L.append("")
    L.append(f"  table -> {out / 'table.csv'}")
    L.append(f"  raw   -> {out / 'raw'}/")
    L.append("=" * 72)
    L.append("This folder GOES INTO GIT. Run it after the runs and commit it --")
    L.append("runs/ is in .gitignore, the results live nowhere else.")

    text = "\n".join(L)
    print(text)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
