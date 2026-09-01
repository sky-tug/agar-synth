#!/usr/bin/env python3
"""
08_count_distribution.py -- how many colonies are on a plate, REALLY?

WHY THIS EXISTS
---------------
Decision 3.17 declared the colony count lognormal. That family was chosen on the
10-plate demo package, where 10 numbers cannot distinguish one heavy-tailed family
from another. `layout.py fit` estimates the two parameters like this:

    ln_loc = median(ln n)      <- deliberately ROBUST to a single crowded plate
    ln_std = std(ln n)         <- NOT robust

The comment above those lines names the danger ("the mean is affected by a single
crowded plate") and then defends only half of the pair.

On the FULL 2987-plate train split the consequence became measurable:

    real       median 13    mean 19.47
    generated  median 12    mean 27.61      <- +42 %

The median is reproduced exactly; the mean is not. In a lognormal the mean depends
on sigma EXPONENTIALLY (mean/median = exp(sigma^2/2)), so a sigma that is somewhat
too large inflates the mean a great deal while leaving the median untouched --
which is precisely the observed signature.

WHY THE MEAN MATTERS MORE THAN THE MEDIAN HERE
----------------------------------------------
The substitution curve compares "N real images" against "N synthetic images". If a
synthetic image carries 42 % more labelled objects than a real one, the synthetic
arm receives more supervision per image, and the curve is biased IN FAVOUR OF
SYNTHETIC DATA -- the paper's central claim, measured with a loaded ruler.

Note that `layout.py validate` reports the MEDIAN and never the mean, so the
project's own gate could not see this. That is a finding about the check, not only
about the data.

WHAT THIS SCRIPT DECIDES
------------------------
It does not assume a family. It asks which of three candidate models reproduces the
real distribution -- median, MEAN, tail and range together:

    A  current      lognormal(median(ln n), std(ln n))
    B  robust       lognormal(median(ln n), IQR(ln n) / 1.349)
    C  empirical    inverse-CDF over the quantile grid
                    -- the idiom layout.py ALREADY uses for `radial` and `size`;
                       `count` is the only place left with a parametric family

Every model is sampled through the same clip to [min, max] that `recipe()` applies,
so what is compared is what layout.py would actually produce.

It also separates the two competing explanations of the +42 %:

    sampling noise   -> the real mean falls inside the 5-95 % band of the model's
                        own replicates at the real sample size
    fit error        -> it falls outside

Read-only with respect to the project. Writes one CSV.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

# the quantile grid layout.py uses for radial / size
QUANTILES = [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]

# IQR of a normal distribution = 1.349 * sigma
IQR_TO_SIGMA = 1.3489795


def read_counts(list_path: Path, label_root: Path) -> tuple[np.ndarray, int, int]:
    """Colonies per plate, counted exactly the way layout.py._read_labels does:
    a label file with no valid line is DROPPED, not counted as zero."""
    stems = [l.strip() for l in list_path.read_text().split("\n") if l.strip()]
    counts, missing, empty = [], 0, 0
    for s in stems:
        lab = label_root / (Path(s).stem + ".txt")
        if not lab.exists():
            missing += 1
            continue
        n = sum(1 for line in lab.read_text().split("\n") if len(line.split()) == 5)
        if n == 0:
            empty += 1
            continue
        counts.append(n)
    return np.asarray(counts, dtype=float), missing, empty


def draw_A(rng, ns, size):
    """current: lognormal, sigma = std(ln n)"""
    ln = np.log(ns)
    mu, sd = float(np.median(ln)), float(ln.std(ddof=1))
    x = np.round(np.exp(rng.normal(mu, sd, size)))
    return np.clip(x, ns.min(), ns.max()), dict(ln_loc=mu, ln_std=sd)


def draw_B(rng, ns, size):
    """robust: lognormal, sigma = IQR(ln n) / 1.349"""
    ln = np.log(ns)
    mu = float(np.median(ln))
    sd = float(np.percentile(ln, 75) - np.percentile(ln, 25)) / IQR_TO_SIGMA
    x = np.round(np.exp(rng.normal(mu, sd, size)))
    return np.clip(x, ns.min(), ns.max()), dict(ln_loc=mu, ln_std=sd)


def draw_C(rng, ns, size):
    """empirical inverse-CDF over the quantile grid -- layout.py's own idiom"""
    q = QUANTILES
    v = [float(np.percentile(ns, p)) for p in q]
    u = rng.random(size) * 100.0
    x = np.round(np.interp(u, q, v))
    return np.clip(x, ns.min(), ns.max()), dict(q=q, values=[round(z, 2) for z in v])


MODELS = [("A  current  lognormal std(ln n)", draw_A),
          ("B  robust   lognormal IQR/1.349", draw_B),
          ("C  empirical inverse-CDF       ", draw_C)]


def summary(a: np.ndarray) -> dict:
    return dict(n=len(a), min=float(a.min()), p25=float(np.percentile(a, 25)),
                median=float(np.median(a)), p75=float(np.percentile(a, 75)),
                p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
                max=float(a.max()), mean=float(a.mean()), total=float(a.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True,
                    help="e.g. data/processed/lists/train.txt -- the SAME list "
                         "layout.py fit was given, otherwise the levels do not match")
    ap.add_argument("--labels", default="data/processed/labels")
    ap.add_argument("--out", default=None,
                    help="per-plate counts CSV "
                         "(default: data/processed/reports/count_distribution.csv)")
    ap.add_argument("--pop", type=int, default=200_000,
                    help="draws for the population estimate of each model")
    ap.add_argument("--reps", type=int, default=200,
                    help="replicates AT THE REAL SAMPLE SIZE, for the noise band")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lst, lbl = Path(args.list), Path(args.labels)
    if not lst.exists():
        sys.exit(f"ERROR: list not found: {lst}")
    if not lbl.is_dir():
        sys.exit(f"ERROR: label directory not found: {lbl}")

    ns, missing, empty = read_counts(lst, lbl)
    if len(ns) < 50:
        sys.exit(f"ERROR: only {len(ns)} plates could be read. "
                 "That is enough to pick a FAMILY, not to estimate parameters.")

    out_csv = (Path(args.out) if args.out
               else Path("data/processed/reports/count_distribution.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    real = summary(ns)
    ln = np.log(ns)

    print("=" * 74)
    print("COLONY COUNT DISTRIBUTION")
    print("=" * 74)
    print(f"  list                 : {lst}")
    print(f"  plates read          : {len(ns)}"
          + (f"   ({missing} label missing, {empty} empty -- both dropped)"
             if (missing or empty) else ""))
    print()

    # ---------------------------------------------------------------- real
    print("  --- REAL ---")
    print(f"      min {real['min']:.0f}   p25 {real['p25']:.0f}   "
          f"median {real['median']:.0f}   p75 {real['p75']:.0f}   "
          f"p90 {real['p90']:.0f}   p99 {real['p99']:.0f}   max {real['max']:.0f}")
    print(f"      mean {real['mean']:.2f}      total colonies {real['total']:.0f}")
    print(f"      mean / median = {real['mean']/real['median']:.3f}")
    print()

    # ------------------------------------------------- is ln(n) normal at all?
    sd_plain = float(ln.std(ddof=1))
    sd_robust = float(np.percentile(ln, 75) - np.percentile(ln, 25)) / IQR_TO_SIGMA
    sd_implied = math.sqrt(2 * math.log(real["mean"] / real["median"]))
    print("  --- IS ln(n) NORMAL? three estimates of the SAME sigma ---")
    print(f"      std(ln n)                       = {sd_plain:.4f}   (what fit uses)")
    print(f"      IQR(ln n) / 1.349               = {sd_robust:.4f}   (robust)")
    print(f"      sigma implied by the real mean  = {sd_implied:.4f}")
    print("      For a truly lognormal sample all three agree. They do not have to")
    print("      agree exactly -- but a large gap means the FAMILY is wrong, not")
    print("      just the estimator.")
    print()
    print("      normal-quantile comparison of ln(n)  (z = (q - median) / sigma)")
    print(f"      {'p':>5}  {'ln(n)':>8}  {'z observed':>11}  {'z normal':>9}")
    for p, z_th in ((1, -2.326), (5, -1.645), (10, -1.282), (25, -0.674),
                    (50, 0.0), (75, 0.674), (90, 1.282), (95, 1.645), (99, 2.326)):
        q = float(np.percentile(ln, p))
        print(f"      {p:>5}  {q:>8.3f}  {(q - np.median(ln)) / sd_plain:>11.3f}  {z_th:>9.3f}")
    print()

    # ------------------------------------------------------------- the models
    print("  --- CANDIDATE MODELS, sampled through the same clip recipe() applies ---")
    print(f"      {'model':<34} {'median':>7} {'mean':>8} {'p90':>6} {'p99':>6} {'max':>6}")
    print(f"      {'REAL':<34} {real['median']:>7.0f} {real['mean']:>8.2f} "
          f"{real['p90']:>6.0f} {real['p99']:>6.0f} {real['max']:>6.0f}")
    pop = {}
    for name, fn in MODELS:
        x, params = fn(np.random.default_rng(args.seed + 1), ns, args.pop)
        s = summary(x)
        pop[name] = (s, params)
        print(f"      {name:<34} {s['median']:>7.0f} {s['mean']:>8.2f} "
              f"{s['p90']:>6.0f} {s['p99']:>6.0f} {s['max']:>6.0f}")
    print()

    # ------------------------------------- noise band at the real sample size
    print(f"  --- NOISE vs FIT ERROR: {args.reps} replicates of n = {len(ns)} ---")
    print(f"      real mean = {real['mean']:.2f}")
    print(f"      {'model':<34} {'mean of means':>14} {'5-95 % band':>18}  verdict")
    for name, fn in MODELS:
        means = []
        for r in range(args.reps):
            x, _ = fn(np.random.default_rng(args.seed + 1000 + r), ns, len(ns))
            means.append(x.mean())
        m = np.asarray(means)
        lo, hi = float(np.percentile(m, 5)), float(np.percentile(m, 95))
        inside = lo <= real["mean"] <= hi
        print(f"      {name:<34} {m.mean():>14.2f} {f'[{lo:.2f} - {hi:.2f}]':>18}  "
              f"{'OK  real mean inside' if inside else 'FAIL  real mean OUTSIDE'}")
    print()

    # ------------------------------------------- what it costs downstream
    print("  --- DOWNSTREAM COST at 58200 synthetic plates ---")
    print(f"      {'model':<34} {'colonies':>12}  {'vs real':>9}")
    real_rate = real["mean"]
    print(f"      {'REAL rate':<34} {58200*real_rate:>12.0f}  {'--':>9}")
    for name, _ in MODELS:
        s = pop[name][0]
        print(f"      {name:<34} {58200*s['mean']:>12.0f}  "
              f"{(s['mean']/real_rate-1)*100:>+8.1f}%")
    print()

    print("  --- READ IT LIKE THIS ---")
    print("   * a model is acceptable only if it reproduces the median AND the mean.")
    print("     The median alone is what validate already checks, and it is what")
    print("     let this through.")
    print("   * if B lands on the real mean, the FAMILY is fine and only the sigma")
    print("     estimator was wrong -> one-line fix in layout.py fit.")
    print("   * if only C lands on it, the lognormal family does not hold on the full")
    print("     data -> decision 3.17 reopens, and count moves to the inverse-CDF")
    print("     that radial and size already use.")
    print("   * whichever wins, `fit` must be re-run: gamma was calibrated against a")
    print("     colony density that is about to change, so the touching rate has to")
    print("     be re-calibrated with it.")
    print()

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plate_index", "colonies"])
        w.writerows(enumerate(ns.astype(int).tolist()))
    print(f"  per-plate counts ({len(ns)}) -> {out_csv}")
    print("=" * 74)


if __name__ == "__main__":
    main()
