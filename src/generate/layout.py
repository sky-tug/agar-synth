#!/usr/bin/env python3
"""
layout.py -- solves the "where, how big, which species" question on synthetic plates.

The first step of Phase 3. NO diffusion. This script only produces coordinates:
because the box coordinates are known BEFORE generation, the label comes into
existence by itself.

Three subcommands:
  fit       extracts distribution parameters from real labels -> layout_<level>.json
  sample    generates synthetic layouts from those parameters -> plan_*.json + labels_*.txt
  validate  compares the generated layout against real data   -> console report

Related decisions (decisions.md):
  3.5   --level is mandatory, there is NO default  (anti-leakage)
  3.7   outputs carry the level in their name
  3.10  plate circle is fixed: center = image center, R = 0.465 * width
  3.11  angle uniform, radius empirical inverse-CDF
  3.12  overlap is allowed, overlap depth is clipped at 0.63
  3.13  mask circular, label box square
  3.14  size is per class
  3.16  one dominant species + a small number of secondary species
  3.17  count is lognormal
  3.18  --naive = ablation A3
  3.19  Strauss interaction parameter gamma, auto-calibrated to the real touching rate
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter
from pathlib import Path

import numpy as np

# --- decision 3.10 ----------------------------------------------------------
PLATE_R_RATIO = 0.465       # R / image_width
# --- decision 3.12 ----------------------------------------------------------
MAX_OVERLAP = 0.63          # 0 = tangential, 1 = fully nested
# --- decision 3.13 ----------------------------------------------------------
ELLIPSE_JITTER = 0.03       # 96.9% of the boxes are exactly square; the rest get this much deviation
# --- sampling ---------------------------------------------------------------
PLACEMENT_TRIES = 400       # position rejection attempts for one colony
CALIB_PLATES    = 250       # number of plates simulated during gamma calibration

QUANTILES = [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]   # quantile grid


# ===========================================================================
# FIT
# ===========================================================================

def _read_labels(stem_paths, label_root: Path, classes):
    """Read YOLO labels. Returns: list of plates, each plate = list of colony dicts."""
    plates = []
    for p in stem_paths:
        lab = label_root / (Path(p).stem + ".txt")
        if not lab.exists():
            print(f"  WARNING: label missing, skipped: {lab}", file=sys.stderr)
            continue
        col = []
        for line in lab.read_text().split("\n"):
            if not line.strip():
                continue
            c, xc, yc, w, h = line.split()
            col.append(dict(cls=classes[int(c)], xc=float(xc), yc=float(yc),
                            w=float(w), h=float(h)))
        if col:
            plates.append(dict(stem=Path(p).stem, colonies=col))
    return plates


def _touching_rate(plates):
    """Percentage of colonies touching at least one neighbour. plates: [[(x,y,diameter),...],...]"""
    touching = total = 0
    for col in plates:
        if len(col) < 2:
            total += len(col); continue
        P = np.array([[k[0], k[1]] for k in col]); S = np.array([k[2] / 2 for k in col])
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        np.fill_diagonal(D, 1e9)
        touching += (D < (S[:, None] + S[None, :])).any(1).sum(); total += len(col)
    return 100.0 * touching / max(total, 1)


def _touching_ci(plates, seed=7, B=2000):
    """Plate-level bootstrap 90% confidence interval of the touching rate.
    The measure is extremely correlated within a plate -- the effective sample size
    is the number of PLATES, not COLONIES. With 10 plates the interval comes out
    very wide; this is not a bug, it is information."""
    rng = np.random.default_rng(seed)
    n = len(plates)
    v = [_touching_rate([plates[i] for i in rng.integers(0, n, n)]) for _ in range(B)]
    return float(np.percentile(v, 5)), float(np.percentile(v, 95))


def _calibrate_gamma(params, target, seed=12345):
    """Fit gamma to the real touching rate via binary search (decision 3.19).

    The recipes (count/class/diameter) are produced once and reused VERBATIM in
    every gamma trial. That way the only thing that changes when gamma changes is
    the placement; otherwise the random stream shifts and the search settles into
    noise."""
    r_rng = np.random.default_rng(seed)
    recipes = [recipe(r_rng, params) for _ in range(CALIB_PLATES)]

    def try_gamma(g):
        rng = np.random.default_rng(seed + 1)
        pl = [[(k["xc"], k["yc"], k["diameter"])
               for k in one_plate(rng, params, gamma=g, prepared_recipe=t)[0]] for t in recipes]
        return _touching_rate(pl)

    lo, hi = 0.0, 1.0
    rate_hi = try_gamma(hi)
    if rate_hi <= target:                  # below the target even at gamma=1
        return 1.0, rate_hi, False
    for _ in range(9):
        mid = (lo + hi) / 2
        if try_gamma(mid) < target: lo = mid
        else:                       hi = mid
    g = (lo + hi) / 2
    return g, try_gamma(g), True


def fit(args):
    classes = Path(args.classes).read_text().split()
    file_list = [l for l in Path(args.list).read_text().split("\n") if l.strip()]
    plates = _read_labels(file_list, Path(args.labels), classes)
    if not plates:
        sys.exit("ERROR: no labels could be read.")

    R = PLATE_R_RATIO                      # in normalized units (image width = 1)
    cx = cy = 0.5

    # --- radial distribution (decision 3.11) -------------------------------
    r_norm, size_by_class, composition, counts = [], {}, [], []
    for pl in plates:
        counts.append(len(pl["colonies"]))
        cnt = Counter(k["cls"] for k in pl["colonies"])
        composition.append({s: n / len(pl["colonies"]) for s, n in cnt.items()})
        for k in pl["colonies"]:
            r_norm.append(math.hypot(k["xc"] - cx, k["yc"] - cy) / R)
            size_by_class.setdefault(k["cls"], []).append((k["w"] + k["h"]) / 2)
    r_norm = np.clip(np.array(r_norm), 0, 1.0)

    # --- count (decision 3.17) ---------------------------------------------
    ns = np.array(counts, dtype=float)
    ln = np.log(ns)

    params = {
        "level": args.level,
        "source": {
            "list": str(args.list),
            "n_plates": len(plates),
            "n_colonies": int(len(r_norm)),
            "WARNING": ("Number of plates < 50. These parameters are enough to choose a "
                        "distribution FAMILY, not enough for estimation.") if len(plates) < 50 else None,
        },
        "plate": {"r_ratio": PLATE_R_RATIO, "center": [cx, cy]},
        "radial": {"q": QUANTILES, "values": [float(np.percentile(r_norm, q)) for q in QUANTILES]},
        # lognormal median = exp(mu). The MEDIAN of ln(n) is used for mu:
        # with few plates the mean is affected by a single crowded plate (n=125).
        "count": {"family": "lognormal", "ln_loc": float(np.median(ln)),
                  "ln_std": float(ln.std(ddof=1)) if len(ln) > 1 else 0.3,
                  "min": int(ns.min()), "max": int(ns.max())},
        "size": {s: {"q": QUANTILES, "values": [float(np.percentile(v, q)) for q in QUANTILES],
                     "n": len(v)}
                 for s, v in sorted(size_by_class.items())},
        "composition": composition,
        "touching": {"max_overlap": MAX_OVERLAP, "gamma": 1.0},
        "unit": "fraction of image width (YOLO normalized)",
    }

    # --- decision 3.19: calibrate gamma to the real touching rate -----------
    real_pl = [[(k["xc"], k["yc"], (k["w"] + k["h"]) / 2) for k in pl["colonies"]]
               for pl in plates]
    target = _touching_rate(real_pl)
    ci = _touching_ci(real_pl)
    g, achieved, ok = _calibrate_gamma(params, target)
    params["touching"].update(gamma=round(g, 4), target_touch_rate=round(target, 2),
                              target_ci90=[round(ci[0], 2), round(ci[1], 2)],
                              achieved_touch_rate=round(achieved, 2), calibration_ok=ok,
                              provisional=len(plates) < 50)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(params, indent=2, ensure_ascii=False))

    print(f"[fit] level={args.level}  {len(plates)} plates / {len(r_norm)} colonies")
    print(f"      count: median {np.median(ns):.0f}  range {ns.min():.0f}-{ns.max():.0f}")
    print(f"      radial median r/R = {np.median(r_norm):.3f}")
    for s, v in sorted(size_by_class.items()):
        print(f"      {s:<14} n={len(v):5d}  median {np.median(v)*2048:6.1f} px (at 2048)")
    d = params["touching"]
    print(f"      touching: real {d['target_touch_rate']:.1f}% "
          f"(CI90 {d['target_ci90'][0]:.1f}%-{d['target_ci90'][1]:.1f}%) "
          f"-> {d['achieved_touch_rate']:.1f}% with gamma={d['gamma']:.3f}")
    if d["provisional"]:
        print(f"  [!] gamma is PROVISIONAL: with {len(plates)} plates the confidence interval "
              f"of the target is {d['target_ci90'][1]-d['target_ci90'][0]:.0f} points wide. "
              f"The mechanism is right, the number is noise. It will be refitted on the full data.")
    if not d["calibration_ok"]:
        print("  [!] the target touching rate could not be reached even at gamma=1 -- the plate is "
              "SPARSE compared to reality. The count or size distribution should be checked.")
    if params["source"]["WARNING"]:
        print(f"  [!] {params['source']['WARNING']}")
    print(f"      -> {out}")


# ===========================================================================
# SAMPLE
# ===========================================================================

def _inverse_cdf(rng, q, values, size):
    """Empirical inverse-CDF sampling (decision 3.11)."""
    u = rng.random(size) * 100.0
    return np.interp(u, q, values)


def _overlap_depth(d, a, b):
    """Interpenetration depth of two circles: 0 = tangential, 1 = fully nested."""
    t = a + b
    return 0.0 if d >= t else 1.0 - d / t


def recipe(rng, params, naive=False):
    """Produces the 'what' of a plate: colony count, classes, diameters.
    Kept SEPARATE from placement -- during calibration the recipe has to stay fixed
    while gamma changes (common random numbers / variance reduction)."""
    cnt = params["count"]
    n = int(round(math.exp(rng.normal(cnt["ln_loc"], cnt["ln_std"]))))
    n = max(cnt["min"], min(cnt["max"], n))

    if naive:                                   # A3: the class distribution is flattened too
        labels = list(rng.choice(list(params["size"].keys()), size=n))
    else:                                       # decision 3.16
        comp = params["composition"][rng.integers(len(params["composition"]))]
        species = list(comp.keys())
        share = np.array([comp[t] for t in species], dtype=float); share /= share.sum()
        labels = list(rng.choice(species, size=n, p=share))

    diameters = []                              # decision 3.14
    for s_ in labels:
        b = params["size"][rng.choice(list(params["size"].keys()))] if naive else params["size"][s_]
        diameters.append(float(_inverse_cdf(rng, b["q"], b["values"], 1)[0]))
    return labels, np.array(diameters)


def one_plate(rng, params, naive=False, gamma=None, prepared_recipe=None):
    """Produces the layout of a single synthetic plate.

    gamma (decision 3.19): the Strauss interaction parameter. If a candidate position
    TOUCHES another colony it is accepted with probability gamma, and with probability
    1-gamma it is rejected and retried. gamma=1 -> pure random (comes out very crowded),
    gamma=0 -> no touching at all. It is fitted to the real touching rate inside `fit`
    by binary search.
    """
    if gamma is None:
        gamma = params.get("touching", {}).get("gamma", 1.0)
    R, (cx, cy) = params["plate"]["r_ratio"], params["plate"]["center"]

    labels, diameters = prepared_recipe if prepared_recipe is not None else recipe(rng, params, naive)

    # position ---------------------------------------------------------------
    order = np.argsort(-diameters)      # largest to smallest: placement gets easier on a crowded plate
    placed, forced, dropped = [], 0, 0
    for i in order:
        half_d = diameters[i] / 2.0
        fallback_valid = None     # geometrically acceptable but rejected by gamma
        fallback_violating, fallback_score = None, 1e9   # best candidate that exceeds the overlap limit
        for _ in range(PLACEMENT_TRIES):
            if naive:
                # A3: no plate circle, uniform over the whole square
                x, y = rng.random(), rng.random()
            else:
                r = float(_inverse_cdf(rng, params["radial"]["q"], params["radial"]["values"], 1)[0])
                th = rng.random() * 2 * math.pi
                x, y = cx + r * R * math.cos(th), cy + r * R * math.sin(th)
                if math.hypot(x - cx, y - cy) + half_d > R:
                    continue                      # a colony may not spill outside the plate
            if not (half_d <= x <= 1 - half_d and half_d <= y <= 1 - half_d):
                continue                          # the box may not spill outside the image
            if naive:
                placed.append((x, y, diameters[i], labels[i]))
                fallback_valid = fallback_violating = None; break
            worst = 0.0
            for (px, py, pd, _) in placed:
                worst = max(worst, _overlap_depth(math.hypot(x - px, y - py), half_d, pd / 2.0))
                if worst > MAX_OVERLAP:
                    break
            if worst > MAX_OVERLAP:                # hard limit (decision 3.12)
                if worst < fallback_score:
                    fallback_violating, fallback_score = (x, y), worst
                continue
            if worst == 0.0 or rng.random() < gamma:  # Strauss acceptance (decision 3.19)
                placed.append((x, y, diameters[i], labels[i]))
                fallback_valid = fallback_violating = None; break
            if fallback_valid is None:              # gamma rejected it but the position is valid
                fallback_valid = (x, y)
        else:
            # PLACEMENT_TRIES exhausted. Do NOT SILENTLY DROP the colony -- that would
            # distort the count distribution. First take the valid position that gamma
            # rejected, and if there is none either, take the least violating one.
            fb = fallback_valid or fallback_violating
            if fb is not None:
                placed.append((fb[0], fb[1], diameters[i], labels[i]))
                forced += 1
            else:
                dropped += 1

    # 5) box: circular mask, square label (decision 3.13) --------------------
    col = []
    for (x, y, d, s) in placed:
        j = 1.0 + rng.normal(0, ELLIPSE_JITTER)
        w, h = d * j, d / j
        col.append(dict(cls=s, xc=round(x, 6), yc=round(y, 6),
                        w=round(w, 6), h=round(h, 6),
                        mask="circle", diameter=round(d, 6)))
    return col, forced, dropped


def sample(args):
    params = json.loads(Path(args.params).read_text())
    if params["level"] != args.level:
        sys.exit(f"ERROR (decision 3.7): the parameter file has level={params['level']}, "
                 f"--level {args.level} was given. The levels must match.")
    classes = Path(args.classes).read_text().split()
    rng = np.random.default_rng(args.seed)

    out = Path(args.out); (out / "plans").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    prefix = "a3naive" if args.naive else f"s{args.level}"

    total, n_forced, n_dropped = 0, 0, 0
    for i in range(args.count):
        col, f, d = one_plate(rng, params, naive=args.naive)
        total += len(col); n_forced += f; n_dropped += d
        name = f"{prefix}_{args.seed}_{i:05d}"
        (out / "plans" / f"{name}.json").write_text(json.dumps(
            dict(name=name, level=args.level, naive=args.naive, seed=args.seed,
                 plate=params["plate"], colonies=col), indent=1, ensure_ascii=False))
        (out / "labels" / f"{name}.txt").write_text("\n".join(
            f"{classes.index(k['cls'])} {k['xc']:.6f} {k['yc']:.6f} {k['w']:.6f} {k['h']:.6f}"
            for k in col) + "\n")

    print(f"[sample] {args.count} plates / {total} colonies  (avg {total/args.count:.1f})")
    print(f"         mode: {'A3 NAIVE (ablation)' if args.naive else 'main line'}  "
          f"level={args.level}  seed={args.seed}")
    if n_forced:
        print(f"  WARNING: {n_forced} colonies ({100*n_forced/total:.1f}%) could not find a "
              f"comfortable spot in {PLACEMENT_TRIES} tries and were put at a fallback position.")
    if n_dropped:
        print(f"  [!] {n_dropped} colonies ({100*n_dropped/(total+n_dropped):.1f}%) could NOT be placed "
              f"at all. The count distribution shifts downwards -- PLACEMENT_TRIES must be increased.")
    print(f"         -> {out}")


# ===========================================================================
# VALIDATE
# ===========================================================================

def _stats(plates, R, cx, cy):
    """plates: [[(xc,yc,diameter), ...], ...] -> dictionary of measures"""
    rn, touching, total, counts, overlaps = [], 0, 0, [], []
    for col in plates:
        if not col: continue
        counts.append(len(col))
        P = np.array([[k[0], k[1]] for k in col])
        S = np.array([k[2] / 2 for k in col])
        for k in col:
            rn.append(math.hypot(k[0] - cx, k[1] - cy) / R)
        if len(col) > 1:
            D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
            np.fill_diagonal(D, 1e9)
            T = S[:, None] + S[None, :]
            touch = D < T
            touching += touch.any(1).sum(); total += len(col)
            ii, jj = np.where(np.triu(touch, 1))
            overlaps += [1 - D[a, b] / T[a, b] for a, b in zip(ii, jj)]
    rn = np.array(rn); counts = np.array(counts)
    # Clark-Evans
    ce = []
    for col in plates:
        if len(col) < 5: continue
        P = np.array([[k[0], k[1]] for k in col])
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        np.fill_diagonal(D, 1e9)
        ce.append(D.min(1).mean() / (0.5 / math.sqrt(len(col) / (math.pi * R ** 2))))
    return dict(
        n_median=float(np.median(counts)), n_min=int(counts.min()), n_max=int(counts.max()),
        r_median=float(np.median(rn)), r_sq_mean=float(np.mean(rn ** 2)),
        outer_ring=float(100 * (rn >= math.sqrt(0.8)).mean()),
        touch_rate=float(100 * touching / max(total, 1)),
        overlap_median=float(np.median(overlaps)) if overlaps else 0.0,
        overlap_max=float(np.max(overlaps)) if overlaps else 0.0,
        ce_median=float(np.median(ce)) if ce else float("nan"),
    )


def validate(args):
    params = json.loads(Path(args.params).read_text())
    R, (cx, cy) = params["plate"]["r_ratio"], params["plate"]["center"]
    classes = Path(args.classes).read_text().split()

    real = []
    for p in [l for l in Path(args.list).read_text().split("\n") if l.strip()]:
        lab = Path(args.labels) / (Path(p).stem + ".txt")
        if not lab.exists(): continue
        real.append([(float(s.split()[1]), float(s.split()[2]),
                      (float(s.split()[3]) + float(s.split()[4])) / 2)
                     for s in lab.read_text().split("\n") if s.strip()])

    generated = []
    for f in sorted((Path(args.generated) / "plans").glob("*.json")):
        j = json.loads(f.read_text())
        generated.append([(k["xc"], k["yc"], k["diameter"]) for k in j["colonies"]])

    g, u = _stats(real, R, cx, cy), _stats(generated, R, cx, cy)
    label = {"n_median": "colony count (median)", "n_min": "colony count (min)",
             "n_max": "colony count (max)", "r_median": "r/R median",
             "r_sq_mean": "(r/R)^2 mean", "outer_ring": "% of colonies in the outer 20% of the area",
             "touch_rate": "% touching colonies  <-- decision 3.12",
             "overlap_median": "overlap depth median",
             "overlap_max": "overlap depth max", "ce_median": "Clark-Evans median"}
    print(f"\n{'measure':<38} {'REAL':>10} {'GENERATED':>10}   diff")
    print("-" * 74)
    for k in label:
        gv, uv = g[k], u[k]
        f = "" if gv == 0 else f"{100*(uv-gv)/abs(gv):+.0f}%"
        marker = ""
        if k == "touch_rate":
            lo, hi = params["touching"].get("target_ci90", [gv - 3, gv + 3])
            marker = ("  [ok] inside the CI90 of the real data" if lo <= uv <= hi
                      else f"  [!] OUTSIDE CI90 ({lo:.0f}%-{hi:.0f}%)")
        print(f"{label[k]:<38} {gv:>10.2f} {uv:>10.2f}  {f:>6}{marker}")
    print(f"\nreal: {len(real)} plates - generated: {len(generated)} plates")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, level=True):
        if level:
            # decision 3.5: NO DEFAULT. It does not run unless the level is stated.
            p.add_argument("--level", type=int, required=True, choices=[10, 25, 50, 100],
                           help="real data level. MANDATORY (decision 3.5) -- "
                                "no default, anti-leakage.")
        p.add_argument("--classes", default="data/processed/classes.txt")

    p = sub.add_parser("fit", help="extract the distribution from real labels")
    common(p)
    p.add_argument("--list", required=True, help="e.g. data/processed/lists/train_25.txt")
    p.add_argument("--labels", default="data/processed/labels")
    p.add_argument("--out", required=True, help="e.g. src/generate/layout_25.json")
    p.set_defaults(fn=fit)

    p = sub.add_parser("sample", help="generate a synthetic layout")
    common(p)
    p.add_argument("--params", required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--naive", action="store_true",
                   help="ablation A3 (decision 3.18): no plate circle, size from a single pool, "
                        "no touching constraint")
    p.set_defaults(fn=sample)

    p = sub.add_parser("validate", help="compare the generated data against the real data")
    common(p, level=False)
    p.add_argument("--params", required=True)
    p.add_argument("--list", required=True)
    p.add_argument("--labels", default="data/processed/labels")
    p.add_argument("--generated", required=True)
    p.set_defaults(fn=validate)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
