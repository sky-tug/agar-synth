#!/usr/bin/env python3
"""
Sanity test of the generation pipeline -- layout.py + mask.py.

WHY IT EXISTS (decision 3.37):
  `metrics.py` has 33 tests because its own AP computation was not trusted.
  `layout.py` contains at least as much treacherous mathematics:
    - empirical inverse-CDF sampling (quantile grid + np.interp)
    - Strauss gamma binary search
    - the empirical count histogram (decision 3.84; it was a lognormal fit until
      that decision, and the fit inflated the MEAN by 42% while leaving the
      median exactly right)
  And these files will produce THE DATA OF THE PAPER.

  The `validate` subcommand is a COMPARISON REPORT, not a unit test: it compares
  aggregate statistics. A wrong quantile interpolation can keep the median right
  while breaking the tail, and `validate` may not catch it.

Usage:
    python src/generate/test_generate.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mask as M            # noqa: E402
import layout as L          # noqa: E402
import tiles as T           # noqa: E402
import species_check as S   # noqa: E402
import inpaint as I         # noqa: E402

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'OK ' if condition else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def sample_params(gamma=1.0, n_min=20, n_max=20, diameter=0.05):
    """A parameter set with a fixed recipe, a single class and no variability.
    Count and size are fixed so that 'what will be generated' is known in the tests."""
    q = list(L.QUANTILES)
    return {
        "level": 100,
        "plate": {"r_ratio": L.PLATE_R_RATIO, "center": [0.5, 0.5]},
        # radius: p0=0, p100=0.9  -> spread over the disc but not pressed against the edge
        "radial": {"q": q, "values": [0.9 * (v / 100.0) for v in q]},
        # Decision 3.84: the count is an empirical histogram, not a lognormal.
        # values/counts with equal frequencies over [n_min, n_max]; when
        # n_min == n_max (the usual case here) the count is exactly fixed, which
        # is what these tests want -- "what will be generated" must be known.
        "count": {"family": "empirical",
                  "values": list(range(n_min, n_max + 1)),
                  "counts": [1] * (n_max - n_min + 1),
                  "min": n_min, "max": n_max},
        "size": {"S.aureus": {"q": q, "values": [diameter] * len(q), "n": 100}},
        "composition": [{"S.aureus": 1.0}],
        "touching": {"max_overlap": L.MAX_OVERLAP, "gamma": gamma},
    }


# ===========================================================================

def test_inverse_cdf():
    print("\n1) Empirical inverse-CDF sampling (decision 3.11)")
    rng = np.random.default_rng(0)
    # uniform U(0,1): the quantile grid = the value itself
    q = [0, 10, 25, 50, 75, 90, 100]
    values = [v / 100.0 for v in q]
    x = L._inverse_cdf(rng, q, values, 200_000)
    check("median of U(0,1) is 0.50", abs(np.median(x) - 0.5) < 0.01, f"{np.median(x):.4f}")
    check("p90 of U(0,1) = 0.90", abs(np.percentile(x, 90) - 0.9) < 0.01,
          f"{np.percentile(x, 90):.4f}")
    check("does not go outside the range", x.min() >= 0 and x.max() <= 1,
          f"[{x.min():.4f}, {x.max():.4f}]")

    # non-linear grid: a kink at p50
    q2 = [0, 50, 100]
    d2 = [0.0, 0.1, 1.0]          # lower half squeezed into 0-0.1, upper half into 0.1-1
    y = L._inverse_cdf(rng, q2, d2, 200_000)
    check("median 0.10 on a kinked grid", abs(np.median(y) - 0.1) < 0.005,
          f"{np.median(y):.4f}")
    check("p25 = 0.05 on a kinked grid", abs(np.percentile(y, 25) - 0.05) < 0.005,
          f"{np.percentile(y, 25):.4f}")


def test_overlap_depth():
    print("\n2) Overlap depth")
    check("tangential -> 0.0", L._overlap_depth(2.0, 1.0, 1.0) == 0.0)
    check("disjoint -> 0.0", L._overlap_depth(5.0, 1.0, 1.0) == 0.0)
    check("coincident centers -> 1.0", L._overlap_depth(0.0, 1.0, 1.0) == 1.0)
    check("half -> 0.5", abs(L._overlap_depth(1.0, 1.0, 1.0) - 0.5) < 1e-12)


def test_touching_rate():
    print("\n3) Touching rate -- a hand-computed case")
    # 4 colonies: two touching, two disjoint  -> 50%
    col = [(0.10, 0.10, 0.06),      # center distance 0.05 < 0.03+0.03 = 0.06 -> touching
           (0.15, 0.10, 0.06),
           (0.50, 0.50, 0.02),
           (0.90, 0.90, 0.02)]
    o = L._touching_rate([col])
    check("2/4 touching -> 50%", abs(o - 50.0) < 1e-9, f"{o:.1f}%")
    check("single colony -> 0%", L._touching_rate([[(0.5, 0.5, 0.02)]]) == 0.0)


def test_count_limits():
    print("\n4) Colony count: the empirical histogram is reproduced (decisions 3.17, 3.84)")
    # The shape that broke the lognormal fit: a dense low mode plus a thin, far
    # upper tail. Median 13, mean 15.01 -- deliberately far apart, because 3.84
    # was caught precisely by the median staying right while the mean ran away.
    values = [1, 2, 5, 8, 13, 20, 35, 60, 120, 225]
    counts = [40, 90, 260, 400, 520, 380, 150, 45, 8, 2]
    exp_mean = sum(v * c for v, c in zip(values, counts)) / sum(counts)
    params = sample_params()
    params["count"] = {"family": "empirical", "values": values, "counts": counts,
                       "min": min(values), "max": max(values)}
    rng = np.random.default_rng(1)
    ns = [len(L.recipe(rng, params)[0]) for _ in range(8000)]

    check("no plate is below the min", min(ns) >= min(values), f"min={min(ns)}")
    check("no plate is above the max", max(ns) <= max(values), f"max={max(ns)}")
    # The point of the histogram: ONLY observed values may come out. An
    # inverse-CDF interpolation would sprinkle counts like 97 between 60 and 120,
    # into a range where the real data has almost no mass (decision 3.84).
    check("only values present in the histogram are produced",
          set(ns) <= set(values), f"{len(set(ns))} distinct")
    check("the MEAN is reproduced -- the row that was missing before 3.84",
          abs(np.mean(ns) - exp_mean) < 0.5, f"{np.mean(ns):.2f} vs {exp_mean:.2f}")
    check("the median is reproduced too", np.median(ns) == 13, f"median={np.median(ns):.0f}")


def test_legacy_count_rejected():
    print("\n4b) A pre-3.84 parameter file must STOP the program (decision 3.84)")
    # Until now nothing tested this guard. A lognormal file that slipped through
    # would put ~40% too many colonies on every plate and bias the substitution
    # curve in favour of synthetic data -- silently.
    params = sample_params()
    params["count"] = {"family": "lognormal", "ln_loc": math.log(20), "ln_std": 1.0,
                       "min": 12, "max": 125}
    rng = np.random.default_rng(0)
    stopped = False
    try:
        L.recipe(rng, params)
    except SystemExit:
        stopped = True
    check("a lognormal parameter file is refused, not silently used", stopped)

    # A file with no count family at all must stop as well.
    params2 = sample_params()
    params2["count"] = {"min": 12, "max": 125}
    stopped2 = False
    try:
        L.recipe(np.random.default_rng(0), params2)
    except SystemExit:
        stopped2 = True
    check("a file with no count family is refused", stopped2)


def test_gamma_extremes():
    print("\n5) Extreme values of the Strauss gamma (decision 3.19)")
    # gamma = 0: a touching position must NEVER be accepted
    params0 = sample_params(gamma=0.0, n_min=25, n_max=25, diameter=0.05)
    rng = np.random.default_rng(2)
    pl0 = [[(k["xc"], k["yc"], k["diameter"]) for k in L.one_plate(rng, params0, gamma=0.0)[0]]
           for _ in range(40)]
    o0 = L._touching_rate(pl0)

    params1 = sample_params(gamma=1.0, n_min=25, n_max=25, diameter=0.05)
    rng = np.random.default_rng(2)
    pl1 = [[(k["xc"], k["yc"], k["diameter"]) for k in L.one_plate(rng, params1, gamma=1.0)[0]]
           for _ in range(40)]
    o1 = L._touching_rate(pl1)

    check("gamma=0 -> touching rate very low", o0 < 5.0, f"{o0:.1f}%")
    check("gamma=1 -> clearly higher than gamma=0", o1 > o0 + 10,
          f"gamma1={o1:.1f}%  gamma0={o0:.1f}%")


def test_gamma_monotone():
    print("\n6) Does the touching rate rise as gamma grows (the binary search assumes this)")
    params = sample_params(n_min=25, n_max=25, diameter=0.05)
    rates = []
    for g in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rng = np.random.default_rng(3)
        pl = [[(k["xc"], k["yc"], k["diameter"]) for k in L.one_plate(rng, params, gamma=g)[0]]
              for _ in range(40)]
        rates.append(L._touching_rate(pl))
    increasing = all(rates[i] <= rates[i + 1] + 1.0 for i in range(len(rates) - 1))
    check("touching rate increases monotonically with gamma", increasing,
          " -> ".join(f"{o:.1f}" for o in rates))


def test_hard_limit():
    print("\n7) Is the hard overlap limit never exceeded (decision 3.12)")
    params = sample_params(gamma=1.0, n_min=45, n_max=45, diameter=0.06)   # deliberately crowded
    rng = np.random.default_rng(4)
    deepest = 0.0
    for _ in range(30):
        col, forced, dropped = L.one_plate(rng, params)
        P = np.array([[k["xc"], k["yc"]] for k in col])
        S = np.array([k["diameter"] / 2 for k in col])
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        np.fill_diagonal(D, 1e9)
        T = S[:, None] + S[None, :]
        ii, jj = np.triu_indices(len(col), 1)
        d = 1 - D[ii, jj] / T[ii, jj]
        if len(d):
            deepest = max(deepest, d.max())
    check(f"deepest overlap <= {L.MAX_OVERLAP} (+tolerance)",
          deepest <= L.MAX_OVERLAP + 0.02, f"{deepest:.3f}")


def test_bounds():
    print("\n8) Are the colonies inside the plate and inside the image (decision 3.10)")
    params = sample_params(n_min=30, n_max=30, diameter=0.05)
    rng = np.random.default_rng(5)
    R = params["plate"]["r_ratio"]
    outside_plate = outside_square = 0
    for _ in range(50):
        for k in L.one_plate(rng, params)[0]:
            if math.hypot(k["xc"] - 0.5, k["yc"] - 0.5) + k["diameter"] / 2 > R + 1e-9:
                outside_plate += 1
            if not (0 <= k["xc"] - k["w"] / 2 and k["xc"] + k["w"] / 2 <= 1
                    and 0 <= k["yc"] - k["h"] / 2 and k["yc"] + k["h"] / 2 <= 1):
                outside_square += 1
    check("no colony spills outside the plate disc", outside_plate == 0, f"{outside_plate}")
    check("no label spills outside the image", outside_square == 0, f"{outside_square}")


def test_no_silent_drop():
    print("\n9) Is a colony silently dropped (decision 3.22)")
    params = sample_params(gamma=0.05, n_min=35, n_max=35, diameter=0.07)  # deliberately hard
    rng = np.random.default_rng(6)
    missing = total_dropped = 0
    for _ in range(30):
        col, forced, dropped = L.one_plate(rng, params)
        total_dropped += dropped
        if len(col) + dropped != 35:
            missing += 1
    check("generated + dropped = requested (no silent loss)", missing == 0, f"{missing} plates")
    check("drops are reported (the counter exists even when it is zero)",
          isinstance(total_dropped, int), f"dropped={total_dropped}")


def test_deterministic():
    print("\n10) Same seed -> same output")
    params = sample_params(n_min=25, n_max=25)
    a = L.one_plate(np.random.default_rng(7), params)[0]
    b = L.one_plate(np.random.default_rng(7), params)[0]
    c = L.one_plate(np.random.default_rng(8), params)[0]
    check("seed 7 twice -> identical", json.dumps(a, sort_keys=True) ==
          json.dumps(b, sort_keys=True))
    check("seed 8 -> different", json.dumps(a, sort_keys=True) !=
          json.dumps(c, sort_keys=True))


def test_naive_ablation():
    print("\n11) Does the A3 naive mode differ from the main line (decision 3.18)")
    params = sample_params(n_min=30, n_max=30, diameter=0.05)
    params["size"]["E.coli"] = {"q": list(L.QUANTILES), "values": [0.15] * len(L.QUANTILES), "n": 50}
    R = params["plate"]["r_ratio"]

    rng = np.random.default_rng(9)
    main_line = [k for _ in range(30) for k in L.one_plate(rng, params)[0]]
    rng = np.random.default_rng(9)
    naive = [k for _ in range(30) for k in L.one_plate(rng, params, naive=True)[0]]

    r_main = np.array([math.hypot(k["xc"] - .5, k["yc"] - .5) / R for k in main_line])
    r_naive = np.array([math.hypot(k["xc"] - .5, k["yc"] - .5) / R for k in naive])
    check("the main line stays on the plate disc", r_main.max() <= 1.0, f"max {r_main.max():.3f}")
    check("the naive mode spills OUTSIDE the plate", (r_naive > 1.0).mean() > 0.10,
          f"{100*(r_naive > 1.0).mean():.1f}% outside")

    # in naive mode size is not conditioned on class -> S.aureus can get an E.coli diameter
    diameters_s = {round(k["diameter"], 4) for k in naive if k["cls"] == "S.aureus"}
    check("in naive mode size is NOT conditioned on class", len(diameters_s) > 1,
          f"{len(diameters_s)} different diameters on S.aureus")


def test_level_lock():
    print("\n12) Does a level mismatch stop the program (decision 3.7)")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        params = sample_params(); params["level"] = 25
        (td / "p.json").write_text(json.dumps(params))
        (td / "classes.txt").write_text("S.aureus\n")

        class A:
            level = 10                       # parameters say 25, 10 was requested
            params = str(td / "p.json")
            classes = str(td / "classes.txt")
            count, seed, out, naive = 1, 0, str(td / "out"), False
        try:
            L.sample(A())
            check("stops on a level mismatch", False, "did not stop!")
        except SystemExit as e:
            check("stops on a level mismatch", "3.7" in str(e), str(e)[:60])


def test_mask_area():
    print("\n13) Is the mask area equal to the analytic disc area (decision 3.24)")
    check("SYNTH_MARGIN == 1.00 (zero label error guarantee)",
          M.SYNTH_MARGIN == 1.00, f"{M.SYNTH_MARGIN}")

    W = H = 2048
    diameter = 0.08
    plan = {"colonies": [{"xc": 0.5, "yc": 0.5, "diameter": diameter}]}
    synth, erase, combined = M.build_masks(plan, [], W, H)
    expected = math.pi * (diameter / 2 * W) ** 2
    observed = (synth > 0).sum()
    check("single colony: white area ~ pi r^2 (1% tolerance)",
          abs(observed - expected) / expected < 0.01,
          f"observed {observed}  expected {expected:.0f}")
    check("erase mask is empty (no colony on the background)", (erase > 0).sum() == 0)
    check("combined = synthetic", int((combined > 0).sum()) == int(observed))


def test_mask_plate_clipping():
    print("\n14) Is the outside of the plate disc kept closed to diffusion (decision 3.10)")
    W = H = 1024
    # plate radius 0.465 -> a colony 0.49 away from the center is entirely outside
    plan = {"colonies": [{"xc": 0.99, "yc": 0.5, "diameter": 0.01}]}
    synth, _, _ = M.build_masks(plan, [], W, H)
    check("a colony outside the plate does not enter the mask", (synth > 0).sum() == 0,
          f"{(synth > 0).sum()} pixels")

    plan2 = {"colonies": [{"xc": 0.5, "yc": 0.5, "diameter": 0.02}]}
    synth2, _, _ = M.build_masks(plan2, [], W, H)
    check("a colony inside the plate is in the mask", (synth2 > 0).sum() > 0)


def test_mask_erase():
    print("\n15) Is the erase region LARGER than the real colony (decision 3.25)")
    W = H = 2048
    diameter = 0.05
    plan = {"colonies": []}
    background = [(0.5, 0.5, diameter, diameter)]     # (xc, yc, w, h)
    synth, erase, combined = M.build_masks(plan, background, W, H)
    real_area = math.pi * (diameter / 2 * W) ** 2
    ratio = (erase > 0).sum() / real_area
    check(f"erase area ~ ERASE_MARGIN^2 = {M.ERASE_MARGIN**2:.2f} times",
          abs(ratio - M.ERASE_MARGIN ** 2) / M.ERASE_MARGIN ** 2 < 0.05,
          f"measured {ratio:.2f}x")
    check("the erase area is larger than the real colony", ratio > 1.0)


def test_tiling():
    """Decision 3.51 / 3.52 -- the tiling layer that makes native-resolution
    inpainting possible, and the invariant that no colony is ever split."""
    print("\n16) Tiling -- no colony may be split across tiles")
    W = H = 2048
    rng = np.random.default_rng(11)
    par = sample_params(n_min=40, n_max=40, diameter=0.05)

    split = 0
    uncovered = 0
    counts = []
    for _ in range(25):
        colonies = L.one_plate(rng, par)[0]
        tiles = T.place_tiles(colonies, W, H, tile=512)
        counts.append(len(tiles))
        boxes = T.colony_boxes(colonies, W, H)
        # every colony must be wholly inside EXACTLY ONE tile
        for i, b in enumerate(boxes):
            owners = [t for t in tiles if t.contains_box(*b)]
            if not owners:
                uncovered += 1
            claimed = [t for t in tiles if i in t.colony_idx]
            if len(claimed) != 1:
                split += 1
    check("every colony is wholly inside a tile", uncovered == 0, f"{uncovered} uncovered")
    check("every colony is claimed by exactly one tile", split == 0, f"{split} split")
    check("tiles stay inside the image",
          all(t.x0 >= 0 and t.y0 >= 0 and t.x1 <= W and t.y1 <= H
              for t in tiles), "")
    check("tiles/plate is in the measured band (12-22 at 512)",
          12 <= np.mean(counts) <= 22, f"mean {np.mean(counts):.1f}")

    # a colony bigger than the tile must ERROR, not be dropped (decision 3.22)
    big = [{"xc": 0.5, "yc": 0.5, "diameter": 0.40, "cls": "E.coli", "w": 0.4, "h": 0.4}]
    try:
        T.place_tiles(big, W, H, tile=512)
        check("oversized colony raises instead of being dropped", False, "no error")
    except ValueError as e:
        check("oversized colony raises instead of being dropped",
              "tile" in str(e).lower(), str(e)[:50])

    print("\n17) Compositing -- nothing outside the mask may change")
    base = rng.integers(0, 255, (W, H, 3), dtype=np.uint8)
    before = base.copy()
    colonies = L.one_plate(rng, par)[0]
    plan = {"colonies": colonies}
    syn, erase, both = M.build_masks(plan, [], W, H)
    tiles = T.place_tiles(colonies, W, H, tile=512)
    gen = np.zeros((512, 512, 3), dtype=np.uint8)      # deliberately black
    for t in tiles:
        T.composite(base, gen, both, t, feather=0)
    disagree_outside = int((base[both == 0] != before[both == 0]).sum())
    changed_inside = int((base[both > 0] != before[both > 0]).sum())
    check("pixels OUTSIDE the mask are byte-identical", disagree_outside == 0,
          f"{disagree_outside} changed")
    check("pixels INSIDE the mask did change", changed_inside > 0,
          f"{changed_inside} changed")

    w = T.blend_weights(64, 16)
    check("blend weights are 1.0 in the interior", abs(w[32, 32] - 1.0) < 1e-6)
    check("blend weights fall to ~0 at the edge", w[0, 0] < 1e-6, f"{w[0,0]:.2e}")
    check("blend weights are symmetric",
          np.allclose(w, w[::-1, :]) and np.allclose(w, w[:, ::-1]))


def test_species_check():
    """Decision 3.68 -- the class-fidelity measurement. These checks matter
    because the statistic it reports decides whether the multi-class arms of the
    paper mean anything, and a measurement that is wrong in a *plausible* way is
    the most expensive kind of bug in this repository."""
    print("\n18) species_check -- class fidelity measurement")

    import cv2

    def plate(spec, W=900, H=900, bg=90):
        """Build a plate with known colonies. spec = [(cls, cx, cy, r, bgr, noise)]"""
        im = np.full((H, W, 3), bg, np.uint8)
        lines = []
        for c, cx, cy, r, bgr, noise in spec:
            cv2.circle(im, (cx, cy), r, bgr, -1)
            if noise:
                sub = im[cy - r:cy + r, cx - r:cx + r].astype(np.int16)
                rr = np.random.default_rng(cx * 7 + cy).integers(-noise, noise + 1,
                                                                 sub.shape)
                im[cy - r:cy + r, cx - r:cx + r] = np.clip(sub + rr, 0, 255).astype(np.uint8)
            lines.append(f"{c} {cx/W:.6f} {cy/H:.6f} {2*r/W:.6f} {2*r/H:.6f}")
        return im, "\n".join(lines)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "images").mkdir(); (td / "labels").mkdir()
        # class 0 = yellow (high R-B), class 1 = grey (R-B ~ 0)
        spec = []
        for i in range(6):
            for j in range(6):
                cx, cy = 80 + i * 145, 80 + j * 145
                if j < 3:
                    spec.append((0, cx, cy, 40, (40, 170, 200), 0))   # B,G,R -> R-B=160
                else:
                    spec.append((1, cx, cy, 40, (150, 150, 150), 0))  # R-B = 0
        im, lab = plate(spec)   # 36 colonies, 18 per class (>= MIN_N)
        cv2.imwrite(str(td / "images" / "p0.jpg"), im,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
        (td / "labels" / "p0.txt").write_text(lab)
        counters = {"missing_image": 0, "missing_label": 0,
                    "clipped": 0, "too_small": 0}
        rows = S.plate_stats(td / "images" / "p0.jpg", td / "labels" / "p0.txt",
                             ["yellowsp", "greysp"], counters)
        check("every labelled colony is measured", len(rows) == 36,
              f"{len(rows)}/36")
        med = S.medians(rows)
        yel = med.get("yellowsp", {}).get("yellowness", 0)
        gre = med.get("greysp", {}).get("yellowness", 0)
        check("yellowness recovers the planted colour difference",
              yel > 140 and abs(gre) < 12, f"yellow={yel:.0f} grey={gre:.0f}")
        check("contrast is positive for a colony brighter than the medium",
              med["greysp"]["contrast"] > 40,
              f"{med['greysp']['contrast']:.1f}")

    # A statistic that is identical on both sides must give ordering 1.0, and a
    # reversed one must give -1.0. Without this, "the class channel works" could
    # be printed by an implementation that always returns +1.
    a = {"x": {"yellowness": 1, "contrast": 1, "texture": 1},
         "y": {"yellowness": 2, "contrast": 2, "texture": 2},
         "z": {"yellowness": 3, "contrast": 3, "texture": 3}}
    rev = {"x": {"yellowness": 3, "contrast": 3, "texture": 3},
           "y": {"yellowness": 2, "contrast": 2, "texture": 2},
           "z": {"yellowness": 1, "contrast": 1, "texture": 1}}
    check("ordering = +1 when the ranking is identical",
          abs(S.ordering(a, a)["yellowness"] - 1.0) < 1e-9)
    check("ordering = -1 when the ranking is reversed",
          abs(S.ordering(a, rev)["yellowness"] + 1.0) < 1e-9)

    # Separability must COLLAPSE when the classes are made identical. This is the
    # verdict that caught the texture artefact; if it could not distinguish
    # "separated" from "collapsed" it would have reported a false pass.
    rng = np.random.default_rng(3)
    def cloud(cls, centre, n=40):
        return [{"cls": cls, "plate": f"p{i % 5}",
                 "yellowness": centre[0] + rng.normal(0, 1),
                 "contrast": centre[1] + rng.normal(0, 1),
                 "texture": centre[2] + rng.normal(0, 1)} for i in range(n)]
    far = cloud("a", (0, 0, 0)) + cloud("b", (20, 20, 20))
    near = cloud("a", (0, 0, 0)) + cloud("b", (0, 0, 0))
    sf = S.separability(far, ["a", "b"])
    sn = S.separability(near, ["a", "b"])
    check("separability is large for well-separated classes", sf > 3, f"{sf:.2f}")
    check("separability collapses for identical classes", sn < 1.0, f"{sn:.2f}")
    check("separability ranks separated above collapsed", sf > 5 * sn,
          f"{sf:.2f} vs {sn:.2f}")

    # A class below MIN_N must be ABSENT from the medians, not silently averaged
    # in with a meaningless value (design principle 2).
    thin = cloud("a", (0, 0, 0), n=S.MIN_N - 1)
    check("a class below MIN_N is dropped, not reported",
          "a" not in S.medians(thin))


def test_canvas_scaling():
    """Decision 3.70 -- the canvas may be rescaled per tile so that a colony
    occupies a constant number of canvas pixels. The measurement that motivated
    it: real colonies get smoother as they grow (rank correlation -0.57),
    generated ones do not (-0.18)."""
    print("\n19) Canvas scaling -- texture must follow colony size")

    def col(d):                       # d = diameter in pixels, W = 2048
        return {"diameter": d / 2048, "cls": "E.coli"}

    W = 2048
    small = [col(29)] * 4             # S.aureus
    large = [col(155)] * 4            # P.aeruginosa
    idx = [0, 1, 2, 3]

    c_small = I.canvas_for(512, 1.0, I.TARGET_DIAMETER, small, idx, W)
    c_large = I.canvas_for(512, 1.0, I.TARGET_DIAMETER, large, idx, W)
    check("a small colony gets a LARGER canvas than a big one",
          c_small > c_large, f"{c_small} vs {c_large}")
    check("the canvas stays within the clamp",
          I.MIN_CANVAS <= c_small <= I.MAX_CANVAS
          and I.MIN_CANVAS <= c_large <= I.MAX_CANVAS,
          f"{I.MIN_CANVAS}-{I.MAX_CANVAS}")
    check("the canvas is a multiple of 8 (the VAE stride)",
          c_small % 8 == 0 and c_large % 8 == 0)

    # Off by default: the original behaviour must be bit-identical, otherwise
    # every number measured before this decision silently stops comparing.
    check("target_diameter=0 and gen_scale=1 leave the canvas untouched",
          I.canvas_for(512, 1.0, 0, small, idx, W) == 512)
    check("a flat gen_scale scales the canvas",
          I.canvas_for(512, 0.5, 0, small, idx, W) == 256)

    # An empty tile has no colony to measure; it must not crash or divide by 0.
    check("a tile with no colonies falls back to gen_scale",
          I.canvas_for(512, 1.0, I.TARGET_DIAMETER, [], [], W) == 512)


def main():
    print("=" * 62)
    print("GENERATION PIPELINE SANITY TEST  (layout.py + mask.py)")
    print("=" * 62)
    for f in (test_inverse_cdf, test_overlap_depth, test_touching_rate, test_count_limits,
              test_legacy_count_rejected,
              test_gamma_extremes, test_gamma_monotone, test_hard_limit, test_bounds,
              test_no_silent_drop, test_deterministic, test_naive_ablation,
              test_level_lock, test_mask_area, test_mask_plate_clipping,
              test_mask_erase, test_tiling, test_species_check,
              test_canvas_scaling):
        f()
    print("\n" + "=" * 62)
    print(f"PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
    if FAILED:
        for k in FAILED:
            print(f"  ! {k}")
        print("=" * 62)
        sys.exit(1)
    print("The generation pipeline is reliable.")
    print("=" * 62)


if __name__ == "__main__":
    main()
