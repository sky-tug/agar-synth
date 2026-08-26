#!/usr/bin/env python3
"""
07_plate_edge_check.py -- where is the plate edge, REALLY?

WHY THIS EXISTS
---------------
Decision 3.10 fixed the plate circle at  R = 0.465 * width , centred on the image
centre. layout.py samples synthetic colony positions inside exactly that circle.

But analyze_manifest.py normalises the radius by  0.5 * width  instead, so the
same word ("radius") means two different things in two files (audit finding 9).
On the demo package the largest observed radius was 0.876, so the check never
fired and the discrepancy stayed invisible.

On the FULL data it fired: 16 colony centres came out at r > 1 in the 0.5*W unit,
and analyze_manifest printed "the plate radius may be wrong".

Two explanations, and they lead to opposite actions:
  (a) the 0.465 constant is wrong  -> layout.py places synthetic colonies in the
      wrong circle, and every generated plate is subtly off
  (b) the constant is right and those boxes are AGAR annotation noise
      -> change nothing, report it as a limitation

This script decides between them with data. It does NOT assume a plate radius;
it asks which radius actually contains the colonies.

WHAT IT PRINTS
--------------
  * the radius distribution of all colony CENTRES (in the 0.5*W unit)
  * the radius that contains 99 / 99.9 / 100 % of the centres
  * the same for the OUTER corner of the box (a colony may touch the rim)
  * how the two conventions (0.465 vs 0.5) compare against that
  * the outlier images, written to a CSV so they can be looked at by eye

Read-only. Writes one CSV, touches nothing else.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_ORDER = ["S.aureus", "B.subtilis", "P.aeruginosa", "E.coli", "C.albicans"]

# Decision 3.10 -- the constant layout.py generates with
R_LAYOUT = 0.465
# The unit analyze_manifest.py reports in (half the image width)
R_MANIFEST = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data/processed")
    ap.add_argument("--out", default=None,
                    help="outlier CSV (default: <data>/reports/plate_edge_outliers.csv)")
    args = ap.parse_args()

    data = Path(args.data).expanduser().resolve()
    lbl_dir, img_dir = data / "labels", data / "images"
    if not lbl_dir.is_dir():
        sys.exit(f"ERROR: not found: {lbl_dir}")

    out_csv = Path(args.out) if args.out else data / "reports" / "plate_edge_outliers.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    r_centre: list[float] = []      # centre distance, in the 0.5*W unit
    r_outer: list[float] = []       # farthest box corner, same unit
    rows: list[tuple] = []          # outlier records
    n_img = n_box = 0
    skipped = 0

    for lp in sorted(lbl_dir.glob("*.txt")):
        stem = lp.stem
        ip = img_dir / f"{stem}.jpg"
        try:
            with Image.open(ip) as im:
                W, H = im.size
        except Exception:
            skipped += 1
            continue
        n_img += 1

        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c, xc, yc, bw, bh = int(parts[0]), *[float(v) for v in parts[1:]]
            n_box += 1

            # normalised coordinates are already relative to W/H, so a square
            # image makes the unit isotropic; guard the non-square case
            dx = (xc - 0.5) * W
            dy = (yc - 0.5) * H
            unit = W / 2.0                       # half-width == the 0.5*W unit
            rc = math.hypot(dx, dy) / unit

            # farthest corner of the box from the plate centre
            cx = abs(dx) + bw * W / 2.0
            cy = abs(dy) + bh * H / 2.0
            ro = math.hypot(cx, cy) / unit

            r_centre.append(rc)
            r_outer.append(ro)

            if rc > R_LAYOUT / R_MANIFEST:       # outside the layout.py circle
                rows.append((stem, CLASS_ORDER[c] if c < len(CLASS_ORDER) else c,
                             round(rc, 4), round(ro, 4),
                             round(rc * R_MANIFEST / R_LAYOUT, 4),
                             round(bw * W, 1), round(bh * H, 1)))

    rc_arr = np.asarray(r_centre)
    ro_arr = np.asarray(r_outer)

    def q(a, p):
        return float(np.percentile(a, p))

    print("=" * 62)
    print("PLATE EDGE CHECK")
    print("=" * 62)
    print(f"  images / boxes           : {n_img} / {n_box}"
          + (f"   ({skipped} images unreadable)" if skipped else ""))
    print()
    print("  --- colony CENTRE radius, in the 0.5*W unit ---")
    for p in (50, 90, 99, 99.9):
        print(f"      p{p:<6} = {q(rc_arr, p):.4f}")
    print(f"      max    = {rc_arr.max():.4f}")
    print()
    print("  --- OUTER box corner, same unit (a colony may touch the rim) ---")
    for p in (99, 99.9):
        print(f"      p{p:<6} = {q(ro_arr, p):.4f}")
    print(f"      max    = {ro_arr.max():.4f}")
    print()

    # How do the two conventions score?
    lim = R_LAYOUT / R_MANIFEST                  # 0.465 expressed in the 0.5*W unit
    out_c = int((rc_arr > lim).sum())
    out_m = int((rc_arr > 1.0).sum())
    print("  --- how many centres fall OUTSIDE each convention ---")
    print(f"      R = 0.465*W (layout.py, decision 3.10) -> {out_c} boxes"
          f"  ({100*out_c/n_box:.3f} %)")
    print(f"      R = 0.500*W (analyze_manifest unit)    -> {out_m} boxes"
          f"  ({100*out_m/n_box:.3f} %)")
    print()

    # The radius the data itself asks for
    print("  --- the radius the DATA asks for (centres) ---")
    for p in (99, 99.9, 100):
        rr = rc_arr.max() if p == 100 else q(rc_arr, p)
        print(f"      contains {p:>5}% of centres -> R = {rr*R_MANIFEST:.4f} * W")
    print()

    print("  --- READ IT LIKE THIS ---")
    print("   * if the bulk stops well before 0.465*W and only a handful exceed it,")
    print("     the constant is FINE and the outliers are AGAR annotation noise")
    print("     -> change nothing, report it as a limitation")
    print("   * if a few percent sit outside, the plate model is wrong and")
    print("     layout.py generates in the wrong circle -> decision 3.10 reopens")
    print()

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "cls", "r_over_halfW", "outer_r_over_halfW",
                    "r_over_plateR", "box_w_px", "box_h_px"])
        w.writerows(sorted(rows, key=lambda r: -r[2]))

    print(f"  outliers ({len(rows)}) -> {out_csv}")
    print("  LOOK AT THE TOP ROWS BY EYE before deciding.")
    print("=" * 62)


if __name__ == "__main__":
    main()
