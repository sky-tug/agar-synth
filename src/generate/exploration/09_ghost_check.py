#!/usr/bin/env python3
"""
09_ghost_check.py -- did the generator paint colonies where none are labelled?

WHY THIS EXISTS
---------------
Decision 3.25 erases the real colonies out of a reused background, so that the
only colonies left on a synthetic plate are the ones layout.py planned and
mask.py labelled. Decision 3.60 then splits generation into two passes so the
model is told which holes to FILL and which to leave as plain agar.

If either mechanism leaks, the plate contains a colony-shaped object with NO
BOX. That is a false negative baked into the training data: the detector is
shown a colony and taught that it is background. It is the single worst thing
this pipeline can produce, and the gate for it has been sitting at
"~0, measured on ONE demo plate" since Phase 3.

This script measures it on the real pilot output.

WHAT COUNTS AS A GHOST
----------------------
A connected piece of ERASE-MASK area that

  * no synthetic label box claims  (a planned colony is allowed to be placed
    near an erased real one purely by chance -- on the first plate inspected,
    2 of the 4 largest erase regions were exactly that, at 36% and 29% box
    overlap, and calling them ghosts would have been wrong), and
  * contains a compact BRIGHT object relative to the plate's own agar.

The brightness threshold is per plate and robust: median + 6 * 1.4826 * MAD of
the agar pixels (plate disc, minus every erase region, minus every label box).
Per plate, because illumination varies between AGAR sessions; robust, because
the reference region can still contain specks.

WHAT THIS IS NOT
----------------
This is a SCREEN, not a detector. It answers "is there something bright and
compact here that nothing labelled", which is cheap, CPU-only, and needs no
model. It cannot tell a colony from a bright plate artefact, a lid reflection
or a scratch, so a positive is a CANDIDATE, not a verdict -- that is why every
hit is written out as a crop.

The strong version of this measurement is to run the G100 detector over the
synthetic plates and count boxes that fall outside the label set. That needs
the GPU. This screen is what can be run while the GPU is busy, and it bounds
the problem: if the screen finds nothing, the detector will not find much
either.

Read-only. Writes one CSV and (optionally) crop images of every candidate.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# decision 3.10
PLATE_R_RATIO = 0.465

# A region smaller than this is a sliver left over where a label box clipped a
# larger region; it carries no evidence either way.
MIN_REGION_PX = 300
# A candidate must have at least this many bright pixels. A colony that would
# matter to a detector at imgsz=1280 is far larger than this; single bright
# specks are not colonies.
MIN_BRIGHT_PX = 200
# Robust sigmas above the agar median before a pixel counts as "bright".
BRIGHT_SIGMAS = 6.0
# An erase region this fraction covered by a synthetic label box is explained by
# a PLANNED colony that happened to land there. Reported separately, because
# calling one of those a ghost is the obvious way to get a scary wrong number.
CLAIMED_FRAC = 0.25


def label_boxes(path: Path, W: int, H: int):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        s = line.split()
        if len(s) != 5:
            continue
        x, y, w, h = (float(v) for v in s[1:])
        out.append((x * W, y * H, w * W, h * H))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True, choices=[10, 25, 50, 100],
                    help="decision 3.5 -- no default")
    ap.add_argument("--build", required=True, help="mask.py build output (masks/)")
    ap.add_argument("--generated", required=True, help="inpaint.py run output (images/ + labels/)")
    ap.add_argument("--out", default=None, help="CSV (default: <generated>/ghost_check.csv)")
    ap.add_argument("--crops", default=None,
                    help="write a crop of every candidate here so it can be judged "
                         "by eye (default: <generated>/ghost_crops)")
    args = ap.parse_args()

    build, gen = Path(args.build), Path(args.generated)
    masks = sorted((build / "masks").glob("*_erase.png"))
    if not masks:
        sys.exit(f"ERROR: no *_erase.png under {build}/masks. "
                 f"mask.py build writes them unconditionally since decision 3.85 -- "
                 f"if they are absent this build predates that and must be redone.")

    out_csv = Path(args.out) if args.out else gen / "ghost_check.csv"
    crop_dir = Path(args.crops) if args.crops else gen / "ghost_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_plates = n_regions = n_unclaimed = n_ghost = 0
    plates_hit = 0
    coincidence = 0            # unlabelled-region count avoided by the box test

    for mp in masks:
        name = mp.name.replace("_erase.png", "")
        img_p = gen / "images" / f"{name}.jpg"
        if not img_p.exists():
            continue
        syn = cv2.imread(str(img_p))
        er = cv2.imread(str(mp), 0)
        if syn is None or er is None:
            continue
        H, W = syn.shape[:2]
        n_plates += 1

        claimed = np.zeros((H, W), np.uint8)
        for x, y, w, h in label_boxes(gen / "labels" / f"{name}.txt", W, H):
            cv2.rectangle(claimed, (int(x - w / 2), int(y - h / 2)),
                          (int(x + w / 2), int(y + h / 2)), 255, -1)

        gray = cv2.cvtColor(syn, cv2.COLOR_BGR2GRAY).astype(np.float32)

        disc = np.zeros((H, W), np.uint8)
        cv2.circle(disc, (W // 2, H // 2), int(PLATE_R_RATIO * W), 255, -1)
        agar = (disc > 0) & (er == 0) & (claimed == 0)
        if agar.sum() < 10000:
            print(f"  [!] {name}: too little clean agar to set a threshold, skipped")
            continue
        med = float(np.median(gray[agar]))
        mad = float(np.median(np.abs(gray[agar] - med))) + 1e-6
        thr = med + BRIGHT_SIGMAS * 1.4826 * mad

        # Every erase region, and how much of it a synthetic label already claims.
        #
        # This is NOT done by comparing component counts before and after removing
        # the labelled area: taking a bite out of a region does not remove the
        # region, and can even SPLIT it into two, so the difference of the two
        # counts is meaningless and can come out negative. Measure the overlap
        # per region instead. (The first version of this file printed that
        # difference; on the first plate it read 0 while two of the four largest
        # regions were in fact 36% and 29% claimed.)
        na, laba, sta, _ = cv2.connectedComponentsWithStats((er > 0).astype(np.uint8))
        n_regions += max(na - 1, 0)
        for i in range(1, na):
            if sta[i, 4] < MIN_REGION_PX:
                continue
            if (claimed[laba == i] > 0).mean() >= CLAIMED_FRAC:
                coincidence += 1

        free = ((er > 0) & (claimed == 0)).astype(np.uint8)
        nf, labf, stf, cef = cv2.connectedComponentsWithStats(free)

        hits = 0
        for i in range(1, nf):
            if stf[i, 4] < MIN_REGION_PX:
                continue
            n_unclaimed += 1
            m = labf == i
            bright = int((gray[m] > thr).sum())
            if bright < MIN_BRIGHT_PX:
                continue
            hits += 1
            cx, cy = cef[i]
            s = 200
            x0 = int(max(0, min(cx - s // 2, W - s)))
            y0 = int(max(0, min(cy - s // 2, H - s)))
            cv2.imwrite(str(crop_dir / f"{name}_r{i:03d}.png"),
                        syn[y0:y0 + s, x0:x0 + s])
            rows.append([name, i, int(stf[i, 4]), bright, round(thr - med, 1),
                         int(cx), int(cy)])
        n_ghost += hits
        if hits:
            plates_hit += 1

    print("=" * 70)
    print("GHOST COLONY SCREEN")
    print("=" * 70)
    print(f"  plates                                  : {n_plates}")
    print(f"  erase regions (all)                     : {n_regions}")
    # NOT "regions with zero label overlap" -- it is the number of leftover
    # PIECES of erase area big enough to hide a colony once every label box has
    # been cut out. A region 30% claimed contributes its remaining 70% here.
    print(f"  unlabelled leftover pieces >= {MIN_REGION_PX} px    : {n_unclaimed}")
    print(f"  ... >={int(CLAIMED_FRAC*100)}% covered by a synthetic box     : {coincidence}"
          f"   <- a PLANNED colony landed there")
    print()
    print(f"  BRIGHT + UNCLAIMED = ghost candidates   : {n_ghost}")
    if n_unclaimed:
        print(f"  rate                                    : "
              f"{100 * n_ghost / n_unclaimed:.2f}% of unclaimed regions")
    print(f"  plates with at least one                : {plates_hit}/{n_plates}")
    print()
    print("  --- READ IT LIKE THIS ---")
    print("   * 0 candidates  -> the erase pass is holding on this many plates.")
    print("     Report the DENOMINATOR too: '0 of N regions' is a measurement,")
    print("     '0' alone is not.")
    print("   * a handful      -> LOOK AT THE CROPS. A lid reflection is not a")
    print("     ghost, and this screen cannot tell the difference.")
    print("   * many           -> the two-pass split is leaking; decision 3.60")
    print("     reopens before any grid is generated.")
    print("   * either way this is a SCREEN. The detector-based count over the")
    print("     synthetic plates is the real gate and still has to be run.")
    print()

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plate", "region", "region_px", "bright_px",
                    "threshold_over_agar", "cx", "cy"])
        w.writerows(rows)
    print(f"  candidates -> {out_csv}")
    print(f"  crops      -> {crop_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
