#!/usr/bin/env python3
"""
Draws the YOLO labels onto the images -- for verification BY EYE.

This step must not be skipped. A single error in the box coordinates (top-left
vs center, a value that was not normalized, the wrong image size) travels
silently through training and comes back later as "why is my model not
learning".

Usage:
    python check_labels.py --data data/processed --n 30
    python check_labels.py --data data/processed --n 10 --class-name C.albicans
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

CLASS_ORDER = ["S.aureus", "B.subtilis", "P.aeruginosa", "E.coli", "C.albicans"]
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]

MAX_SIDE = 1400   # shrink after drawing -- to browse the folder quickly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="convert.py output folder")
    ap.add_argument("--n", type=int, default=30, help="number of images to draw")
    ap.add_argument("--out", default=None, help="output folder (default: <data>/viz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-name", default=None,
                    help="sample only from the images that contain this class")
    args = ap.parse_args()

    data = Path(args.data).expanduser().resolve()
    manifest = data / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"ERROR: {manifest} does not exist. run convert.py first.")

    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.class_name:
        rows = [r for r in rows if int(r.get(f"n_{args.class_name}", 0) or 0) > 0]
        if not rows:
            sys.exit(f"ERROR: no image contains {args.class_name}")

    # --- Sampling: we want to see at least a few examples of every class ---
    rnd = random.Random(args.seed)
    selected, selected_stems = [], set()

    if not args.class_name:
        quota = max(1, args.n // (len(CLASS_ORDER) * 2))
        for c in CLASS_ORDER:
            candidates = [r for r in rows if int(r.get(f"n_{c}", 0) or 0) > 0
                          and r["stem"] not in selected_stems]
            rnd.shuffle(candidates)
            for r in candidates[:quota]:
                selected.append(r)
                selected_stems.add(r["stem"])

    remaining = [r for r in rows if r["stem"] not in selected_stems]
    rnd.shuffle(remaining)
    selected += remaining[: max(0, args.n - len(selected))]
    selected = selected[: args.n]

    outdir = Path(args.out) if args.out else data / "viz"
    outdir.mkdir(parents=True, exist_ok=True)

    warn_counts = Counter()
    total_boxes = 0

    for r in selected:
        img_path = Path(r["image_path"])
        lbl_path = Path(r["label_path"])
        if not img_path.exists() or not lbl_path.exists():
            warn_counts["missing_file"] += 1
            continue

        im = Image.open(img_path).convert("RGB")
        W, H = im.size

        # --- Size consistency: does the manifest agree with the real file? ---
        if (W, H) != (int(r["width"]), int(r["height"])):
            warn_counts["size_mismatch"] += 1
            print(f"  ! size does not match: {img_path.name} "
                  f"manifest={r['width']}x{r['height']} file={W}x{H}")

        draw = ImageDraw.Draw(im)
        thickness = max(2, round(min(W, H) / 500))

        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cid, xc, yc, bw, bh = line.split()
            cid = int(cid)
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            total_boxes += 1

            # --- Automatic sanity checks ---
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                warn_counts["out_of_range"] += 1
            if bw * W < 4 or bh * H < 4:
                warn_counts["box_too_small"] += 1
            # Decision 3.45: the check above only looks at whether the center and
            # the width are between 0 and 1; a box with xc=0.99, bw=0.1 PASSES
            # but reaches out to 1.04 of the image. It should not happen because
            # convert.py clips -- and the job of this file is exactly to verify
            # that "should not happen".
            if (xc - bw / 2 < -1e-6 or xc + bw / 2 > 1 + 1e-6
                    or yc - bh / 2 < -1e-6 or yc + bh / 2 > 1 + 1e-6):
                warn_counts["outside_edge"] += 1

            x0 = (xc - bw / 2) * W
            y0 = (yc - bh / 2) * H
            x1 = (xc + bw / 2) * W
            y1 = (yc + bh / 2) * H
            draw.rectangle([x0, y0, x1, y1],
                           outline=COLORS[cid % len(COLORS)], width=thickness)

        # --- Small legend (which color is which class) ---
        yy = 8
        for i, c in enumerate(CLASS_ORDER):
            if int(r.get(f"n_{c}", 0) or 0) > 0:
                draw.rectangle([8, yy, 8 + 26, yy + 20], fill=COLORS[i])
                draw.text((42, yy + 4), f"{c}  ({r[f'n_{c}']})", fill="white",
                          stroke_width=2, stroke_fill="black")
                yy += 26

        im.thumbnail((MAX_SIDE, MAX_SIDE))
        im.save(outdir / f"{r['stem']}_boxed.jpg", quality=88)

    print("\n" + "=" * 58)
    print(f"  images drawn  : {len(selected)}")
    print(f"  boxes drawn   : {total_boxes}")
    print(f"  output folder : {outdir}")
    if warn_counts:
        print("\n  WARNINGS:")
        for k, v in warn_counts.items():
            print(f"    {k}: {v}")
    else:
        print("\n  no warnings from the automatic checks")
    print("=" * 58)
    print("""
NOW OPEN THE FOLDER AND LOOK AT IT WITH YOUR OWN EYES. What you are looking for:
  - Are the boxes sitting on the colonies, or are they shifted?
  - Are the small colonies (S.aureus, C.albicans) boxed, or were they missed?
  - Are there boxes outside the plate / landing on empty background?
  - Is the same colony boxed more than once?
These 10 minutes save 3 days further down the road.
""")


if __name__ == "__main__":
    main()
