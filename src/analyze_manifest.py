#!/usr/bin/env python3
"""
Descriptive analysis and quality control from manifest.csv + YOLO labels.

It does two jobs at the same time:
  1) DESCRIPTIVE STATISTICS FOR THE PAPER (outline 3.2 data table)
  2) LAYOUT STATISTICS FOR PHASE 3 (colony count, size, radial position)
  3) QUALITY CONTROL: duplicate boxes, boxes outside the plate, tiny boxes

Output: CSV tables + PNG plots under <data>/reports/

Usage:
    python analyze_manifest.py --data data/processed
    python analyze_manifest.py --data data/processed --iou-dup 0.53
"""

import argparse
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_ORDER = ["S.aureus", "B.subtilis", "P.aeruginosa", "E.coli", "C.albicans"]

# Decision 3.49: 'radius' used to be normalized by the HALF WIDTH of the image
# and the comment claimed "the plate fits the square exactly". Wrong: the
# measured plate radius is R = 0.465 * width (decision 3.10). So the real plate
# edge corresponded to 0.930 in that unit, but the checks sat at 0.95 and 1.0
# -> the "outside the plate" check could NEVER fire. On top of that, layout.py
# was normalizing the same word by 0.465; the two files were using the same
# term with two different meanings.
# Now both are relative to the PLATE RADIUS: 0 = center, 1 = plate edge.
PLATE_R_RATIO = 0.465


def iou(a, b):
    """a, b = (x0, y0, x1, y1) in pixels"""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_boxes(man: pd.DataFrame) -> pd.DataFrame:
    """Expands all label files into a single box table."""
    rows = []
    for r in man.itertuples():
        W, H = r.width, r.height
        for line in Path(r.label_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cid, xc, yc, bw, bh = line.split()
            cid = int(cid)
            xc, yc, bw, bh = (float(v) for v in (xc, yc, bw, bh))
            px_w, px_h = bw * W, bh * H
            # distance of the center from the plate center, normalized by the
            # PLATE RADIUS (decision 3.49) -- same unit as layout.py
            dx = (xc - 0.5) / PLATE_R_RATIO
            dy = (yc - 0.5) / PLATE_R_RATIO
            rows.append({
                "stem": r.stem,
                "cls": CLASS_ORDER[cid],
                "cid": cid,
                "img_w": W, "img_h": H,
                "px_w": px_w, "px_h": px_h,
                "px_size": max(px_w, px_h),
                "area_px": px_w * px_h,
                "aspect": px_w / px_h if px_h else np.nan,
                "xc": xc, "yc": yc,
                "radius": float(np.hypot(dx, dy)),   # 0 = center, 1 = PLATE EDGE
                "radius_outer": float(np.hypot(dx, dy)) + (max(px_w, px_h) / 2)
                / (PLATE_R_RATIO * W),               # outer edge of the colony
                "x0": (xc - bw / 2) * W, "y0": (yc - bh / 2) * H,
                "x1": (xc + bw / 2) * W, "y1": (yc + bh / 2) * H,
            })
    return pd.DataFrame(rows)


def coco_size_bucket(area):
    """COCO convention: small <32^2, medium <96^2, large above"""
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    # Decision 3.76: 0.50 was calibrated on the demo package, where the highest
    # real pair reached 0.450 -- a margin of only 0.05. On the full data the real
    # IoU distribution gives p99.9 = 0.479, so 0.50 sits INSIDE the normal range
    # and crowded plates trip it. The threshold now comes from the data, not a
    # guess. This is a REPORTING threshold: the suspects are listed, never removed.
    # The 20 pairs it finds are 0.024% of 83208 boxes and were verified by eye --
    # AGAR annotation noise (duplicate boxes, boxes on the rim reflection).
    ap.add_argument("--iou-dup", type=float, default=0.53,
                    help="box pairs above this threshold are 'duplicate suspects'")
    ap.add_argument("--tiny-px", type=float, default=8,
                    help="boxes below this size are flagged")
    args = ap.parse_args()

    data = Path(args.data).expanduser().resolve()
    man_path = data / "manifest.csv"
    if not man_path.exists():
        sys.exit(f"ERROR: {man_path} does not exist. run convert.py first.")

    man = pd.read_csv(man_path)
    box = load_boxes(man)
    rep = data / "reports"
    rep.mkdir(exist_ok=True)

    print(f"\n{len(man)} images, {len(box)} boxes read\n")

    # ================= 1. PER-CLASS SUMMARY (paper table) ===================
    g = box.groupby("cls")
    cls_tab = pd.DataFrame({
        "boxes": g.size(),
        "images": g["stem"].nunique(),
        "size_min_px": g["px_size"].min().round(1),
        "size_median_px": g["px_size"].median().round(1),
        "size_p95_px": g["px_size"].quantile(0.95).round(1),
        "size_max_px": g["px_size"].max().round(1),
        "aspect_median": g["aspect"].median().round(3),
    }).reindex(CLASS_ORDER).fillna(0)
    cls_tab["box_share_%"] = (cls_tab["boxes"] / len(box) * 100).round(1)
    cls_tab.to_csv(rep / "class_summary.csv")
    print("=== PER-CLASS SUMMARY ===")
    print(cls_tab.to_string(), "\n")

    # ================= 2. COCO SIZE BREAKDOWN =================
    box["size_bucket"] = box["area_px"].map(coco_size_bucket)
    size_tab = pd.crosstab(box["cls"], box["size_bucket"]).reindex(
        index=CLASS_ORDER, columns=["small", "medium", "large"]).fillna(0).astype(int)
    size_tab.to_csv(rep / "size_breakdown.csv")
    print("=== COCO SIZE BREAKDOWN (the mAP breakdown is reported on this axis) ===")
    print(size_tab.to_string(), "\n")

    # ================= 3. COLONIES PER IMAGE (Phase 3 input) ================
    per_img = box.groupby("stem").size()
    print("=== COLONIES PER IMAGE ===")
    print(f"  min {per_img.min()} | median {per_img.median():.0f} | "
          f"mean {per_img.mean():.1f} | p95 {per_img.quantile(.95):.0f} | "
          f"max {per_img.max()}\n")
    per_img.rename("n_box").to_csv(rep / "colonies_per_image.csv")

    # ================= 4. SPECIES COMBINATIONS ==============================
    combo = man["classes"].value_counts().rename("images")
    combo.to_csv(rep / "species_combinations.csv")
    print("=== SPECIES COMBINATIONS (the layout distribution must be derived "
          "conditional on this) ===")
    print(combo.to_string(), "\n")

    # ================= 5. QUALITY CONTROL ===================================
    print("=== QUALITY CONTROL ===")

    tiny = box[box["px_size"] < args.tiny_px]
    print(f"  boxes below {args.tiny_px}px         : {len(tiny)}")

    edge = box[(box["x0"] <= 1) | (box["y0"] <= 1) |
               (box["x1"] >= box["img_w"] - 1) | (box["y1"] >= box["img_h"] - 1)]
    print(f"  touching the image edge : {len(edge)}")

    # r = 1.0 is the PLATE EDGE (decision 3.49). Even when the colony CENTER is
    # inside the plate, its OUTER EDGE can stick out; the two are reported
    # separately.
    print(f"  center r > 0.90         : {(box['radius'] > 0.90).sum()}")

    # Decision 3.77: this check used to shout "PROBLEM (the plate radius may be
    # wrong)" on a SINGLE box outside the plate. On the full data it fired on 16
    # boxes out of 83208 -- 0.019%, one in 5200 -- and those were verified BY EYE
    # to be AGAR annotation noise, not a wrong constant (e.g. 13672: two identical
    # boxes on the dark rim band with no colony under them).
    # exploration/07_plate_edge_check.py measured the radius the data itself asks
    # for: 99.9% of the centres sit inside 0.4561*W, and the constant is 0.465*W.
    # A check that cries wolf at one in five thousand teaches you to ignore it.
    # It now fires only when the share is large enough to mean a wrong plate model.
    OUTSIDE_TOL = 0.005                      # 0.5% of all boxes
    n_out = int((box["radius"] > 1.0).sum())
    frac_out = n_out / max(len(box), 1)
    tag = ("   <-- PROBLEM (the plate radius may be wrong)"
           if frac_out > OUTSIDE_TOL
           else f"   ({frac_out*100:.3f}% -- annotation noise, below the {OUTSIDE_TOL*100:.1f}% tolerance)")
    print(f"  center outside plate r>1: {n_out}{tag}")

    n_out_e = int((box["radius_outer"] > 1.0).sum())
    print(f"  OUTER EDGE beyond plate : {n_out_e}"
          f"   ({n_out_e/max(len(box),1)*100:.1f}% -- normal, a colony can touch the edge)")

    # duplicate box suspects -- within an image, vectorized IoU
    dup_rows = []
    all_iou = []          # all pairwise IoUs greater than zero -- for threshold calibration
    for stem, sub in box.groupby("stem"):
        b = sub[["x0", "y0", "x1", "y1"]].to_numpy(dtype=float)
        n = len(b)
        if n < 2:
            continue
        ix0 = np.maximum(b[:, None, 0], b[None, :, 0])
        iy0 = np.maximum(b[:, None, 1], b[None, :, 1])
        ix1 = np.minimum(b[:, None, 2], b[None, :, 2])
        iy1 = np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area[:, None] + area[None, :] - inter
        m = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        upper = np.triu(m, k=1)
        all_iou.extend(upper[upper > 0].tolist())
        ii, jj = np.where(upper >= args.iou_dup)
        cls_arr = sub["cls"].to_numpy()
        for a, c in zip(ii, jj):
            dup_rows.append({"stem": stem, "iou": round(float(m[a, c]), 3),
                             "cls_a": cls_arr[a], "cls_b": cls_arr[c]})
    dup = pd.DataFrame(dup_rows)
    n_dup_img = dup["stem"].nunique() if len(dup) else 0
    print(f"  box pairs with IoU >= {args.iou_dup} : {len(dup)}  "
          f"(in {n_dup_img} images)")
    # Decision 3.50: the 0.5 threshold was a GUESS. Real colonies do touch each
    # other (39%, overlap depth up to 0.63 -- decision 3.12), which means pairs
    # with a high IoU are LEGITIMATE. In the demo package the real maximum IoU
    # came out as 0.450: the margin is only 0.05. Pick the threshold from the
    # tail of the observed distribution, not by guessing.
    if len(all_iou):
        a = np.array(all_iou)
        print(f"  --- real IoU distribution (those > 0, n={len(a)}) ---")
        for q in (50, 90, 95, 99, 99.9):
            print(f"      p{q:<5} = {np.percentile(a, q):.3f}")
        print(f"      max   = {a.max():.3f}")
        suggested = round(float(np.percentile(a, 99.9)) + 0.05, 2)
        if args.iou_dup < suggested:
            print(f"  ! --iou-dup {args.iou_dup} looks low. Observed p99.9 "
                  f"{np.percentile(a, 99.9):.3f}; suggested threshold >= {suggested}")
    if len(dup):
        dup.sort_values("iou", ascending=False).to_csv(rep / "duplicate_suspects.csv",
                                                       index=False)
        print(f"    -> draw the first rows of {rep/'duplicate_suspects.csv'}")
        print(f"       one by one with check_labels.py and verify them BY EYE")
    print()

    # ================= 6. PLOTS =============================================
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    for c in CLASS_ORDER:
        s = box.loc[box["cls"] == c, "px_size"]
        if len(s):
            ax[0, 0].hist(s, bins=60, alpha=.55, label=f"{c} (n={len(s)})")
    ax[0, 0].set(title="Colony size distribution", xlabel="box side (px)",
                 ylabel="number of boxes")
    ax[0, 0].legend(fontsize=7)

    ax[0, 1].hist(per_img, bins=40, color="#4363d8")
    ax[0, 1].set(title="Colonies per image",
                 xlabel="colonies", ylabel="images")

    for c in CLASS_ORDER:
        s = box.loc[box["cls"] == c, "radius"]
        if len(s):
            ax[1, 0].hist(s, bins=40, alpha=.5, density=True, label=c)
    ax[1, 0].axvline(1.0, color="k", ls="--", lw=1)
    ax[1, 0].set(title="Radial position (0=center, 1=edge) -- Phase 3 layout input",
                 xlabel="normalized radius", ylabel="density")
    ax[1, 0].legend(fontsize=7)

    sample = box.sample(min(6000, len(box)), random_state=0)
    ax[1, 1].scatter(sample["xc"], sample["yc"], s=2, alpha=.25,
                     c=sample["cid"], cmap="tab10")
    ax[1, 1].set(title="Spatial distribution of colony centers",
                 xlabel="x (normalized)", ylabel="y (normalized)")
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(rep / "data_summary.png", dpi=140)
    print(f"plots  -> {rep/'data_summary.png'}")
    print(f"tables -> {rep}/\n")


if __name__ == "__main__":
    main()
