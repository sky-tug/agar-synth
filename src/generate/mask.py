#!/usr/bin/env python3
"""
mask.py -- prepares the mask and the background that will be handed to diffusion,
starting from a layout plan.

The second step of Phase 3. There is STILL no diffusion. This script produces
these three things:
  1. background : a real plate image (from the train subset of that level)
  2. mask       : the binary region that diffusion will fill in
  3. label      : already produced by `layout.py`, copied over here

There are two kinds of region in the mask:
  SYNTHETIC  colony discs -- new colonies will be drawn here
  ERASE      the area on top of the REAL colonies of the background plate -- these
             will be painted over and destroyed

The second one is critical: AGAR has no empty plates. If we use the background
as it is, unlabelled real colonies remain in the image; the model learns
"colony = background". That amounts to silently injecting false negatives into
the synthetic data.

Two subcommands:
  pool   selects and scores the background pool of a level -> bg_pool_<level>.json
  build  plan + background -> mask PNG + background + label

Related decisions (decisions.md): 3.4 - 3.5 - 3.7 - 3.10 - 3.13 - 3.24 - 3.25 - 3.26
"""
from __future__ import annotations
import argparse, json, math, shutil, sys
from pathlib import Path

import numpy as np
import cv2

PLATE_R_RATIO = 0.465         # decision 3.10

# --- decision 3.24 ----------------------------------------------------------
# The synthetic mask is the labelled disc ITSELF. NO dilation.
# Because diffusion can only paint the inside of the mask, a generated colony
# CANNOT OVERFLOW its label box. The "zero label error" claim rests on this.
SYNTH_MARGIN = 1.00

# --- decision 3.25 ----------------------------------------------------------
# The erase mask, on the other hand, is dilated GENEROUSLY: the shadow/halo of the
# real colony must go too.
ERASE_MARGIN = 1.35           # can be changed with --erase-margin
ERASE_BLUR = 9                # edge smoothing (odd number)


# ===========================================================================
# POOL
# ===========================================================================

def _plate_mask(w, h):
    """The plate disc (decision 3.10). Its outside is never opened to diffusion."""
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), int(PLATE_R_RATIO * w), 255, -1)
    return m


def _read_labels(p: Path):
    if not p.exists():
        return []
    return [tuple(map(float, s.split()[1:5])) for s in p.read_text().split("\n") if s.strip()]


def _species_distribution(stems, label_root: Path, classes):
    """The per-plate distribution of the dominant species over a set of plates."""
    cnt = {s: 0 for s in classes}
    for st in stems:
        p = label_root / f"{st}.txt"
        if not p.exists():
            continue
        c = {}
        for line in p.read_text().split("\n"):
            if line.strip():
                i = int(line.split()[0]); c[i] = c.get(i, 0) + 1
        if c:
            cnt[classes[max(c, key=c.get)]] += 1
    t = sum(cnt.values()) or 1
    return {s: n / t for s, n in cnt.items()}, t


def pool(args):
    global ERASE_MARGIN
    ERASE_MARGIN = args.erase_margin
    file_list = [l for l in Path(args.list).read_text().split("\n") if l.strip()]
    records = []
    for path in file_list:
        stem = Path(path).stem
        boxes = _read_labels(Path(args.labels) / f"{stem}.txt")
        # the ratio of the area to be erased to the plate disc -- smaller is a better background
        area = sum(math.pi * ((w + h) / 4 * ERASE_MARGIN) ** 2 for _, _, w, h in boxes)
        plate_area = math.pi * PLATE_R_RATIO ** 2
        records.append(dict(stem=stem, n_colonies=len(boxes),
                            erase_ratio=round(area / plate_area, 4),
                            largest=round(max(((w + h) / 2 for _, _, w, h in boxes), default=0), 4)))
    records.sort(key=lambda k: k["erase_ratio"])

    threshold = args.threshold
    selected = [k for k in records if k["erase_ratio"] <= threshold]
    if not selected:
        sys.exit(f"ERROR: no plate satisfies erase_ratio <= {threshold}. "
                 f"The best is {records[0]['stem']} ({records[0]['erase_ratio']}). "
                 f"Raise --threshold but read the risk in decision 3.26.")

    # --- decision 3.27: is the pool biased with respect to species? --------
    classes = Path(args.classes).read_text().split()
    lab_root = Path(args.labels)
    d_pool, n_p = _species_distribution([k["stem"] for k in selected], lab_root, classes)
    d_all, n_a = _species_distribution([k["stem"] for k in records], lab_root, classes)
    bias = {s: round(d_pool[s] - d_all[s], 3) for s in classes}
    worst = max(bias.values(), key=abs) if bias else 0.0

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        level=args.level, source_list=str(args.list), threshold=threshold,
        erase_margin=ERASE_MARGIN, pool_size=len(selected), n_candidates=len(records),
        species_bias=bias, pool_species_share=d_pool, all_species_share=d_all,
        pool=[k["stem"] for k in selected], details=records), indent=2, ensure_ascii=False))

    print(f"[pool] level={args.level}  {len(selected)}/{len(records)} plates selected "
          f"(erase_ratio <= {threshold})")
    print(f"       cleanest: " + ", ".join(f"{k['stem']}({k['erase_ratio']:.3f})"
                                           for k in selected[:5]))
    print(f"       dirtiest selected: {selected[-1]['stem']} ({selected[-1]['erase_ratio']:.3f})")
    print("       dominant species share (pool vs all):")
    for s_ in classes:
        mark = "  <-- biased" if abs(bias[s_]) >= 0.15 else ""
        print(f"         {s_:<14} {100*d_pool[s_]:5.1f}%  vs  {100*d_all[s_]:5.1f}%"
              f"   ({bias[s_]:+.2f}){mark}")
    if len(selected) < 20:
        print(f"  [!] The pool has {len(selected)} plates. The same background will be used over and "
              f"over; synthetic diversity drops and FID/KID will catch it (decision 3.26).")
    if abs(worst) >= 0.15:
        print(f"  [!] THE POOL IS BIASED WITH RESPECT TO SPECIES (decision 3.27). The threshold looks "
              f"at the AREA to be erased; the area is determined by colony SIZE, and size is determined "
              f"by SPECIES. So the threshold is silently selecting the plates of small-colony species. "
              f"If the background of the synthetic data comes from a single species family, this becomes "
              f"a systematic shift.")
    print(f"       -> {out}")


# ===========================================================================
# BUILD
# ===========================================================================

def build_masks(plan, bg_boxes, w, h):
    """Returns: (synthetic_mask, erase_mask, combined_mask) -- all uint8 0/255."""
    synth = np.zeros((h, w), np.uint8)
    for k in plan["colonies"]:
        r = k["diameter"] / 2 * SYNTH_MARGIN
        cv2.circle(synth, (int(round(k["xc"] * w)), int(round(k["yc"] * h))),
                   max(1, int(round(r * w))), 255, -1)

    erase = np.zeros((h, w), np.uint8)
    for (xc, yc, bw, bh) in bg_boxes:
        r = (bw + bh) / 4 * ERASE_MARGIN
        cv2.circle(erase, (int(round(xc * w)), int(round(yc * h))),
                   max(1, int(round(r * w))), 255, -1)
    if ERASE_BLUR:
        erase = (cv2.GaussianBlur(erase, (ERASE_BLUR, ERASE_BLUR), 0) > 40).astype(np.uint8) * 255

    plate = _plate_mask(w, h)
    synth = cv2.bitwise_and(synth, plate)     # a colony cannot go outside the plate
    erase = cv2.bitwise_and(erase, plate)
    return synth, erase, cv2.bitwise_or(synth, erase)


def build(args):
    global ERASE_MARGIN
    pool_data = json.loads(Path(args.pool).read_text())
    ERASE_MARGIN = pool_data.get("erase_margin", ERASE_MARGIN)   # whichever margin the pool was selected with
    if pool_data["level"] != args.level:
        sys.exit(f"ERROR (decision 3.7): pool level={pool_data['level']}, --level {args.level}.")
    plans = sorted((Path(args.plans) / "plans").glob("*.json"))
    if not plans:
        sys.exit(f"ERROR: no plan under {args.plans}/plans.")

    rng = np.random.default_rng(args.seed)
    C = Path(args.out)
    for sub in ("masks", "backgrounds", "labels", "provenance"):
        (C / sub).mkdir(parents=True, exist_ok=True)

    usage, coverage, overflow = {}, [], 0
    for pf in plans:
        plan = json.loads(pf.read_text())
        if plan["level"] != args.level:
            sys.exit(f"ERROR (decision 3.7): {pf.name} level={plan['level']}.")
        name = plan["name"]

        bg = pool_data["pool"][int(rng.integers(len(pool_data["pool"])))]
        usage[bg] = usage.get(bg, 0) + 1

        img = cv2.imread(str(Path(args.images) / f"{bg}.jpg"))
        if img is None:
            sys.exit(f"ERROR: could not read background: {bg}.jpg")
        h, w = img.shape[:2]
        bg_boxes = _read_labels(Path(args.labels) / f"{bg}.txt")
        synth, erase, combined = build_masks(plan, bg_boxes, w, h)

        # how much of the synthetic discs overlaps the erase region (informational)
        coverage.append(float((cv2.bitwise_and(synth, erase) > 0).sum() / max((synth > 0).sum(), 1)))
        # is there a colony whose label spills outside the image (there should not be)
        for k in plan["colonies"]:
            if not (0 <= k["xc"] - k["w"] / 2 and k["xc"] + k["w"] / 2 <= 1
                    and 0 <= k["yc"] - k["h"] / 2 and k["yc"] + k["h"] / 2 <= 1):
                overflow += 1

        cv2.imwrite(str(C / "masks" / f"{name}.png"), combined)
        if args.split_masks:
            cv2.imwrite(str(C / "masks" / f"{name}_synthetic.png"), synth)
            cv2.imwrite(str(C / "masks" / f"{name}_erase.png"), erase)
        cv2.imwrite(str(C / "backgrounds" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 96])
        shutil.copy(Path(args.plans) / "labels" / f"{name}.txt", C / "labels" / f"{name}.txt")
        (C / "provenance" / f"{name}.json").write_text(json.dumps(dict(
            name=name, level=args.level, background=bg,
            background_source=pool_data["source_list"],   # leakage audit trail (decision 3.4)
            n_synthetic=len(plan["colonies"]),
            n_erased_real=len(bg_boxes),
            synth_margin=SYNTH_MARGIN, erase_margin=ERASE_MARGIN), indent=1, ensure_ascii=False))

    n = len(plans)
    repeat = max(usage.values())
    print(f"[build] {n} plates - level={args.level} - pool of {len(pool_data['pool'])} backgrounds")
    print(f"        background repetition: at most {repeat} times ({100*repeat/n:.1f}%), "
          f"{len(pool_data['pool'])-len(usage)} plates unused")
    print(f"        part of the synthetic disc overlapping the erase region: "
          f"avg {100*np.mean(coverage):.1f}%")
    if overflow:
        print(f"  [!] {overflow} labels spill outside the image -- layout.py should be checked.")
    if repeat / n > 0.15:
        print(f"  [!] One background carries {100*repeat/n:.0f}% of the generation. "
              f"Diversity risk (decision 3.26).")
    print(f"        -> {C}")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        # decision 3.5: NO default
        p.add_argument("--level", type=int, required=True, choices=[10, 25, 50, 100],
                       help="MANDATORY (decision 3.5). The background pool depends on the level too (3.4).")
        p.add_argument("--labels", default="data/processed/labels")

    p = sub.add_parser("pool", help="select the background pool of the level")
    common(p)
    p.add_argument("--list", required=True, help="e.g. data/processed/lists/train_25.txt")
    p.add_argument("--classes", default="data/processed/classes.txt")
    p.add_argument("--erase-margin", type=float, default=ERASE_MARGIN,
                   help="radius multiplier when erasing a real colony. The erased area grows "
                        "with the SQUARE of this.")
    p.add_argument("--threshold", type=float, default=0.06,
                   help="upper limit for the ratio of the area to be erased to the plate disc (default 0.06)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=pool)

    p = sub.add_parser("build", help="build mask + background + label")
    common(p)
    p.add_argument("--plans", required=True, help="output of layout.py sample")
    p.add_argument("--pool", required=True)
    p.add_argument("--images", default="data/processed/images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-masks", action="store_true",
                   help="also write the synthetic and erase masks separately (for visual checking)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=build)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
