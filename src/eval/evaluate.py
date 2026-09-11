#!/usr/bin/env python3
"""
Evaluation CLI -- runs the SAME WAY on EVERY run.

What it produces (under runs/<name>/eval/):
  summary.json       one-line machine-readable result (the grid table is built from it)
  class_ap.csv       class x (AP, AP50, AP75) + GT counts
  size_ap.csv        all / small / medium / large breakdown
  counting.csv       MAE / RMSE / sMAPE / ME / r  -- total and per class
  conf_curve.csv     conf threshold sweep on VAL (only when --split val)

Two modes of operation:
  A) with --weights: loads the Ultralytics model, produces the predictions itself
  B) with --pred-dir: reads ready-made YOLO-format prediction files
     (for the second detector / external model comparison -- the measurement
     code stays the same)

conf threshold rule (LEAKAGE PREVENTION):
  Counting metrics require a conf threshold. This threshold cannot be searched
  on TEST.
    1) first  run with --split val  -> the selected threshold is written to summary.json
    2) then   run with --split test --conf-thr <that value>
  If --conf-thr is not given while --split test, the program stops with an ERROR.

Usage:
    python src/eval/evaluate.py --weights runs/G100_s0/weights/best.pt \\
        --data data/processed --split val --out runs/G100_s0/eval
    python src/eval/evaluate.py --weights runs/G100_s0/weights/best.pt \\
        --data data/processed --split test --conf-thr 0.35 --out runs/G100_s0/eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.metrics import (  # noqa: E402
    AREA_RANGES_COCO, CLASS_ORDER, ImageAnno, counting_metrics,
    evaluate_detection, read_yolo_txt, select_conf_threshold,
)

# Predictions BELOW this threshold are never recorded at all. It is kept low
# for mAP (COCO also uses every detection); the counting threshold is applied
# separately.
MIN_CONF = 0.001

# Decision 3.48: max_det and nms_iou used to be HARD-CODED HERE (1000 / 0.7)
# and the `train.max_det` / `train.nms_iou` values in base.yaml were not read
# anywhere -- changing the config did nothing. In Phase 5, when we say "we are
# freezing the protocol", the thing that gets frozen has to be the thing that
# actually runs. They are now read from the config; the values below are only
# a FALLBACK.
FALLBACK_MAX_DET = 1000    # AGAR has up to 125 colonies on a plate -> keep it generous
FALLBACK_NMS_IOU = 0.7

# Decision 3.94: a (class, size) cell with fewer GT boxes than this is left out
# of the size-band mAP that is reported ALONGSIDE the COCO-identical one. Read
# from eval.min_gt_for_cell; the value below is only a fallback.
FALLBACK_MIN_GT = 10


def stems_of(data: Path, split: str) -> list[str]:
    # Same rule as train.py: paths are resolved relative to ROOT, not relative
    # to the working directory. The previous version used Path("splits") -> it
    # only worked from the repo root, and if some other directory contained a
    # "splits/" it could read the WRONG split list.
    p = ROOT / "splits" / f"{split}.txt"
    if not p.exists():
        sys.exit(f"ERROR: {p} does not exist. first run  python src/make_splits.py --data {data}")
    s = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not s:
        sys.exit(f"ERROR: {p} is empty. In the demo package val/test come out empty -- the full data is required.")
    return s


def image_path_of(data: Path, stem: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        c = data / "images" / f"{stem}{ext}"
        if c.exists():
            return c
    sys.exit(f"ERROR: no image found for {stem} ({data / 'images'})")


def load_gt(data: Path, stems: list[str]) -> list[ImageAnno]:
    from PIL import Image
    out = []
    for s in stems:
        ip = image_path_of(data, s)
        with Image.open(ip) as im:
            W, H = im.size
        gc, gb, _ = read_yolo_txt(data / "labels" / f"{s}.txt", W, H)
        out.append(ImageAnno(stem=s, width=W, height=H, gt_cls=gc, gt_box=gb))
    return out


def fill_preds_from_dir(images, pred_dir: Path):
    """Ready-made predictions: <stem>.txt, line = cls xc yc w h conf (normalized)."""
    for im in images:
        dc, db, ds = read_yolo_txt(pred_dir / f"{im.stem}.txt",
                                   im.width, im.height, with_conf=True)
        im.dt_cls, im.dt_box, im.dt_conf = dc, db, ds


def read_protocol(config: Path):
    """Read the inference parameters from base.yaml (decision 3.48)."""
    if not config.exists():
        print(f"  ! {config} does not exist -- using fallback values "
              f"(max_det={FALLBACK_MAX_DET}, nms_iou={FALLBACK_NMS_IOU})", file=sys.stderr)
        return FALLBACK_MAX_DET, FALLBACK_NMS_IOU
    import yaml
    t = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("train", {}) or {}
    return int(t.get("max_det", FALLBACK_MAX_DET)), float(t.get("nms_iou", FALLBACK_NMS_IOU))


def read_locked_conf(config: Path):
    """
    The counting conf threshold frozen into the protocol (decision 3.88).

    It is selected ONCE on VAL and is the ruler for every run afterwards. Before
    Phase 5 the value lived only in the documentation ("conf_thr = 0.35, LOCKED
    TO THE PROTOCOL") while `eval.conf_thr` in base.yaml stayed null and was read
    by nobody -- a methodological rule that existed only as a comment.
    Returns None when the config does not lock a value.
    """
    if not config.exists():
        return None
    import yaml
    e = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("eval", {}) or {}
    v = e.get("conf_thr")
    return None if v is None else float(v)


def read_min_gt(config: Path) -> int:
    """
    Smallest GT count a (class, size) cell needs before its AP is averaged in
    (decision 3.94). 0 disables the rule.
    """
    if not config.exists():
        return FALLBACK_MIN_GT
    import yaml
    e = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("eval", {}) or {}
    v = e.get("min_gt_for_cell")
    return FALLBACK_MIN_GT if v is None else int(v)


def map_with_min_gt(ap, n_gt, min_gt: int):
    """
    mAP over a size band, ignoring classes with too few ground-truth boxes.

    WHY (decision 3.94): COCO averages per-class AP over the classes present in
    a size band, whatever their support. In AGAR's val split S.aureus has
    exactly ONE large box. Its AP is then the binary outcome of a single
    detection -- 0.8 in one seed, 0.0 in the next -- and it carries a full
    quarter of mAP_large. Measured across five G100 seeds, mAP_large has
    sigma 0.0668, 18.8x the primary metric; that spread is very largely this
    one box (3.93d).

    COCO is not wrong, it is faithful to its own definition, and `metrics.py`
    stays identical to it -- the two-implementation agreement of Phase 2 must
    not be broken. This function is a SECOND, reported alongside, reading.

    A class with zero boxes in the band is already NaN and out of the average;
    a class with one box was not. Zero and one do not deserve different
    treatment, and that inconsistency is what this rule removes.

    Returns (value, dropped) where dropped is [(class_index, n_gt), ...] for
    cells that had boxes but too few of them.
    """
    ap = np.asarray(ap, dtype=float)
    n = np.asarray(n_gt)
    if min_gt <= 0:
        keep = np.ones(len(n), dtype=bool)
    else:
        keep = n >= min_gt
    dropped = [(i, int(n[i])) for i in range(len(n))
               if not keep[i] and n[i] > 0 and not np.isnan(ap[i])]
    vals = ap[keep]
    vals = vals[~np.isnan(vals)]
    return (float(vals.mean()) if vals.size else float("nan")), dropped


def fill_preds_from_model(images, data: Path, weights: str, imgsz: int,
                          device: str, batch: int, half: bool,
                          max_det: int, nms_iou: float):
    from ultralytics import YOLO
    model = YOLO(weights)
    paths = [str(image_path_of(data, im.stem)) for im in images]

    # Ultralytics 8.4 deprecated `half` in favour of `quantize` and prints a
    # warning EVERY time the kwarg is passed -- once per chunk, so ~75 identical
    # lines for a 640-image split. It looks like a crash and invites a Ctrl+C on
    # a run that is perfectly healthy. The kwarg is now only passed when it was
    # actually asked for, which is the default-off path for the whole grid.
    pred_kw = dict(imgsz=imgsz, conf=MIN_CONF, iou=nms_iou, max_det=max_det,
                   device=device, verbose=False, stream=False)
    if half:
        pred_kw["half"] = True

    n_chunks = (len(paths) + batch - 1) // batch
    print(f"inference: {len(paths)} images, batch {batch} -> {n_chunks} chunks")

    t0 = time.perf_counter()
    boxes_all = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        res = model.predict(chunk, **pred_kw)
        boxes_all.extend(res)
        # A silent run is indistinguishable from a hung one. Show progress.
        done = min(i + batch, len(paths))
        print(f"\r  {done}/{len(paths)}  "
              f"({done / len(paths) * 100:5.1f}%)", end="", flush=True)
    print()
    infer_s = time.perf_counter() - t0

    # Predictions are matched to images BY PATH, not BY POSITION.
    # The previous version used zip(images, boxes_all): if a single image is
    # skipped, zip silently stops at the shorter one and ALL images AFTER THAT
    # POINT get matched with the wrong predictions. There is no error message,
    # and mAP goes nuts.
    if len(boxes_all) != len(images):
        sys.exit(f"ERROR: {len(images)} images were sent, {len(boxes_all)} predictions "
                 f"came back. Ultralytics may have skipped some images "
                 f"(a corrupt file?). The matching is not trustworthy, stopped.")

    by_stem = {}
    for r in boxes_all:
        st = Path(getattr(r, "path", "") or "").stem
        if st in by_stem:
            sys.exit(f"ERROR: two predictions came back for the same stem: {st}. "
                     f"Image names must be unique.")
        by_stem[st] = r

    for im in images:
        r = by_stem.get(im.stem)
        if r is None:
            sys.exit(f"ERROR: no prediction found for {im.stem}. "
                     f"(Is .path missing from the Ultralytics results? check the version)")
        b = r.boxes
        if b is None or len(b) == 0:
            im.dt_cls = np.zeros(0, int)
            im.dt_box = np.zeros((0, 4))
            im.dt_conf = np.zeros(0)
            continue
        im.dt_cls = b.cls.cpu().numpy().astype(int)
        im.dt_box = b.xyxy.cpu().numpy().astype(float)
        im.dt_conf = b.conf.cpu().numpy().astype(float)

    return {"inference_sec": round(infer_s, 3),
            "ms_per_image": round(1000 * infer_s / max(len(images), 1), 2)}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--weights", help="Ultralytics .pt")
    ap.add_argument("--pred-dir", help="folder of ready-made YOLO-format predictions")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"),
                    help="max_det and nms_iou are read from here (decision 3.48)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--conf-thr", type=float, default=None,
                    help="counting threshold. MANDATORY for --split test (the one selected on val)")
    ap.add_argument("--override-conf-thr", action="store_true",
                    help="deliberately evaluate with a conf threshold OTHER than the "
                         "one locked in the config (eval.conf_thr). Without this flag "
                         "a mismatch STOPS the program (decision 3.88)")
    ap.add_argument("--tag", default="", help="run name to be written into summary.json")
    args = ap.parse_args()

    if bool(args.weights) == bool(args.pred_dir):
        sys.exit("ERROR: give --weights OR --pred-dir (not both).")
    if args.split == "test" and args.conf_thr is None:
        sys.exit("ERROR: the conf threshold CANNOT BE SEARCHED on the test set.\n"
                 "  first: --split val   (selects the threshold)\n"
                 "  then:  --split test --conf-thr <selected value>")
    if args.split == "train" and args.conf_thr is None:
        # Searching the threshold on train is not leakage, but it is meaningless
        # (the model has seen that data). It used to leave selected=None, which
        # raised a TypeError inside counting_metrics.
        sys.exit("ERROR: --conf-thr is MANDATORY for --split train.\n"
                 "  The threshold is NOT SEARCHED on the training set (the model has "
                 "seen that data, the selection comes out optimistic).\n"
                 "  Give the threshold selected on VAL: --conf-thr <value>")

    # --- the locked ruler (decision 3.88) -----------------------------------
    # The threshold still has to be TYPED on the command line: that is what keeps
    # the leakage rule visible at the call site. What is new is that a typo can no
    # longer pass silently -- across 61 runs a single `0.3` instead of `0.35`
    # would move one arm's counting metrics and nothing would say so.
    locked_conf = read_locked_conf(Path(args.config))
    if (locked_conf is not None and args.conf_thr is not None
            and abs(args.conf_thr - locked_conf) > 1e-9):
        msg = (f"conf threshold MISMATCH: --conf-thr {args.conf_thr}, but the protocol "
               f"locks {locked_conf}\n  ({args.config} -> eval.conf_thr)")
        if not args.override_conf_thr:
            sys.exit(f"ERROR (decision 3.88): {msg}\n"
                     "  The ruler must be IDENTICAL in all 61 runs of the grid.\n"
                     "  If the difference is deliberate: --override-conf-thr")
        print("\033[1;33m! %s\033[0m" % msg)
        print("\033[1;33m!   --override-conf-thr given -- continuing DELIBERATELY. "
              "summary.json records it.\033[0m")
    if locked_conf is None and args.split in ("test", "train"):
        print("\033[1;33m%s\033[0m" % (
            "! eval.conf_thr is null in the config: the protocol locks no threshold,\n"
            "  so a typo in --conf-thr CANNOT be caught (decision 3.88)."))

    data = Path(args.data).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    stems = stems_of(data, args.split)
    print(f"{args.split}: {len(stems)} images")

    images = load_gt(data, stems)
    timing = {}
    if args.pred_dir:
        fill_preds_from_dir(images, Path(args.pred_dir).resolve())
    else:
        max_det, nms_iou = read_protocol(Path(args.config))
        print(f"protocol: max_det={max_det}  nms_iou={nms_iou}  ({args.config})")
        timing = fill_preds_from_model(images, data, args.weights, args.imgsz,
                                       args.device, args.batch, args.half,
                                       max_det, nms_iou)

    # ---------------- Detection metrics ----------------
    det = evaluate_detection(images, n_classes=len(CLASS_ORDER),
                             area_ranges=AREA_RANGES_COCO)

    import pandas as pd

    class_tab = pd.DataFrame({
        "AP50-95": det["ap"]["all"],
        "AP50": det["ap50"]["all"],
        "AP75": det["ap75"]["all"],
        "AP_small": det["ap"]["small"],
        "AP_medium": det["ap"]["medium"],
        "AP_large": det["ap"]["large"],
        "n_GT": det["n_gt"]["all"],
        "n_GT_small": det["n_gt"]["small"],
        "n_GT_medium": det["n_gt"]["medium"],
        "n_GT_large": det["n_gt"]["large"],
    }, index=CLASS_ORDER).round(4)
    class_tab.to_csv(out / "class_ap.csv")

    # --- size bands, two readings side by side (decision 3.94) ---
    min_gt = read_min_gt(Path(args.config))
    filtered, dropped_cells = {}, []
    for a in AREA_RANGES_COCO:
        v, drop = map_with_min_gt(det["ap"][a], det["n_gt"][a], min_gt)
        filtered[a] = v
        dropped_cells += [{"class": CLASS_ORDER[i], "band": a, "n_GT": n}
                          for i, n in drop]

    size_tab = pd.DataFrame({
        a: {"mAP50-95": det["map"][a], "mAP50": det["map50"][a],
            "mAP75": det["map75"][a],
            f"mAP50-95_minGT{min_gt}": filtered[a],
            "n_GT": int(det["n_gt"][a].sum())}
        for a in AREA_RANGES_COCO
    }).T.round(4)
    size_tab.to_csv(out / "size_ap.csv")

    # ---------------- Counting metrics ----------------
    if args.split == "val" and args.conf_thr is None:
        selected, curve = select_conf_threshold(images)
        pd.DataFrame(curve, columns=["conf_thr", "MAE"]).to_csv(
            out / "conf_curve.csv", index=False)
        print(f"\nconf threshold selected on VAL: {selected}  "
              f"-> give --conf-thr {selected} in the test run")
        if locked_conf is not None and abs(selected - locked_conf) > 1e-9:
            # Not an error: searching on VAL is legitimate. But if a fresh search
            # lands somewhere else, that is worth seeing rather than discovering
            # later in a table (decision 3.88).
            print("\033[1;33m%s\033[0m" % (
                f"! the value selected on VAL ({selected}) differs from the one locked "
                f"in the protocol ({locked_conf}).\n"
                f"  The grid keeps using {locked_conf}. This is information, not an error."))
    else:
        selected = args.conf_thr

    cnt = counting_metrics(images, selected)
    cnt_tab = pd.DataFrame({"total": cnt["total"], **cnt["per_class"]}).T.round(4)
    cnt_tab.to_csv(out / "counting.csv")

    # ---------------- Summary ----------------
    summary = {
        "tag": args.tag or Path(args.out).parent.name,
        "split": args.split,
        "n_images": len(images),
        "n_gt_boxes": int(det["n_gt"]["all"].sum()),
        "imgsz": args.imgsz,
        "max_det": None if args.pred_dir else max_det,
        "nms_iou": None if args.pred_dir else nms_iou,
        "conf_thr_count": selected,
        "conf_thr_locked": locked_conf,                       # decision 3.88
        "conf_thr_overridden": bool(args.override_conf_thr),
        # --- primary metric ---
        "mAP50-95": det["map"]["all"],
        "mAP50": det["map50"]["all"],
        "mAP75": det["map75"]["all"],
        "mAP_small": det["map"]["small"],
        "mAP_medium": det["map"]["medium"],
        "mAP_large": det["map"]["large"],
        # --- same bands, thin cells left out (3.94). The three above stay
        #     COCO-identical; these are the ones the paper's size table uses.
        "min_gt_for_cell": min_gt,
        "mAP_small_minGT": filtered["small"],
        "mAP_medium_minGT": filtered["medium"],
        "mAP_large_minGT": filtered["large"],
        "cells_dropped": dropped_cells,
        # --- primary metric per class ---
        **{f"AP_{c}": float(det["ap"]["all"][i]) for i, c in enumerate(CLASS_ORDER)},
        # --- counting ---
        "MAE": cnt["total"]["MAE"],
        "RMSE": cnt["total"]["RMSE"],
        "sMAPE": cnt["total"]["sMAPE"],
        "ME": cnt["total"]["ME"],
        "count_r": cnt["total"]["r"],
        **timing,
    }
    summary = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in summary.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    # ---------------- To the screen ----------------
    print("\n=== SIZE BREAKDOWN ===")
    print(size_tab.to_string())
    if dropped_cells:
        # Loud on purpose (principle 2). A cell this thin quietly averaged into
        # mAP is exactly what made mAP_large unusable before 3.94.
        print("\033[1;33m%s\033[0m" % (
            f"! size cells with fewer than {min_gt} GT boxes, left out of "
            f"the mAP50-95_minGT{min_gt} column:"))
        for c in dropped_cells:
            print("\033[1;33m%s\033[0m" % (
                f"    {c['class']}/{c['band']}  n_GT={c['n_GT']}"))
        for a in AREA_RANGES_COCO:
            if any(c["band"] == a for c in dropped_cells):
                print("\033[1;33m%s\033[0m" % (
                    f"    {a}: {det['map'][a]:.4f} (COCO) -> "
                    f"{filtered[a]:.4f} (minGT{min_gt})"))
    print("\n=== PER CLASS ===")
    print(class_tab[["AP50-95", "AP50", "AP_small", "AP_medium", "AP_large",
                     "n_GT"]].to_string())
    print(f"\n=== COUNTING (conf={selected}) ===")
    print(cnt_tab[["MAE", "RMSE", "sMAPE", "ME", "true_total",
                   "pred_total"]].to_string())
    print(f"\nPRIMARY METRIC  mAP50-95 = {summary['mAP50-95']:.4f}"
          f"   (small {summary['mAP_small']:.4f})")
    print(f"summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
