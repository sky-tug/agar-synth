#!/usr/bin/env python3
"""
Measurement core -- INDEPENDENT of Ultralytics.

Why we have our own implementation:
  1. Ultralytics' val() does not report size-stratified AP (small/medium/large).
     The paper's central claim is about small colonies; this breakdown is
     mandatory.
  2. MAE / sMAPE (colony counting) do not exist in Ultralytics at all.
  3. The second-detector control (RT-DETR, YOLO11) requires the SAME measurement
     code to run. A detector-dependent metric invalidates any cross-detector
     comparison.
  4. The primary metric must live "inside the box" -- when the protocol is
     frozen (Phase 5), the measurement code freezes with it.

The AP computation is algorithmically identical to COCO
(cocoeval.evaluateImg / accumulate):
  - IoU thresholds 0.50:0.05:0.95 (10 thresholds)
  - 101-point interpolation
  - GTs outside the area range are "ignore"; detections matched to them are too
  - unmatched detections outside the area range are also ignored

Verification: src/eval/test_metrics.py (cross-checked against pycocotools when
available, to 1e-4).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants -- PART OF THE PROTOCOL. Changing any of these requires an entry in
# decisions.md (the decision log) with the rationale.
# ---------------------------------------------------------------------------

CLASS_ORDER = ["S.aureus", "B.subtilis", "P.aeruginosa", "E.coli", "C.albicans"]

IOU_THRS = np.linspace(0.5, 0.95, 10)          # COCO standard
REC_THRS = np.linspace(0.0, 1.0, 101)          # 101-point interpolation

# COCO area ranges, in ORIGINAL image pixels -- NOT in imgsz. evaluate.py reads
# width/height from the image file itself so this holds. If it ever used imgsz
# the whole size breakdown (the paper's main argument) would silently shift.
# Meaningful for AGAR: S.aureus median 29px -> 841px^2 (small),
# E.coli median 128px -> 16384px^2 (large).
AREA_RANGES_COCO = {
    "all":    (0.0, 1e10),
    "small":  (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large":  (96.0 ** 2, 1e10),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageAnno:
    """Ground truth + detections for one image. Boxes are xyxy, ABSOLUTE pixels."""
    stem: str
    width: int
    height: int
    gt_cls: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    gt_box: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    dt_cls: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    dt_box: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    dt_conf: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_yolo_txt(path: Path, W: int, H: int, with_conf: bool = False):
    """
    YOLO format -> (cls[N], xyxy[N,4], conf[N] | None), ABSOLUTE pixels.
    Line: cls xc yc w h [conf]   (all normalised)

    W and H must be the ORIGINAL image dimensions. Swapping them produces
    plausible-looking but wrong boxes; test_metrics.py guards this with a
    deliberately non-square (800x400) case.
    """
    cls, box, conf = [], [], []
    if not path.exists():
        return (np.zeros(0, int), np.zeros((0, 4)),
                np.zeros(0) if with_conf else None)

    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        xc, yc, w, h = (float(v) for v in p[1:5])
        cls.append(c)
        box.append([(xc - w / 2) * W, (yc - h / 2) * H,
                    (xc + w / 2) * W, (yc + h / 2) * H])
        if with_conf:
            conf.append(float(p[5]) if len(p) > 5 else 1.0)

    return (np.asarray(cls, int),
            np.asarray(box, float).reshape(-1, 4),
            np.asarray(conf, float) if with_conf else None)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a[N,4], b[M,4] xyxy -> IoU[N,M]"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def box_area(b: np.ndarray) -> np.ndarray:
    if len(b) == 0:
        return np.zeros(0)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


# ---------------------------------------------------------------------------
# COCO evaluateImg equivalent: matching for one (image, class, area range)
# ---------------------------------------------------------------------------

def _match_one(gt_box, dt_box, dt_conf, area_rng, iou_thrs):
    """
    Matching identical to COCO's cocoeval.evaluateImg.
    Returns: sorted dt_conf, dtm[T,D] (matched?), dtIg[T,D] (ignored?),
             npig (number of GTs that count)
    """
    T = len(iou_thrs)
    G, D = len(gt_box), len(dt_box)

    gt_ig = np.zeros(G, dtype=bool)
    if G:
        ga = box_area(gt_box)
        gt_ig = (ga < area_rng[0]) | (ga > area_rng[1])
        # COCO: non-ignored GTs come first so matching prefers them.
        # This is the "ignore" mechanism: when computing e.g. small-object AP we
        # do NOT delete large GTs -- deleting them would turn a correct large
        # detection into a false positive and unfairly depress the small-object
        # score. Marking them ignore means no reward and no penalty.
        order_g = np.argsort(gt_ig, kind="stable")
        gt_box, gt_ig = gt_box[order_g], gt_ig[order_g]

    npig = int((~gt_ig).sum())

    if D == 0:
        return np.zeros(0), np.zeros((T, 0), bool), np.zeros((T, 0), bool), npig

    order_d = np.argsort(-dt_conf, kind="stable")
    dt_box, dt_conf = dt_box[order_d], dt_conf[order_d]

    ious = iou_matrix(dt_box, gt_box)          # [D, G]
    dtm = np.zeros((T, D), dtype=bool)
    dtIg = np.zeros((T, D), dtype=bool)

    for ti, thr in enumerate(iou_thrs):
        gtm = np.full(G, -1, dtype=int)
        for di in range(D):
            best_iou = min(thr, 1 - 1e-10)
            m = -1
            for gi in range(G):
                if gtm[gi] >= 0:                       # this GT is already taken
                    continue
                if m > -1 and (not gt_ig[m]) and gt_ig[gi]:
                    break                              # stop before falling into ignored GTs
                if ious[di, gi] < best_iou:
                    continue
                best_iou = ious[di, gi]
                m = gi
            if m == -1:
                continue
            dtm[ti, di] = True
            dtIg[ti, di] = gt_ig[m]
            gtm[m] = di

    # unmatched detections outside the area range do not count either
    da = box_area(dt_box)
    out_of_range = (da < area_rng[0]) | (da > area_rng[1])
    dtIg |= (~dtm) & out_of_range[None, :]

    return dt_conf, dtm, dtIg, npig


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_detection(images, n_classes=len(CLASS_ORDER),
                       area_ranges=None, iou_thrs=IOU_THRS):
    """
    images: list of ImageAnno
    Returns dict:
      ap[area][class_id]        -> AP averaged over IoU thresholds
      ap50[area][class_id]      -> AP at IoU 0.50
      ap75[area][class_id]
      map[area]                 -> mean over classes that have GT
      map50[area], map75[area]
      n_gt[area][class_id]
    """
    area_ranges = area_ranges or AREA_RANGES_COCO
    T = len(iou_thrs)
    R = len(REC_THRS)

    out = {"ap": {}, "ap50": {}, "ap75": {}, "n_gt": {},
           "map": {}, "map50": {}, "map75": {}}

    i50 = int(np.argmin(np.abs(iou_thrs - 0.50)))
    i75 = int(np.argmin(np.abs(iou_thrs - 0.75)))

    for aname, arng in area_ranges.items():
        ap_per_cls = np.full((T, n_classes), np.nan)
        ngt_per_cls = np.zeros(n_classes, dtype=int)

        for c in range(n_classes):
            confs, dtms, dtIgs = [], [], []
            npig_total = 0

            for im in images:
                g = im.gt_box[im.gt_cls == c] if len(im.gt_cls) else np.zeros((0, 4))
                dmask = im.dt_cls == c if len(im.dt_cls) else np.zeros(0, bool)
                d = im.dt_box[dmask] if len(im.dt_cls) else np.zeros((0, 4))
                s = im.dt_conf[dmask] if len(im.dt_cls) else np.zeros(0)

                conf, dtm, dtIg, npig = _match_one(g, d, s, arng, iou_thrs)
                npig_total += npig
                if len(conf):
                    confs.append(conf)
                    dtms.append(dtm)
                    dtIgs.append(dtIg)

            ngt_per_cls[c] = npig_total
            if npig_total == 0:
                continue                       # class absent in this area -> AP undefined

            if confs:
                conf = np.concatenate(confs)
                dtm = np.concatenate(dtms, axis=1)
                dtIg = np.concatenate(dtIgs, axis=1)
                # Sorted across ALL images, not per image. The precision-recall
                # curve is drawn over one globally ranked detection list; doing
                # it per image and averaging gives a different (wrong) number.
                # This is the most common bug in hand-written AP code.
                order = np.argsort(-conf, kind="stable")
                dtm, dtIg = dtm[:, order], dtIg[:, order]
            else:
                dtm = np.zeros((T, 0), bool)
                dtIg = np.zeros((T, 0), bool)

            for ti in range(T):
                tp = dtm[ti] & ~dtIg[ti]
                fp = (~dtm[ti]) & ~dtIg[ti]
                tp_c = np.cumsum(tp)
                fp_c = np.cumsum(fp)

                rc = tp_c / npig_total
                pr = tp_c / np.maximum(tp_c + fp_c, np.finfo(float).eps)

                # COCO: make the precision curve monotonically non-increasing
                pr = np.concatenate([pr, [0.0]])
                for i in range(len(pr) - 2, -1, -1):
                    pr[i] = max(pr[i], pr[i + 1])
                pr = pr[:-1]

                idx = np.searchsorted(rc, REC_THRS, side="left")
                q = np.zeros(R)
                valid = idx < len(pr)
                q[valid] = pr[idx[valid]]
                ap_per_cls[ti, c] = q.mean()

        with warnings.catch_warnings():
            # Class/area cells with no GT are NaN -- "Mean of empty slice" is the
            # expected state here, not a problem worth warning about.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out["ap"][aname] = np.nanmean(ap_per_cls, axis=0)
            out["ap50"][aname] = ap_per_cls[i50]
            out["ap75"][aname] = ap_per_cls[i75]
            out["n_gt"][aname] = ngt_per_cls
            out["map"][aname] = float(np.nanmean(ap_per_cls)) \
                if np.isfinite(ap_per_cls).any() else float("nan")
            out["map50"][aname] = float(np.nanmean(ap_per_cls[i50])) \
                if np.isfinite(ap_per_cls[i50]).any() else float("nan")
            out["map75"][aname] = float(np.nanmean(ap_per_cls[i75])) \
                if np.isfinite(ap_per_cls[i75]).any() else float("nan")

    return out


# ---------------------------------------------------------------------------
# Colony counting -- in microbiology this is the ACTUAL task; mAP is a proxy
# ---------------------------------------------------------------------------

def counting_metrics(images, conf_thr: float, n_classes=len(CLASS_ORDER)):
    """
    Predicted vs. true colony count per image.

    WARNING: conf_thr must NOT be selected on the test set. It is chosen on VAL,
    frozen, and applied to test. evaluate.py enforces this at runtime.

    sMAPE definition (symmetric, bounded 0-200%):
        200 * |p - g| / (|p| + |g|),   0 when p = g = 0
    Plain MAPE diverges on empty plates; sMAPE does not.
    """
    g_tot, p_tot = [], []
    g_cls = {c: [] for c in range(n_classes)}
    p_cls = {c: [] for c in range(n_classes)}

    for im in images:
        keep = im.dt_conf >= conf_thr if len(im.dt_conf) else np.zeros(0, bool)
        dcls = im.dt_cls[keep] if len(im.dt_cls) else np.zeros(0, int)
        g_tot.append(len(im.gt_cls))
        p_tot.append(int(keep.sum()))
        for c in range(n_classes):
            g_cls[c].append(int((im.gt_cls == c).sum()))
            p_cls[c].append(int((dcls == c).sum()))

    def block(g, p):
        g = np.asarray(g, float)
        p = np.asarray(p, float)
        if len(g) == 0:
            return {k: float("nan") for k in
                    ("MAE", "RMSE", "sMAPE", "ME", "n_images",
                     "true_total", "pred_total", "r")}
        err = p - g
        denom = np.abs(p) + np.abs(g)
        smape = np.where(denom > 0, 200.0 * np.abs(err) / np.maximum(denom, 1e-12), 0.0)
        r = float("nan")
        if len(g) > 1 and g.std() > 0 and p.std() > 0:
            r = float(np.corrcoef(g, p)[0, 1])
        return {
            "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "sMAPE": float(smape.mean()),
            # Signed mean error. MAE hides BIAS: telling a microbiologist
            # "the model systematically undercounts" is more useful than
            # "MAE is 4.2".  + means overcounting.
            "ME": float(err.mean()),
            "n_images": int(len(g)),
            "true_total": int(g.sum()),
            "pred_total": int(p.sum()),
            "r": r,
        }

    res = {"conf_thr": float(conf_thr), "total": block(g_tot, p_tot), "per_class": {}}
    for c in range(n_classes):
        res["per_class"][CLASS_ORDER[c]] = block(g_cls[c], p_cls[c])
    return res


def select_conf_threshold(images, grid=None, metric="MAE",
                          n_classes=len(CLASS_ORDER)):
    """
    Find the confidence threshold that minimises counting error ON VAL.
    The returned value goes into the protocol; it is never re-searched on test.
    """
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.91, 0.05), 2)
    best, best_v = None, np.inf
    curve = []
    for t in grid:
        m = counting_metrics(images, float(t), n_classes)["total"][metric]
        curve.append((float(t), float(m)))
        if m < best_v:
            best, best_v = float(t), float(m)
    return best, curve
