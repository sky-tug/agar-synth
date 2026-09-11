#!/usr/bin/env python3
"""
Sanity test of the metrics code.

Rule: do not trust a metric you wrote yourself before you have tested it with
cases whose correct answer you know BY HAND. Roadmap Phase 2: "After writing
it, test it with a synthetic sanity case (does it give the correct result on a
count you know by hand)."

Usage:
    python src/eval/test_metrics.py

If pycocotools is installed the AP values are cross-checked against it as well
(optional).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import (  # noqa: E402
    AREA_RANGES_COCO, CLASS_ORDER, ImageAnno, counting_metrics,
    evaluate_detection, iou_matrix, read_yolo_txt, select_conf_threshold,
)

PASSED, FAILED = [], []


# Under pytest a failed check MUST fail the test (decision 3.94).
#
# Before this line existed, `check` only appended to a list and returned. Run
# from the command line `main()` looked at FAILED and exited 1, but pytest does
# not call `main()` -- it calls the `test_*` functions one by one, sees no
# exception, and reports green. The two test files hold 104 checks inside 30
# test functions; "30 passed" meant "30 functions ran without crashing", not
# "104 checks found what they expected". Principle 1, fourth occurrence: the
# rule existed in the code but had no verdict.
_UNDER_PYTEST = "pytest" in sys.modules


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'OK ' if condition else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if _UNDER_PYTEST:
        assert condition, name + (f" -- {detail}" if detail else "")


def box(x, y, w, h):
    return [x, y, x + w, y + h]


# ---------------------------------------------------------------------------

def test_iou():
    print("\n1) IoU")
    a = np.array([box(0, 0, 10, 10)], float)
    b = np.array([box(0, 0, 10, 10), box(5, 0, 10, 10), box(20, 20, 10, 10)], float)
    m = iou_matrix(a, b)
    check("same box -> 1.0", abs(m[0, 0] - 1.0) < 1e-9, f"{m[0,0]:.4f}")
    # half overlap: intersection 50, union 150 -> 1/3
    check("half overlap -> 1/3", abs(m[0, 1] - 1 / 3) < 1e-9, f"{m[0,1]:.4f}")
    check("disjoint -> 0.0", m[0, 2] == 0.0)
    check("empty input does not blow up", iou_matrix(np.zeros((0, 4)), b).shape == (0, 3))


def test_perfect_prediction():
    print("\n2) Perfect prediction -> mAP = 1.0")
    ims = []
    rng = np.random.default_rng(0)
    for i in range(4):
        n = 6
        xy = rng.integers(50, 1900, size=(n, 2))
        bx = np.array([box(x, y, 40, 40) for x, y in xy], float)
        cls = rng.integers(0, 5, size=n)
        ims.append(ImageAnno(f"im{i}", 2048, 2048,
                             gt_cls=cls, gt_box=bx,
                             dt_cls=cls.copy(), dt_box=bx.copy(),
                             dt_conf=np.full(n, 0.9)))
    r = evaluate_detection(ims)
    check("mAP50-95 = 1.0", abs(r["map"]["all"] - 1.0) < 1e-6, f"{r['map']['all']:.6f}")
    check("mAP50 = 1.0", abs(r["map50"]["all"] - 1.0) < 1e-6)


def test_half_detected():
    print("\n3) Half of the GT found, no false alarms at all -> AP ~ 0.5")
    n = 20
    bx = np.array([box(100 * i + 10, 500, 40, 40) for i in range(n)], float)
    cls = np.zeros(n, int)
    im = ImageAnno("a", 4096, 2048, gt_cls=cls, gt_box=bx,
                   dt_cls=cls[:10], dt_box=bx[:10], dt_conf=np.full(10, 0.9))
    r = evaluate_detection([im], n_classes=1)
    ap = r["ap"]["all"][0]
    check("AP ~ 0.5 (recall ceiling 0.5, precision 1.0)",
          abs(ap - 0.5) < 0.02, f"AP={ap:.4f}")


def test_iou_threshold():
    print("\n4) IoU threshold sensitivity")
    # GT 100x100, prediction shifted by 10px -> IoU = 90*100 / (2*10000 - 9000) = 0.818
    g = np.array([box(0, 0, 100, 100)], float)
    d = np.array([box(10, 0, 100, 100)], float)
    iou = iou_matrix(d, g)[0, 0]
    check("expected IoU 0.8182", abs(iou - 0.8181818) < 1e-5, f"{iou:.4f}")
    im = ImageAnno("a", 512, 512, gt_cls=np.zeros(1, int), gt_box=g,
                   dt_cls=np.zeros(1, int), dt_box=d, dt_conf=np.array([0.9]))
    r = evaluate_detection([im], n_classes=1)
    # IoU 0.50..0.80 -> matches (7 thresholds), 0.85/0.90/0.95 -> does not match
    check("AP50 = 1.0", abs(r["ap50"]["all"][0] - 1.0) < 1e-6)
    check("AP50-95 = 7/10", abs(r["ap"]["all"][0] - 0.7) < 1e-6,
          f"{r['ap']['all'][0]:.4f}")


def test_false_alarm():
    print("\n5) Does a false alarm lower the precision")
    g = np.array([box(0, 0, 40, 40)], float)
    d = np.array([box(0, 0, 40, 40), box(500, 500, 40, 40)], float)
    im = ImageAnno("a", 1024, 1024, gt_cls=np.zeros(1, int), gt_box=g,
                   dt_cls=np.zeros(2, int), dt_box=d, dt_conf=np.array([0.9, 0.8]))
    r = evaluate_detection([im], n_classes=1)
    check("a low-scoring FP does not lower AP (the recall ceiling was not exceeded)",
          abs(r["ap50"]["all"][0] - 1.0) < 1e-6, f"{r['ap50']['all'][0]:.4f}")

    im2 = ImageAnno("a", 1024, 1024, gt_cls=np.zeros(1, int), gt_box=g,
                    dt_cls=np.zeros(2, int), dt_box=d[::-1].copy(),
                    dt_conf=np.array([0.9, 0.8]))
    r2 = evaluate_detection([im2], n_classes=1)
    check("a high-scoring FP does lower AP",
          r2["ap50"]["all"][0] < 0.6, f"{r2['ap50']['all'][0]:.4f}")


def test_size_breakdown():
    print("\n6) Size breakdown -- scenario where the small colonies were missed")
    # 10 small (20x20 = 400px^2 -> small), 10 large (150x150 -> large)
    small = np.array([box(100 * i + 10, 100, 20, 20) for i in range(10)], float)
    large = np.array([box(100 * i + 10, 600, 150, 150) for i in range(10)], float)
    gt = np.vstack([small, large])
    cls = np.zeros(20, int)
    # only the large ones were found
    im = ImageAnno("a", 2048, 2048, gt_cls=cls, gt_box=gt,
                   dt_cls=np.zeros(10, int), dt_box=large.copy(),
                   dt_conf=np.full(10, 0.9))
    r = evaluate_detection([im], n_classes=1)
    check("small AP = 0", abs(r["ap"]["small"][0] - 0.0) < 1e-6,
          f"{r['ap']['small'][0]:.4f}")
    check("large AP = 1", abs(r["ap"]["large"][0] - 1.0) < 1e-6,
          f"{r['ap']['large'][0]:.4f}")
    check("all AP ~ 0.5", abs(r["ap"]["all"][0] - 0.5) < 0.02,
          f"{r['ap']['all'][0]:.4f}")
    check("small GT count 10", r["n_gt"]["small"][0] == 10)
    check("large GT count 10", r["n_gt"]["large"][0] == 10)


def test_counting():
    print("\n7) Counting metrics -- a case computed by hand")
    # 3 images: ground truth 10 / 20 / 30, prediction 12 / 18 / 30
    ims = []
    for gn, pn in [(10, 12), (20, 18), (30, 30)]:
        gb = np.array([box(10 * i, 10, 5, 5) for i in range(gn)], float)
        db = np.array([box(10 * i, 10, 5, 5) for i in range(pn)], float)
        ims.append(ImageAnno("x", 2048, 2048,
                             gt_cls=np.zeros(gn, int), gt_box=gb,
                             dt_cls=np.zeros(pn, int), dt_box=db,
                             dt_conf=np.full(pn, 0.9)))
    m = counting_metrics(ims, conf_thr=0.25, n_classes=1)["total"]
    # MAE = (2 + 2 + 0)/3 = 1.3333
    check("MAE = 4/3", abs(m["MAE"] - 4 / 3) < 1e-9, f"{m['MAE']:.4f}")
    # RMSE = sqrt((4+4+0)/3) = 1.6330
    check("RMSE = sqrt(8/3)", abs(m["RMSE"] - np.sqrt(8 / 3)) < 1e-9,
          f"{m['RMSE']:.4f}")
    # sMAPE = mean(200*2/22, 200*2/38, 0) = mean(18.1818, 10.5263, 0) = 9.5694
    expected = (200 * 2 / 22 + 200 * 2 / 38 + 0) / 3
    check("sMAPE same as computed by hand", abs(m["sMAPE"] - expected) < 1e-9,
          f"{m['sMAPE']:.4f} (expected {expected:.4f})")
    # ME = (2 - 2 + 0)/3 = 0  -> no bias
    check("ME = 0 (no bias)", abs(m["ME"]) < 1e-9)
    check("the totals are correct",
          m["true_total"] == 60 and m["pred_total"] == 60)


def test_counting_empty():
    print("\n8) Counting -- zero/zero and one-sided zero")
    empty = ImageAnno("b", 512, 512)
    m = counting_metrics([empty], conf_thr=0.25, n_classes=1)["total"]
    check("0 vs 0 -> sMAPE 0 (not NaN)", m["sMAPE"] == 0.0)
    check("0 vs 0 -> MAE 0", m["MAE"] == 0.0)

    one_sided = ImageAnno("c", 512, 512,
                          gt_cls=np.zeros(0, int), gt_box=np.zeros((0, 4)),
                          dt_cls=np.zeros(3, int),
                          dt_box=np.array([box(i * 10, 10, 5, 5) for i in range(3)], float),
                          dt_conf=np.full(3, 0.9))
    m2 = counting_metrics([one_sided], conf_thr=0.25, n_classes=1)["total"]
    check("ground truth 0, prediction 3 -> sMAPE 200 (upper bound)",
          abs(m2["sMAPE"] - 200) < 1e-9, f"{m2['sMAPE']:.1f}")


def test_conf_threshold():
    print("\n9) Does the conf threshold selection find the right point on VAL")
    # 20 real colonies; 20 correct predictions at conf=0.9, 30 false alarms at conf=0.3
    gb = np.array([box(50 * i, 100, 20, 20) for i in range(20)], float)
    fp = np.array([box(50 * i, 900, 20, 20) for i in range(30)], float)
    im = ImageAnno("v", 2048, 2048,
                   gt_cls=np.zeros(20, int), gt_box=gb,
                   dt_cls=np.zeros(50, int), dt_box=np.vstack([gb, fp]),
                   dt_conf=np.concatenate([np.full(20, 0.9), np.full(30, 0.3)]))
    best, curve = select_conf_threshold([im], n_classes=1)
    check("the threshold goes above 0.3", best > 0.3, f"selected={best}")
    m = counting_metrics([im], best, n_classes=1)["total"]
    check("MAE = 0 at the selected threshold", m["MAE"] == 0.0)


def test_read_yolo_txt():
    """
    Decision 3.38: the file reader was NOT being tested at all.
    All the other 27 checks build ImageAnno directly with PIXEL boxes, i.e. they
    skip the only place where the normalized -> pixel conversion lives. If W and
    H were swapped there, or the conf column shifted, no test would have caught
    it. W != H was chosen deliberately -- on a square image a W/H swap would be
    invisible.
    """
    print("\n10) read_yolo_txt -- normalized <-> pixel conversion")
    import tempfile
    W, H = 800, 400                      # W != H : makes the swap bug visible

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.txt"
        # center (0.5, 0.5), width 0.25, height 0.50
        # -> pixels: x 300..500 , y 100..300
        p.write_text("2 0.5 0.5 0.25 0.5\n", encoding="utf-8")
        cls, bx, conf = read_yolo_txt(p, W, H)
        check("class id is correct", cls.tolist() == [2], f"{cls.tolist()}")
        check("xyxy pixels correct (W!=H)",
              np.allclose(bx[0], [300.0, 100.0, 500.0, 300.0]),
              f"{bx[0].tolist()}")
        check("conf is None when not requested", conf is None)

        # does a W/H swap really give a different result (is the test meaningful)
        _, box_swapped, _ = read_yolo_txt(p, H, W)
        check("a W/H swap produces a different box (the test is sensitive)",
              not np.allclose(bx[0], box_swapped[0]))

        # reading with conf
        p2 = Path(td) / "b.txt"
        p2.write_text("0 0.5 0.5 0.1 0.1 0.87\n"
                      "1 0.25 0.75 0.1 0.1\n", encoding="utf-8")   # no conf on line 2
        cls2, box2, conf2 = read_yolo_txt(p2, W, H, with_conf=True)
        check("conf was read", abs(conf2[0] - 0.87) < 1e-9, f"{conf2[0]}")
        check("1.0 is assumed when there is no conf", conf2[1] == 1.0, f"{conf2[1]}")

        # round trip: pixel -> normalized -> pixel
        xc = (box2[0][0] + box2[0][2]) / 2 / W
        yc = (box2[0][1] + box2[0][3]) / 2 / H
        check("the round trip preserves the center",
              abs(xc - 0.5) < 1e-9 and abs(yc - 0.5) < 1e-9, f"({xc:.6f}, {yc:.6f})")

        # malformed and incomplete inputs
        p3 = Path(td) / "c.txt"
        p3.write_text("\n0 0.5 0.5\n0 0.5 0.5 0.1 0.1\n\n", encoding="utf-8")
        cls3, box3, _ = read_yolo_txt(p3, W, H)
        check("a line with missing fields is skipped", len(cls3) == 1, f"{len(cls3)} lines")

        cls4, box4, conf4 = read_yolo_txt(Path(td) / "missing.txt", W, H, with_conf=True)
        check("returns empty when the file does not exist",
              len(cls4) == 0 and box4.shape == (0, 4) and len(conf4) == 0)


def test_pycocotools():
    print("\n11) cross-validation against pycocotools (optional)")
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("  [SKIP] pycocotools is not installed "
              "(with pip install pycocotools this test runs too)")
        return

    rng = np.random.default_rng(7)
    ims, coco_imgs, coco_anns, coco_dts = [], [], [], []
    ann_id = 1
    for i in range(6):
        n = int(rng.integers(5, 15))
        xy = rng.integers(20, 1900, size=(n, 2))
        wh = rng.integers(15, 200, size=(n, 2))
        gb = np.array([box(x, y, w, h) for (x, y), (w, h) in zip(xy, wh)], float)
        gc = rng.integers(0, 3, size=n)
        # predictions: shift some, drop some, add a few FPs
        keep = rng.random(n) > 0.25
        db = gb[keep] + rng.normal(0, 6, size=(int(keep.sum()), 4))
        dc = gc[keep]
        ds = rng.uniform(0.3, 0.99, size=int(keep.sum()))
        nfp = int(rng.integers(0, 4))
        if nfp:
            fx = rng.integers(20, 1900, size=(nfp, 2))
            db = np.vstack([db, np.array([box(x, y, 40, 40) for x, y in fx], float)])
            dc = np.concatenate([dc, rng.integers(0, 3, size=nfp)])
            ds = np.concatenate([ds, rng.uniform(0.1, 0.5, size=nfp)])

        ims.append(ImageAnno(f"i{i}", 2048, 2048, gt_cls=gc, gt_box=gb,
                             dt_cls=dc, dt_box=db, dt_conf=ds))

        coco_imgs.append({"id": i, "width": 2048, "height": 2048})
        for c, b in zip(gc, gb):
            coco_anns.append({"id": ann_id, "image_id": i, "category_id": int(c),
                              "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                              "area": float((b[2] - b[0]) * (b[3] - b[1])),
                              "iscrowd": 0})
            ann_id += 1
        for c, b, s in zip(dc, db, ds):
            coco_dts.append({"image_id": i, "category_id": int(c),
                             "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                             "score": float(s)})

    gt_json = {"images": coco_imgs, "annotations": coco_anns,
               "categories": [{"id": c, "name": str(c)} for c in range(3)]}

    import contextlib
    import io
    coco = COCO()
    coco.dataset = gt_json
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
        dt = coco.loadRes(coco_dts)
        E = COCOeval(coco, dt, "bbox")
        E.evaluate(); E.accumulate(); E.summarize()

    ours = evaluate_detection(ims, n_classes=3)
    pairs = [("mAP50-95", ours["map"]["all"], E.stats[0]),
             ("mAP50", ours["map50"]["all"], E.stats[1]),
             ("mAP75", ours["map75"]["all"], E.stats[2]),
             ("mAP small", ours["map"]["small"], E.stats[3]),
             ("mAP medium", ours["map"]["medium"], E.stats[4]),
             ("mAP large", ours["map"]["large"], E.stats[5])]
    for name, a, b in pairs:
        if np.isnan(b) or b < 0:
            print(f"  [SKIP] {name}: undefined on the COCO side")
            continue
        check(f"{name} same as COCO", abs(a - b) < 1e-4,
              f"ours={a:.5f} coco={b:.5f}")


def test_min_gt_cell():
    """
    A (class, size) cell with almost no ground truth must not be averaged into
    the size-band mAP (decision 3.94).

    The numbers below are the real large-object row of G50_s2 on val. S.aureus
    has exactly ONE large box there; its AP is the binary outcome of a single
    detection and it was carrying a quarter of mAP_large.
    """
    from src.eval.evaluate import map_with_min_gt   # noqa: E402
    print("\n[min_gt_for_cell]")
    nan = float("nan")
    ap = [0.0, 0.6827, 0.7218, 0.7661, nan]     # C.albicans has no large box
    n_gt = [1, 524, 1962, 3506, 0]

    v, dropped = map_with_min_gt(ap, n_gt, 10)
    check("thin cell is left out", abs(v - 0.723533) < 1e-4, f"got {v:.6f}")
    check("thin cell is reported", dropped == [(0, 1)], str(dropped))

    v0, d0 = map_with_min_gt(ap, n_gt, 0)
    check("min_gt=0 reproduces COCO", abs(v0 - 0.542650) < 1e-4, f"got {v0:.6f}")
    check("min_gt=0 drops nothing", d0 == [], str(d0))

    ve, de = map_with_min_gt([nan] * 5, [0] * 5, 10)
    check("a band with no boxes is nan", ve != ve, str(ve))
    check("an empty cell is not 'dropped'", de == [], str(de))

    vk, dk = map_with_min_gt([0.5, 0.6], [100, 200], 10)
    check("every cell above the floor", abs(vk - 0.55) < 1e-9, f"got {vk:.6f}")
    check("nothing dropped when all pass", dk == [], str(dk))


def main():
    print("=" * 62)
    print("SANITY TEST OF THE METRICS CODE")
    print("=" * 62)
    for f in (test_iou, test_perfect_prediction, test_half_detected, test_iou_threshold,
              test_false_alarm, test_size_breakdown, test_counting, test_counting_empty,
              test_conf_threshold, test_read_yolo_txt, test_min_gt_cell,
              test_pycocotools):
        f()
    print("\n" + "=" * 62)
    print(f"PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
    if FAILED:
        for k in FAILED:
            print(f"  ! {k}")
        print("=" * 62)
        sys.exit(1)
    print("The metrics code is reliable. You can move on to the baseline.")
    print("=" * 62)


if __name__ == "__main__":
    main()
