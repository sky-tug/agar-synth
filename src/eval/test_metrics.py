#!/usr/bin/env python3
"""
Olcum kodunun sanity testi.

Kural: kendi yazdigin metrige, dogru cevabini ELLE bildigin vakalarla
guvenmeden once inanma. Yol haritasi Faz 2: "Yazdiktan sonra sentetik bir
sanity vaka ile test et (elle bildigin bir sayim uzerinde dogru sonucu
veriyor mu)."

Kullanim:
    python src/eval/test_metrics.py

pycocotools kuruluysa AP degerleri ona karsi da kiyaslanir (opsiyonel).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import (  # noqa: E402
    AREA_RANGES_COCO, CLASS_ORDER, ImageAnno, counting_metrics,
    evaluate_detection, iou_matrix, select_conf_threshold,
)

GECTI, KALDI = [], []


def kontrol(ad, kosul, detay=""):
    (GECTI if kosul else KALDI).append(ad)
    print(f"  [{'OK ' if kosul else 'HATA'}] {ad}" + (f"   {detay}" if detay else ""))


def kutu(x, y, w, h):
    return [x, y, x + w, y + h]


# ---------------------------------------------------------------------------

def test_iou():
    print("\n1) IoU")
    a = np.array([kutu(0, 0, 10, 10)], float)
    b = np.array([kutu(0, 0, 10, 10), kutu(5, 0, 10, 10), kutu(20, 20, 10, 10)], float)
    m = iou_matrix(a, b)
    kontrol("ayni kutu -> 1.0", abs(m[0, 0] - 1.0) < 1e-9, f"{m[0,0]:.4f}")
    # yarim ortusme: kesisim 50, birlesim 150 -> 1/3
    kontrol("yarim ortusme -> 1/3", abs(m[0, 1] - 1 / 3) < 1e-9, f"{m[0,1]:.4f}")
    kontrol("ayrik -> 0.0", m[0, 2] == 0.0)
    kontrol("bos girdi patlamiyor", iou_matrix(np.zeros((0, 4)), b).shape == (0, 3))


def test_mukemmel_tahmin():
    print("\n2) Mukemmel tahmin -> mAP = 1.0")
    ims = []
    rng = np.random.default_rng(0)
    for i in range(4):
        n = 6
        xy = rng.integers(50, 1900, size=(n, 2))
        box = np.array([kutu(x, y, 40, 40) for x, y in xy], float)
        cls = rng.integers(0, 5, size=n)
        ims.append(ImageAnno(f"im{i}", 2048, 2048,
                             gt_cls=cls, gt_box=box,
                             dt_cls=cls.copy(), dt_box=box.copy(),
                             dt_conf=np.full(n, 0.9)))
    r = evaluate_detection(ims)
    kontrol("mAP50-95 = 1.0", abs(r["map"]["all"] - 1.0) < 1e-6, f"{r['map']['all']:.6f}")
    kontrol("mAP50 = 1.0", abs(r["map50"]["all"] - 1.0) < 1e-6)


def test_yarim_tespit():
    print("\n3) GT'nin yarisi bulunmus, hic yanlis alarm yok -> AP ~ 0.5")
    n = 20
    box = np.array([kutu(100 * i + 10, 500, 40, 40) for i in range(n)], float)
    cls = np.zeros(n, int)
    im = ImageAnno("a", 4096, 2048, gt_cls=cls, gt_box=box,
                   dt_cls=cls[:10], dt_box=box[:10], dt_conf=np.full(10, 0.9))
    r = evaluate_detection([im], n_classes=1)
    ap = r["ap"]["all"][0]
    kontrol("AP ~ 0.5 (recall tavani 0.5, precision 1.0)",
            abs(ap - 0.5) < 0.02, f"AP={ap:.4f}")


def test_iou_esigi():
    print("\n4) IoU esigi duyarliligi")
    # GT 100x100, tahmin 10px kaymis -> IoU = 90*100 / (2*10000 - 9000) = 0.818
    g = np.array([kutu(0, 0, 100, 100)], float)
    d = np.array([kutu(10, 0, 100, 100)], float)
    iou = iou_matrix(d, g)[0, 0]
    kontrol("beklenen IoU 0.8182", abs(iou - 0.8181818) < 1e-5, f"{iou:.4f}")
    im = ImageAnno("a", 512, 512, gt_cls=np.zeros(1, int), gt_box=g,
                   dt_cls=np.zeros(1, int), dt_box=d, dt_conf=np.array([0.9]))
    r = evaluate_detection([im], n_classes=1)
    # IoU 0.50..0.80 -> eslesir (7 esik), 0.85/0.90/0.95 -> eslesmez
    kontrol("AP50 = 1.0", abs(r["ap50"]["all"][0] - 1.0) < 1e-6)
    kontrol("AP50-95 = 7/10", abs(r["ap"]["all"][0] - 0.7) < 1e-6,
            f"{r['ap']['all'][0]:.4f}")


def test_yanlis_alarm():
    print("\n5) Yanlis alarm precision'i dusuruyor mu")
    g = np.array([kutu(0, 0, 40, 40)], float)
    d = np.array([kutu(0, 0, 40, 40), kutu(500, 500, 40, 40)], float)
    im = ImageAnno("a", 1024, 1024, gt_cls=np.zeros(1, int), gt_box=g,
                   dt_cls=np.zeros(2, int), dt_box=d, dt_conf=np.array([0.9, 0.8]))
    r = evaluate_detection([im], n_classes=1)
    kontrol("dusuk skorlu FP AP'yi dusurmuyor (recall tavani asilmadi)",
            abs(r["ap50"]["all"][0] - 1.0) < 1e-6, f"{r['ap50']['all'][0]:.4f}")

    im2 = ImageAnno("a", 1024, 1024, gt_cls=np.zeros(1, int), gt_box=g,
                    dt_cls=np.zeros(2, int), dt_box=d[::-1].copy(),
                    dt_conf=np.array([0.9, 0.8]))
    r2 = evaluate_detection([im2], n_classes=1)
    kontrol("yuksek skorlu FP AP'yi dusuruyor",
            r2["ap50"]["all"][0] < 0.6, f"{r2['ap50']['all'][0]:.4f}")


def test_boyut_kirilimi():
    print("\n6) Boyut kirilimi -- kucuk koloniler kacirilmis senaryosu")
    # 10 kucuk (20x20 = 400px^2 -> small), 10 buyuk (150x150 -> large)
    kucuk = np.array([kutu(100 * i + 10, 100, 20, 20) for i in range(10)], float)
    buyuk = np.array([kutu(100 * i + 10, 600, 150, 150) for i in range(10)], float)
    gt = np.vstack([kucuk, buyuk])
    cls = np.zeros(20, int)
    # sadece buyukleri bulmus
    im = ImageAnno("a", 2048, 2048, gt_cls=cls, gt_box=gt,
                   dt_cls=np.zeros(10, int), dt_box=buyuk.copy(),
                   dt_conf=np.full(10, 0.9))
    r = evaluate_detection([im], n_classes=1)
    kontrol("small AP = 0", abs(r["ap"]["small"][0] - 0.0) < 1e-6,
            f"{r['ap']['small'][0]:.4f}")
    kontrol("large AP = 1", abs(r["ap"]["large"][0] - 1.0) < 1e-6,
            f"{r['ap']['large'][0]:.4f}")
    kontrol("all AP ~ 0.5", abs(r["ap"]["all"][0] - 0.5) < 0.02,
            f"{r['ap']['all'][0]:.4f}")
    kontrol("small GT sayisi 10", r["n_gt"]["small"][0] == 10)
    kontrol("large GT sayisi 10", r["n_gt"]["large"][0] == 10)


def test_sayim():
    print("\n7) Sayim metrikleri -- elle hesaplanan vaka")
    # 3 goruntu: gercek 10 / 20 / 30, tahmin 12 / 18 / 30
    ims = []
    for gn, pn in [(10, 12), (20, 18), (30, 30)]:
        gb = np.array([kutu(10 * i, 10, 5, 5) for i in range(gn)], float)
        db = np.array([kutu(10 * i, 10, 5, 5) for i in range(pn)], float)
        ims.append(ImageAnno("x", 2048, 2048,
                             gt_cls=np.zeros(gn, int), gt_box=gb,
                             dt_cls=np.zeros(pn, int), dt_box=db,
                             dt_conf=np.full(pn, 0.9)))
    m = counting_metrics(ims, conf_thr=0.25, n_classes=1)["toplam"]
    # MAE = (2 + 2 + 0)/3 = 1.3333
    kontrol("MAE = 4/3", abs(m["MAE"] - 4 / 3) < 1e-9, f"{m['MAE']:.4f}")
    # RMSE = sqrt((4+4+0)/3) = 1.6330
    kontrol("RMSE = sqrt(8/3)", abs(m["RMSE"] - np.sqrt(8 / 3)) < 1e-9,
            f"{m['RMSE']:.4f}")
    # sMAPE = mean(200*2/22, 200*2/38, 0) = mean(18.1818, 10.5263, 0) = 9.5694
    bek = (200 * 2 / 22 + 200 * 2 / 38 + 0) / 3
    kontrol("sMAPE elle hesapla ayni", abs(m["sMAPE"] - bek) < 1e-9,
            f"{m['sMAPE']:.4f} (beklenen {bek:.4f})")
    # ME = (2 - 2 + 0)/3 = 0  -> yanlilik yok
    kontrol("ME = 0 (yanlilik yok)", abs(m["ME"]) < 1e-9)
    kontrol("toplamlar dogru",
            m["gercek_toplam"] == 60 and m["tahmin_toplam"] == 60)


def test_sayim_bos():
    print("\n8) Sayim -- sifir/sifir ve tek tarafli sifir")
    bos = ImageAnno("b", 512, 512)
    m = counting_metrics([bos], conf_thr=0.25, n_classes=1)["toplam"]
    kontrol("0 vs 0 -> sMAPE 0 (NaN degil)", m["sMAPE"] == 0.0)
    kontrol("0 vs 0 -> MAE 0", m["MAE"] == 0.0)

    tek = ImageAnno("c", 512, 512,
                    gt_cls=np.zeros(0, int), gt_box=np.zeros((0, 4)),
                    dt_cls=np.zeros(3, int),
                    dt_box=np.array([kutu(i * 10, 10, 5, 5) for i in range(3)], float),
                    dt_conf=np.full(3, 0.9))
    m2 = counting_metrics([tek], conf_thr=0.25, n_classes=1)["toplam"]
    kontrol("gercek 0, tahmin 3 -> sMAPE 200 (ust sinir)", abs(m2["sMAPE"] - 200) < 1e-9,
            f"{m2['sMAPE']:.1f}")


def test_conf_esigi():
    print("\n9) conf esigi secimi VAL'da dogru noktayi buluyor mu")
    # 20 gercek koloni; 20 dogru tahmin conf=0.9, 30 yanlis alarm conf=0.3
    gb = np.array([kutu(50 * i, 100, 20, 20) for i in range(20)], float)
    fp = np.array([kutu(50 * i, 900, 20, 20) for i in range(30)], float)
    im = ImageAnno("v", 2048, 2048,
                   gt_cls=np.zeros(20, int), gt_box=gb,
                   dt_cls=np.zeros(50, int), dt_box=np.vstack([gb, fp]),
                   dt_conf=np.concatenate([np.full(20, 0.9), np.full(30, 0.3)]))
    best, egri = select_conf_threshold([im], n_classes=1)
    kontrol("esik 0.3'un ustune cikiyor", best > 0.3, f"secilen={best}")
    m = counting_metrics([im], best, n_classes=1)["toplam"]
    kontrol("secilen esikte MAE = 0", m["MAE"] == 0.0)


def test_pycocotools():
    print("\n10) pycocotools ile capraz dogrulama (opsiyonel)")
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("  [ATLA] pycocotools kurulu degil "
              "(pip install pycocotools ile bu test de kosar)")
        return

    rng = np.random.default_rng(7)
    ims, coco_imgs, coco_anns, coco_dts = [], [], [], []
    ann_id = 1
    for i in range(6):
        n = int(rng.integers(5, 15))
        xy = rng.integers(20, 1900, size=(n, 2))
        wh = rng.integers(15, 200, size=(n, 2))
        gb = np.array([kutu(x, y, w, h) for (x, y), (w, h) in zip(xy, wh)], float)
        gc = rng.integers(0, 3, size=n)
        # tahminler: bazilarini kaydir, bazilarini at, birkac FP ekle
        tut = rng.random(n) > 0.25
        db = gb[tut] + rng.normal(0, 6, size=(int(tut.sum()), 4))
        dc = gc[tut]
        ds = rng.uniform(0.3, 0.99, size=int(tut.sum()))
        nfp = int(rng.integers(0, 4))
        if nfp:
            fx = rng.integers(20, 1900, size=(nfp, 2))
            db = np.vstack([db, np.array([kutu(x, y, 40, 40) for x, y in fx], float)])
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

    bizim = evaluate_detection(ims, n_classes=3)
    esler = [("mAP50-95", bizim["map"]["all"], E.stats[0]),
             ("mAP50", bizim["map50"]["all"], E.stats[1]),
             ("mAP75", bizim["map75"]["all"], E.stats[2]),
             ("mAP small", bizim["map"]["small"], E.stats[3]),
             ("mAP medium", bizim["map"]["medium"], E.stats[4]),
             ("mAP large", bizim["map"]["large"], E.stats[5])]
    for ad, a, b in esler:
        if np.isnan(b) or b < 0:
            print(f"  [ATLA] {ad}: COCO tarafinda tanimsiz")
            continue
        kontrol(f"{ad} COCO ile ayni", abs(a - b) < 1e-4,
                f"bizim={a:.5f} coco={b:.5f}")


def main():
    print("=" * 62)
    print("OLCUM KODU SANITY TESTI")
    print("=" * 62)
    for f in (test_iou, test_mukemmel_tahmin, test_yarim_tespit, test_iou_esigi,
              test_yanlis_alarm, test_boyut_kirilimi, test_sayim, test_sayim_bos,
              test_conf_esigi, test_pycocotools):
        f()
    print("\n" + "=" * 62)
    print(f"GECTI: {len(GECTI)}   KALDI: {len(KALDI)}")
    if KALDI:
        for k in KALDI:
            print(f"  ! {k}")
        print("=" * 62)
        sys.exit(1)
    print("Olcum kodu guvenilir. Baseline'a gecebilirsin.")
    print("=" * 62)


if __name__ == "__main__":
    main()
