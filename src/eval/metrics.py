#!/usr/bin/env python3
"""
Olcum cekirdegi -- Ultralytics'ten BAGIMSIZ.

Neden kendi implementasyonumuz var:
  1. Ultralytics'in val() ciktisi boyut bazli AP (small/medium/large) vermiyor.
     Makalenin ana argumani kucuk koloniler uzerinde; bu kirilim sart.
  2. MAE / sMAPE (koloni sayimi) Ultralytics'te hic yok.
  3. Ikinci detektor kontrolu (RT-DETR, YOLO11) icin AYNI olcum kodunun
     calismasi gerekiyor. Detektore bagli bir metrik, detektorler arasi
     karsilastirmayi gecersiz kilar.
  4. Ana metrik "kutunun icinde" olmali -- protokol donduruldugunda
     olcum kodu da donar.

AP hesabi COCO ile birebir ayni algoritma (cocoeval.evaluateImg/accumulate):
  - IoU esikleri 0.50:0.05:0.95 (10 esik)
  - 101 noktali interpolasyon
  - alan araligi disindaki GT'ler "ignore", onlarla eslesen tespitler de ignore
  - eslesmeyen ve alan araligi disinda kalan tespitler de ignore

Dogrulama: src/eval/test_metrics.py  (pycocotools varsa ona karsi da kiyaslar)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Sabitler -- PROTOKOLUN PARCASI, degistirilirse decisions.md'ye not dusulecek
# ---------------------------------------------------------------------------

CLASS_ORDER = ["S.aureus", "B.subtilis", "P.aeruginosa", "E.coli", "C.albicans"]

IOU_THRS = np.linspace(0.5, 0.95, 10)          # COCO standardi
REC_THRS = np.linspace(0.0, 1.0, 101)          # 101 noktali interpolasyon

# COCO alan araliklari (ORIJINAL goruntu pikseli cinsinden, imgsz degil).
# AGAR icin anlamli: S.aureus medyan 29px -> 841px^2 (small),
# E.coli medyan 128px -> 16384px^2 (large).
AREA_RANGES_COCO = {
    "all":    (0.0, 1e10),
    "small":  (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large":  (96.0 ** 2, 1e10),
}


# ---------------------------------------------------------------------------
# Veri yapilari
# ---------------------------------------------------------------------------

@dataclass
class ImageAnno:
    """Tek goruntunun gercek kutulari + tespitleri. Kutular xyxy, MUTLAK piksel."""
    stem: str
    width: int
    height: int
    gt_cls: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    gt_box: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    dt_cls: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    dt_box: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    dt_conf: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------------------
# G/C okuma
# ---------------------------------------------------------------------------

def read_yolo_txt(path: Path, W: int, H: int, with_conf: bool = False):
    """
    YOLO formati -> (cls[N], xyxy[N,4], conf[N] | None), MUTLAK piksel.
    Satir: cls xc yc w h [conf]   (hepsi normalize)
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
# COCO evaluateImg esdegeri: tek (goruntu, sinif, alan araligi) icin eslestirme
# ---------------------------------------------------------------------------

def _match_one(gt_box, dt_box, dt_conf, area_rng, iou_thrs):
    """
    COCO cocoeval.evaluateImg ile birebir ayni eslestirme.
    Doner: dt_conf_sirali, dtm[T,D] (eslesme var mi), dtIg[T,D], npig (sayilan GT)
    """
    T = len(iou_thrs)
    G, D = len(gt_box), len(dt_box)

    gt_ig = np.zeros(G, dtype=bool)
    if G:
        ga = box_area(gt_box)
        gt_ig = (ga < area_rng[0]) | (ga > area_rng[1])
        # COCO: ignore olmayan GT'ler once gelsin (eslestirme onlari tercih etsin)
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
                if gtm[gi] >= 0:                       # bu GT zaten alindi
                    continue
                if m > -1 and (not gt_ig[m]) and gt_ig[gi]:
                    break                              # ignore'a dusmeden dur
                if ious[di, gi] < best_iou:
                    continue
                best_iou = ious[di, gi]
                m = gi
            if m == -1:
                continue
            dtm[ti, di] = True
            dtIg[ti, di] = gt_ig[m]
            gtm[m] = di

    # eslesmeyen ve alan araligi disinda kalan tespitler de sayilmaz
    da = box_area(dt_box)
    out_of_range = (da < area_rng[0]) | (da > area_rng[1])
    dtIg |= (~dtm) & out_of_range[None, :]

    return dt_conf, dtm, dtIg, npig


# ---------------------------------------------------------------------------
# Ana degerlendirme
# ---------------------------------------------------------------------------

def evaluate_detection(images, n_classes=len(CLASS_ORDER),
                       area_ranges=None, iou_thrs=IOU_THRS):
    """
    images: ImageAnno listesi
    Doner: dict
      ap[alan][sinif_id]        -> IoU esikleri uzerinde ortalama AP
      ap50[alan][sinif_id]      -> IoU 0.50'de AP
      ap75[alan][sinif_id]
      map[alan]                 -> siniflar uzerinde ortalama (GT'si olan siniflar)
      map50[alan], map75[alan]
      n_gt[alan][sinif_id]
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
                continue                       # bu sinif bu alanda yok -> AP tanimsiz

            if confs:
                conf = np.concatenate(confs)
                dtm = np.concatenate(dtms, axis=1)
                dtIg = np.concatenate(dtIgs, axis=1)
                order = np.argsort(-conf, kind="stable")   # TUM goruntuler boyunca
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

                # COCO: precision egrisini monoton azalan hale getir
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
            # GT'si olmayan sinif/alan hucreleri NaN -- "Mean of empty slice"
            # beklenen durum, uyari degil.
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
# Koloni sayimi -- mikrobiyolojide ASIL is bu, mAP vekil olcut
# ---------------------------------------------------------------------------

def counting_metrics(images, conf_thr: float, n_classes=len(CLASS_ORDER)):
    """
    Goruntu basina tahmin edilen koloni sayisi vs gercek.

    DIKKAT: conf_thr TEST kumesinde secilemez. VAL'da secilip donduruldu,
    teste oyle uygulanir. evaluate.py bunu zorunlu tutuyor.

    sMAPE tanimi (simetrik, %0-200 arasi):
        200 * |p - g| / (|p| + |g|),   p = g = 0 ise 0
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
                    ("MAE", "RMSE", "sMAPE", "ME", "n_goruntu",
                     "gercek_toplam", "tahmin_toplam", "r")}
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
            "ME": float(err.mean()),                 # + = fazla sayiyor
            "n_goruntu": int(len(g)),
            "gercek_toplam": int(g.sum()),
            "tahmin_toplam": int(p.sum()),
            "r": r,
        }

    res = {"conf_thr": float(conf_thr), "toplam": block(g_tot, p_tot), "sinif": {}}
    for c in range(n_classes):
        res["sinif"][CLASS_ORDER[c]] = block(g_cls[c], p_cls[c])
    return res


def select_conf_threshold(images, grid=None, metric="MAE",
                          n_classes=len(CLASS_ORDER)):
    """
    VAL kumesinde sayim hatasini en aza indiren conf esigini bulur.
    Donen deger PROTOKOLE yazilir; test kumesinde bir daha aranmaz.
    """
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.91, 0.05), 2)
    best, best_v = None, np.inf
    egri = []
    for t in grid:
        m = counting_metrics(images, float(t), n_classes)["toplam"][metric]
        egri.append((float(t), float(m)))
        if m < best_v:
            best, best_v = float(t), float(m)
    return best, egri
