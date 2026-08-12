#!/usr/bin/env python3
"""
Degerlendirme CLI'si -- HER kosuda ayni sekilde calisir.

Ne uretir (runs/<name>/eval/ altina):
  ozet.json          tek satirlik makine-okunur sonuc (grid tablosu bundan kurulur)
  sinif_ap.csv       sinif x (AP, AP50, AP75) + GT sayilari
  boyut_ap.csv       all / small / medium / large kirilimi
  sayim.csv          MAE / RMSE / sMAPE / ME / r  -- toplam ve sinif bazli
  conf_egrisi.csv    VAL'da conf esigi taramasi (sadece --split val ise)

Iki calisma bicimi:
  A) --weights ile: Ultralytics modelini yukler, tahminleri kendisi uretir
  B) --pred-dir ile: hazir YOLO-format tahmin dosyalarindan okur
     (ikinci detektor / dis model karsilastirmasi icin -- olcum kodu ayni kalir)

conf esigi kurali (SIZINTI ONLEME):
  Sayim metrikleri bir conf esigi gerektirir. Bu esik TEST'te aranamaz.
    1) once  --split val  ile kosturulur  -> secilen esik ozet.json'a yazilir
    2) sonra --split test --conf-thr <o deger>  ile kosturulur
  --split test iken --conf-thr verilmezse program HATA verip durur.

Kullanim:
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import (  # noqa: E402
    AREA_RANGES_COCO, CLASS_ORDER, ImageAnno, counting_metrics,
    evaluate_detection, read_yolo_txt, select_conf_threshold,
)

# Tahminler bu esigin ALTINDA hic kaydedilmez. mAP icin dusuk tutulur
# (COCO da tum tespitleri kullanir); sayim esigi ayrica uygulanir.
MIN_CONF = 0.001
MAX_DET = 1000          # AGAR'da bir plakta 125'e kadar koloni var -> bol tut


def stems_of(data: Path, split: str) -> list[str]:
    p = Path("splits") / f"{split}.txt"
    if not p.exists():
        sys.exit(f"HATA: {p} yok. once  python src/make_splits.py --data {data}")
    s = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not s:
        sys.exit(f"HATA: {p} bos. Demo pakette val/test bos cikar -- tam veri gerekli.")
    return s


def image_path_of(data: Path, stem: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        c = data / "images" / f"{stem}{ext}"
        if c.exists():
            return c
    sys.exit(f"HATA: {stem} icin goruntu bulunamadi ({data / 'images'})")


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
    """Hazir tahminler: <stem>.txt, satir = cls xc yc w h conf (normalize)."""
    for im in images:
        dc, db, ds = read_yolo_txt(pred_dir / f"{im.stem}.txt",
                                   im.width, im.height, with_conf=True)
        im.dt_cls, im.dt_box, im.dt_conf = dc, db, ds


def fill_preds_from_model(images, data: Path, weights: str, imgsz: int,
                          device: str, batch: int, half: bool):
    from ultralytics import YOLO
    model = YOLO(weights)
    paths = [str(image_path_of(data, im.stem)) for im in images]

    t0 = time.perf_counter()
    boxes_all = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        res = model.predict(chunk, imgsz=imgsz, conf=MIN_CONF, iou=0.7,
                            max_det=MAX_DET, device=device, half=half,
                            verbose=False, stream=False)
        boxes_all.extend(res)
    cikarim_s = time.perf_counter() - t0

    for im, r in zip(images, boxes_all):
        b = r.boxes
        if b is None or len(b) == 0:
            im.dt_cls = np.zeros(0, int)
            im.dt_box = np.zeros((0, 4))
            im.dt_conf = np.zeros(0)
            continue
        im.dt_cls = b.cls.cpu().numpy().astype(int)
        im.dt_box = b.xyxy.cpu().numpy().astype(float)
        im.dt_conf = b.conf.cpu().numpy().astype(float)

    return {"cikarim_saniye": round(cikarim_s, 3),
            "goruntu_basina_ms": round(1000 * cikarim_s / max(len(images), 1), 2)}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--weights", help="Ultralytics .pt")
    ap.add_argument("--pred-dir", help="hazir YOLO-format tahmin klasoru")
    ap.add_argument("--out", required=True, help="cikti klasoru")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--conf-thr", type=float, default=None,
                    help="sayim esigi. --split test icin ZORUNLU (val'da secilmis olan)")
    ap.add_argument("--etiket", default="", help="ozet.json'a yazilacak kosu adi")
    args = ap.parse_args()

    if bool(args.weights) == bool(args.pred_dir):
        sys.exit("HATA: --weights VEYA --pred-dir ver (ikisi birden degil).")
    if args.split == "test" and args.conf_thr is None:
        sys.exit("HATA: test kumesinde conf esigi ARANAMAZ.\n"
                 "  once: --split val   (esigi secer)\n"
                 "  sonra: --split test --conf-thr <secilen deger>")

    data = Path(args.data).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    stems = stems_of(data, args.split)
    print(f"{args.split}: {len(stems)} goruntu")

    images = load_gt(data, stems)
    zaman = {}
    if args.pred_dir:
        fill_preds_from_dir(images, Path(args.pred_dir).resolve())
    else:
        zaman = fill_preds_from_model(images, data, args.weights, args.imgsz,
                                      args.device, args.batch, args.half)

    # ---------------- Tespit metrikleri ----------------
    det = evaluate_detection(images, n_classes=len(CLASS_ORDER),
                             area_ranges=AREA_RANGES_COCO)

    import pandas as pd

    sinif_tab = pd.DataFrame({
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
    sinif_tab.to_csv(out / "sinif_ap.csv")

    boyut_tab = pd.DataFrame({
        a: {"mAP50-95": det["map"][a], "mAP50": det["map50"][a],
            "mAP75": det["map75"][a], "n_GT": int(det["n_gt"][a].sum())}
        for a in AREA_RANGES_COCO
    }).T.round(4)
    boyut_tab.to_csv(out / "boyut_ap.csv")

    # ---------------- Sayim metrikleri ----------------
    if args.split == "val" and args.conf_thr is None:
        secilen, egri = select_conf_threshold(images)
        pd.DataFrame(egri, columns=["conf_thr", "MAE"]).to_csv(
            out / "conf_egrisi.csv", index=False)
        print(f"\nVAL'da secilen conf esigi: {secilen}  "
              f"-> test kosusunda --conf-thr {secilen} ver")
    else:
        secilen = args.conf_thr

    say = counting_metrics(images, secilen)
    say_tab = pd.DataFrame({"toplam": say["toplam"], **say["sinif"]}).T.round(4)
    say_tab.to_csv(out / "sayim.csv")

    # ---------------- Ozet ----------------
    ozet = {
        "etiket": args.etiket or Path(args.out).parent.name,
        "split": args.split,
        "n_goruntu": len(images),
        "n_gt_kutu": int(det["n_gt"]["all"].sum()),
        "imgsz": args.imgsz,
        "conf_thr_sayim": secilen,
        # --- ana metrik ---
        "mAP50-95": det["map"]["all"],
        "mAP50": det["map50"]["all"],
        "mAP75": det["map75"]["all"],
        "mAP_small": det["map"]["small"],
        "mAP_medium": det["map"]["medium"],
        "mAP_large": det["map"]["large"],
        # --- sinif bazli ana metrik ---
        **{f"AP_{c}": float(det["ap"]["all"][i]) for i, c in enumerate(CLASS_ORDER)},
        # --- sayim ---
        "MAE": say["toplam"]["MAE"],
        "RMSE": say["toplam"]["RMSE"],
        "sMAPE": say["toplam"]["sMAPE"],
        "ME": say["toplam"]["ME"],
        "sayim_r": say["toplam"]["r"],
        **zaman,
    }
    ozet = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in ozet.items()}
    (out / "ozet.json").write_text(json.dumps(ozet, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    # ---------------- Ekrana ----------------
    print("\n=== BOYUT KIRILIMI ===")
    print(boyut_tab.to_string())
    print("\n=== SINIF BAZLI ===")
    print(sinif_tab[["AP50-95", "AP50", "AP_small", "AP_medium", "AP_large",
                     "n_GT"]].to_string())
    print(f"\n=== SAYIM (conf={secilen}) ===")
    print(say_tab[["MAE", "RMSE", "sMAPE", "ME", "gercek_toplam",
                   "tahmin_toplam"]].to_string())
    print(f"\nANA METRIK  mAP50-95 = {ozet['mAP50-95']:.4f}"
          f"   (small {ozet['mAP_small']:.4f})")
    print(f"ozet -> {out / 'ozet.json'}")


if __name__ == "__main__":
    main()
