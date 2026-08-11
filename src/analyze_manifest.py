#!/usr/bin/env python3
"""
manifest.csv + YOLO etiketlerinden betimsel analiz ve kalite kontrolu.

Iki isi ayni anda yapiyor:
  1) MAKALE ICIN betimsel istatistik (outline 3.2 veri tablosu)
  2) FAZ 3 ICIN yerlesim istatistikleri (koloni sayisi, boyut, radyal konum)
  3) KALITE KONTROLU: mukerrer kutu, plak disi kutu, minik kutu

Cikti: <data>/reports/  altinda CSV tablolari + PNG grafikler

Kullanim:
    python analyze_manifest.py --data data/processed
    python analyze_manifest.py --data data/processed --iou-dup 0.5
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


def iou(a, b):
    """a, b = (x0, y0, x1, y1) piksel"""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_boxes(man: pd.DataFrame) -> pd.DataFrame:
    """Tum etiket dosyalarini tek bir kutu tablosuna acar."""
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
            # merkezin plak merkezinden uzakligi, yarim genislige gore normalize
            dx, dy = (xc - 0.5) * 2, (yc - 0.5) * 2
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
                "radius": float(np.hypot(dx, dy)),   # 0 = merkez, 1 = kenar
                "x0": (xc - bw / 2) * W, "y0": (yc - bh / 2) * H,
                "x1": (xc + bw / 2) * W, "y1": (yc + bh / 2) * H,
            })
    return pd.DataFrame(rows)


def coco_size_bucket(area):
    """COCO konvansiyonu: small <32^2, medium <96^2, large ustu"""
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--iou-dup", type=float, default=0.5,
                    help="bu esigin ustundeki kutu ciftleri 'mukerrer suphesi'")
    ap.add_argument("--tiny-px", type=float, default=8,
                    help="bu boyutun altindaki kutular isaretlenir")
    args = ap.parse_args()

    data = Path(args.data).expanduser().resolve()
    man_path = data / "manifest.csv"
    if not man_path.exists():
        sys.exit(f"HATA: {man_path} yok. once convert.py calistir.")

    man = pd.read_csv(man_path)
    box = load_boxes(man)
    rep = data / "reports"
    rep.mkdir(exist_ok=True)

    print(f"\n{len(man)} goruntu, {len(box)} kutu okundu\n")

    # ================= 1. SINIF BAZLI OZET (makale tablosu) =================
    g = box.groupby("cls")
    cls_tab = pd.DataFrame({
        "kutu": g.size(),
        "goruntu": g["stem"].nunique(),
        "boyut_min_px": g["px_size"].min().round(1),
        "boyut_medyan_px": g["px_size"].median().round(1),
        "boyut_p95_px": g["px_size"].quantile(0.95).round(1),
        "boyut_max_px": g["px_size"].max().round(1),
        "en_boy_medyan": g["aspect"].median().round(3),
    }).reindex(CLASS_ORDER).fillna(0)
    cls_tab["kutu_payi_%"] = (cls_tab["kutu"] / len(box) * 100).round(1)
    cls_tab.to_csv(rep / "sinif_ozeti.csv")
    print("=== SINIF BAZLI OZET ===")
    print(cls_tab.to_string(), "\n")

    # ================= 2. COCO BOYUT KIRILIMI =================
    box["boyut_grubu"] = box["area_px"].map(coco_size_bucket)
    size_tab = pd.crosstab(box["cls"], box["boyut_grubu"]).reindex(
        index=CLASS_ORDER, columns=["small", "medium", "large"]).fillna(0).astype(int)
    size_tab.to_csv(rep / "boyut_kirilimi.csv")
    print("=== COCO BOYUT KIRILIMI (mAP kirilimi bu eksende raporlanacak) ===")
    print(size_tab.to_string(), "\n")

    # ================= 3. GORUNTU BASINA KOLONI (Faz 3 girdisi) =============
    per_img = box.groupby("stem").size()
    print("=== GORUNTU BASINA KOLONI SAYISI ===")
    print(f"  min {per_img.min()} | medyan {per_img.median():.0f} | "
          f"ortalama {per_img.mean():.1f} | p95 {per_img.quantile(.95):.0f} | "
          f"max {per_img.max()}\n")
    per_img.rename("n_box").to_csv(rep / "goruntu_basina_koloni.csv")

    # ================= 4. TUR KOMBINASYONLARI ===============================
    combo = man["classes"].value_counts().rename("goruntu")
    combo.to_csv(rep / "tur_kombinasyonlari.csv")
    print("=== TUR KOMBINASYONLARI (yerlesim dagilimi buna kosullu cikarilmali) ===")
    print(combo.to_string(), "\n")

    # ================= 5. KALITE KONTROLU ===================================
    print("=== KALITE KONTROLU ===")

    tiny = box[box["px_size"] < args.tiny_px]
    print(f"  {args.tiny_px}px altinda kutu : {len(tiny)}")

    kenar = box[(box["x0"] <= 1) | (box["y0"] <= 1) |
                (box["x1"] >= box["img_w"] - 1) | (box["y1"] >= box["img_h"] - 1)]
    print(f"  goruntu kenarina degen  : {len(kenar)}")

    # plak dairesinin disi: plak kareye tam oturuyor, r>1 ise kose bolgesi
    print(f"  kenar bandi (r > 0.95)  : {(box['radius'] > 0.95).sum()}")
    print(f"  plak dairesi disi (r>1) : {(box['radius'] > 1.0).sum()}")

    # mukerrer kutu suphesi -- goruntu ici, vektorize IoU
    dup_rows = []
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
        ii, jj = np.where(np.triu(m, k=1) >= args.iou_dup)
        cls_arr = sub["cls"].to_numpy()
        for a, c in zip(ii, jj):
            dup_rows.append({"stem": stem, "iou": round(float(m[a, c]), 3),
                             "cls_a": cls_arr[a], "cls_b": cls_arr[c]})
    dup = pd.DataFrame(dup_rows)
    n_dup_img = dup["stem"].nunique() if len(dup) else 0
    print(f"  IoU >= {args.iou_dup} kutu cifti      : {len(dup)}  "
          f"({n_dup_img} goruntude)")
    if len(dup):
        dup.sort_values("iou", ascending=False).to_csv(rep / "mukerrer_supheli.csv",
                                                       index=False)
        print(f"    -> {rep/'mukerrer_supheli.csv'} icindeki ilk satirlari")
        print(f"       check_labels.py ile tek tek cizdirip GOZLE dogrula")
    print()

    # ================= 6. GRAFIKLER =========================================
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    for c in CLASS_ORDER:
        s = box.loc[box["cls"] == c, "px_size"]
        if len(s):
            ax[0, 0].hist(s, bins=60, alpha=.55, label=f"{c} (n={len(s)})")
    ax[0, 0].set(title="Koloni boyutu dagilimi", xlabel="kutu kenari (px)",
                 ylabel="kutu sayisi")
    ax[0, 0].legend(fontsize=7)

    ax[0, 1].hist(per_img, bins=40, color="#4363d8")
    ax[0, 1].set(title="Goruntu basina koloni sayisi",
                 xlabel="koloni", ylabel="goruntu")

    for c in CLASS_ORDER:
        s = box.loc[box["cls"] == c, "radius"]
        if len(s):
            ax[1, 0].hist(s, bins=40, alpha=.5, density=True, label=c)
    ax[1, 0].axvline(1.0, color="k", ls="--", lw=1)
    ax[1, 0].set(title="Radyal konum (0=merkez, 1=kenar) — Faz 3 yerlesim girdisi",
                 xlabel="normalize yaricap", ylabel="yogunluk")
    ax[1, 0].legend(fontsize=7)

    sample = box.sample(min(6000, len(box)), random_state=0)
    ax[1, 1].scatter(sample["xc"], sample["yc"], s=2, alpha=.25,
                     c=sample["cid"], cmap="tab10")
    ax[1, 1].set(title="Koloni merkezlerinin uzamsal dagilimi",
                 xlabel="x (normalize)", ylabel="y (normalize)")
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(rep / "veri_ozeti.png", dpi=140)
    print(f"grafikler -> {rep/'veri_ozeti.png'}")
    print(f"tablolar  -> {rep}/\n")


if __name__ == "__main__":
    main()