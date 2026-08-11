#!/usr/bin/env python3
"""
AGAR JSON -> YOLO donusumu.

Ne yapar:
  - AGAR kokunde *.json dosyalarini tarar
  - Outline 3.2 filtrelerini uygular (lower-resolution + countable + 5 mikroorganizma)
  - defects / contamination iceren goruntuleri TAMAMEN atar
  - Goruntu boyutunu dosyadan okur (JSON'da yok, 2048x2048 varsayilmaz)
  - out/labels/<id>.txt  ve  out/images/<id>.<ext> (symlink) uretir
  - out/manifest.csv uretir -> bolme, alt orneklem ve makale veri tablosunun kaynagi

Kullanim:
    python convert.py --src /yol/AGAR_representative --out data/processed
    python convert.py --src ... --out ... --copy     # symlink yerine kopyala
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# --- SABIT SINIF SIRASI -- proje boyunca degismeyecek ---------------------
CLASS_ORDER = ["S.aureus", "B.subtilis", "P.aeruginosa", "E.coli", "C.albicans"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_ORDER)}

# Kapsam disi birakilan artefakt siniflari
ARTIFACT_CLASSES = {"defects", "contamination"}

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def normalize_class(name: str) -> str:
    """'S. aureus' -> 'S.aureus'. AGAR'da her iki yazim da gorulebiliyor."""
    return name.replace(" ", "").strip()


def find_image(json_path: Path):
    """JSON ile ayni isimli goruntu dosyasini bulur."""
    for ext in IMG_EXTS:
        cand = json_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="AGAR kok klasoru")
    ap.add_argument("--out", required=True, help="cikti klasoru")
    ap.add_argument("--background", default="lower-resolution",
                    help="tutulacak alt kume (varsayilan: lower-resolution). "
                         "'all' dersen filtre uygulanmaz")
    ap.add_argument("--copy", action="store_true",
                    help="goruntuleri symlink yerine kopyala")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not src.is_dir():
        sys.exit(f"HATA: kaynak klasor yok: {src}")

    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    json_files = sorted(src.rglob("*.json"))
    if not json_files:
        sys.exit(f"HATA: {src} altinda hic JSON bulunamadi")

    # Sayaclar -- her elenen goruntunun sebebi kayit altinda
    n = Counter()
    class_boxes = Counter()          # sinif basina kutu sayisi
    unknown_classes = Counter()      # beklenmeyen sinif isimleri
    rows = []

    for jp in json_files:
        n["json_toplam"] += 1

        try:
            meta = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            n["elendi_bozuk_json"] += 1
            print(f"  ! bozuk JSON atlandi: {jp.name} ({e})")
            continue

        # --- Filtre 1: alt kume ---
        bg = meta.get("background")
        if args.background != "all" and bg != args.background:
            n["elendi_alt_kume"] += 1
            continue

        labels = meta.get("labels") or []

        # --- Filtre 2: countable ---
        # AGAR'da koloni seviyesinde etiket YALNIZCA countable goruntulerde var.
        # uncountable ve empty goruntulerde labels listesi bos gelir.
        if len(labels) == 0:
            n["elendi_countable_degil"] += 1
            continue

        # --- Filtre 3: artefakt sinifi iceren goruntuler tamamen atilir ---
        # (kutuyu silip goruntuyu tutmak modele "burada nesne yok" diye ogretir)
        present = {normalize_class(l.get("class", "")) for l in labels}
        present |= {normalize_class(c) for c in (meta.get("classes") or [])}
        if present & ARTIFACT_CLASSES:
            n["elendi_artefakt"] += 1
            continue

        bilinmeyen = present - set(CLASS_ORDER) - ARTIFACT_CLASSES
        if bilinmeyen:
            for b in bilinmeyen:
                unknown_classes[b] += 1
            n["elendi_bilinmeyen_sinif"] += 1
            continue

        # --- Goruntu dosyasi ve BOYUTU (JSON'da yok, dosyadan okunuyor) ---
        img_path = find_image(jp)
        if img_path is None:
            n["elendi_goruntu_yok"] += 1
            print(f"  ! goruntu bulunamadi: {jp.name}")
            continue
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception as e:
            n["elendi_goruntu_acilamadi"] += 1
            print(f"  ! goruntu acilamadi: {img_path.name} ({e})")
            continue

        # --- Kutu donusumu ---
        lines = []
        per_class = defaultdict(int)
        atilan_kutu = 0
        kirpilan_kutu = 0

        for l in labels:
            cname = normalize_class(l.get("class", ""))
            cid = CLASS_TO_ID.get(cname)
            if cid is None:
                atilan_kutu += 1
                continue

            x, y = float(l["x"]), float(l["y"])          # sol ust kose
            w, h = float(l["width"]), float(l["height"])

            # goruntu sinirlarina kirp
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(W), x + w), min(float(H), y + h)
            if (x0, y0, x1, y1) != (x, y, x + w, y + h):
                kirpilan_kutu += 1

            bw, bh = x1 - x0, y1 - y0
            if bw <= 1 or bh <= 1:      # bozuk / sifir alanli kutu
                atilan_kutu += 1
                continue

            xc = (x0 + bw / 2) / W
            yc = (y0 + bh / 2) / H
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw / W:.6f} {bh / H:.6f}")
            per_class[cname] += 1
            class_boxes[cname] += 1

        if not lines:
            n["elendi_gecerli_kutu_yok"] += 1
            continue

        n["kutu_atildi"] += atilan_kutu
        n["kutu_kirpildi"] += kirpilan_kutu

        # --- Yazim ---
        stem = img_path.stem
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")

        dst_img = out / "images" / img_path.name
        if not dst_img.exists():
            if args.copy:
                shutil.copy2(img_path, dst_img)
            else:
                dst_img.symlink_to(img_path)

        rows.append({
            "sample_id": meta.get("sample_id", stem),
            "stem": stem,
            "image_path": str(img_path),
            "label_path": str(out / "labels" / f"{stem}.txt"),
            "width": W,
            "height": H,
            "background": bg,
            "n_boxes": len(lines),
            "colonies_number": meta.get("colonies_number", ""),
            "classes": "|".join(sorted(per_class)),
            "n_classes": len(per_class),
            **{f"n_{c}": per_class.get(c, 0) for c in CLASS_ORDER},
        })
        n["tutuldu"] += 1

    if not rows:
        sys.exit("HATA: filtrelerden gecen hic goruntu kalmadi. "
                 "--background degerini kontrol et.")

    # --- manifest.csv ---
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    # --- sinif isimleri dosyasi (YOLO icin) ---
    (out / "classes.txt").write_text("\n".join(CLASS_ORDER) + "\n", encoding="utf-8")

    # --- OZET ---
    print("\n" + "=" * 58)
    print("DONUSUM OZETI")
    print("=" * 58)
    print(f"  taranan JSON            : {n['json_toplam']}")
    print(f"  elendi / alt kume       : {n['elendi_alt_kume']}")
    print(f"  elendi / countable degil: {n['elendi_countable_degil']}")
    print(f"  elendi / artefakt sinifi: {n['elendi_artefakt']}   <- makalede raporlanacak")
    print(f"  elendi / bilinmeyen sinif: {n['elendi_bilinmeyen_sinif']}")
    print(f"  elendi / goruntu yok    : {n['elendi_goruntu_yok']}")
    print(f"  elendi / gecerli kutu yok: {n['elendi_gecerli_kutu_yok']}")
    print(f"  TUTULAN GORUNTU         : {n['tutuldu']}")
    print(f"  toplam kutu             : {sum(class_boxes.values())}")
    print(f"  kirpilan kutu           : {n['kutu_kirpildi']}")
    print(f"  atilan kutu (bozuk)     : {n['kutu_atildi']}")
    print("\n  sinif basina kutu:")
    for c in CLASS_ORDER:
        print(f"    {CLASS_TO_ID[c]}  {c:<16} {class_boxes.get(c, 0)}")
    if unknown_classes:
        print("\n  ! beklenmeyen sinif isimleri (kontrol et):")
        for k, v in unknown_classes.most_common():
            print(f"    {k}: {v}")
    print(f"\n  manifest -> {manifest}")
    print("=" * 58)
    print("\nSIRADAKI ADIM: python check_labels.py --data",
          out, "--n 30\n")


if __name__ == "__main__":
    main()
