#!/usr/bin/env python3
"""
YOLO etiketlerini goruntu uzerine cizer -- GOZLE dogrulama icin.

Bu adim atlanmamali. Kutu koordinatlarindaki tek bir hata (sol-ust vs merkez,
normalize edilmemis deger, yanlis goruntu boyutu) egitimde sessizce ilerler ve
sonra "modelim neden ogrenmiyor" olarak geri doner.

Kullanim:
    python check_labels.py --data data/processed --n 30
    python check_labels.py --data data/processed --n 10 --class-name C.albicans
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

CLASS_ORDER = ["S.aureus", "B.subtilis", "P.aeruginosa", "E.coli", "C.albicans"]
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]

MAX_SIDE = 1400   # cizim sonrasi kucultme -- klasorde hizli gezinmek icin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="convert.py cikti klasoru")
    ap.add_argument("--n", type=int, default=30, help="cizilecek goruntu sayisi")
    ap.add_argument("--out", default=None, help="cikti klasoru (varsayilan: <data>/viz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-name", default=None,
                    help="yalnizca bu sinifi iceren goruntulerden ornekle")
    args = ap.parse_args()

    data = Path(args.data).expanduser().resolve()
    manifest = data / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"HATA: {manifest} yok. once convert.py calistir.")

    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.class_name:
        rows = [r for r in rows if int(r.get(f"n_{args.class_name}", 0) or 0) > 0]
        if not rows:
            sys.exit(f"HATA: {args.class_name} iceren goruntu yok")

    # --- Ornekleme: her siniftan en az birkac ornek gormek istiyoruz ---
    rnd = random.Random(args.seed)
    secilen, secilen_stems = [], set()

    if not args.class_name:
        pay = max(1, args.n // (len(CLASS_ORDER) * 2))
        for c in CLASS_ORDER:
            aday = [r for r in rows if int(r.get(f"n_{c}", 0) or 0) > 0
                    and r["stem"] not in secilen_stems]
            rnd.shuffle(aday)
            for r in aday[:pay]:
                secilen.append(r)
                secilen_stems.add(r["stem"])

    kalan = [r for r in rows if r["stem"] not in secilen_stems]
    rnd.shuffle(kalan)
    secilen += kalan[: max(0, args.n - len(secilen))]
    secilen = secilen[: args.n]

    outdir = Path(args.out) if args.out else data / "viz"
    outdir.mkdir(parents=True, exist_ok=True)

    uyari = Counter()
    toplam_kutu = 0

    for r in secilen:
        img_path = Path(r["image_path"])
        lbl_path = Path(r["label_path"])
        if not img_path.exists() or not lbl_path.exists():
            uyari["dosya_yok"] += 1
            continue

        im = Image.open(img_path).convert("RGB")
        W, H = im.size

        # --- Boyut tutarliligi: manifest ile gercek dosya ayni mi? ---
        if (W, H) != (int(r["width"]), int(r["height"])):
            uyari["boyut_uyusmazligi"] += 1
            print(f"  ! boyut uyusmuyor: {img_path.name} "
                  f"manifest={r['width']}x{r['height']} dosya={W}x{H}")

        draw = ImageDraw.Draw(im)
        kalinlik = max(2, round(min(W, H) / 500))

        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cid, xc, yc, bw, bh = line.split()
            cid = int(cid)
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            toplam_kutu += 1

            # --- Otomatik saglik kontrolleri ---
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                uyari["aralik_disi"] += 1
            if bw * W < 4 or bh * H < 4:
                uyari["cok_kucuk_kutu"] += 1

            x0 = (xc - bw / 2) * W
            y0 = (yc - bh / 2) * H
            x1 = (xc + bw / 2) * W
            y1 = (yc + bh / 2) * H
            draw.rectangle([x0, y0, x1, y1],
                           outline=COLORS[cid % len(COLORS)], width=kalinlik)

        # --- Kucuk lejant (hangi renk hangi sinif) ---
        yy = 8
        for i, c in enumerate(CLASS_ORDER):
            if int(r.get(f"n_{c}", 0) or 0) > 0:
                draw.rectangle([8, yy, 8 + 26, yy + 20], fill=COLORS[i])
                draw.text((42, yy + 4), f"{c}  ({r[f'n_{c}']})", fill="white",
                          stroke_width=2, stroke_fill="black")
                yy += 26

        im.thumbnail((MAX_SIDE, MAX_SIDE))
        im.save(outdir / f"{r['stem']}_kutulu.jpg", quality=88)

    print("\n" + "=" * 58)
    print(f"  cizilen goruntu : {len(secilen)}")
    print(f"  cizilen kutu    : {toplam_kutu}")
    print(f"  cikti klasoru   : {outdir}")
    if uyari:
        print("\n  UYARILAR:")
        for k, v in uyari.items():
            print(f"    {k}: {v}")
    else:
        print("\n  otomatik kontrollerde uyari yok")
    print("=" * 58)
    print("""
SIMDI KLASORU AC VE GOZLE BAK. Aradigin sey:
  - Kutular kolonilerin uzerinde mi, kayik mi?
  - Kucuk koloniler (S.aureus, C.albicans) kutulanmis mi, kacmis mi?
  - Plak disinda / bosa dusen kutu var mi?
  - Ayni koloni birden fazla kutulanmis mi?
Bu 10 dakika, ilerideki 3 gunu kurtarir.
""")


if __name__ == "__main__":
    main()
