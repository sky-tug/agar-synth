#!/usr/bin/env python3
"""
GPU butce hesabi -- Faz 2'nin kapisi: "~85 kosu butceye sigiyor mu?"

Nasil calisir:
  Tek bir olculmus kosudan (tercihen G100, tek seed, tam egitim) yola cikip
  tum gridi olceklendirir. Maliyet modeli:

      sure  ~  n_train_goruntu  x  epoch_sayisi  x  (imgsz / olculen_imgsz)^2

  Ilk carpan dogrusal (bir epoch = bir gecis), ucuncusu karesel (piksel sayisi).
  Bu bir tahmin; gercek olcumler geldikce --olcum ile guncelle.

Kullanim:
    # G100 kosusundan sonra
    python scripts/butce.py --olcum runs/G100_s0/olcum.json

    # elle: 500 goruntu, epoch basina 90 saniye, 150 epoch
    python scripts/butce.py --n-train 500 --epoch-s 90 --epochs 150 --imgsz 1280

    # butce sinirina karsi ve alternatif senaryolar
    python scripts/butce.py --olcum runs/G100_s0/olcum.json --butce 200 --senaryo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# GRID TANIMI -- yol haritasi Faz 5/6/7
# (ad, egitim kumesi buyuklugu / tam veri orani, hangi faz)
# "+S" kollarinda toplam egitim kumesi tam veri buyuklugune tamamlanir.
#
# 12 Agustos 2026 -- hedef dergi belli oldu: Muhendislik Bilimleri ve Tasarim
# Dergisi (JESD, TR Dizin). Kapsam buna gore kisildi:
#   - klasik kol: 3 seviye -> yalnizca G25 (ikame egrisinin orta noktasi)
#   - miktar taramasi: 0.5x/1x/2x/4x -> 0.5x ve 2x  (4x tek basina en pahali
#     kalemdi: 3.25 tam-kosu esdegeri x 3 seed)
#   - ikinci detektor kontrolu: kapsam disi (dergi bunu beklemiyor)
#   - XAI kolu EKLENDI (hoca onayladi) -- egitim yok, yalnizca cikarim
# Eski genis plan asagida --kollar ile hala secilebilir.
# ---------------------------------------------------------------------------

ANA_GRID = [
    ("G100",   1.00), ("G50",  0.50), ("G50+S", 1.00), ("G25", 0.25),
    ("G25+S",  1.00), ("G10",  0.10), ("G10+S", 1.00), ("S100", 1.00),
]

# Kisilmis: yalnizca G25 seviyesinde. Difuzyon vs klasik sorusunun cevabi
# icin ikame egrisinin orta noktasi yeterli.
KLASIK = [("B_G25", 0.25), ("C_G25", 0.25)]
KLASIK_GENIS = [
    ("B_G50", 0.50), ("B_G25", 0.25), ("B_G10", 0.10),
    ("C_G50", 0.50), ("C_G25", 0.25), ("C_G10", 0.10),
]

# Sentetik miktar taramasi: G25 tabani (0.25) + sentetik.
# 1x = gercegi tam veriye tamamlayan miktar (0.75); zaten ana gridde G25+S var.
MIKTAR = [("G25+S_0.5x", 0.25 + 0.375), ("G25+S_2x", 0.25 + 1.50)]
MIKTAR_GENIS = [
    ("G25+S_0.5x", 0.25 + 0.375),
    ("G25+S_2x",   0.25 + 1.50),
    ("G25+S_4x",   0.25 + 3.00),
]

# Hoca: "Uc farkli senaryo secip kullanabilirsin. Maske sabit alinabilir."
ABLASYON = [("A1_LoRAsiz", 1.00), ("A2_arka_plan", 1.00), ("A3_rastgele_yerlesim", 1.00)]

IKINCI_DETEKTOR = [("YOLO11_G25", 0.25), ("YOLO11_G25+S", 1.00)]

KOLLAR = {
    "ana_grid":        (ANA_GRID, 5, "Faz 6"),
    "klasik":          (KLASIK, 3, "Faz 6"),
    "miktar_taramasi": (MIKTAR, 3, "Faz 6"),
    "ablasyon":        (ABLASYON, 3, "Faz 7"),
    # --- varsayilan disi, --kollar ile acilir ---
    "klasik_genis":    (KLASIK_GENIS, 3, "Faz 6"),
    "miktar_genis":    (MIKTAR_GENIS, 3, "Faz 6"),
    "ikinci_detektor": (IKINCI_DETEKTOR, 2, "Faz 7"),
}

# Varsayilan olarak hesaplanan kollar (JESD kapsami)
VARSAYILAN_KOLLAR = ["ana_grid", "klasik", "miktar_taramasi", "ablasyon"]

# XAI kolu (Grad-CAM/++): egitim YOK, mevcut agirliklarla cikarim.
# Maliyeti egitimin yaninda ihmal edilebilir ama sifir degil -- gosterilsin.
XAI_KONFIG = ["G100", "G25", "G25+S", "S100"]      # karsilastirilacak modeller
XAI_GORUNTU = 200                                   # test kumesinden ornek


def biçim(saat: float) -> str:
    if saat < 1:
        return f"{saat*60:.0f} dk"
    if saat < 48:
        return f"{saat:.1f} sa"
    return f"{saat/24:.1f} gun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--olcum", help="runs/<name>/olcum.json")
    ap.add_argument("--n-train", type=int, help="olculen kosunun egitim goruntu sayisi")
    ap.add_argument("--epoch-s", type=float, help="olculen epoch basina saniye")
    ap.add_argument("--epochs", type=int, default=150, help="gridde kosulacak epoch")
    ap.add_argument("--imgsz", type=int, help="gridde kullanilacak imgsz")
    ap.add_argument("--olculen-imgsz", type=int, help="olcumun yapildigi imgsz")
    ap.add_argument("--tam-veri", type=int, default=None,
                    help="tam AGAR countable+lower-res goruntu sayisi. "
                         "Olcum alt kumede yapildiysa buradan olceklenir.")
    ap.add_argument("--butce", type=float, help="elindeki GPU-saat")
    ap.add_argument("--kollar", default=",".join(VARSAYILAN_KOLLAR),
                    help="virgul ile. varsayilan (JESD kapsami): "
                         + ",".join(VARSAYILAN_KOLLAR)
                         + "  |  hepsi: " + ",".join(KOLLAR))
    ap.add_argument("--xai", action="store_true",
                    help="XAI kolunu (Grad-CAM cikarimi) da hesaba kat")
    ap.add_argument("--xai-s-goruntu", type=float, default=1.5,
                    help="bir goruntu icin Grad-CAM++ cikarim suresi (s)")
    ap.add_argument("--senaryo", action="store_true",
                    help="sigmazsa kisma senaryolarini da goster")
    # uretim (Faz 3) maliyeti
    ap.add_argument("--uretim-s-goruntu", type=float, default=0.0,
                    help="bir sentetik goruntunun inpainting suresi (s)")
    ap.add_argument("--lora-saat", type=float, default=0.0,
                    help="seviye basina bir LoRA egitimi (saat). 3 LoRA kosacak.")
    ap.add_argument("--sentetik-seed-basina", action="store_true",
                    help="her seed icin AYRI sentetik kume uret. Istatistiksel "
                         "olarak daha temiz (uretim rastgeleligi de varyansa "
                         "girer) ama uretim maliyetini seed sayisi kadar carpar.")
    args = ap.parse_args()

    # ---------------- olculen taban ----------------
    if args.olcum:
        m = json.loads(Path(args.olcum).read_text(encoding="utf-8"))
        n_olculen = m["n_train"]
        epoch_s = m["epoch_basina_s"]
        olculen_imgsz = m["imgsz"]
        kaynak = f"{m['kosu']} ({m.get('epochs_gerceklesen')} epoch, " \
                 f"{m.get('tepe_vram_gb')} GB VRAM)"
        if m.get("duman_testi"):
            print("! UYARI: bu bir DUMAN TESTI olcumu. Kisa kosularda epoch basina\n"
                  "  sure isinma yuzunden yaniltici olur. G100 tam egitimi bekle.\n")
    elif args.n_train and args.epoch_s:
        n_olculen, epoch_s = args.n_train, args.epoch_s
        olculen_imgsz = args.olculen_imgsz or args.imgsz or 1280
        kaynak = "elle girilen degerler"
    else:
        sys.exit("HATA: --olcum VEYA (--n-train ve --epoch-s) ver.")

    imgsz = args.imgsz or olculen_imgsz
    olculen_imgsz = args.olculen_imgsz or olculen_imgsz
    n_tam = args.tam_veri or n_olculen

    # bir "tam veri buyuklugunde" kosunun saniyesi
    s_per_img_epoch = epoch_s / max(n_olculen, 1)
    imgsz_carpani = (imgsz / olculen_imgsz) ** 2
    tam_kosu_s = s_per_img_epoch * n_tam * args.epochs * imgsz_carpani

    print("=" * 70)
    print("GPU BUTCE HESABI")
    print("=" * 70)
    print(f"olcum kaynagi        : {kaynak}")
    print(f"olculen              : {n_olculen} goruntu, {epoch_s:.1f} s/epoch, "
          f"imgsz {olculen_imgsz}")
    print(f"grid varsayimi       : {n_tam} goruntu (tam veri), {args.epochs} epoch, "
          f"imgsz {imgsz}")
    if imgsz_carpani != 1:
        print(f"imgsz olcekleme      : x{imgsz_carpani:.2f}  "
              f"({olculen_imgsz} -> {imgsz}, karesel)")
    print(f"tam boyutlu 1 kosu   : {biçim(tam_kosu_s/3600)}")
    print()

    secili = [k.strip() for k in args.kollar.split(",") if k.strip()]
    toplam_saat, toplam_kosu = 0.0, 0
    satirlar = []

    for kol in secili:
        if kol not in KOLLAR:
            sys.exit(f"HATA: bilinmeyen kol '{kol}'. Secenekler: {list(KOLLAR)}")
        konfigler, seed, faz = KOLLAR[kol]
        pay = sum(f for _, f in konfigler)
        kosu = len(konfigler) * seed
        saat = pay * seed * tam_kosu_s / 3600
        toplam_saat += saat
        toplam_kosu += kosu
        satirlar.append((kol, faz, len(konfigler), seed, kosu, pay * seed, saat))

    w = max(len(s[0]) for s in satirlar)
    print(f"{'kol':<{w}}  {'faz':<6} {'konf':>5} {'seed':>5} {'kosu':>5} "
          f"{'tam-kosu esd.':>13} {'GPU-saat':>10}")
    print("-" * 70)
    for ad, faz, nk, sd, kosu, esd, saat in satirlar:
        print(f"{ad:<{w}}  {faz:<6} {nk:>5} {sd:>5} {kosu:>5} "
              f"{esd:>13.2f} {saat:>10.1f}")
    print("-" * 70)
    print(f"{'TOPLAM (egitim)':<{w}}  {'':<6} {'':>5} {'':>5} {toplam_kosu:>5} "
          f"{'':>13} {toplam_saat:>10.1f}")

    # ---------------- uretim maliyeti (Faz 3) ----------------
    uretim_saat = 0.0
    if args.uretim_s_goruntu or args.lora_saat:
        # Bir konfigurasyonun sentetik ihtiyaci = toplam egitim kumesi - gercek pay.
        # Gercek pay konfigurasyon adindaki G<sayi>'dan okunur; "+S" yoksa 0 sentetik.
        import re

        def sentetik_ihtiyaci(ad, toplam):
            if "+S" not in ad and not ad.startswith("S100"):
                return 0.0
            if ad.startswith("S100"):
                return toplam                      # tamami sentetik
            m = re.search(r"G(\d+)", ad)
            gercek = int(m.group(1)) / 100 if m else 0.0
            return max(0.0, toplam - gercek)

        sentetik_pay = 0.0
        for kol in secili:
            konfigler, _, _ = KOLLAR[kol]
            sentetik_pay += sum(sentetik_ihtiyaci(a, f) for a, f in konfigler)

        # Sentetik kume seed'ler arasinda paylasilirsa bir kez uretilir.
        # Seed basina ayri uretmek istatistiksel olarak daha temiz (uretim
        # rastgeleligi de varyansa girer) ama maliyeti seed sayisi kadar carpar.
        carpan = 1
        if args.sentetik_seed_basina:
            carpan = max(sd for _, sd, _ in (KOLLAR[k] for k in secili))

        n_sentetik = int(round(sentetik_pay * n_tam)) * carpan
        uretim_saat = n_sentetik * args.uretim_s_goruntu / 3600 + 3 * args.lora_saat
        print()
        print(f"uretim (Faz 3)       : {n_sentetik} sentetik goruntu "
              f"x {args.uretim_s_goruntu} s  +  3 LoRA x {args.lora_saat} sa"
              + (f"   [seed basina ayri uretim, x{carpan}]" if carpan > 1
                 else "   [tek sentetik kume, seed'ler paylasiyor]"))
        print(f"                     = {uretim_saat:.1f} GPU-saat")

    # ---------------- XAI kolu (egitim yok, cikarim) ----------------
    xai_saat = 0.0
    if args.xai:
        xai_saat = len(XAI_KONFIG) * XAI_GORUNTU * args.xai_s_goruntu / 3600
        print()
        print(f"XAI (Grad-CAM/++)    : {len(XAI_KONFIG)} model "
              f"({', '.join(XAI_KONFIG)}) x {XAI_GORUNTU} goruntu "
              f"x {args.xai_s_goruntu} s")
        print(f"                     = {xai_saat:.2f} GPU-saat   "
              f"(egitim yok -- mevcut agirliklarla cikarim)")

    genel = toplam_saat + uretim_saat + xai_saat
    print()
    print(f"GENEL TOPLAM         : {biçim(genel)}  ({toplam_kosu} egitim kosusu)")

    # ---------------- butce karsilastirmasi ----------------
    if args.butce:
        print()
        print("=" * 70)
        if genel <= args.butce:
            print(f"KAPI: SIGIYOR.  {genel:.1f} / {args.butce:.0f} GPU-saat "
                  f"(%{100*genel/args.butce:.0f} dolu)")
            print("Faz 3'e gecebilirsin.")
        else:
            print(f"KAPI: SIGMIYOR.  {genel:.1f} / {args.butce:.0f} GPU-saat "
                  f"-- {genel - args.butce:.1f} saat fazla")
            print("\nYol haritasinin onerdigi kisma sirasi "
                  "(ana grid EN SONA kalir, o makalenin belkemigi):")
            for ad, tarif, kazanc in [
                ("miktar_taramasi", "seed 3 -> 1", None),
                ("klasik", "seed 3 -> 2", None),
                ("ablasyon", "tek seviye, seed 3 -> 2", None),
                ("ikinci_detektor", "tek konfigurasyon", None),
                ("ana_grid", "seed 5 -> 3 (SON CARE -- n=3 ile std savunmasi zayiflar)",
                 None),
            ]:
                if ad in secili:
                    print(f"  - {ad:<16} {tarif}")
        print("=" * 70)

    if args.senaryo:
        print("\n" + "=" * 70)
        print("SENARYOLAR")
        print("=" * 70)
        senaryolar = [
            ("tam plan (yukaridaki)", genel),
            ("ana grid seed 5->3", genel - (
                sum(f for _, f in ANA_GRID) * 2 * tam_kosu_s / 3600)
                if "ana_grid" in secili else genel),
            ("yan kollar seed 3->2", genel - (
                (sum(f for _, f in KLASIK) + sum(f for _, f in MIKTAR))
                * 1 * tam_kosu_s / 3600)),
            ("epoch 150->100", genel * 100 / args.epochs),
            ("imgsz 1280->1024 (! kucuk koloni riski)", genel * (1024 / imgsz) ** 2),
            ("imgsz 1280->640  (!! C.albicans 8.6 px, onerilmez)",
             genel * (640 / imgsz) ** 2),
        ]
        for ad, s in senaryolar:
            isaret = ""
            if args.butce:
                isaret = "  <- sigar" if s <= args.butce else ""
            print(f"  {ad:<45} {biçim(s):>10}{isaret}")
        print("=" * 70)

    print("\nNot: bu bir TAHMIN. Her yeni olcum geldiginde --olcum ile guncelle;")
    print("bolum 6 (hesaplama maliyeti) tablosu gercek olcumlerden yazilacak.")


if __name__ == "__main__":
    main()
