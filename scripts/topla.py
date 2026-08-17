#!/usr/bin/env python3
"""
Kosu sonuclarini toplar -- makalenin sonuc tablosu buradan cikacak.

NEDEN VAR (karar 3.33):
  `runs/` .gitignore'da (ve olmali: agirliklar GB'larca). Ama runs/ altinda
  yalnizca agirlik yok -- her kosunun `olcum.json`'u (sure, VRAM, GPU-saat,
  git commit) ve `eval*/ozet.json`'u (mAP, sinif AP, sayim hatalari) da orada.
  Yani MAKALENIN BUTUN SONUCLARI surum kontrolunun disinda kaliyor.

  Bu dosyalar birkac KB ama YENIDEN URETILEMEZ: 500+ GPU-saat surer.
  Dizustu olurse, ya da BİDB sunucusuna gecince, hicbir yerde yoklar.

  `topla.py` hepsini tarayip `sonuclar/` altina yaziyor. `sonuclar/` git'e
  GIRIYOR. Ayrica 61 kosuluk grid tablosunu bir sekilde kurman gerekiyordu --
  bu iki ihtiyaci birden cozuyor.

Kullanim:
    python scripts/topla.py                      # runs/ -> sonuclar/
    python scripts/topla.py --runs runs --cikti sonuclar
    python scripts/topla.py --duman-dahil        # duman testlerini de al

Cikti:
    sonuclar/tablo.csv       her kosu bir satir -- ana tablo
    sonuclar/ham/<kosu>.json olcum + degerlendirme, kosu basina birlestirilmis
    sonuclar/ozet.txt        ekrana basilan ozetin kopyasi
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]

# tablo sutun sirasi -- makalenin tablosuna yakin dursun
SUTUNLAR = [
    "kosu", "seviye", "seed", "overlay", "split",
    "mAP50-95", "mAP50", "mAP75",
    "mAP_small", "mAP_medium", "mAP_large",
    "AP_S.aureus", "AP_B.subtilis", "AP_P.aeruginosa", "AP_E.coli", "AP_C.albicans",
    "MAE", "RMSE", "sMAPE", "ME", "sayim_r", "conf_thr_sayim",
    "n_train", "n_val", "n_goruntu", "n_gt_kutu",
    "imgsz", "batch", "epochs_planlanan", "epochs_gerceklesen",
    "toplam_sure_dk", "epoch_basina_s", "gpu_saat", "tepe_vram_gb",
    "goruntu_saniye", "goruntu_basina_ms",
    "kart", "torch", "git_commit", "tarih_utc", "duman_testi",
]


def _oku(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ! okunamadi: {p}  ({e})", file=sys.stderr)
        return None


def topla(runs: Path, duman_dahil: bool):
    kayitlar = []
    for kosu_dizin in sorted(d for d in runs.iterdir() if d.is_dir()):
        olcum = _oku(kosu_dizin / "olcum.json") if (kosu_dizin / "olcum.json").exists() else None

        # eval*/ozet.json  -- eval, eval_val, eval_test hepsini yakala
        ozetler = {}
        for ed in sorted(kosu_dizin.glob("eval*")):
            oz = ed / "ozet.json"
            if oz.exists():
                d = _oku(oz)
                if d:
                    ozetler[d.get("split", ed.name)] = d

        if olcum is None and not ozetler:
            continue                       # bos klasor, atla
        if olcum and olcum.get("duman_testi") and not duman_dahil:
            continue

        kayitlar.append({"kosu": kosu_dizin.name, "olcum": olcum, "ozet": ozetler})
    return kayitlar


def satirlastir(kayitlar):
    """Her (kosu, split) cifti icin bir satir. Degerlendirme yoksa yalnizca olcum."""
    satirlar = []
    for k in kayitlar:
        o = k["olcum"] or {}
        ort = o.get("ortam", {})
        taban = {
            "kosu": k["kosu"],
            "seviye": o.get("seviye"), "seed": o.get("seed"),
            "overlay": Path(o["overlay"]).stem if o.get("overlay") else "base",
            "n_train": o.get("n_train"), "n_val": o.get("n_val"),
            "imgsz": o.get("imgsz"), "batch": o.get("batch"),
            "epochs_planlanan": o.get("epochs_planlanan"),
            "epochs_gerceklesen": o.get("epochs_gerceklesen"),
            "toplam_sure_dk": o.get("toplam_sure_dk"),
            "epoch_basina_s": o.get("epoch_basina_s"),
            "gpu_saat": o.get("gpu_saat"), "tepe_vram_gb": o.get("tepe_vram_gb"),
            "goruntu_saniye": o.get("goruntu_saniye"),
            "kart": ort.get("kart"), "torch": ort.get("torch"),
            "git_commit": o.get("git_commit"), "tarih_utc": o.get("tarih_utc"),
            "duman_testi": o.get("duman_testi"),
        }
        if not k["ozet"]:
            satirlar.append({**taban, "split": None})
            continue
        for split, e in sorted(k["ozet"].items()):
            r = dict(taban)
            r["split"] = split
            for c in SUTUNLAR:
                if c in e and c not in r:
                    r[c] = e[c]
            satirlar.append(r)
    return satirlar


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(KOK / "runs"))
    ap.add_argument("--cikti", default=str(KOK / "sonuclar"))
    ap.add_argument("--duman-dahil", action="store_true",
                    help="duman testi olarak isaretli kosulari da tabloya al")
    a = ap.parse_args()

    runs = Path(a.runs).expanduser().resolve()
    if not runs.is_dir():
        sys.exit(f"HATA: {runs} yok.")
    cikti = Path(a.cikti).expanduser().resolve()
    (cikti / "ham").mkdir(parents=True, exist_ok=True)

    kayitlar = topla(runs, a.duman_dahil)
    if not kayitlar:
        sys.exit(f"HATA: {runs} altinda olcum/degerlendirme bulunamadi.\n"
                 f"  (duman testlerini dahil etmek icin --duman-dahil)")

    # --- ham kopyalar: kosu basina tek dosya, git'e girer ---
    for k in kayitlar:
        (cikti / "ham" / f"{k['kosu']}.json").write_text(
            json.dumps({"olcum": k["olcum"], "degerlendirme": k["ozet"]},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    # --- tablo ---
    satirlar = satirlastir(kayitlar)
    try:
        import pandas as pd
        df = pd.DataFrame(satirlar)
        for c in SUTUNLAR:
            if c not in df.columns:
                df[c] = None
        df = df[SUTUNLAR]
        df.to_csv(cikti / "tablo.csv", index=False)
    except ImportError:
        import csv
        with (cikti / "tablo.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUTUNLAR, extrasaction="ignore")
            w.writeheader()
            w.writerows(satirlar)
        df = None

    # --- ozet ---
    L = []
    L.append("=" * 72)
    L.append(f"TOPLANAN SONUCLAR   ({datetime.now(timezone.utc).isoformat(timespec='seconds')})")
    L.append("=" * 72)
    L.append(f"  kaynak            : {runs}")
    L.append(f"  kosu              : {len(kayitlar)}")
    L.append(f"  tablo satiri      : {len(satirlar)}  (kosu x split)")

    olculen = [k for k in kayitlar if k["olcum"]]
    gpu = sum(k["olcum"].get("gpu_saat") or 0 for k in olculen)
    L.append(f"  toplam GPU-saat   : {gpu:.2f}")
    degerlendirilen = [k for k in kayitlar if k["ozet"]]
    L.append(f"  degerlendirilmis  : {len(degerlendirilen)} / {len(kayitlar)}")

    eksik = [k["kosu"] for k in kayitlar if not k["ozet"]]
    if eksik:
        L.append(f"  ! degerlendirmesi olmayan kosular: {', '.join(eksik)}")
    eksik_olcum = [k["kosu"] for k in kayitlar if not k["olcum"]]
    if eksik_olcum:
        L.append(f"  🔴 olcum.json'u OLMAYAN kosular: {', '.join(eksik_olcum)}")
        L.append("     (train.py yerine dogrudan YOLO ile mi kosuldu? sure/VRAM kaybi)")

    # ana metrigi olan test satirlari
    test = [s for s in satirlar if s.get("split") == "test" and s.get("mAP50-95") is not None]
    if test:
        L.append("")
        L.append("  --- TEST SONUCLARI (ana metrik) ---")
        L.append(f"  {'kosu':<22} {'mAP50-95':>9} {'small':>8} {'MAE':>7}")
        for s in sorted(test, key=lambda r: -(r.get("mAP50-95") or 0)):
            L.append(f"  {s['kosu']:<22} {s['mAP50-95']:>9.4f} "
                     f"{(s.get('mAP_small') or float('nan')):>8.4f} "
                     f"{(s.get('MAE') or float('nan')):>7.2f}")
    L.append("")
    L.append(f"  tablo -> {cikti / 'tablo.csv'}")
    L.append(f"  ham   -> {cikti / 'ham'}/")
    L.append("=" * 72)
    L.append("Bu klasor GIT'E GIRIYOR. Kosulardan sonra calistir ve commit et --")
    L.append("runs/ gitignore'da, sonuclar buradan baska hicbir yerde durmuyor.")

    metin = "\n".join(L)
    print(metin)
    (cikti / "ozet.txt").write_text(metin + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
