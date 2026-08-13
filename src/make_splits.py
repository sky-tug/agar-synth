#!/usr/bin/env python3
"""
Bolme (train/val/test) ve ic ice alt orneklem listelerini uretir.

Tasarim kararlari:
  - Bolme GORUNTU seviyesinde (ayni plaktan parcalar farkli kumelere dusmesin)
  - Tur kombinasyonuna gore TABAKALI (stratified): %10 alt kumesinde C.albicans
    kaybolursa seviyeler arasi fark sinif dengesizliginden gelir, veri miktarindan degil
  - IC ICE: %10 subset %25'in, %25 %50'nin, %50 %100'un ALT KUMESI
  - Listeler splits/ altina STEM olarak yazilir -> git'e commit edilir, tasinabilir
  - Ultralytics'in okuyacagi mutlak yollu listeler data/.../lists/ altina uretilir
    (bunlar gitignore'da; her makinede yeniden uretilir)

Kullanim:
    python src/make_splits.py --data data/processed --seed 42
    python src/make_splits.py --data data/processed --val 0.15 --test 0.15
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_ORDER = ["S.aureus", "B.subtilis", "P.aeruginosa", "E.coli", "C.albicans"]
LEVELS = [50, 25, 10]          # ic ice alt orneklem seviyeleri (% train icinde)


def stratified_split(df, val_frac, test_frac, rng):
    """Tabaka = tur kombinasyonu. Her tabaka kendi icinde bolunur."""
    train, val, test = [], [], []
    for _, grp in df.groupby("classes", sort=True):
        idx = grp["stem"].tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        # cok kucuk tabakalarda train'in bos kalmamasini garanti et
        while n - n_test - n_val < 1 and (n_test + n_val) > 0:
            if n_test >= n_val:
                n_test -= 1
            else:
                n_val -= 1
        test += idx[:n_test]
        val += idx[n_test:n_test + n_val]
        train += idx[n_test + n_val:]
    return sorted(train), sorted(val), sorted(test)


def nested_subsamples(train_stems, df, rng):
    """
    %50 ⊂ %100, %25 ⊂ %50, %10 ⊂ %25 olacak sekilde kademeli daraltma.
    Her adimda tabakali orneklem.
    """
    lookup = df.set_index("stem")["classes"].to_dict()
    subsets = {}
    havuz = list(train_stems)
    onceki_oran = 100
    for lvl in LEVELS:
        oran = lvl / onceki_oran          # bir onceki seviyeye gore pay
        secilen = []
        by_stratum = {}
        for s in havuz:
            by_stratum.setdefault(lookup[s], []).append(s)
        for _, items in sorted(by_stratum.items()):
            items = sorted(items)
            rng.shuffle(items)
            k = max(1, int(round(len(items) * oran)))
            secilen += items[:k]
        secilen = sorted(secilen)
        subsets[lvl] = secilen
        havuz = secilen                    # sonraki seviye BUNUN alt kumesi olacak
        onceki_oran = lvl
    return subsets


def class_table(stems, df):
    """Bir kume icin sinif basina kutu sayisi + goruntu sayisi."""
    sub = df[df["stem"].isin(stems)]
    row = {"goruntu": len(sub)}
    toplam = 0
    for c in CLASS_ORDER:
        v = int(sub[f"n_{c}"].sum())
        row[c] = v
        toplam += v
    row["kutu"] = toplam
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = Path(args.data).expanduser().resolve()
    man_path = data / "manifest.csv"
    if not man_path.exists():
        sys.exit(f"HATA: {man_path} yok. once convert.py calistir.")

    df = pd.read_csv(man_path)
    df["stem"] = df["stem"].astype(str)
    rng = np.random.default_rng(args.seed)

    class _R:                       # np Generator ile shuffle icin ince sarmalayici
        @staticmethod
        def shuffle(x):
            perm = rng.permutation(len(x))
            x[:] = [x[i] for i in perm]

    train, val, test = stratified_split(df, args.val, args.test, _R)
    subsets = nested_subsamples(train, df, _R)

    # ---------------- Yazim: stem listeleri (git'e giriyor) ----------------
    sd = Path(args.splits_dir)
    sd.mkdir(parents=True, exist_ok=True)

    def write(name, stems):
        (sd / f"{name}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")

    write("train", train)
    write("val", val)
    write("test", test)
    for lvl, stems in subsets.items():
        write(f"train_{lvl}", stems)

    # ---------------- Ultralytics icin mutlak yollu listeler ---------------
    #
    # DIKKAT -- Ultralytics etiket dosyasini, goruntu yolundaki son
    # '/images/' parcasini '/labels/' ile degistirerek arar.
    # Bu yuzden listeler HAM AGAR klasorunu degil, convert.py'nin urettigi
    # data/processed/images/ altini gostermek ZORUNDA.
    # Ham yol yazilirsa etiket bulunamaz, Ultralytics uyarir ama durmaz ve
    # model "bu goruntulerde nesne yok" diye ogrenir. Sessiz, olumcul hata.
    lists = data / "lists"
    lists.mkdir(exist_ok=True)
    img_dir = data / "images"
    path_of = {}
    for s, raw in df.set_index("stem")["image_path"].to_dict().items():
        p = img_dir / Path(raw).name          # convert.py buraya symlink attı
        if not p.exists():                    # uzanti farkliysa ara
            eslesen = sorted(img_dir.glob(f"{s}.*"))
            if not eslesen:
                sys.exit(f"HATA: {img_dir} altinda {s} icin goruntu yok. "
                         f"once convert.py calistir.")
            p = eslesen[0]
        path_of[s] = p

    def write_paths(name, stems):
        # DIKKAT: .resolve() KULLANMA. data/processed/images/ altindaki dosyalar
        # ham AGAR klasorune SYMLINK; resolve() onlari takip eder ve yolu ham
        # klasore cevirir -> '/images/' parcasi kaybolur -> Ultralytics etiketi
        # bulamaz. path_of zaten mutlak yol (data .resolve() edilmis durumda).
        (lists / f"{name}.txt").write_text(
            "\n".join(str(path_of[s]) for s in stems) + "\n",
            encoding="utf-8")

    for name, stems in [("train", train), ("val", val), ("test", test)]:
        write_paths(name, stems)
    for lvl, stems in subsets.items():
        write_paths(f"train_{lvl}", stems)

    # ---------------- Dogrulama ----------------
    tamam = True

    print("\n=== ULTRALYTICS ETIKET COZUMLEME KONTROLU ===")
    # Kontrol, DOSYAYA YAZILAN satirlarin AYNISI uzerinden yapilmali.
    # (Onceki surum path_of'u kontrol ediyordu ama dosyaya resolve() edilmis
    #  hali yaziliyordu; kontrol gecti, egitim patladi.)
    yazilan = [l.strip() for l in
               (lists / "train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    ham_yol = [y for y in yazilan if "/images/" not in y]
    eksik = [y for y in yazilan[:200]
             if not Path(y.replace("/images/", "/labels/")).with_suffix(".txt").exists()]
    print(f"  listede '/images/' parcasi olmayan satir: {len(ham_yol)}"
          f"{'  <-- SORUN (symlink resolve edilmis olabilir)' if ham_yol else ''}")
    print(f"  '/images/' -> '/labels/' ile etiket bulunamayan: {len(eksik)}"
          f"{'  <-- SORUN' if eksik else ''}")
    if yazilan:
        print(f"  ornek satir: {yazilan[0]}")
    tamam &= not eksik and not ham_yol

    print("\n=== IC ICE OLMA KONTROLU ===")
    zincir = [("train_10", subsets[10], "train_25", subsets[25]),
              ("train_25", subsets[25], "train_50", subsets[50]),
              ("train_50", subsets[50], "train", train)]
    for an, a, bn, b in zincir:
        ok = set(a).issubset(set(b))
        tamam &= ok
        print(f"  {an} ⊂ {bn} : {'EVET' if ok else 'HAYIR  <-- SORUN'}")

    print("\n=== SIZINTI KONTROLU (kumeler kesismiyor mu) ===")
    for an, a, bn, b in [("train", train, "val", val), ("train", train, "test", test),
                         ("val", val, "test", test)]:
        kesisim = set(a) & set(b)
        tamam &= not kesisim
        print(f"  {an} ∩ {bn} : {len(kesisim)} goruntu"
              f"{'  <-- SORUN' if kesisim else ''}")

    # ---------------- Dagilim tablosu ----------------
    rows = {}
    for name, stems in [("train", train), ("val", val), ("test", test)]:
        rows[name] = class_table(stems, df)
    for lvl in LEVELS:
        rows[f"train_{lvl}"] = class_table(subsets[lvl], df)
    tab = pd.DataFrame(rows).T
    tab.to_csv(sd / "dagilim.csv")

    print("\n=== KUME BASINA DAGILIM (kutu sayisi) ===")
    print(tab.to_string())

    # sinif paylarinin seviyeler arasi korunup korunmadigi
    pay = tab[CLASS_ORDER].div(tab["kutu"], axis=0).mul(100).round(1)
    print("\n=== SINIF PAYLARI (%) -- seviyeler arasi benzer olmali ===")
    print(pay.to_string())

    print(f"\nstem listeleri -> {sd}/        (git'e commit et)")
    print(f"yol listeleri  -> {lists}/   (gitignore, yeniden uretilebilir)")

    # ---------------- Kucuk veri uyarilari ----------------
    uyari = []
    if len(val) == 0 or len(test) == 0:
        uyari.append(
            f"val={len(val)}, test={len(test)} -- kume BOS. Tabakalar cok kucuk "
            "(her tur kombinasyonundan birkac goruntu var). Tam veride duzelir; "
            "demo pakette bolme anlamli degil.")
    seviyeler = [("train", train)] + [(f"train_{l}", subsets[l]) for l in LEVELS]
    for (an, a), (bn, b) in zip(seviyeler, seviyeler[1:]):
        if len(a) == len(b):
            uyari.append(f"{an} ve {bn} ayni buyuklukte ({len(a)}) -- her tabakadan "
                         "en az 1 goruntu tutuldugu icin daralma duruyor. "
                         "Tam veride sorun olmaz.")
    if uyari:
        print("\n=== UYARILAR ===")
        for u in uyari:
            print(f"  ! {u}")

    print(f"\nDurum: {'TAMAM' if tamam else 'SORUN VAR -- yukariya bak'}\n")


if __name__ == "__main__":
    main()