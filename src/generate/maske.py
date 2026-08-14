#!/usr/bin/env python3
"""
maske.py — yerlesim planindan difuzyona verilecek maskeyi ve arka plani hazirlar.

Faz 3'un ikinci adimi. Difuzyon HALA yok. Bu betik su uc seyi uretir:
  1. arka plan  : gercek bir plak goruntusu (o seviyenin train alt kumesinden)
  2. maske      : difuzyonun dolduracagi ikili bolge
  3. etiket     : zaten `yerlesim.py` uretti, buraya kopyalanir

Maskede iki tur bolge var:
  SENTETIK  koloni diskleri — buraya yeni koloni cizdirilecek
  SILME     arka plan plagindaki GERCEK kolonilerin uzeri — bunlar boyanip yok edilecek

Ikincisi kritik: AGAR'da bos plak yok. Arka plani oldugu gibi kullanirsak
goruntude etiketsiz gercek koloniler kalir; model "koloni = arka plan" ogrenir.
Bu, sentetik veriye sessizce yanlis negatif enjekte etmektir.

Iki alt komut:
  havuz  bir seviyenin arka plan havuzunu secer ve puanlar -> bg_havuz_<seviye>.json
  uret   plan + arka plan -> maske PNG + arka plan + etiket

Ilgili kararlar (decisions.md): 3.4 · 3.5 · 3.7 · 3.10 · 3.13 · 3.24 · 3.25 · 3.26
"""
from __future__ import annotations
import argparse, json, math, shutil, sys
from pathlib import Path

import numpy as np
import cv2

PLAK_R_ORANI = 0.465          # karar 3.10

# --- karar 3.24 -------------------------------------------------------------
# Sentetik maske = etiketlenen diskin TA KENDISI. Genisletme YOK.
# Difuzyon yalnizca maskenin icini boyayabildigi icin uretilen koloni etiket
# kutusunu TASIYAMAZ. "Etiket hatasi sifir" iddiasi buna dayaniyor.
SENTETIK_PAY = 1.00

# --- karar 3.25 -------------------------------------------------------------
# Silme maskesi ise CÖMERT genisletilir: gercek koloninin golgesi/halesi de gitmeli.
SILME_PAY = 1.35              # --silme-pay ile degistirilebilir
SILME_BULANIK = 9             # kenar yumusatma (tek sayi)


# ===========================================================================
# HAVUZ
# ===========================================================================

def _plak_maskesi(w, h):
    """Plak diski (karar 3.10). Disi difuzyona hic acilmaz."""
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), int(PLAK_R_ORANI * w), 255, -1)
    return m


def _etiket_oku(p: Path):
    if not p.exists():
        return []
    return [tuple(map(float, s.split()[1:5])) for s in p.read_text().split("\n") if s.strip()]


def _sinif_dagilimi(stemler, etiket_kok: Path, siniflar):
    """Bir plak kumesinde plak-basina baskin turun dagilimi."""
    say = {s: 0 for s in siniflar}
    for st in stemler:
        p = etiket_kok / f"{st}.txt"
        if not p.exists():
            continue
        c = {}
        for satir in p.read_text().split("\n"):
            if satir.strip():
                i = int(satir.split()[0]); c[i] = c.get(i, 0) + 1
        if c:
            say[siniflar[max(c, key=c.get)]] += 1
    t = sum(say.values()) or 1
    return {s: n / t for s, n in say.items()}, t


def havuz(args):
    global SILME_PAY
    SILME_PAY = args.silme_pay
    liste = [l for l in Path(args.liste).read_text().split("\n") if l.strip()]
    kayit = []
    for yol in liste:
        stem = Path(yol).stem
        kut = _etiket_oku(Path(args.etiketler) / f"{stem}.txt")
        # silinecek alanin plak diskine orani — kucuk olan iyi arka plandir
        alan = sum(math.pi * ((w + h) / 4 * SILME_PAY) ** 2 for _, _, w, h in kut)
        plak_alan = math.pi * PLAK_R_ORANI ** 2
        kayit.append(dict(stem=stem, koloni=len(kut),
                          silme_orani=round(alan / plak_alan, 4),
                          en_buyuk=round(max(((w + h) / 2 for _, _, w, h in kut), default=0), 4)))
    kayit.sort(key=lambda k: k["silme_orani"])

    esik = args.esik
    secili = [k for k in kayit if k["silme_orani"] <= esik]
    if not secili:
        sys.exit(f"HATA: hicbir plak silme_orani <= {esik} kosulunu saglamiyor. "
                 f"En iyisi {kayit[0]['stem']} ({kayit[0]['silme_orani']}). "
                 f"--esik yukselt ama karar 3.26'daki riski oku.")

    # --- karar 3.27: havuz sinif bakimindan yanli mi? ----------------------
    siniflar = Path(args.siniflar).read_text().split()
    ek = Path(args.etiketler)
    d_havuz, n_h = _sinif_dagilimi([k["stem"] for k in secili], ek, siniflar)
    d_tum, n_t = _sinif_dagilimi([k["stem"] for k in kayit], ek, siniflar)
    yanlilik = {s: round(d_havuz[s] - d_tum[s], 3) for s in siniflar}
    en_kotu = max(yanlilik.values(), key=abs) if yanlilik else 0.0

    cikti = Path(args.cikti); cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps(dict(
        seviye=args.seviye, kaynak_liste=str(args.liste), esik=esik,
        silme_pay=SILME_PAY, havuz_boyu=len(secili), aday_boyu=len(kayit),
        sinif_yanliligi=yanlilik, havuz_sinif_payi=d_havuz, tum_sinif_payi=d_tum,
        havuz=[k["stem"] for k in secili], ayrinti=kayit), indent=2, ensure_ascii=False))

    print(f"[havuz] seviye={args.seviye}  {len(secili)}/{len(kayit)} plak secildi "
          f"(silme_orani <= {esik})")
    print(f"        en temiz: " + ", ".join(f"{k['stem']}({k['silme_orani']:.3f})"
                                            for k in secili[:5]))
    print(f"        en kirli secili: {secili[-1]['stem']} ({secili[-1]['silme_orani']:.3f})")
    print("        baskin tur payi (havuz vs tumu):")
    for s_ in siniflar:
        im = "  <-- yanli" if abs(yanlilik[s_]) >= 0.15 else ""
        print(f"          {s_:<14} %{100*d_havuz[s_]:5.1f}  vs  %{100*d_tum[s_]:5.1f}"
              f"   ({yanlilik[s_]:+.2f}){im}")
    if len(secili) < 20:
        print(f"  🔴 Havuz {len(secili)} plak. Ayni arka plan defalarca kullanilacak; "
              f"sentetik cesitlilik duser ve FID/KID bunu yakalar (karar 3.26).")
    if abs(en_kotu) >= 0.15:
        print(f"  🔴 HAVUZ SINIF BAKIMINDAN YANLI (karar 3.27). Esik, silinecek ALANA "
              f"bakiyor; alan koloni BOYUTUYLA, boyut da SINIFLA belirleniyor. Yani esik "
              f"sessizce kucuk-koloni turlerinin plaklarini seciyor. Sentetik verinin "
              f"arka plani tek bir tur ailesinden gelirse bu sistematik bir kayma olur.")
    print(f"        -> {cikti}")


# ===========================================================================
# URET
# ===========================================================================

def maske_uret(plan, arka_kut, w, h):
    """Donen: (sentetik_maske, silme_maskesi, birlesik_maske) — hepsi uint8 0/255."""
    sen = np.zeros((h, w), np.uint8)
    for k in plan["koloniler"]:
        r = k["cap"] / 2 * SENTETIK_PAY
        cv2.circle(sen, (int(round(k["xc"] * w)), int(round(k["yc"] * h))),
                   max(1, int(round(r * w))), 255, -1)

    sil = np.zeros((h, w), np.uint8)
    for (xc, yc, bw, bh) in arka_kut:
        r = (bw + bh) / 4 * SILME_PAY
        cv2.circle(sil, (int(round(xc * w)), int(round(yc * h))),
                   max(1, int(round(r * w))), 255, -1)
    if SILME_BULANIK:
        sil = (cv2.GaussianBlur(sil, (SILME_BULANIK, SILME_BULANIK), 0) > 40).astype(np.uint8) * 255

    plak = _plak_maskesi(w, h)
    sen = cv2.bitwise_and(sen, plak)          # koloni plagin disina cikamaz
    sil = cv2.bitwise_and(sil, plak)
    return sen, sil, cv2.bitwise_or(sen, sil)


def uret(args):
    global SILME_PAY
    hav = json.loads(Path(args.havuz).read_text())
    SILME_PAY = hav.get("silme_pay", SILME_PAY)   # havuz hangi payla secildiyse o
    if hav["seviye"] != args.seviye:
        sys.exit(f"HATA (karar 3.7): havuz seviye={hav['seviye']}, --seviye {args.seviye}.")
    planlar = sorted((Path(args.planlar) / "planlar").glob("*.json"))
    if not planlar:
        sys.exit(f"HATA: {args.planlar}/planlar altinda plan yok.")

    rng = np.random.default_rng(args.seed)
    C = Path(args.cikti)
    for alt in ("maske", "arkaplan", "labels", "kayit"):
        (C / alt).mkdir(parents=True, exist_ok=True)

    kullanim, kapsam, tasan = {}, [], 0
    for pf in planlar:
        plan = json.loads(pf.read_text())
        if plan["seviye"] != args.seviye:
            sys.exit(f"HATA (karar 3.7): {pf.name} seviye={plan['seviye']}.")
        ad = plan["ad"]

        bg = hav["havuz"][int(rng.integers(len(hav["havuz"])))]
        kullanim[bg] = kullanim.get(bg, 0) + 1

        img = cv2.imread(str(Path(args.goruntuler) / f"{bg}.jpg"))
        if img is None:
            sys.exit(f"HATA: arka plan okunamadi: {bg}.jpg")
        h, w = img.shape[:2]
        arka_kut = _etiket_oku(Path(args.etiketler) / f"{bg}.txt")
        sen, sil, birlesik = maske_uret(plan, arka_kut, w, h)

        # sentetik disklerin ne kadari silme bolgesiyle cakisiyor (bilgi amacli)
        kapsam.append(float((cv2.bitwise_and(sen, sil) > 0).sum() / max((sen > 0).sum(), 1)))
        # etiketi goruntu disina tasan koloni var mi (olmamali)
        for k in plan["koloniler"]:
            if not (0 <= k["xc"] - k["w"] / 2 and k["xc"] + k["w"] / 2 <= 1
                    and 0 <= k["yc"] - k["h"] / 2 and k["yc"] + k["h"] / 2 <= 1):
                tasan += 1

        cv2.imwrite(str(C / "maske" / f"{ad}.png"), birlesik)
        if args.maske_ayri:
            cv2.imwrite(str(C / "maske" / f"{ad}_sentetik.png"), sen)
            cv2.imwrite(str(C / "maske" / f"{ad}_silme.png"), sil)
        cv2.imwrite(str(C / "arkaplan" / f"{ad}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 96])
        shutil.copy(Path(args.planlar) / "labels" / f"{ad}.txt", C / "labels" / f"{ad}.txt")
        (C / "kayit" / f"{ad}.json").write_text(json.dumps(dict(
            ad=ad, seviye=args.seviye, arka_plan=bg,
            arka_plan_kaynagi=hav["kaynak_liste"],       # sizinti denetim izi (karar 3.4)
            sentetik_koloni=len(plan["koloniler"]),
            silinen_gercek_koloni=len(arka_kut),
            sentetik_pay=SENTETIK_PAY, silme_pay=SILME_PAY), indent=1, ensure_ascii=False))

    n = len(planlar)
    tekrar = max(kullanim.values())
    print(f"[uret] {n} plak · seviye={args.seviye} · havuz {len(hav['havuz'])} arka plan")
    print(f"       arka plan tekrari: en cok {tekrar} kez ({100*tekrar/n:.1f}%), "
          f"kullanilmayan {len(hav['havuz'])-len(kullanim)} plak")
    print(f"       sentetik diskin silme bolgesiyle cakisan kismi: "
          f"ort %{100*np.mean(kapsam):.1f}")
    if tasan:
        print(f"  🔴 {tasan} etiket goruntu disina tasiyor — yerlesim.py kontrol edilmeli.")
    if tekrar / n > 0.15:
        print(f"  🔴 Bir arka plan uretimin %{100*tekrar/n:.0f}'ini tasiyor. "
              f"Cesitlilik riski (karar 3.26).")
    print(f"       -> {C}")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    alt = ap.add_subparsers(dest="komut", required=True)

    def ortak(p):
        # karar 3.5: varsayilan YOK
        p.add_argument("--seviye", type=int, required=True, choices=[10, 25, 50, 100],
                       help="ZORUNLU (karar 3.5). Arka plan havuzu da seviyeye bagli (3.4).")
        p.add_argument("--etiketler", default="data/processed/labels")

    p = alt.add_parser("havuz", help="seviyenin arka plan havuzunu sec")
    ortak(p)
    p.add_argument("--liste", required=True, help="orn. data/processed/lists/train_25.txt")
    p.add_argument("--siniflar", default="data/processed/classes.txt")
    p.add_argument("--silme-pay", type=float, default=SILME_PAY,
                   help="gercek koloniyi silerken yaricap carpani. Silinen alan bunun "
                        "KARESIYLE buyur.")
    p.add_argument("--esik", type=float, default=0.06,
                   help="silinecek alanin plak diskine orani ust siniri (varsayilan 0.06)")
    p.add_argument("--cikti", required=True)
    p.set_defaults(fn=havuz)

    p = alt.add_parser("uret", help="maske + arka plan + etiket uret")
    ortak(p)
    p.add_argument("--planlar", required=True, help="yerlesim.py ornekle ciktisi")
    p.add_argument("--havuz", required=True)
    p.add_argument("--goruntuler", default="data/processed/images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maske-ayri", action="store_true",
                   help="sentetik ve silme maskelerini ayrica yaz (gozle kontrol icin)")
    p.add_argument("--cikti", required=True)
    p.set_defaults(fn=uret)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
