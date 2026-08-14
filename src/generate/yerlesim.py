#!/usr/bin/env python3
"""
yerlesim.py — sentetik plaklarda "nereye, ne buyuklukte, hangi sinif" sorusunu cozer.

Faz 3'un ilk adimi. Difuzyon YOK. Bu betik yalnizca koordinat uretir:
kutu koordinatlari uretimden ONCE belli oldugu icin etiket kendiliginden olusur.

Uc alt komut:
  uydur   gercek etiketlerden dagilim parametrelerini cikarir  -> yerlesim_<seviye>.json
  ornekle o parametrelerden sentetik yerlesimler uretir        -> plan_*.json + etiket_*.txt
  dogrula uretilen yerlesimi gercek veriyle karsilastirir       -> konsol raporu

Ilgili kararlar (decisions.md):
  3.5   --seviye zorunlu, varsayilan YOK  (sizinti karsiti)
  3.7   ciktilar seviyeyi adinda tasir
  3.10  plak cemberi sabit: merkez = goruntu merkezi, R = 0.465 * genislik
  3.11  aci duzgun, yaricap ampirik ters-CDF
  3.12  cakismaya izin var, ortusme derinligi 0.63'te kirpilir
  3.13  maske dairesel, etiket kutusu kare
  3.14  boyut sinif basina ayri
  3.16  bir baskin tur + az sayida ikincil tur
  3.17  sayim lognormal
  3.18  --naif = ablasyon A3
  3.19  Strauss etkilesim parametresi gamma, gercek degme oranina otomatik kalibre
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter
from pathlib import Path

import numpy as np

# --- karar 3.10 -------------------------------------------------------------
PLAK_R_ORANI = 0.465        # R / goruntu_genisligi
# --- karar 3.12 -------------------------------------------------------------
MAKS_ORTUSME = 0.63         # 0 = tegetsel, 1 = tam ic ice
# --- karar 3.13 -------------------------------------------------------------
ELIPS_JITTER = 0.03         # kutularin %96.9'u tam kare; kalanina bu kadar sapma
# --- ornekleme --------------------------------------------------------------
DENEME_SAYISI = 400         # bir koloni icin konum reddetme denemesi
KALIBRE_PLAK  = 250         # gamma kalibrasyonunda simule edilen plak sayisi

KDEMET = [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]   # quantile izgarasi


# ===========================================================================
# UYDUR
# ===========================================================================

def _etiket_oku(stem_yollari, etiket_kok: Path, siniflar):
    """YOLO etiketlerini oku. Donen: plak listesi, her plak = koloni dict listesi."""
    plaklar = []
    for p in stem_yollari:
        et = etiket_kok / (Path(p).stem + ".txt")
        if not et.exists():
            print(f"  UYARI: etiket yok, atlandi: {et}", file=sys.stderr)
            continue
        kol = []
        for satir in et.read_text().split("\n"):
            if not satir.strip():
                continue
            c, xc, yc, w, h = satir.split()
            kol.append(dict(sinif=siniflar[int(c)], xc=float(xc), yc=float(yc),
                            w=float(w), h=float(h)))
        if kol:
            plaklar.append(dict(stem=Path(p).stem, koloniler=kol))
    return plaklar


def _degen_oran(plaklar):
    """Bir komsusuna degen koloni yuzdesi. plaklar: [[(x,y,cap),...],...]"""
    degen = tum = 0
    for kol in plaklar:
        if len(kol) < 2:
            tum += len(kol); continue
        P = np.array([[k[0], k[1]] for k in kol]); S = np.array([k[2] / 2 for k in kol])
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        np.fill_diagonal(D, 1e9)
        degen += (D < (S[:, None] + S[None, :])).any(1).sum(); tum += len(kol)
    return 100.0 * degen / max(tum, 1)


def _degen_gb(plaklar, seed=7, B=2000):
    """Degme oraninin plak-duzeyinde onyukleme (bootstrap) %90 guven araligi.
    Olcut plak icinde son derece korele — etkin ornek buyuklugu KOLONI degil PLAK
    sayisidir. 10 plakla aralik cok genis cikar; bu bir hata degil, bilgi."""
    rng = np.random.default_rng(seed)
    n = len(plaklar)
    v = [_degen_oran([plaklar[i] for i in rng.integers(0, n, n)]) for _ in range(B)]
    return float(np.percentile(v, 5)), float(np.percentile(v, 95))


def _kalibre_gamma(par, hedef, seed=12345):
    """gamma'yi ikili aramayla gercek degme oranina oturt (karar 3.19).

    Tarifler (sayim/sinif/cap) bir kez uretilip her gamma denemesinde AYNEN
    kullaniliyor. Boylece gamma degistiginde degisen tek sey yerlestirme oluyor;
    aksi halde rastgele akis kayar ve arama gurultuye oturur."""
    t_rng = np.random.default_rng(seed)
    tarifler = [tarif(t_rng, par) for _ in range(KALIBRE_PLAK)]

    def dene(g):
        rng = np.random.default_rng(seed + 1)
        pl = [[(k["xc"], k["yc"], k["cap"])
               for k in bir_plak(rng, par, gamma=g, hazir_tarif=t)[0]] for t in tarifler]
        return _degen_oran(pl)

    lo, hi = 0.0, 1.0
    o_hi = dene(hi)
    if o_hi <= hedef:                      # gamma=1'de bile hedefin altinda
        return 1.0, o_hi, False
    for _ in range(9):
        mid = (lo + hi) / 2
        if dene(mid) < hedef: lo = mid
        else:                 hi = mid
    g = (lo + hi) / 2
    return g, dene(g), True


def uydur(args):
    siniflar = Path(args.siniflar).read_text().split()
    liste = [l for l in Path(args.liste).read_text().split("\n") if l.strip()]
    plaklar = _etiket_oku(liste, Path(args.etiketler), siniflar)
    if not plaklar:
        sys.exit("HATA: hic etiket okunamadi.")

    R = PLAK_R_ORANI                       # normalize birimde (goruntu genisligi = 1)
    cx = cy = 0.5

    # --- radyal dagilim (karar 3.11) ---------------------------------------
    r_norm, boyut_sinif, kompozisyon, sayimlar = [], {}, [], []
    for pl in plaklar:
        sayimlar.append(len(pl["koloniler"]))
        say = Counter(k["sinif"] for k in pl["koloniler"])
        kompozisyon.append({s: n / len(pl["koloniler"]) for s, n in say.items()})
        for k in pl["koloniler"]:
            r_norm.append(math.hypot(k["xc"] - cx, k["yc"] - cy) / R)
            boyut_sinif.setdefault(k["sinif"], []).append((k["w"] + k["h"]) / 2)
    r_norm = np.clip(np.array(r_norm), 0, 1.0)

    # --- sayim (karar 3.17) -------------------------------------------------
    ns = np.array(sayimlar, dtype=float)
    ln = np.log(ns)

    par = {
        "seviye": args.seviye,
        "kaynak": {
            "liste": str(args.liste),
            "plak_sayisi": len(plaklar),
            "koloni_sayisi": int(len(r_norm)),
            "UYARI": ("Plak sayisi < 50. Bu parametreler dagilim AILESI secmeye yeter, "
                      "kestirim icin yetmez.") if len(plaklar) < 50 else None,
        },
        "plak": {"R_orani": PLAK_R_ORANI, "merkez": [cx, cy]},
        "radyal": {"q": KDEMET, "deger": [float(np.percentile(r_norm, q)) for q in KDEMET]},
        # lognormal medyani = exp(mu). mu icin ln(n)'in MEDYANI kullaniliyor:
        # az sayida plakla ortalama, tek bir kalabalik plaktan (n=125) etkileniyor.
        "sayim": {"aile": "lognormal", "ln_ort": float(np.median(ln)),
                  "ln_std": float(ln.std(ddof=1)) if len(ln) > 1 else 0.3,
                  "min": int(ns.min()), "maks": int(ns.max())},
        "boyut": {s: {"q": KDEMET, "deger": [float(np.percentile(v, q)) for q in KDEMET],
                      "n": len(v)}
                  for s, v in sorted(boyut_sinif.items())},
        "kompozisyon": kompozisyon,
        "degme": {"maks_ortusme": MAKS_ORTUSME, "gamma": 1.0},
        "birim": "goruntu genisliginin kesri (YOLO normalize)",
    }

    # --- karar 3.19: gamma'yi gercek degme oranina kalibre et ---------------
    gercek_pl = [[(k["xc"], k["yc"], (k["w"] + k["h"]) / 2) for k in pl["koloniler"]]
                 for pl in plaklar]
    hedef = _degen_oran(gercek_pl)
    gb = _degen_gb(gercek_pl)
    g, ulasilan, basarili = _kalibre_gamma(par, hedef)
    par["degme"].update(gamma=round(g, 4), hedef_degen_oran=round(hedef, 2),
                        hedef_gb90=[round(gb[0], 2), round(gb[1], 2)],
                        kalibre_degen_oran=round(ulasilan, 2), kalibre_basarili=basarili,
                        gecici=len(plaklar) < 50)
    cikti = Path(args.cikti)
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps(par, indent=2, ensure_ascii=False))

    print(f"[uydur] seviye={args.seviye}  {len(plaklar)} plak / {len(r_norm)} koloni")
    print(f"        sayim: medyan {np.median(ns):.0f}  aralik {ns.min():.0f}-{ns.max():.0f}")
    print(f"        radyal medyan r/R = {np.median(r_norm):.3f}")
    for s, v in sorted(boyut_sinif.items()):
        print(f"        {s:<14} n={len(v):5d}  medyan {np.median(v)*2048:6.1f} px (2048'de)")
    d = par["degme"]
    print(f"        degme: gercek %{d['hedef_degen_oran']:.1f} "
          f"(GB90 %{d['hedef_gb90'][0]:.1f}-%{d['hedef_gb90'][1]:.1f}) "
          f"-> gamma={d['gamma']:.3f} ile %{d['kalibre_degen_oran']:.1f}")
    if d["gecici"]:
        print(f"  🔴 gamma GECICI: {len(plaklar)} plakta hedefin guven araligi "
              f"{d['hedef_gb90'][1]-d['hedef_gb90'][0]:.0f} puan genisliginde. "
              f"Mekanizma dogru, sayi gurultu. Tam veride tekrar uydurulacak.")
    if not d["kalibre_basarili"]:
        print("  🔴 gamma=1'de bile hedef degme orani asilamadi — plak gercege gore SEYREK. "
              "Sayim veya boyut dagilimi kontrol edilmeli.")
    if par["kaynak"]["UYARI"]:
        print(f"  🔴 {par['kaynak']['UYARI']}")
    print(f"        -> {cikti}")


# ===========================================================================
# ORNEKLE
# ===========================================================================

def _ters_cdf(rng, q, deger, boyut):
    """Ampirik ters-CDF ornekleme (karar 3.11)."""
    u = rng.random(boyut) * 100.0
    return np.interp(u, q, deger)


def _ortusme(d, a, b):
    """Iki dairenin ic ice gecme derinligi: 0 = tegetsel, 1 = tam ic ice."""
    t = a + b
    return 0.0 if d >= t else 1.0 - d / t


def tarif(rng, par, naif=False):
    """Bir plagin 'ne'sini uretir: koloni sayisi, siniflar, caplar.
    Yerlestirmeden AYRI tutuluyor — kalibrasyonda gamma degisirken tarifin sabit
    kalmasi gerekiyor (ortak rastgele sayilar / varyans azaltma)."""
    sy = par["sayim"]
    n = int(round(math.exp(rng.normal(sy["ln_ort"], sy["ln_std"]))))
    n = max(sy["min"], min(sy["maks"], n))

    if naif:                                    # A3: sinif dagilimi da duzlestirilir
        etiketler = list(rng.choice(list(par["boyut"].keys()), size=n))
    else:                                       # karar 3.16
        komp = par["kompozisyon"][rng.integers(len(par["kompozisyon"]))]
        turler = list(komp.keys())
        pay = np.array([komp[t] for t in turler], dtype=float); pay /= pay.sum()
        etiketler = list(rng.choice(turler, size=n, p=pay))

    caplar = []                                 # karar 3.14
    for s_ in etiketler:
        b = par["boyut"][rng.choice(list(par["boyut"].keys()))] if naif else par["boyut"][s_]
        caplar.append(float(_ters_cdf(rng, b["q"], b["deger"], 1)[0]))
    return etiketler, np.array(caplar)


def bir_plak(rng, par, naif=False, gamma=None, hazir_tarif=None):
    """Tek bir sentetik plagin yerlesimini uretir.

    gamma (karar 3.19): Strauss etkilesim parametresi. Bir aday konum baska bir
    koloniye DEGIYORSA gamma olasilikla kabul edilir, 1-gamma olasilikla reddedilip
    yeniden denenir. gamma=1 -> saf rastgele (cok kalabalik cikar), gamma=0 -> hic
    degme yok. Gercek degme oranina `uydur` icinde ikili aramayla oturtulur.
    """
    if gamma is None:
        gamma = par.get("degme", {}).get("gamma", 1.0)
    R, (cx, cy) = par["plak"]["R_orani"], par["plak"]["merkez"]

    etiketler, caplar = hazir_tarif if hazir_tarif is not None else tarif(rng, par, naif)

    # konum ------------------------------------------------------------------
    sira = np.argsort(-caplar)          # buyukten kucuge: sikisik plakta yerlesim kolaylasir
    yerlesik, zorlanan, dusen = [], 0, 0
    for i in sira:
        yc_cap = caplar[i] / 2.0
        yedek_gecerli = None      # geometrik olarak uygun ama gamma reddetti
        yedek_ihlal, yedek_skor = None, 1e9   # ortusme sinirini asan en iyi aday
        for _ in range(DENEME_SAYISI):
            if naif:
                # A3: plak cemberi yok, tum kareye duzgun
                x, y = rng.random(), rng.random()
            else:
                r = float(_ters_cdf(rng, par["radyal"]["q"], par["radyal"]["deger"], 1)[0])
                th = rng.random() * 2 * math.pi
                x, y = cx + r * R * math.cos(th), cy + r * R * math.sin(th)
                if math.hypot(x - cx, y - cy) + yc_cap > R:
                    continue                      # koloni plagin disina tasamaz
            if not (yc_cap <= x <= 1 - yc_cap and yc_cap <= y <= 1 - yc_cap):
                continue                          # kutu goruntu disina tasamaz
            if naif:
                yerlesik.append((x, y, caplar[i], etiketler[i]))
                yedek_gecerli = yedek_ihlal = None; break
            enb = 0.0
            for (px, py, pc, _) in yerlesik:
                enb = max(enb, _ortusme(math.hypot(x - px, y - py), yc_cap, pc / 2.0))
                if enb > MAKS_ORTUSME:
                    break
            if enb > MAKS_ORTUSME:                 # sert sinir (karar 3.12)
                if enb < yedek_skor:
                    yedek_ihlal, yedek_skor = (x, y), enb
                continue
            if enb == 0.0 or rng.random() < gamma:  # Strauss kabulu (karar 3.19)
                yerlesik.append((x, y, caplar[i], etiketler[i]))
                yedek_gecerli = yedek_ihlal = None; break
            if yedek_gecerli is None:               # gamma reddetti ama konum gecerli
                yedek_gecerli = (x, y)
        else:
            # DENEME_SAYISI doldu. Koloniyi SESSIZCE DUSURME — sayim dagilimi bozulur.
            # Once gamma'nin reddettigi gecerli konumu, o da yoksa en az ihlal edeni al.
            ye = yedek_gecerli or yedek_ihlal
            if ye is not None:
                yerlesik.append((ye[0], ye[1], caplar[i], etiketler[i]))
                zorlanan += 1
            else:
                dusen += 1

    # 5) kutu: dairesel maske, kare etiket (karar 3.13) ----------------------
    kol = []
    for (x, y, c, s) in yerlesik:
        j = 1.0 + rng.normal(0, ELIPS_JITTER)
        w, h = c * j, c / j
        kol.append(dict(sinif=s, xc=round(x, 6), yc=round(y, 6),
                        w=round(w, 6), h=round(h, 6),
                        maske="daire", cap=round(c, 6)))
    return kol, zorlanan, dusen


def ornekle(args):
    par = json.loads(Path(args.parametre).read_text())
    if par["seviye"] != args.seviye:
        sys.exit(f"HATA (karar 3.7): parametre dosyasi seviye={par['seviye']}, "
                 f"--seviye {args.seviye} verildi. Seviyeler eslesmeli.")
    siniflar = Path(args.siniflar).read_text().split()
    rng = np.random.default_rng(args.seed)

    cikti = Path(args.cikti); (cikti / "planlar").mkdir(parents=True, exist_ok=True)
    (cikti / "labels").mkdir(parents=True, exist_ok=True)
    onek = "a3naif" if args.naif else f"s{args.seviye}"

    toplam, zor, dus = 0, 0, 0
    for i in range(args.adet):
        kol, z, d = bir_plak(rng, par, naif=args.naif)
        toplam += len(kol); zor += z; dus += d
        ad = f"{onek}_{args.seed}_{i:05d}"
        (cikti / "planlar" / f"{ad}.json").write_text(json.dumps(
            dict(ad=ad, seviye=args.seviye, naif=args.naif, seed=args.seed,
                 plak=par["plak"], koloniler=kol), indent=1, ensure_ascii=False))
        (cikti / "labels" / f"{ad}.txt").write_text("\n".join(
            f"{siniflar.index(k['sinif'])} {k['xc']:.6f} {k['yc']:.6f} {k['w']:.6f} {k['h']:.6f}"
            for k in kol) + "\n")

    print(f"[ornekle] {args.adet} plak / {toplam} koloni  (ort {toplam/args.adet:.1f})")
    print(f"          mod: {'A3 NAIF (ablasyon)' if args.naif else 'ana hat'}  "
          f"seviye={args.seviye}  seed={args.seed}")
    if zor:
        print(f"  UYARI: {zor} koloni (%{100*zor/toplam:.1f}) {DENEME_SAYISI} denemede "
              f"rahat yer bulamadi, yedek konuma yerlestirildi.")
    if dus:
        print(f"  🔴 {dus} koloni (%{100*dus/(toplam+dus):.1f}) HIC yerlestirilemedi. "
              f"Sayim dagilimi asagi kayar — DENEME_SAYISI artirilmali.")
    print(f"          -> {cikti}")


# ===========================================================================
# DOGRULA
# ===========================================================================

def _istatistik(plaklar, R, cx, cy):
    """plaklar: [[(xc,yc,cap), ...], ...] -> olcut sozlugu"""
    rn, degen, tum, sayim, ort = [], 0, 0, [], []
    for kol in plaklar:
        if not kol: continue
        sayim.append(len(kol))
        P = np.array([[k[0], k[1]] for k in kol])
        S = np.array([k[2] / 2 for k in kol])
        for k in kol:
            rn.append(math.hypot(k[0] - cx, k[1] - cy) / R)
        if len(kol) > 1:
            D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
            np.fill_diagonal(D, 1e9)
            T = S[:, None] + S[None, :]
            deg = D < T
            degen += deg.any(1).sum(); tum += len(kol)
            ii, jj = np.where(np.triu(deg, 1))
            ort += [1 - D[a, b] / T[a, b] for a, b in zip(ii, jj)]
    rn = np.array(rn); sayim = np.array(sayim)
    # Clark-Evans
    ce = []
    for kol in plaklar:
        if len(kol) < 5: continue
        P = np.array([[k[0], k[1]] for k in kol])
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        np.fill_diagonal(D, 1e9)
        ce.append(D.min(1).mean() / (0.5 / math.sqrt(len(kol) / (math.pi * R ** 2))))
    return dict(
        n_medyan=float(np.median(sayim)), n_min=int(sayim.min()), n_maks=int(sayim.max()),
        r_medyan=float(np.median(rn)), r_kare_ort=float(np.mean(rn ** 2)),
        dis_halka=float(100 * (rn >= math.sqrt(0.8)).mean()),
        degen_oran=float(100 * degen / max(tum, 1)),
        ortusme_medyan=float(np.median(ort)) if ort else 0.0,
        ortusme_maks=float(np.max(ort)) if ort else 0.0,
        ce_medyan=float(np.median(ce)) if ce else float("nan"),
    )


def dogrula(args):
    par = json.loads(Path(args.parametre).read_text())
    R, (cx, cy) = par["plak"]["R_orani"], par["plak"]["merkez"]
    siniflar = Path(args.siniflar).read_text().split()

    gercek = []
    for p in [l for l in Path(args.liste).read_text().split("\n") if l.strip()]:
        et = Path(args.etiketler) / (Path(p).stem + ".txt")
        if not et.exists(): continue
        gercek.append([(float(s.split()[1]), float(s.split()[2]),
                        (float(s.split()[3]) + float(s.split()[4])) / 2)
                       for s in et.read_text().split("\n") if s.strip()])

    uret = []
    for f in sorted((Path(args.uretilen) / "planlar").glob("*.json")):
        j = json.loads(f.read_text())
        uret.append([(k["xc"], k["yc"], k["cap"]) for k in j["koloniler"]])

    g, u = _istatistik(gercek, R, cx, cy), _istatistik(uret, R, cx, cy)
    ad = {"n_medyan": "koloni sayisi (medyan)", "n_min": "koloni sayisi (min)",
          "n_maks": "koloni sayisi (maks)", "r_medyan": "r/R medyan",
          "r_kare_ort": "(r/R)^2 ortalama", "dis_halka": "dis %20 alanda koloni %",
          "degen_oran": "degen koloni %  <-- karar 3.12", "ortusme_medyan": "ortusme derinligi medyan",
          "ortusme_maks": "ortusme derinligi maks", "ce_medyan": "Clark-Evans medyan"}
    print(f"\n{'olcut':<38} {'GERCEK':>10} {'URETILEN':>10}   fark")
    print("-" * 74)
    for k in ad:
        gv, uv = g[k], u[k]
        f = "" if gv == 0 else f"{100*(uv-gv)/abs(gv):+.0f}%"
        isaret = ""
        if k == "degen_oran":
            lo, hi = par["degme"].get("hedef_gb90", [gv - 3, gv + 3])
            isaret = ("  ✅ gercegin GB90'i icinde" if lo <= uv <= hi
                      else f"  🔴 GB90 DISI (%{lo:.0f}-%{hi:.0f})")
        print(f"{ad[k]:<38} {gv:>10.2f} {uv:>10.2f}  {f:>6}{isaret}")
    print(f"\ngercek: {len(gercek)} plak · uretilen: {len(uret)} plak")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    alt = ap.add_subparsers(dest="komut", required=True)

    def ortak(p, seviye=True):
        if seviye:
            # karar 3.5: VARSAYILAN YOK. Seviye belirtilmeden calismaz.
            p.add_argument("--seviye", type=int, required=True, choices=[10, 25, 50, 100],
                           help="gercek veri seviyesi. ZORUNLU (karar 3.5) — "
                                "varsayilan yok, sizinti karsiti.")
        p.add_argument("--siniflar", default="data/processed/classes.txt")

    p = alt.add_parser("uydur", help="gercek etiketlerden dagilim cikar")
    ortak(p)
    p.add_argument("--liste", required=True, help="orn. data/processed/lists/train_25.txt")
    p.add_argument("--etiketler", default="data/processed/labels")
    p.add_argument("--cikti", required=True, help="orn. src/generate/yerlesim_25.json")
    p.set_defaults(fn=uydur)

    p = alt.add_parser("ornekle", help="sentetik yerlesim uret")
    ortak(p)
    p.add_argument("--parametre", required=True)
    p.add_argument("--adet", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cikti", required=True)
    p.add_argument("--naif", action="store_true",
                   help="ablasyon A3 (karar 3.18): plak cemberi yok, boyut tek havuzdan, "
                        "degme kisiti yok")
    p.set_defaults(fn=ornekle)

    p = alt.add_parser("dogrula", help="uretileni gercekle karsilastir")
    ortak(p, seviye=False)
    p.add_argument("--parametre", required=True)
    p.add_argument("--liste", required=True)
    p.add_argument("--etiketler", default="data/processed/labels")
    p.add_argument("--uretilen", required=True)
    p.set_defaults(fn=dogrula)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
