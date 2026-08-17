#!/usr/bin/env python3
"""
Uretim hattinin sanity testi -- yerlesim.py + maske.py.

NEDEN VAR (karar 3.37):
  `metrics.py`'nin 33 testi var cunku kendi AP hesabina guvenilmiyordu.
  `yerlesim.py`'de en az onun kadar sinsi matematik var:
    - ampirik ters-CDF ornekleme (quantile izgarasi + np.interp)
    - Strauss gamma ikili aramasi
    - lognormal parametre kestirimi
  Ve bu dosyalar MAKALENIN VERISINI uretecek.

  `dogrula` alt komutu bir KARSILASTIRMA RAPORU, birim testi degil: toplu
  istatistikleri kiyasliyor. Yanlis bir quantile interpolasyonu medyani dogru
  tutup kuyrugu bozabilir ve `dogrula` bunu yakalamayabilir.

Kullanim:
    python src/generate/test_uretim.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maske as M          # noqa: E402
import yerlesim as Y       # noqa: E402

GECTI, KALDI = [], []


def kontrol(ad, kosul, detay=""):
    (GECTI if kosul else KALDI).append(ad)
    print(f"  [{'OK ' if kosul else 'HATA'}] {ad}" + (f"   {detay}" if detay else ""))


def ornek_par(gamma=1.0, n_min=20, n_maks=20, cap=0.05):
    """Sabit tarifli, tek sinifli, oynak olmayan bir parametre seti.
    Testlerde 'ne uretilecegi' belli olsun diye sayim ve boyut sabitlendi."""
    q = list(Y.KDEMET)
    return {
        "seviye": 100,
        "plak": {"R_orani": Y.PLAK_R_ORANI, "merkez": [0.5, 0.5]},
        # yaricap: p0=0, p100=0.9  -> diske yayilmis ama kenara dayanmiyor
        "radyal": {"q": q, "deger": [0.9 * (v / 100.0) for v in q]},
        "sayim": {"aile": "lognormal", "ln_ort": math.log(n_min), "ln_std": 0.0,
                  "min": n_min, "maks": n_maks},
        "boyut": {"S.aureus": {"q": q, "deger": [cap] * len(q), "n": 100}},
        "kompozisyon": [{"S.aureus": 1.0}],
        "degme": {"maks_ortusme": Y.MAKS_ORTUSME, "gamma": gamma},
    }


# ===========================================================================

def test_ters_cdf():
    print("\n1) Ampirik ters-CDF ornekleme (karar 3.11)")
    rng = np.random.default_rng(0)
    # duzgun U(0,1): quantile izgarasi = degerin kendisi
    q = [0, 10, 25, 50, 75, 90, 100]
    deger = [v / 100.0 for v in q]
    x = Y._ters_cdf(rng, q, deger, 200_000)
    kontrol("U(0,1) medyani 0.50", abs(np.median(x) - 0.5) < 0.01, f"{np.median(x):.4f}")
    kontrol("U(0,1) p90 = 0.90", abs(np.percentile(x, 90) - 0.9) < 0.01,
            f"{np.percentile(x, 90):.4f}")
    kontrol("aralik disina cikmiyor", x.min() >= 0 and x.max() <= 1,
            f"[{x.min():.4f}, {x.max():.4f}]")

    # dogrusal olmayan izgara: p50'de kirilma
    q2 = [0, 50, 100]
    d2 = [0.0, 0.1, 1.0]          # alt yarisi 0-0.1'e, ust yarisi 0.1-1'e sikisik
    y = Y._ters_cdf(rng, q2, d2, 200_000)
    kontrol("kirikli izgarada medyan 0.10", abs(np.median(y) - 0.1) < 0.005,
            f"{np.median(y):.4f}")
    kontrol("kirikli izgarada p25 = 0.05", abs(np.percentile(y, 25) - 0.05) < 0.005,
            f"{np.percentile(y, 25):.4f}")


def test_ortusme():
    print("\n2) Ortusme derinligi")
    kontrol("tegetsel -> 0.0", Y._ortusme(2.0, 1.0, 1.0) == 0.0)
    kontrol("ayrik -> 0.0", Y._ortusme(5.0, 1.0, 1.0) == 0.0)
    kontrol("merkezler cakisik -> 1.0", Y._ortusme(0.0, 1.0, 1.0) == 1.0)
    kontrol("yarisi -> 0.5", abs(Y._ortusme(1.0, 1.0, 1.0) - 0.5) < 1e-12)


def test_degen_oran():
    print("\n3) Degme orani -- elle bilinen vaka")
    # 4 koloni: ikisi degiyor, ikisi ayrik  -> %50
    kol = [(0.10, 0.10, 0.06),      # merkez mesafesi 0.05 < 0.03+0.03 = 0.06 -> degiyor
           (0.15, 0.10, 0.06),
           (0.50, 0.50, 0.02),
           (0.90, 0.90, 0.02)]
    o = Y._degen_oran([kol])
    kontrol("2/4 degiyor -> %50", abs(o - 50.0) < 1e-9, f"%{o:.1f}")
    kontrol("tek koloni -> %0", Y._degen_oran([[(0.5, 0.5, 0.02)]]) == 0.0)


def test_sayim_siniri():
    print("\n4) Koloni sayisi min/maks sinirinda kaliyor mu (karar 3.17)")
    par = ornek_par(); par["sayim"].update(ln_ort=math.log(40), ln_std=1.5,
                                           min=12, maks=125)
    rng = np.random.default_rng(1)
    ns = [len(Y.tarif(rng, par)[0]) for _ in range(3000)]
    kontrol("hicbir plak min altinda degil", min(ns) >= 12, f"min={min(ns)}")
    kontrol("hicbir plak maks ustunde degil", max(ns) <= 125, f"maks={max(ns)}")
    kontrol("medyan ~40 (ln_ort = ln 40)", abs(np.median(ns) - 40) <= 2,
            f"medyan={np.median(ns):.0f}")


def test_gamma_uclari():
    print("\n5) Strauss gamma uc degerleri (karar 3.19)")
    # gamma = 0: degen konum ASLA kabul edilmemeli
    par0 = ornek_par(gamma=0.0, n_min=25, n_maks=25, cap=0.05)
    rng = np.random.default_rng(2)
    pl0 = [[(k["xc"], k["yc"], k["cap"]) for k in Y.bir_plak(rng, par0, gamma=0.0)[0]]
           for _ in range(40)]
    o0 = Y._degen_oran(pl0)

    par1 = ornek_par(gamma=1.0, n_min=25, n_maks=25, cap=0.05)
    rng = np.random.default_rng(2)
    pl1 = [[(k["xc"], k["yc"], k["cap"]) for k in Y.bir_plak(rng, par1, gamma=1.0)[0]]
           for _ in range(40)]
    o1 = Y._degen_oran(pl1)

    kontrol("gamma=0 -> degme orani cok dusuk", o0 < 5.0, f"%{o0:.1f}")
    kontrol("gamma=1 -> gamma=0'dan belirgin yuksek", o1 > o0 + 10,
            f"gamma1=%{o1:.1f}  gamma0=%{o0:.1f}")


def test_gamma_monoton():
    print("\n6) gamma buyudukce degme orani artiyor mu (ikili arama bunu varsayiyor)")
    par = ornek_par(n_min=25, n_maks=25, cap=0.05)
    oranlar = []
    for g in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rng = np.random.default_rng(3)
        pl = [[(k["xc"], k["yc"], k["cap"]) for k in Y.bir_plak(rng, par, gamma=g)[0]]
              for _ in range(40)]
        oranlar.append(Y._degen_oran(pl))
    artan = all(oranlar[i] <= oranlar[i + 1] + 1.0 for i in range(len(oranlar) - 1))
    kontrol("degme orani gamma ile monoton artiyor", artan,
            " -> ".join(f"{o:.1f}" for o in oranlar))


def test_sert_sinir():
    print("\n7) Ortusme sert siniri asilmiyor mu (karar 3.12)")
    par = ornek_par(gamma=1.0, n_min=45, n_maks=45, cap=0.06)   # bilerek kalabalik
    rng = np.random.default_rng(4)
    en_derin = 0.0
    for _ in range(30):
        kol, zor, dus = Y.bir_plak(rng, par)
        P = np.array([[k["xc"], k["yc"]] for k in kol])
        S = np.array([k["cap"] / 2 for k in kol])
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        np.fill_diagonal(D, 1e9)
        T = S[:, None] + S[None, :]
        ii, jj = np.triu_indices(len(kol), 1)
        d = 1 - D[ii, jj] / T[ii, jj]
        if len(d):
            en_derin = max(en_derin, d.max())
    kontrol(f"en derin ortusme <= {Y.MAKS_ORTUSME} (+tolerans)",
            en_derin <= Y.MAKS_ORTUSME + 0.02, f"{en_derin:.3f}")


def test_sinirlar():
    print("\n8) Koloniler plagin ve goruntunun icinde mi (karar 3.10)")
    par = ornek_par(n_min=30, n_maks=30, cap=0.05)
    rng = np.random.default_rng(5)
    R = par["plak"]["R_orani"]
    disari_plak = disari_kare = 0
    for _ in range(50):
        for k in Y.bir_plak(rng, par)[0]:
            if math.hypot(k["xc"] - 0.5, k["yc"] - 0.5) + k["cap"] / 2 > R + 1e-9:
                disari_plak += 1
            if not (0 <= k["xc"] - k["w"] / 2 and k["xc"] + k["w"] / 2 <= 1
                    and 0 <= k["yc"] - k["h"] / 2 and k["yc"] + k["h"] / 2 <= 1):
                disari_kare += 1
    kontrol("plak diskinin disina tasan koloni yok", disari_plak == 0, f"{disari_plak}")
    kontrol("goruntu disina tasan etiket yok", disari_kare == 0, f"{disari_kare}")


def test_koloni_dusmuyor():
    print("\n9) Koloni sessizce dusuruluyor mu (karar 3.22)")
    par = ornek_par(gamma=0.05, n_min=35, n_maks=35, cap=0.07)  # bilerek zor
    rng = np.random.default_rng(6)
    eksik = toplam_dusen = 0
    for _ in range(30):
        kol, zor, dus = Y.bir_plak(rng, par)
        toplam_dusen += dus
        if len(kol) + dus != 35:
            eksik += 1
    kontrol("uretilen + dusen = istenen (sessiz kayip yok)", eksik == 0, f"{eksik} plak")
    kontrol("dusenler raporlaniyor (sifir olsa bile sayac var)",
            isinstance(toplam_dusen, int), f"dusen={toplam_dusen}")


def test_belirlenimci():
    print("\n10) Ayni seed -> ayni cikti")
    par = ornek_par(n_min=25, n_maks=25)
    a = Y.bir_plak(np.random.default_rng(7), par)[0]
    b = Y.bir_plak(np.random.default_rng(7), par)[0]
    c = Y.bir_plak(np.random.default_rng(8), par)[0]
    kontrol("seed 7 iki kez -> ayni", json.dumps(a, sort_keys=True) ==
            json.dumps(b, sort_keys=True))
    kontrol("seed 8 -> farkli", json.dumps(a, sort_keys=True) !=
            json.dumps(c, sort_keys=True))


def test_naif_ablasyon():
    print("\n11) A3 naif mod ana hattan farkli mi (karar 3.18)")
    par = ornek_par(n_min=30, n_maks=30, cap=0.05)
    par["boyut"]["E.coli"] = {"q": list(Y.KDEMET), "deger": [0.15] * len(Y.KDEMET), "n": 50}
    R = par["plak"]["R_orani"]

    rng = np.random.default_rng(9)
    ana = [k for _ in range(30) for k in Y.bir_plak(rng, par)[0]]
    rng = np.random.default_rng(9)
    naif = [k for _ in range(30) for k in Y.bir_plak(rng, par, naif=True)[0]]

    r_ana = np.array([math.hypot(k["xc"] - .5, k["yc"] - .5) / R for k in ana])
    r_naif = np.array([math.hypot(k["xc"] - .5, k["yc"] - .5) / R for k in naif])
    kontrol("ana hat plak diskinde kaliyor", r_ana.max() <= 1.0, f"maks {r_ana.max():.3f}")
    kontrol("naif mod plagin DISINA tasiyor", (r_naif > 1.0).mean() > 0.10,
            f"%{100*(r_naif > 1.0).mean():.1f} disarida")

    # naif modda boyut sinifa kosullanmiyor -> S.aureus'a E.coli capi dusebilmeli
    caplar_s = {round(k["cap"], 4) for k in naif if k["sinif"] == "S.aureus"}
    kontrol("naif modda boyut sinifa kosullu DEGIL", len(caplar_s) > 1,
            f"S.aureus'ta {len(caplar_s)} farkli cap")


def test_seviye_kilidi():
    print("\n12) Seviye uyusmazligi programi durduruyor mu (karar 3.7)")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        par = ornek_par(); par["seviye"] = 25
        (td / "p.json").write_text(json.dumps(par))
        (td / "siniflar.txt").write_text("S.aureus\n")

        class A:
            seviye = 10                      # parametre 25, istenen 10
            parametre = str(td / "p.json")
            siniflar = str(td / "siniflar.txt")
            adet, seed, cikti, naif = 1, 0, str(td / "out"), False
        try:
            Y.ornekle(A())
            kontrol("seviye uyusmazliginda duruyor", False, "durmadi!")
        except SystemExit as e:
            kontrol("seviye uyusmazliginda duruyor", "3.7" in str(e), str(e)[:60])


def test_maske_alani():
    print("\n13) Maske alani analitik disk alanina esit mi (karar 3.24)")
    kontrol("SENTETIK_PAY == 1.00 (etiket hatasi sifir garantisi)",
            M.SENTETIK_PAY == 1.00, f"{M.SENTETIK_PAY}")

    W = H = 2048
    cap = 0.08
    plan = {"koloniler": [{"xc": 0.5, "yc": 0.5, "cap": cap}]}
    sen, sil, birlesik = M.maske_uret(plan, [], W, H)
    beklenen = math.pi * (cap / 2 * W) ** 2
    gozlenen = (sen > 0).sum()
    kontrol("tek koloni: beyaz alan ~ pi r^2 (%1 tolerans)",
            abs(gozlenen - beklenen) / beklenen < 0.01,
            f"gozlenen {gozlenen}  beklenen {beklenen:.0f}")
    kontrol("silme maskesi bos (arka planda koloni yok)", (sil > 0).sum() == 0)
    kontrol("birlesik = sentetik", int((birlesik > 0).sum()) == int(gozlenen))


def test_maske_plak_kirpmasi():
    print("\n14) Plak diskinin disi difuzyona acilmiyor mu (karar 3.10)")
    W = H = 1024
    # plak yaricapi 0.465 -> merkeze uzakligi 0.49 olan koloni tamamen disarida
    plan = {"koloniler": [{"xc": 0.99, "yc": 0.5, "cap": 0.01}]}
    sen, _, _ = M.maske_uret(plan, [], W, H)
    kontrol("plak disindaki koloni maskeye girmiyor", (sen > 0).sum() == 0,
            f"{(sen > 0).sum()} piksel")

    plan2 = {"koloniler": [{"xc": 0.5, "yc": 0.5, "cap": 0.02}]}
    sen2, _, _ = M.maske_uret(plan2, [], W, H)
    kontrol("plak icindeki koloni maskede var", (sen2 > 0).sum() > 0)


def test_maske_silme():
    print("\n15) Silme bolgesi gercek koloniden BUYUK mu (karar 3.25)")
    W = H = 2048
    cap = 0.05
    plan = {"koloniler": []}
    arka = [(0.5, 0.5, cap, cap)]                  # (xc, yc, w, h)
    sen, sil, birlesik = M.maske_uret(plan, arka, W, H)
    gercek_alan = math.pi * (cap / 2 * W) ** 2
    oran = (sil > 0).sum() / gercek_alan
    kontrol(f"silme alani ~ SILME_PAY^2 = {M.SILME_PAY**2:.2f} kat",
            abs(oran - M.SILME_PAY ** 2) / M.SILME_PAY ** 2 < 0.05,
            f"olculen {oran:.2f}x")
    kontrol("silme alani gercek koloniden buyuk", oran > 1.0)


def main():
    print("=" * 62)
    print("URETIM HATTI SANITY TESTI  (yerlesim.py + maske.py)")
    print("=" * 62)
    for f in (test_ters_cdf, test_ortusme, test_degen_oran, test_sayim_siniri,
              test_gamma_uclari, test_gamma_monoton, test_sert_sinir, test_sinirlar,
              test_koloni_dusmuyor, test_belirlenimci, test_naif_ablasyon,
              test_seviye_kilidi, test_maske_alani, test_maske_plak_kirpmasi,
              test_maske_silme):
        f()
    print("\n" + "=" * 62)
    print(f"GECTI: {len(GECTI)}   KALDI: {len(KALDI)}")
    if KALDI:
        for k in KALDI:
            print(f"  ! {k}")
        print("=" * 62)
        sys.exit(1)
    print("Uretim hatti guvenilir.")
    print("=" * 62)


if __name__ == "__main__":
    main()
