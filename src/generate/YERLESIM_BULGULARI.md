# Yerlesim kesfi — Faz 3 girdisi

> 14 Agustos 2026. Kaynak: `AGAR_representative/lower-resolution`, **10 plak / 387 koloni**.
> Uretim betikleri: `src/generate/kesif/` altinda.
>
> 🔴 **Bu sayilar gecici.** 10 plak bir dagilim *ailesi* secmeye yeter, *parametre*
> kestirmeye yetmez. Tam veri gelince ayni betikler seviye basina (bkz. karar 3.5)
> yeniden calistirilacak ve buradaki her sayi guncellenecek.

![yerlesim kesfi](yerlesim_kesfi.png)

---

## 1. Plak cemberi

Hough ile 10/10 plakta bulundu. Cok kararli:

| | |
|---|---|
| goruntu | 2048 x 2048 (hepsi) |
| merkez | (1042 ± 5, 1056 ± 15) |
| yaricap R | 950 ± 6 px, yani **R/W = 0.465** |

Tek sapan: `14581` (R=831, R/W=0.406) — en kalabalik plak (n=125) ve tek
`r_norm > 1` ureten plak (maks 1.036). Hough muhtemelen ic bir halkayi yakaladi.
**Uretimde cember goruntuden kestirilmeyecek, sabit alinacak:** merkez = goruntu
merkezi, R = 0.465·W. Tam veride bu sabitin varyansi olculecek.

## 2. Radyal dagilim — duzgun DEGIL, kenar seyrek

Esit **alanli** 5 halka (duzgun dagilimda her biri %20 almali):

| halka (r/R) | gozlenen |
|---|---|
| 0.00–0.45 | %22.0 |
| 0.45–0.63 | %24.3 |
| 0.63–0.77 | %25.1 |
| 0.77–0.89 | %18.6 |
| **0.89–1.00** | **%8.5** |

KS testi (duzgun disk H0): D=0.123, **p = 1.6e-5** → reddedildi.
`r_norm²` ortalamasi 0.444 (duzgun dagilimda 0.500).

**Yorum:** ic %89'luk disk pratikte duzgun; son halka ~yariya dusuyor. Bunun bir
kismi geometrik (koloni merkezi kendi yaricapindan daha kenara gidemez; en buyuk
koloniler ~78 px = 0.08R), kalani fiziksel (menisk, plak duvari).

**Model:** θ ~ U(0, 2π) (KS p=0.44, duzgunluk reddedilemedi) ·
r icin gozlenen `r_norm` dagilimindan **ters-CDF ornekleme**. Parametrik bir
aileye zorlamaya gerek yok, ampirik CDF daha durust ve daha ucuz.

## 3. 🔴 Yerlesim RASTGELE — ablasyon A3'u tehdit ediyor

Clark–Evans oranı (gozlenen ort. en-yakin-komsu ÷ Poisson beklentisi):

| plak | n | CE |
|---|---|---|
| 14627 | 12 | 1.06 |
| 14512 | 15 | 1.05 |
| 14130 | 18 | 1.04 |
| 13895 | 19 | 1.02 |
| 13938 | 32 | 0.93 |
| 14380 | 34 | 1.07 |
| 14410 | 37 | 0.95 |
| 14618 | 40 | 0.90 |
| 14684 | 55 | 1.09 |
| 14581 | 125 | 1.00 |

**10 plagin 9'u 0.90–1.10 bandinda.** Kumelenme yok, duzenli oruntu yok.
Koloniler plak diskine **rastgele** dusuyor. Mikrobiyolojik olarak beklenen
sonuc (suspansiyon homojen yayilir), ama metodolojik bir sorun yaratiyor:

> Ablasyon **A3 "rastgele yerlesim"**, modellenen yerlesime karsi bir kontrol
> olarak tasarlanmisti. Gercek yerlesim zaten rastgeleyse A3 ile ana hat
> neredeyse **ayni sey** olur ve ablasyon bos cikar (3 kosu bosa gider).

A3'un yeniden tanimlanmasi gerekiyor — asagida "Acik karar".

## 4. 🔴 Koloniler birbirine DEGIYOR — %39

| olcut | deger |
|---|---|
| en az bir komsusuna degen koloni | **151 / 387 = %39.0** |
| en yakin komsu ÷ koloni capi, medyan | 1.35 |
| ayni oran, p5 / p10 | 0.47 / 0.56 |
| ortusen ciftlerde ortusme derinligi (0=tegetsel, 1=tam ic ice) | medyan 0.25, p90 0.49, **maks 0.63** |

**Bu, maske ureticisinin en kritik parametresi.** Naif bir "cakisma yok"
reddetme ornekleyicisi yazarsak sentetik plaklar gercektekinden **temiz** olur:
detektor hicbir zaman ortusen/degen koloni gormez, gercek test kumesinde bu
vakalarda basarisiz olur, biz de "difuzyon ise yaramiyor" diye yorumlariz.
Oysa hata uretici modelde degil, **yerlesim modelinde** olur. Sentetik verinin
lehine degil, aleyhine calisan ve tespiti zor bir yanlilik.

**Model:** cakismaya izin ver; ortusme derinligini 0.63'te kirp (gozlenen maks).
Hedef: uretilen plaklarda "en az bir komsusuna degen koloni" orani %35–45 bandi.

## 5. Boyut

Kutu en/boy orani **%96.9 tam kare** (w == h). Kare olmayan 12 kutu var,
sapma kucuk. → Maskeler daire, etiket kutusu kare, ~%3 hafif elips jitter.

| sinif | n | medyan (px) | p10 | p90 | std/medyan |
|---|---|---|---|---|---|
| C.albicans | 32 | 26.5 | 21.2 | 33.7 | 0.17 |
| S.aureus | 152 | 29.0 | 29.0 | 51.0 | 0.34 |
| E.coli | 109 | 128.0 | 116.0 | 171.0 | 0.18 |
| B.subtilis | 53 | 151.0 | 103.0 | 202.0 | 0.30 |
| P.aeruginosa | 41 | 155.0 | 117.0 | 207.0 | 0.25 |

**Iki ayri rejim:** ~27–29 px (C.albicans, S.aureus) ve ~128–155 px
(E.coli, B.subtilis, P.aeruginosa). Arada hicbir sey yok. Boyut sinif
basina ayri ornekleniyor (karar 2.11'in gerekcesiyle ayni: boyut bir sinif ipucu).

Sinif bazli radyal fark zayif ama var (B.subtilis r²=0.349 vs S.aureus r²=0.488
→ B.subtilis biraz daha merkeze toplaniyor). 10 plakta anlamlandirilamaz;
**tam veride tekrar bakilacak**, simdilik radyal dagilim sinifa kosullanmayacak.

## 6. Plak kompozisyonu — karisik plaklar var

| plak | n | kompozisyon |
|---|---|---|
| 13895 | 19 | B.subtilis 19 |
| 13938 | 32 | C.albicans 32 |
| 14512 | 15 | P.aeruginosa 15 |
| 14618 | 40 | S.aureus 40 |
| 14684 | 55 | E.coli 55 |
| 14130 | 18 | P.aeruginosa 15 + **S.aureus 3** |
| 14627 | 12 | P.aeruginosa 11 + **S.aureus 1** |
| 14410 | 37 | E.coli 31 + **S.aureus 6** |
| 14581 | 125 | S.aureus 102 + E.coli 23 |

Oruntu: **bir baskin tur + az sayida ikincil tur** (cogunlukla S.aureus).
Uretici tek-tur varsaymayacak; tur kombinasyonu ve baskin/ikincil orani
`tur_kombinasyonlari.csv` dagilimindan ornekleyecek.

## 7. Koloni sayisi

`n` = 12, 15, 18, 19, 32, 34, 37, 40, 55, 125 · medyan 33 · ort 38.7
`ln(n)`: ort 3.42, std 0.69 → lognormal geri-donusum medyan 30, p10 13, p90 74.

Lognormal aile makul gorunuyor; **parametreler tam veriden gelecek.**

---

## Acik karar — ablasyon A3

Bulgu 3'un sonucu. Uc secenek:

| | A3 tanimi | Sonuc |
|---|---|---|
| (a) | eski hali: diske rastgele | **Bos cikar** — gercek yerlesim zaten rastgele |
| (b) | plak cemberi yok sayilir: tum 2048x2048 kareye duzgun; boyut sinifa kosullanmadan tek havuzdan; degme kisiti yok | Yerlesim modelinin **uc bileseninin hepsini** birden kapatir — tek degiskenli degil ama "yerlesim modeli gerekli mi" sorusunu net yanitlar |
| (c) | yalnizca degme kisiti kapatilir (cakisma serbest / hic cakisma yok) | Tek degiskenli, bulgu 4'u dogrudan test eder, ama dar |

Oneri: **(b)**. Hocanin "uc farkli senaryo secip kullanabilirsin" (karar H.3)
esnekligi buna izin veriyor, ve makalede savunulabilir bir cumle uretiyor:
*"maske kontrollu yerlesim, naif rastgele yerlestirmeye kiyasla X puan getiriyor."*
