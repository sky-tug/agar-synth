# Karar Gunlugu

> Bir karar = bir satir + gerekce. Uc ay sonra "ben bunu neden boyle yapmistim"
> sorusunun cevabi burasi. Hakem "neden X degil de Y?" diye sordugunda da.
>
> Kural: bir kararı DEGISTIRIRSEN eski satiri SILME. Ustunu ciz, yeni satiri
> tarihiyle ekle. Degistirilmis kararlarin gecmisi, kararin kendisi kadar onemli.

---

## Faz 1 — Veri hatti (11 Agustos 2026)

| # | Karar | Gerekce |
|---|---|---|
| 1.1 | Sinif sirasi sabit: `S.aureus`=0, `B.subtilis`=1, `P.aeruginosa`=2, `E.coli`=3, `C.albicans`=4 | Proje boyunca degismeyecek. Degisirse tum etiket dosyalari sessizce bozulur. |
| 1.2 | `defects` / `contamination` iceren goruntuler **tamamen** atiliyor | Kutuyu silip goruntuyu tutmak modele "burada nesne yok" diye ogretir. Kac goruntu gitti makalede raporlanacak. |
| 1.3 | Goruntu boyutu JSON'dan degil **dosyadan** okunuyor | AGAR JSON'unda boyut alani yok. 2048x2048 varsaymak sessiz normalize hatasi uretir. |
| 1.4 | Alan adi `classes` (oneride `microbes` yaziyordu), sinif adi `S.aureus` (bosluksuz) | Demo paketten dogrulandi. Makale metninde duzeltilecek. |
| 1.5 | Bolme tur kombinasyonuna gore **tabakali**, goruntu seviyesinde | %10 alt kumesinde C.albicans kaybolursa seviyeler arasi fark sinif dengesizliginden gelir, veri miktarindan degil. |
| 1.6 | Alt orneklemler **ic ice**: %10 ⊂ %25 ⊂ %50 ⊂ %100 | Degilse farkin veri miktarindan mi, hangi goruntulerin secildiginden mi geldigi ayrilamaz. |
| 1.7 | `splits/` stem listeleri git'e giriyor; mutlak yollu listeler gitmiyor | Tekrarlanabilirligin temeli. Yollar makineye bagli, stem'ler degil. |

---

## Faz 2 — Olcum altyapisi (12-13 Agustos 2026)

### Duzeltmeler

| # | Karar | Gerekce |
|---|---|---|
| 2.0 | **`make_splits.py` duzeltildi**: Ultralytics listeleri artik `data/processed/images/` altini gosteriyor, ham AGAR klasorunu degil | Ultralytics etiketi, yoldaki son `/images/` parcasini `/labels/` ile degistirerek bulur. Ham yol (`data/AGAR_representative/...`) yazildiginda **hicbir etiket bulunamiyordu**. Ultralytics bu durumda durmaz, uyarir ve gecer — model "bu goruntulerde nesne yok" diye ogrenir. Sessiz ve olumcul. `make_splits.py` ve `scripts/train.py` artik ikisi de bu cozumlemeyi acikca kontrol ediyor. |

### Olcum

| # | Karar | Gerekce |
|---|---|---|
| 2.1 | mAP / AP hesabi **Ultralytics'ten bagimsiz**, `src/eval/metrics.py` icinde | (a) Ultralytics boyut bazli AP vermiyor — makalenin ana argumani kucuk koloniler uzerinde. (b) Ikinci detektor kontrolu (RT-DETR/YOLO11) icin ayni olcum kodunun calismasi sart; detektore bagli metrik, detektorler arasi karsilastirmayi gecersiz kilar. |
| 2.2 | AP algoritmasi COCO ile **birebir**: IoU 0.50:0.05:0.95, 101 noktali interpolasyon, alan araligi disi GT'ler ignore | Hakem "kendi metriginizi mi yazdiniz?" diye soracak. Cevap: evet, ve pycocotools'a karsi 1e-4 hassasiyetle dogrulandi (`src/eval/test_metrics.py`, test 10). |
| 2.3 | Boyut kirilimi **COCO araliklari** (32²/96²), orijinal 2048px goruntude | AGAR makalesiyle kiyaslanabilirlik. Demo olcumu: S.aureus'un 113 kutusu small, E.coli'nin 107'si large — ayrim anlamli. |
| 2.4 | Sayim conf esigi **VAL'da secilir**, teste oyle uygulanir. `evaluate.py --split test` esik verilmeden **calismaz** | Test kumesinde esik aramak, test kumesini ayara dahil etmektir. Bu bir sizinti. Kod bunu zorunlu tutuyor, insan hafizasina birakilmiyor. |
| 2.5 | sMAPE tanimi: `200·|p−g| / (|p|+|g|)`, p=g=0 iken 0 | Simetrik, %0–200 arasi sinirli. Bos plaklarda MAPE sonsuza gider; sMAPE gitmez. |
| 2.6 | Sayimda `ME` (ortalama hata, isaretli) de raporlanacak | MAE modelin **yanliligini** gizler. Mikrobiyologa "model sistematik olarak az sayiyor" demek, "MAE 4.2" demekten daha kullanisli. |
| 2.7 | Ana metrik `mAP50-95`. "Fark var" esigi Faz 2 sonunda olculup `configs/base.yaml`'a sabitlenecek | Esik sonuclara bakildiktan sonra secilirse cherry-picking olur. Seed'ler arasi std'den buyuk olmali. |

### Egitim protokolu

| # | Karar | Gerekce |
|---|---|---|
| 2.8 | **`imgsz: 1280`** | Projenin en kritik tek hiperparametresi. AGAR lower-res 2048px; C.albicans medyani 27.5px, S.aureus 29px. imgsz=640'ta bunlar **8.6 px**'e iner — P5 basinin stride'i 8, tespit imkansiza yakin. 1280'de 17.2 px. Bedeli: sure ve VRAM ~ imgsz², yani 640'in 4 kati. **VRAM yetmezse batch dusurulecek, imgsz DUSURULMEYECEK.** |
| 2.9 | Ana gridde augmentation **minimum**: sadece `fliplr`, `flipud`, `hsv_*` | Ultralytics varsayilani `mosaic=1.0` ile gelir. Hicbir sey yapilmazsa G50/G25/G10 kontrol gruplari **zaten** klasik veri artirma kullanir ve "difuzyon vs klasik" karsilastirmasi bastan bulanir. |
| 2.10 | `flipud` **acik** (0.5) | Agar plaginin kanonik yonu yok. Dikey cevirme burada mesru bir donusum — dogal goruntulerin aksine. |
| 2.11 | `scale` / `translate` / `degrees` ana gridde **kapali** | Koloni boyutu bir **sinif ipucu** (S.aureus kucuk, E.coli buyuk). Olcegi rastgele oynatmak sinif sinyalini bozar. Baseline B bu ipucunu bilerek bozuyor — bulgu ona gore yorumlanacak. |
| 2.12 | `copy_paste` ana gridde **kapali**, Baseline C'de acik | Ana gridde acik olursa "difuzyon vs copy-paste" karsilastirmasi tanimsiz hale gelir. |
| 2.13 | Butun augmentation alanlari config'de **acikca** yaziliyor (0.0 olanlar dahil) | Ultralytics varsayilanlari surumden surume degisiyor. Ortuk varsayilan = tekrarlanamazlik. |
| 2.14 | Egitim `scripts/train.py` uzerinden, dogrudan `yolo train` ile degil | Sure / GPU-saat / tepe VRAM **ilk kosudan itibaren** loglanmali (outline bolum 6). Sonradan geri donup olculemez. |
| 2.15 | `max_det: 1000` | AGAR'da bir plakta 125'e kadar koloni var. Varsayilan 300 yeterli gorunuyor ama kalabalik plaklarda kirpma riski sayim metrigini dogrudan bozar. |

### Acik kalanlar

| Konu | Durum |
|---|---|
| `eval.anlamli_fark_esigi` | G100 x 3 seed olculunce `configs/base.yaml`'a yazilacak |
| `eval.conf_thr` | ilk VAL kosusundan gelecek |
| `model.weights: yolo26n.pt` | Ultralytics surumu YOLO26'yi tanimiyorsa `yolo11n.pt`'ye dusulecek — **sessizce degil**, buraya not dusulerek |
| Baseline C `copy_paste` | Ultralytics'in `copy_paste`'i segmentasyon maskesi ister; detect gorevinde sessizce etkisiz kalabilir. Kontrol edilecek; etkisizse offline copy-paste yazilacak. **Etkisiz bir Baseline C, difuzyon lehine sahte kazanc uretir — en tehlikeli hata bu.** |
| Tam AGAR verisi | Basvuru 10 Agustos'ta gonderildi, onay bekleniyor. Faz 1'in kapisi tam veride **tekrar** gecilmeli: demo pakette (10 goruntu, 7 tabaka) val/test bos cikiyor ve train_50 = train_25 = train_10. `make_splits.py` bunu uyari olarak basiyor — kod hatasi degil, veri yetersizligi. 240 goruntuluk sentetik testte bolme dogru calisiyor (ic ice ✓, sizinti yok ✓, sinif paylari korunuyor ✓). |

---

## Hoca cevabi — 13 Agustos 2026

Faz 0'in dort acik sorusu kapandi. Cevaplar ve projeye etkileri:

| # | Karar | Kaynak / etki |
|---|---|---|
| H.1 | **"Kendi kendine ogrenme" AYRI bir deney kolu DEGIL** | Hoca: *"sentetik veri ile egitim ve gercek veri ile test, sentetik ve gercek veriyi karistirarak egitim ve test calismalariyla yapiyoruz."* Yani basliktaki ifade **ikame egrisinin kendisini** tarif ediyor. **Projenin en buyuk belirsizligi kapandi — SSL kolu yok, tasarim degismiyor.** |
| H.2 | 3.4'teki oranlar **serbest**; bulgular agirlikli olarak bu oranlarin testi | Mevcut G100/G50/G25/G10/S100 tasarimi aynen gecerli. Miktar taramasi (0.5x, 2x) bulgular bolumunun merkezine tasiniyor — yan kol degil. |
| H.3 | **Ablasyon duzeltmesi onaylandi.** *"Oradaki ablasyon tablosu ornek. Uc farkli senaryo secip kullanabilirsin. Maske sabit alinabilir."* | Onerdigimiz A1/A2/A3 (LoRA yok / gercek arka plan yok / rastgele yerlesim) aynen kullanilabilir. "Maske sabit" = maske kontrolunun her senaryoda acik kalmasi — zaten teknik zorunluluktu. |
| H.4 | **5 seed.** Grafiklerde **min / maks / ortalama** gosterilecek | `configs/base.yaml` → `eval.raporlama`. Hata cubugu min–maks araligi olacak; std tabloda ayrica verilecek (n=5 ile savunulabilir). |
| H.5 | **XAI eklenecek** — ablasyondan SONRA ayri bolum, Grad-CAM/++ | Ozette 1-2 cumle zorunlu; bolum opsiyoneldi, eklemeye karar verildi. **Egitim yok, yalnizca cikarim** → maliyeti ihmal edilebilir (~0.3 GPU-saat). `configs/base.yaml` → `xai`. |
| H.6 | **Hedef dergi: Muhendislik Bilimleri ve Tasarim Dergisi (JESD)**, Suleyman Demirel Univ., DergiPark | TR Dizin + Scilit + EBSCOhost + SOBIAD + CrossRef. Yilda 4 sayi. Turkce/Ingilizce. Acik erisim CC BY 4.0. Yalnizca ozgun arastirma makalesi. Bildirilen sureler: on inceleme 8 gun, hakem 111 gun, yayin 69 gun. |
| H.7 | **Makale dili: Turkce** | Yazim hizi. Ilerideki bir uluslararasi hedef icin ceviri maliyeti kabul ediliyor. |

### Kapsam kismasi (H.6'nin sonucu)

JESD icin 80 kosuluk grid gereginden genis. Yeni kapsam **61 kosu**:

| Kol | Onceki | Yeni | Gerekce |
|---|---|---|---|
| Ana grid | 8 konf x 5 seed = 40 | **degismedi** | Makalenin belkemigi |
| Klasik kol (Baseline B/C) | 3 seviye x 3 seed = 18 | **G25 x 3 seed = 6** | "Difuzyon vs klasik" sorusu icin ikame egrisinin orta noktasi yeterli |
| Miktar taramasi | 0.5x/2x/4x x 3 = 9 | **0.5x/2x x 3 = 6** | 4x tek basina en pahali kalemdi (3.25 tam-kosu esdegeri x 3 seed) |
| Ablasyon | 3 x 3 = 9 | **degismedi** | Hoca acikca istedi |
| Ikinci detektor | 2 x 2 = 4 | **kapsam disi** | Dergi bunu beklemiyor; genelleme iddiasi makalede sinirlanacak |
| XAI | yok | **eklendi** | Egitim yok, ~0.3 GPU-saat |

Egitim yuku: 62.7 → **46.9 tam-kosu esdegeri** (%25 azalma).
Kisilan kollar `scripts/butce.py` icinde `klasik_genis` / `miktar_genis` /
`ikinci_detektor` olarak duruyor — butce izin verirse geri acilabilir.

### Acik kalan tek nokta

**"ELR"** — hoca XAI baglaminda *"IoU, ELR gibi skorlari da ... verebilirsin"*
demis. ELR standart bir XAI/CAM metrigi olarak dogrulanamadi. En olasi karsiligi
**EBPG (Energy-Based Pointing Game)** — CAM literaturunde IoU ile birlikte
raporlanan olcut budur. Teyit gelene kadar EBPG olarak uygulanacak.
Bir sonraki mailde tek satirla sorulacak.

---

## Faz 2 — hat dogrulamasi ve ilk olcumler (13 Agustos 2026)

Donanim: **RTX 4060 Laptop, 8 GB** (7.6 GB kullanilabilir) · Ultralytics 8.4.118 ·
torch 2.13.0+cu130 · Python 3.12 · ortam: `src/.venv` (conda `agar` DEGIL)

### Duzeltilen iki hata

| # | Sorun | Cozum |
|---|---|---|
| 2.16 | `make_splits.py` liste yazarken `.resolve()` kullaniyordu. `data/processed/images/` altindakiler ham AGAR'a **symlink**; resolve onlari takip edip yolu ham klasore ceviriyor, `/images/` parcasi kayboluyor, Ultralytics etiketi bulamiyor. | `.resolve()` kaldirildi. Ayrica **kontrol de duzeltildi**: onceki surum degiskendeki degeri test ediyordu, dosyaya yazilan degeri degil — kontrol geciyor, egitim patliyordu. Artik yazilan satirlar okunup dogrulaniyor ve ornek bir satir ekrana basiliyor. |
| 2.17 | `max_det: 1000` config'de yaziliyordu ama `model.train()`'e gecirilmiyordu; Ultralytics sessizce 300 kullaniyordu. | `scripts/train.py` artik acikca gonderiyor. Ayni siniftan: `cutmix` (8.4'te eklendi) config'e acikca 0.0 olarak yazildi. |

### Duman testi — hat calisiyor mu?

10 goruntu, 10 epoch, batch 4. **Sonuc: etiketler bulunuyor.**
`10 images, 0 backgrounds, 0 corrupt` + `Instances 387`.
mAP 0 cikti; 30 gradyan adimi icin beklenen durum, tanisal degil.

`yolo26n.pt` Ultralytics 8.4.118'de **mevcut** — `yolo11n.pt` yedegine gerek yok.

### Ezberleme testi — hat OGRENIYOR mu?

Ayni 10 goruntu, 300 epoch, batch 8, patience 1000. 7.2 dakika.

| Sinif | AP50 | AP50-95 | kutu boyutu (medyan px) |
|---|---|---|---|
| **tumu** | **0.947** | **0.767** | |
| S.aureus | 0.951 | 0.723 | 29 |
| B.subtilis | 0.873 | 0.704 | 151 |
| P.aeruginosa | 0.968 | 0.851 | 155 |
| E.coli | 0.981 | 0.820 | 128 |
| C.albicans | 0.963 | 0.738 | 27.5 |

**Karar 2.18 — `imgsz: 1280` ampirik olarak dogrulandi.** En kucuk iki sinif
(C.albicans 27.5 px, S.aureus 29 px) buyuk siniflarla ayni bantta AP aliyor.
1280'de kucuk koloniler ogrenilebiliyor. Bu, protokolun en kritik varsayimiydi;
artik tahmin degil, olcum.

### Donanim olcumleri (bolum 6 icin ilk kayit)

| | |
|---|---|
| tepe VRAM, imgsz 1280 batch 4 | 4.19 GB |
| tepe VRAM, imgsz 1280 batch 8 | **6.05 GB**  (7.6 GB'nin %80'i) |
| cikarim | 8.0 ms / goruntu |
| epoch (10 goruntu, batch 8) | 1.4 s |

**Karar 2.19 — bu donanimda batch 8 tavan.** batch 16 denenmeyecek; 6.05 GB
zaten sinira yakin ve OOM riski uzun kosularda saatler kaybettirir.

### 🔴 BUTCE KAPISI: bu dizustunde SIGMIYOR

Duman olcumu 10 goruntude yapildi; epoch basina sabit maliyet (dogrulama,
dataloader baslatma) amortize olmadigi icin dogrudan olceklendirmek asiri
kotumser. Gercekci aralik, olculen cikarim hizindan (8.0 ms/goruntu) turetildi:

| Varsayim | sa / tam kosu | 61 kosuluk grid |
|---|---|---|
| iyimser (egitim = 3x cikarim) | 8.0 | **375 GPU-saat** |
| orta (4x) | 10.7 | **500 GPU-saat** |
| kotumser (5x) | 13.3 | **625 GPU-saat** |
| duman olcumunden lineer (ust sinir, guvenilmez) | 46.7 | 2187 GPU-saat |

Uretim (Faz 3) ~95 GPU-saat ekliyor. **Toplam ~470–720 GPU-saat**, yani bu
kartta kesintisiz **3–4 hafta**. Bir dizustu bilgisayarda gercekci degil.

**Sonuclar:**

1. **BİDB GPU sunucusu artik kritik yolda.** "Olsa iyi olur" degil, ana gridin
   on sarti. Sorulmasi geciken tek is bu.
2. Dizustu bundan sonra **gelistirme ve duman testi** icin; ana grid degil.
3. `epochs: 150` bir tahmin. 8000 goruntude model muhtemelen daha erken
   yakinsar ve `patience: 50` devreye girer. G100 kosusu bunu olcecek —
   gercek maliyet bu tablodan dusuk cikabilir.
4. Kesin sayi tam veriyle yapilacak G100 kosusundan gelecek. Bu tablo bir
   **aralik**, karar degil.

---

## Faz 3 — Uretim hatti

*(bos)*
