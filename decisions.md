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
Kisilan kollar `scripts/budget.py` icinde `classic_wide` / `amount_wide` /
`second_detector` olarak duruyor — butce izin verirse geri acilabilir.

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

## Faz 3 — Uretim hatti (14 Agustos 2026)

### Sizinti karsiti protokol — uretimden ONCE kilitlendi

Faz 3'un ilk isi kod yazmak degil, **sizinti tanimini kapatmak.** Sonradan
duzeltmek tum sentetik uretimi ve ona bagli 12 kosuyu tekrarlamak demek.

| # | Karar | Gerekce |
|---|---|---|
| 3.1 | **Sizinti tek kanalli degil, UC kanalli.** Bir sentetik goruntu, seviyenin disindaki gercek veriyi su uc yoldan tasiyabilir: (a) LoRA agirliklari, (b) inpainting'in uzerine yapildigi gercek arka plan plagi, (c) yerlesim/boyut dagilimlarinin fit edildigi etiket istatistikleri | REHBER.md yalnizca (a)'yi yaziyordu. (b) ve (c) daha sessiz ama ayni sonucu dogurur: G10+S'in "sadece %10 gercek veri" iddiasi, uretim boru hattinin herhangi bir yerinde %100'luk kumeye bakildiysa cokmus olur. Hakem (a)'yi sorar; (b) ve (c) makale reddedilmeden fark edilmeyebilir — daha tehlikeli olan bu. |
| 3.2 | **Seviye basina ayri LoRA. Dort egitim: `lora_10`, `lora_25`, `lora_50`, `lora_100`.** Her biri yalnizca kendi seviyesinin **train** alt kumesiyle egitilir. REHBER'de "3 egitim" yaziyordu; S100'un da bir LoRA'ya ihtiyaci var, dogru sayi **4**. | Alt orneklemler ic ice (karar 1.6) oldugu icin `lora_10`'un verisi ⊂ `lora_25`'in verisi. Bu sorun degil, tam tersi: ikame egrisinin her noktasi kendi bilgi butcesiyle uretiliyor. |
| 3.3 | LoRA **yalnizca train bolmesinden** egitilir. val ve test hicbir seviyede, hicbir asamada difuzyon modeline gosterilmez. | Test kumesi %100 gercek ve dokunulmamis kalmali (Faz 2 ilkesi). Difuzyon uzerinden dolayli temas da temastir. |
| 3.4 | **Arka plan havuzu seviye basina ayri.** `G25+S`'in sentetik goruntuleri yalnizca `train_25` icindeki plaklarin arka planlarini kullanir. | Plak dokusu, aydinlatma, kenar golgesi — bunlarin hepsi o goruntunun bilgisi. Havuz ortak olursa "sadece %25 gercek veri" cumlesi yanlis olur. |
| 3.5 | **Yerlesim ve boyut dagilimlari da seviye basina fit edilir.** `analyze_manifest.py` cikti tablosu bolme-farkindali calisacak; `layout.py` bir `--level` argumani almadan calismayacak (varsayilan yok, hata verir). | En sinsi kanal bu. "Koloniler plakta soyle dagiliyor, boyut medyani su" bilgisi 8000 goruntuden cikarilirsa, 800 goruntuluk seviyeye ait olmayan bir on bilgi sentetige gomulur. Varsayilan deger koymamak, bunun insan hafizasina birakilmamasini sagliyor (karar 2.4 ile ayni mantik). |
| 3.6 | **S100 ifadesi sabitlendi.** Makalede "gercek veri kullanilmadi" **yazilmayacak**. Kullanilacak ifade: *"S100 kolunda detektor yalnizca sentetik goruntulerle egitilmistir; gercek veri bu kola yalnizca uretici modelin uyarlanmasi (LoRA), arka plan havuzu ve yerlesim istatistikleri uzerinden **dolayli** olarak girmistir."* | Bu cumle makalenin durustluk cipasi. Simdi yazildi ki Faz 8'de unutulmasin. |
| 3.7 | **Isimlendirme kurali — sizintiyi yapisal olarak imkansiz kilar.** Her sentetik urun seviyesini adinda tasir: `data/synthetic/s25/`, `lora_25/`, `bg_pool_25.json`, `layout_25.json`. Farkli sayi tasiyan iki yol ayni uretim komutunda bulusursa `generate` betigi **durur**. | Insan dikkatine degil, dosya sistemine yaslanan bir kontrol. Uc hafta sonra yorgunken yanlis LoRA'yi secmek en olasi hata bicimi. |
| 3.8 | **Ablasyon A1 ("LoRA yok") tanimi:** G25 seviyesinde, uyarlanmamis base difuzyon modeliyle uretim. Arka plan ve yerlesim yine `train_25`'ten gelir — yalnizca LoRA kanali kapatilir. | Ablasyonun tek degiskenli kalmasi icin. Uc kanali birden kapatirsak A1 ile A2 ayrilamaz hale gelir. |
| 3.9 | **Butce sonucu:** LoRA egitimi 1 degil **4 kalem**. Faz 3 icin ongorulen ~95 GPU-saat bu varsayimla yeniden hesaplanacak; `scripts/budget.py`'ye `--lora-count` parametresi eklenecek. | ~470–720 GPU-saatlik toplam tahmin bunu icermiyordu. BİDB talebinde verilecek sayi bu duzeltmeden sonra kesinlesir. |

### Yerlesim modeli — 10 demo plak / 387 koloni uzerinden (14 Agustos)

Ayrinti ve sekil: `src/generate/LAYOUT_FINDINGS.md`. Betikler `src/generate/exploration/`.
**Buradaki tum sayilar gecici** — 10 plak dagilim *ailesi* secmeye yeter, *parametre*
kestirmeye yetmez. Tam veride seviye basina (karar 3.5) tekrar calistirilacak.

| # | Karar | Gerekce |
|---|---|---|
| 3.10 | **Plak cemberi sabit alinir:** merkez = goruntu merkezi, `R = 0.465 · genislik`. Uretimde goruntuden kestirilmez. | 10/10 plakta Hough ile bulundu, R = 950 ± 6 px (2048'de). Tek sapan `14581` (R=831) muhtemelen ic halkayi yakalamis — kestirim hatasi uretim hattina sizmasin. |
| 3.11 | **Yerlesim: θ ~ U(0,2π), r gozlenen `r_norm` dagiliminin ters-CDF'sinden.** Parametrik aileye zorlanmaz. | Aci duzgun (KS p=0.44, reddedilemedi). Yaricap duzgun **degil** (KS D=0.123, p=1.6e-5): ic %89'luk disk duzgun, son halka %20 yerine %8.5 aliyor. Ampirik CDF hem daha durust hem daha ucuz. |
| 3.12 | **🔴 Cakismaya IZIN VERILIR.** Reddetme ornekleyicisi "hic degmesin" kuralini uygulamaz; ortusme derinligi 0.63'te (gozlenen maks) kirpilir. Kalibrasyon hedefi: uretilen plaklarda "en az bir komsusuna degen koloni" orani **%35–45**. | Gercekte kolonilerin **%39'u** en az bir komsusuna degiyor (ortusme derinligi medyan 0.25, p90 0.49, maks 0.63). Naif bir "cakisma yok" kurali sentetik plaklari gercektekinden **temiz** yapar; detektor ortusen koloniyi hic gormez, gercek testte o vakalarda batar ve biz bunu *"difuzyon ise yaramiyor"* diye okuruz. Hata uretici modelde degil yerlesim modelinde olur, ama oyle gorunmez. **Sentetik verinin aleyhine calisan ve tespiti en zor yanlilik bu.** |
| 3.13 | **Maske dairesel, etiket kutusu kare.** ~%3 hafif elips jitter eklenir. | AGAR kutularinin **%96.9'u tam kare** (w == h, 375/387). Kare olmayan 12 kutunun sapmasi kucuk. |
| 3.14 | **Boyut sinif basina ayri orneklenir**, ortak havuzdan degil. | Iki ayri rejim var, arada hicbir sey yok: C.albicans 26.5 px / S.aureus 29 px, buna karsi E.coli 128 / B.subtilis 151 / P.aeruginosa 155. Karar 2.11 ile ayni gerekce: boyut bir **sinif ipucu**. |
| 3.15 | **Radyal dagilim sinifa kosullanmaz** (simdilik). | Zayif bir egilim var (B.subtilis r²=0.349, S.aureus r²=0.488 → B.subtilis biraz daha merkezde) ama 10 plakta anlamlandirilamaz. Tam veride tekrar bakilacak; o zaman kosullanabilir. |
| 3.16 | **Plak kompozisyonu: bir baskin tur + az sayida ikincil tur.** Uretici tek-tur varsaymaz; kombinasyon ve baskin/ikincil orani `species_combinations.csv` dagilimindan orneklenir. | Demo pakette 10 plagin 4'u karisik: `P.aeruginosa 15 + S.aureus 3`, `P.aeruginosa 11 + S.aureus 1`, `E.coli 31 + S.aureus 6`, `S.aureus 102 + E.coli 23`. Ikincil tur cogunlukla S.aureus. |
| 3.17 | **Koloni sayisi lognormal ailesinden.** Demo: `ln(n)` ort 3.42, std 0.69 → medyan 30, p10 13, p90 74. Parametreler tam veriden gelecek. | n = 12…125, sag kuyruklu. Aile secimi 10 plakla yapilabilir, parametre kestirimi yapilamaz. |

### 🔴 Ablasyon A3 yeniden tanimlandi (14 Agustos)

| # | Karar | Gerekce |
|---|---|---|
| 3.18 | ~~A3 = "rastgele yerlesim"~~ **A3 = "naif rastgele yerlestirme"**: plak cemberi yok sayilir (tum 2048x2048 kareye duzgun), boyut sinifa kosullanmadan tek havuzdan cekilir, degme/ortusme kisiti hic uygulanmaz. | **Bulgu: gercek yerlesim ZATEN rastgele.** Clark–Evans orani 10 plagin 9'unda 0.90–1.10 bandinda — kumelenme de yok, duzenli oruntu de yok. Eski tanimiyla A3 ana hattin kopyasi olurdu ve **3 kosu bos giderdi**; ablasyon "fark yok" derdi, bu da yerlesim modelinin gereksiz oldugu anlamina gelmezdi, sadece deneyin bos oldugu anlamina gelirdi. Yeni tanim yerlesim modelinin uc bilesenini birden kapatiyor: tek degiskenli degil, ama **"yerlesim modeli gerekli mi"** sorusunu net yanitliyor ve makalede savunulabilir bir cumle uretiyor: *"maske kontrollu yerlesim, naif rastgele yerlestirmeye kiyasla X puan getiriyor."* Hocanin "uc farkli senaryo secip kullanabilirsin" esnekligi (karar H.3) buna izin veriyor. |

### `layout.py` yazildi (14 Agustos) — kod kararlari

| # | Karar | Gerekce |
|---|---|---|
| 3.19 | **Kisa menzilli itme: Strauss etkilesim parametresi `gamma`.** Bir aday konum baska bir koloniye deger halde ise `gamma` olasilikla kabul edilir, `1-gamma` ile reddedilip yeniden denenir. `gamma` elle secilmez — `fit` icinde ikili aramayla gercek degme oranina **otomatik kalibre edilir**. | Saf rastgele yerlestirme (gamma=1) degme oranini %52 cikardi, gercek %39. Yani gercek yerlesimde **kisa menzilde zayif bir itme var** — Clark–Evans bunu goremiyor cunku ortalama-tabanli bir olcut; kisa menzil yapisina duyarsiz. Strauss sureci (Strauss 1975) bunun standart, alintilanabilir modeli. Demo veride kalibrasyon `gamma=0.317` verdi. |
| 3.20 | **Kalibrasyonda ortak rastgele sayilar.** Tarifler (sayim / sinif / cap) bir kez uretilir, her `gamma` denemesinde aynen kullanilir; degisen tek sey yerlestirmedir. | Ilk surumde gamma reddi rastgele akisi kaydiriyordu; ayni gamma farkli tarifler uretiyor, ikili arama gurultuye oturuyordu. Fonksiyon `recipe()` ve `one_plate()` olarak ayrildi. |
| 3.21 | **Sayim icin lognormal konumu `ln(n)`'in ORTALAMASI degil MEDYANI.** | Ortalama, tek bir kalabalik plaktan (demo'da n=125) asiri etkileniyor; uretilen medyan gercegin %15 altinda kaliyordu. Medyanla fark %6'ya indi. Saglam kestirim. |
| 3.22 | **Koloni sessizce dusurulmez.** Denemeler tukendiginde once gamma'nin reddettigi gecerli konum, o da yoksa en az ihlal eden konum kabul edilir; her iki durum da sayilir ve raporlanir. Hicbiri yoksa `🔴` ile ekrana basilir. | Ilk surumde gamma reddi sonrasi yedek aday tutulmuyordu ve koloni **sessizce dusuyordu** — uretilen sayim dagilimi asagi kayiyordu, hicbir uyari yoktu. Faz 2'deki `make_splits` hatasiyla ayni sinif: sessiz, olumcul, ve ancak metrik tuhaf gelince fark edilir. |
| 3.23 | **`validate` alt komutu**: uretilen yerlesim gercekle 10 olcutte karsilastirilir (sayim, r/R, dis halka, degme orani, ortusme derinligi, Clark–Evans). | "Gozle bakip iyi gorunuyor" bir kabul kriteri degil. Uretim hattinin her degisikliginden sonra bu tablo calistirilacak. |

### 🔴 Demo veride kalibrasyon anlamli degil — mekanizma dogru, sayi gurultu

Gercek degme orani %39.0, ama **plak duzeyinde onyukleme (bootstrap) %90 guven
araligi: %26.7 – %48.2** (10 plak). 22 puan genislik.

Sebep: degme orani plak icinde son derece korele — seyrek plak ~%0, kalabalik plak
~%90. **Etkin ornek buyuklugu koloni sayisi (387) degil, plak sayisi (10).**
Uretimde farkli seed'ler %31.5 ile %36.0 arasinda degisiyor; ikisi de bu araligin icinde.

Sonuc: `gamma = 0.317` **gecici**. `fit` bunu `"provisional": true` alaniyla JSON'a
yaziyor ve ekrana kirmizi uyari basiyor. Tam veri gelince seviye basina yeniden
uydurulacak (karar 3.5). **Bu sayiya dayanan hicbir uretim baslatilmayacak.**

### `mask.py` yazildi (14 Agustos) — maske ve arka plan kararlari

| # | Karar | Gerekce |
|---|---|---|
| 3.24 | **Sentetik maske = etiketlenen diskin TA KENDISI. Pay yok, genisletme yok (`SYNTH_MARGIN = 1.00`).** | Difuzyon yalnizca maskenin **icini** boyayabilir. Maskeyi buyutursek uretilen koloni etiket kutusunu tasabilir ve **"etiket hatasi sifir" iddiasi "yaklasik sifir"a doner** — bu makalenin ana iddiasi, pazarlik konusu degil. Bedeli: koloninin dogal golgesi/halesi kutu disina cikamaz, kenarda dikis izi riski var. Bu bir **uretim kalitesi** sorunu; difuzyon parametreleriyle cozulecek, maskeyi buyuterek degil. Parametre kodda acikca duruyor ki degistiren kisi neyi kaybettigini gorsun. |
| 3.25 | **Arka plandaki GERCEK koloniler silme maskesine alinir** (`ERASE_MARGIN = 1.35`, comert). Maske = sentetik diskler ∪ silme bolgeleri. | 🔴 **REHBER'in "bos/az koloni iceren plaklar toplanir" varsayimi yanlis: AGAR'da bos plak yok.** Demo pakette en temiz plakta bile 32 koloni var. Arka plani oldugu gibi kullanirsak goruntude **etiketsiz gercek koloniler** kalir → model "koloni = arka plan" ogrenir. Bu, sentetik veriye sessizce **yanlis negatif enjekte etmektir** ve karar 3.12'deki tuzagin aynasi: sentetik verinin aleyhine calisir, "difuzyon ise yaramiyor" diye okunur. |
| 3.26 | **Arka plan havuzu `erase_ratio` (silinecek alan ÷ plak diski) esigiyle secilir**, varsayilan 0.06. Havuz < 20 plak ise kirmizi uyari. | Silinecek alan ne kadar buyukse difuzyon o kadar cok "yeni plak uretiyor", o kadar az "gercek arka plan kullaniyor" — ve A2 ablasyonunun ("gercek arka plan yok") anlami zayifliyor. Esik, bu odunlesmeyi tek bir gorunur sayiya baglar. |
| 3.27 | **🔴 `pool` komutu sinif yanliligi kontrolu yapar** ve %15'i asarsa uyarir. | Demo veride ortaya cikti: esik 0.06'da havuzun %100'u kucuk-koloni turlerinden (S.aureus, C.albicans) geldi. **Sebep zincirleme:** esik silinecek ALANA bakar → alan koloni BOYUTUYLA belirlenir → boyut da SINIFLA (karar 3.14: iki ayri rejim, 27 px vs 140 px). Yani masum gorunen bir esik, arka plan havuzunu sessizce tek bir tur ailesine daraltiyor. Plak ortami, aydinlatma ve besiyeri rengi ture gore degisiyorsa sentetik verinin tamami tek bir gorsel ailenin uzerine kuruluyor demektir. Demo'da 10 plak var, tam veride bu daha yumusak cikabilir — **ama kontrol edilmeden gecilmeyecek.** |
| 3.28 | **Her uretilen plagin yaninda `provenance/<name>.json` var:** hangi arka plan, hangi seviyeden, hangi listeden, kac gercek koloni silindi. | Sizinti denetim izi (karar 3.4). Hakem "G25'in arka planlari nereden geldi?" diye sorarsa cevap dosyada; hafizada degil. |

### Demo veride uctan uca calisti

`400 plak · havuz 6 arka plan · en cok kullanilan arka plan uretimin %19.5'i ·
sentetik disklerin %17.1'i silme bolgesiyle cakisiyor · tasan etiket 0`

Uretilen kolonilerin dis kenari: p95 = 0.952·R, maks = 1.000·R.
Gercekte: p95 = 0.927·R, maks = 1.003·R. Kenara yakin koloni orani gercekte %3.1,
uretilende %5.2. Fark, yaricapin koloni boyutundan **bagimsiz** orneklenmesinden
geliyor (gercekte buyuk koloniler kenara pek gitmiyor). Kucuk etki, 10 plakta
duzeltilmeye degmez; tam veride bakilacak.

---

## Faz 3 — difuzyon hatti (18 Agustos 2026)

| # | Karar | Gerekce |
|---|---|---|
| 3.51 | **Inpainting DOGAL COZUNURLUKTE ve PARCALI (tiled) yapilacak.** Plagi 512'ye kucultup uretmek yasak. Varsayilan tile 512, `--tile` ile degistirilebilir. | Difuzyon modeli sabit tuvalde calisir (SD 1.5 icin 512). Plagi 2048'den 512'ye kucultursek **C.albicans 27.5 px'ten 6.8 px'e** iner. Bu, `imgsz=1280` argumaninin (karar 2.8) daha sert hali: orada **detektor** 8.6 px'te zorlaniyordu, burada **uretici** 7 piksellik bir nesne sentezlemeye calisacak ve upscaler dokusunu uyduracak. Sonuctaki sey koloni olmaz; ikame egrisinin kucuk-sinif kollari — makalenin ana iddiasi — bir artefakti olcmus olur. |
| 3.52 | **🔴 Bir koloni ASLA iki tile arasinda bolunemez.** `place_tiles()` bolen bir yerlesim uretirse `AssertionError` firlatiyor; `test_generate.py` bunu uretilen plaklarda dogruluyor. | Koloni tile sinirina denk gelirse iki yarisi **ayri geciste, farkli gurultuden** uretilir. Uyusmazlar: koloninin tam ortasindan bir dikis gecer ve detektor "ortasinda cizgi olan koloni" ogrenir. Etiket dogru kalir, goruntu bozuk olur — yani sessiz. |
| 3.53 | **Uretim maliyet modeli degisti: `saniye/goruntu` DEGIL, `tile/goruntu × saniye/tile`.** `budget.py` `--tiles-per-image` ve `--sec-per-tile` aliyor; eski `--gen-sec-per-image` uyari basiyor. | Eski model 8 s/goruntu varsayiyordu, yani ~1 tile. Olculdu: 512'lik tile ile **16.3 tile/plak**, 768 ile 9.7. Koloniler plaga dagilmis oldugu icin (karar 3.18, Clark–Evans ~1.0) maskeye gore akilli yerlestirme tam kapsama izgarasindan pek tasarruf ettirmiyor (16.3'e karsi 20). Tiling'den kacis yok. |

### 🔴 Butce yeniden hesaplandi — sampler artik bir butce karari

58.200 sentetik goruntu × 16.3 tile = **0.95M inpainting gecisi.**

| tile basina | uretim | GENEL TOPLAM |
|---|---|---|
| 1.5 s (SD1.5, ~20 adim) | 403 GPU-saat | **905 GPU-saat** |
| 0.8 s (~10 adim) | 219 | 721 |
| 0.4 s (LCM/Turbo, 4–8 adim) | 113 | 615 |

17 Agustos tahmini uretim icin 137 saat diyordu. Gercek, adim sayisina gore
**113 ile 403 arasinda** — yani denoising adim sayisi bir kalite ayrintisi degil,
**projenin en buyuk tek butce kaldiraci**. Damitilmis bir sampler (LCM/Turbo)
ile 20 adimli SD1.5 arasindaki fark ~290 GPU-saat.

**Faz 4 pilotunda olculecek ilk sey bu:** tile basina gercek sure ve dusuk adimda
kalite kaybi. Ikisi olculmeden BİDB'ye verilecek sayi tek bir aralik olamaz.

🟡 **Tile boyutu ikinci kaldirac:** 768'e cikmak tile sayisini 16.3'ten 9.7'ye
dusuruyor (%40 tasarruf) ama SD 1.5 512'de egitildi; 768 dagitim disi ve VRAM
maliyeti kareyle artiyor. Pilotta iki boyut da denenecek.


### `inpaint.py` yazildi (18 Agustos, ogleden sonra)

| # | Karar | Gerekce |
|---|---|---|
| 3.54 | **Silme bolgeleri icin negatif istem (negative prompt).** Ayrica `inpaint.py` uretimden sonra maske disini degistirmiyor — kompozit yalnizca maskenin izin verdigi yere yaziyor. | Karar 3.25 gercek kolonileri siliyor; oralar **duz besiyeri** olarak geri gelmeli. Model oraya bir koloni uydurursa goruntude **etiketsiz nesne** olusur — tam da silmenin onlemeye calistigi yanlis negatif, bu sefer ureticinin eliyle. Negatif istem zayif bir onlem, garanti degil; Faz 4'te kalan oran bir detektorle olculecek (acik madde). |
| 3.55 | **`inpaint.py` her kosuda `gen_metrics.json` yaziyor**: tile/goruntu, s/tile (ort + p95), s/goruntu, tepe VRAM, model, adim sayisi. Ve gercek kosuda "58.200 goruntu bu ayarla kac GPU-saat" satirini basiyor. Bir de `bench` alt komutu (tile × adim taramasi). | Karar 3.53: uretim maliyeti iki kaldiraca bagli ve ikisi de ampirik. `budget.py`'ye **tahmin edilmis** bir `--sec-per-tile` verilmeyecek; gercek kosu varken olculmus deger kullanilacak. `bench` Faz 4'te GPU-saat rakami telaffuz edilmeden once kosulacak komut. |
| 3.56 | **`--dry` modu**: difuzyon cagrisi yerine belirlenimci bir sahte boyayici. Tiling, istem uretimi, kompozit, maske kapisi, etiket kopyalama, zamanlama — hepsi gercek calisiyor. | GPU'suz makinede boru hattinin dogrulanabilmesi icin. Ve daha onemlisi: bir tesisat hatasi asla "uretim kalitesi sorunu" sanilmasin diye. Kuru mod ciktisi bilerek cirkin; kimse onu gercek ornek sanmasin. |

**Kuru modda uctan uca dogrulandi** (8 plak, 131 tile, 280 koloni):

| garanti | sonuc |
|---|---|
| maskenin DISI degismiyor | ort \|fark\| 0.11–0.27 (yalnizca JPEG), yapisal degisiklik yok |
| etiket uretimden etkilenmiyor | 8/8 birebir ayni |
| hicbir koloni bosta kalmiyor | 280/280 boyandi |
| tile/goruntu | 16.38 — sabahki bagimsiz olcumle (16.3) tutuyor |

### 🔴 Yeni acik madde: JPEG nesil sayisi uyusmuyor

Yukaridaki dogrulama sirasinda cikti. Sikistirma gecmisi **gercek ve sentetik
goruntuler arasinda ayni degil**:

| goruntu | JPEG nesli |
|---|---|
| gercek AGAR | **1** (veri setinin kendisi) |
| `mask.py` -> `backgrounds/*.jpg` | 2 (q96 ile yeniden kodlaniyor) |
| `inpaint.py` -> `images/*.jpg` | **3** (q95) |

Yani her sentetik goruntu, gercek olanlardan **iki nesil fazla** JPEG gormus.
Olculen etki su an kucuk (ort |fark| 0.17, yuksek-frekans enerjisinde %+0.1)
ama **yon sistematik**: detektor "sentetik = su sikistirma izi" kisayolunu
ogrenebilir, ve o zaman ikame egrisi veri kalitesini degil sikistirma farkini
olcmus olur. FID/KID de ayni sahte sinyali yakalar.

Cozum yonu — Faz 4'te karara baglanacak:
1. `mask.py` arka plani **yeniden kodlamasin**; `inpaint.py` orijinal AGAR
   dosyasini okusun. Nesil 3 -> 2'ye iner.
2. Ya da gercek goruntulere de ayni yeniden kodlama uygulansin (nesil esitlenir,
   ama gercek veri gereksiz yere bozulur).
3. Faz 4 olcumu: **yalnizca arka plan yamalarindan** gercek/sentetik ayirt
   etmeye calisan bir siniflandirici. Basarili olursa kisayol var demektir.

Bu, karar 3.12'deki tuzagin akrabasi: sentetik verinin **lehine** degil,
**gecersizligine** calisan ve gozle gorunmeyen bir fark.


### `adapt.py` yazildi ve gozle kontrol dort sey yakaladi (18 Agustos, aksam)

| # | Karar | Gerekce |
|---|---|---|
| 3.57 | **`adapt.py crops` egitim kumesini once cikarip RAPORLUYOR**, GPU'ya dokunmadan. Tur basina koloni sayisi basiliyor; bir tur hic yoksa kirmizi uyari. | Uyarlama kumesinde olmayan bir tur **uretilemez**. Dusuk seviyelerde gercek risk: `train_10`'dan C.albicans kaybolursa G10+S kolu veri miktarini degil sinif dengesizligini olcmus olur (karar 1.5'in ayni mantigi, bu sefer uretici tarafinda). Tek GPU-saniyesi harcanmadan gorulebilmeli. |
| 3.58 | **Kirpmada tile'a atanan degil, kirpmaya DUSEN TUM koloniler maskelenir.** | Ilk surum yalnizca `t.colony_idx`'i maskeliyordu; komsu koloniler kirpmada **gorunur** kaliyordu. Gozle kontrol bunu aninda gosterdi. Ama uretimde `mask.py` arka plandaki butun gercek kolonileri **zaten silmis** oluyor (karar 3.25) — model hicbir gorunur koloni gormuyor. Yani egitim, modele cikarimda var olmayan bir ipucu (komsunun gorunumunu kopyala) ogretiyordu. |
| 3.59 | **Egitim kumesinin %20'si BOS DELIK** (`--empty-ratio`): koloni icermeyen bir besiyeri yamasi maskelenir, hedef ayni duz besiyeridir, istem "no colonies". | Onceki kurguda kume yalnizca "delik -> koloni" ornegi iceriyordu; model bir deligin **her zaman** koloni oldugunu ogrenirdi. |
| 3.60 | **🔴 Uretim IKI GECIS: once SILME (duz besiyeri), sonra SENTEZ (koloniler).** `mask.py build --split-masks` zorunlu hale geldi; birlesik maske kullanilirsa uyari basiliyor. | 3.59'u uygularken asil sorun ortaya cikti ve **temsil duzeyinde**: tek ikili maskeyle model bir deligin koloni mi bos mu olmasi gerektigini **bilemez** — ikisi de sadece delik. Hicbir negatif istem, iki durumu ayirt etmeyen bir girdiyi duzeltemez. Gecisi ikiye bolmek her soruyu tek anlamli yapiyor: silme gecisi `CLEAN_PROMPT` aliyor, sentez gecisi tur istemini. **Karar 3.54'teki "hayalet koloni" riski boylece bir onlem meselesi olmaktan cikip cozulmus oluyor.** |

**Maliyet sonucu:** iki gecis tile sayisini 16.4'ten **27.8'e** cikardi (demo,
havuz esigi 0.25). Silme gecisinin maliyeti arka planin ne kadar kirli oldugu ile
dogru orantili — yani **karar 3.26'daki havuz esigi artik dogrudan bir butce
parametresi**. Temiz havuz (esik 0.06) hem 3.27'deki sinif yanliligini artiriyor
hem de uretimi ucuzlatiyor; ikisi arasindaki denge Faz 4'te olculecek.

**Gozle kontrol notu:** 3.58, 3.59 ve 3.60'in ucu de kodu okuyarak degil,
**egitim kirpmalarina bakarak** bulundu. `check_labels.py`'nin docstring'indeki
cumle burada da gecerliydi: bu 10 dakika, ilerideki gunleri kurtardi.


### Ilk LoRA egitimi kosuldu (18 Agustos, aksam)

RTX 4060 Laptop, demo paketten 168 kirpma, SD 1.5 inpainting, rank 16.

```
trainable params: 3,188,736 || all params: 862,724,100 || trainable%: 0.3696
1.05 s/adim x 1500 adim = 26 dakika
```

| # | Karar | Gerekce |
|---|---|---|
| 3.61 | **Adim basina loss RAPORLANMAZ, yorumlanmaz.** Yerine (a) son `log_every` adimin kosan ortalamasi, (b) **SABIT kirpmalar uzerinde SABIT zaman adimlarinda** hesaplanan dogrulama loss'u basiliyor. Izlenecek sayi ikincisi. | Difuzyon egitiminde her adim rastgele bir zaman adimi orneklıyor: cok gurultulu bir adimda gurultuyu tahmin etmek kolay (loss dusuk), az gurultulude zor (yuksek). `batch=1` ile bu varyans egitim sinyalini tamamen gomuyor. Ilk kosu ardisik loglarda `0.0006, 0.0147, 0.0016, 0.0241` bastı — **hicbir sey ifade etmiyor**. Bir egri gibi okunursa "model ogrenmiyor" ya da "asiri ogreniyor" diye yanlis teshis konur. Sabit kirpma + sabit zaman adimi ile degisen tek sey model olur. |
| 3.62 | **`--ckpt-every` (varsayilan 500): ara LoRA kayitlari.** Her kayitta dogrulama loss'u da olculuyor. | 168 kirpma icin 1500 adim bir **tahmin**. Az mi cok mu oldugu ancak uretip bakarak anlasilir; ara kayit olmadan her cevap icin bastan egitmek gerekir. Kayitla tek kosu "kac adim" sorusunu cevapliyor, uc kosu degil. |

### 🟢 Olculen: LoRA egitimi tahminden ucuz

| | varsayim | **olculen** |
|---|---|---|
| LoRA basina | 2.0 saat | **0.44 saat** (1.05 s/adim × 1500) |
| 4 LoRA | 8.0 saat | **1.75 saat** |

`budget.py --lora-hours 0.44 --lora-count 4`. Toplam icinde kucuk bir kalem
(~6 saat tasarruf) ama artik **tahmin degil olcum** — ve tam veride kirpma
sayisi artacagi icin adim sayisi da artabilir; oran korunur, mutlak deger degil.

**Guncel toplam** (27.8 tile/goruntu, 0.8 s/tile, olculmus LoRA):
egitim 502 + uretim 361 + XAI 0.3 = **~864 GPU-saat**. `sec-per-tile` hala tek
buyuk bilinmeyen; `inpaint.py bench` onu kapatacak.

### Ilk gercek uretim kosusu — 18 Agustos 2026, aksam

Ilk plak uretildi. Iki sey ortaya cikti; ikisi de "calisti" diye gecistirilecek
turden degil.

| # | Karar | Gerekce |
|---|---|---|
| 3.63 | **LoRA'nin yuklendigi KANITLANIR, varsayilmaz.** `attach_lora()` peft ile yukluyor, `merge_and_unload()` ile taban agirliklara katiyor, ve hedeflenen bir agirligi (`attn1.to_q.weight`) katma oncesi/sonrasi karsilastiriyor. Fark sifirsa program **duruyor**. | Ilk kosuda LoRA **hic yuklenmedi**. `adapt.py` peft formatinda (`base_model.model.*` on ekli) yaziyor, `pipe.load_lora_weights()` diffusers/kohya adlandirmasi (`unet.*`) bekliyor; eslesme bulamayinca `No LoRA keys associated to UNet2DConditionModel found` diye **uyari** basip taban modelle uretmeye devam etti. Yani ilk sentetik plak, agar plagi hic gormemis bir modelden cikti. Bu tam olarak 2. tasarim ilkesinin (sessiz hata) hedefi: 58.200 goruntuluk grid boyle uretilseydi sonuc "difuzyon bu is icin calismiyor" diye okunurdu — makalenin ana iddiasi, bir yukleme hatasi yuzunden yanlis cikardi. |
| 3.64 | **Tile sayisi tahmin degil olcum: 31,0 tile/plak** (varsayim 16,3 idi; `tiles.py` bas yorumundaki tablo uretilmis yerlesimlerden geliyordu). Butce artik bu sayiyla kuruluyor. | Fark %90. 16,3 sayisi koloni boyut dagilimina cok duyarli; `layout_100` demo veriden `E.coli` 128 px, `P.aeruginosa` 155 px, `B.subtilis` 151 px medyanlarla cikti — bu boyutta koloniler tek bir 512'lik pencereye az sayida sigiyor, bolunmeme sarti (3.52) da pencere sayisini artiriyor. Ders 3. ilkenin tekrari: modelin kendi ic tablosu bile olculene kadar tahmindir. |

### 🔴 Olculen: uretim maliyeti tahminden **cok** pahali

| | onceki varsayim | **olculen (20 adim, 512 tile, fp16)** |
|---|---|---|
| tile/goruntu | 27,8 | **31,0** |
| s/tile | 0,8 | **3,19** (p95 3,19) |
| s/goruntu | 22 | **98,96** |
| 58.200 goruntu | 361 GPU-saat | **1.597 GPU-saat** |

Tepe VRAM 2,89 GB — kartin (8 GB) cok altinda, yani hizi sinirlayan sey bellek
degil, denoising adim sayisi. Bu haliyle **uretim tek basina egitimin 3 katina**
cikiyor ve toplam ~2.100 GPU-saat oluyor; BİDB'den istenebilecek bir sayi degil.

Kapatilacak kaldiraclar, ucuzdan pahaliya:
1. **Adim sayisi** — 20 → 8 veya 4 (LCM/Turbo damitilmis orneksleyici). Dogrusal: 4 adim ~320 saat.
2. **Tile boyutu** — 768'de tile/plak ~9,7'ye duser ama tile basina sure artar; net kazanc **olculecek**, varsayilmayacak.
3. **Tile'lari toplu (batch) isleme** — 2,89 GB tepe VRAM ile ayni anda 2-3 tile sigar.
4. Son care: sentetik goruntu sayisini dusurmek — ama bu dogrudan makalenin sorusunu daraltir, once digerleri denenecek.

`inpaint.py bench --tiles 512,768 --steps-list 4,8,20` bir sonraki adim.

### 🟢 Faz 3 kapisi: koloniler gecti

LoRA'li ilk plak uretildi ve **ikna edici**. Yerel cozunurlukte koloniler isinsal
lifli dokuya, hale ve golgeye sahip; kutular kolonilere tam oturuyor (36 kutu,
gozle sapma yok — karar 3.24'un sifir etiket hatasi garantisi goruntude de
tutuyor). LoRA'siz ayni plak (kazara uretilen A1 ablasyon ornegi) neredeyse bos:
soluk saydam diskler ve bir iki tuhaf nesne. Fark, LoRA'nin gerekliligini
gorsel olarak da kanitliyor.

### 🔴 Ama: hayalet koloniler olculdu, ve cok yuksek

Ayni plakta arka planin **silinmesi gereken 35 gercek kolonisinden 23'u geri
geldi** — sari-yesil, etiketsiz nesneler olarak. Bunlar detektor icin yanlis
negatif; egitim verisinde "burada koloni yok" diye ogretilirler.

| silme gecisi | koloni olarak geri gelen bolge |
|---|---|
| difuzyon + `lora_100` | **23 / 35 (%66)** |
| difuzyon, taban model | 6 / 35 (%17) |
| `cv2.inpaint` (Telea, r=7) | **0 / 35 (%0)** |
| (referans: hic dokunulmamis gercek arka plan) | 34 / 35 |

| # | Karar | Gerekce |
|---|---|---|
| 3.65 | **Silme gecisi KLASIK yapilir (`cv2.inpaint`, Telea r=7), difuzyonla degil.** `--erase-method diffusion` karsilastirma tekrar uretilebilsin diye duruyor ama varsayilan degil. | Ustteki tablo. LoRA silme gecisini **daha kotu** yapiyor ve bu kacinilmaz: LoRA tam olarak "besiyerindeki deligi koloniye cevir" diye egitildi, gecis 1 ona tam bunu veriyor. Istemle (prompt) geri alinamaz — agirlik guncellemesi istemden gucludur. Klasik doldurma koloni **uyduramaz**, cunku uyduracak bir sey yok: cevredeki besiyerini ice dogru tasir, "duz besiyeri" zaten bunun tanimi. Ustelik bedava: o plakta silme izgarasi 31 tile'in 12'siydi, gecis tamamen GPU butcesinden cikiyor (**-%39**). Bedeli: delik cevresine gore buyukse klasik doldurma bulaniklastirir. Havuz esigi (erase_ratio ≤ 0.06, karar 3.26) delikleri kucuk tutuyor ama bu bir garanti degil, o yuzden en buyuk delik capi plak basina olculup raporlaniyor (`ERASE_WARN_PX = 120`). |

**Butceye etkisi:** 31 → **19 tile/goruntu**. 58.200 goruntu 1.590 → **~975
GPU-saat**. Adim sayisi kaldiraci hala acik.

**Acik kalan "hayalet koloni" maddesi kapandi** — ama tam olarak degil: %0 bu tek
plakta ve *bu* olcutle (bolge ici ile cevre halka arasindaki kanal farki > 25).
Faz 4 pilotunda ayni sey bir detektorle, coklu plakta tekrar olculecek.

### ✅ Faz 3 kapisi KAPANDI — 18 Agustos 2026

Klasik silmeyle uretilen plak (`g100_v2`) olculdu ve goruldu.

| | difuzyon silme | **klasik silme** | referans (gercek arka plan) |
|---|---|---|---|
| hayalet koloni | 17 / 29 | **0 / 29** | 22 / 29 |
| medyan kanal farki | 30,2 | **1,5** | 41,6 |
| tile/goruntu | 31,0 | **19,0** | — |
| s/goruntu | 98,5 | **60,0** | — |
| 58.200 goruntu | 1.590 GPU-saat | **961 GPU-saat** | — |

Olcum notu: ilk hesap 7/35 vermisti. Yanlisti — sentetik koloninin silme
bolgesiyle **ortustugu** kisimlar (ortalama %5,5, karar 3.28) bolge icini
karartip "hayalet" gibi gosteriyordu. Orasi zaten **etiketli** bir koloni. Olcut
duzeltildi: silme bolgesinin yalnizca sentetik maske disinda kalan pikselleri
sayiliyor. Duzeltilmis sayi 0/29.

`ERASE_WARN_PX` uyarisi 172 px'lik bir bolge icin atesledi; o bolge gozle
kontrol edildi, klasik doldurma orada da temiz. Uyari kalsin — bu sefer yanlis
alarmdi, her zaman olmayabilir.

**Plak gozle:** cerceve, plaka kenari, uzerindeki yazi ve cizikler gercek;
36 kolonin hepsi etiketli; etiketsiz tek bir nesne yok. Tek sorulacak soru
("bu gercek olabilir mi?") icin cevap **evet**.

### 🟢 Olculen: iki butce kaldiraci (bench, 2 plak)

|  tile | adim | tile/goruntu | s/tile | 58.200 goruntu |
|---|---|---|---|---|
| **512** | **4** | 16,0 | **0,897** | **232 saat** |
| 512 | 8 | 16,0 | 1,469 | 380 saat |
| 512 | 20 | 16,0 | 3,208 | 830 saat |
| 768 | 4 | 10,0 | 2,214 | 358 saat |
| 768 | 8 | 10,0 | 3,621 | 585 saat |
| 768 | 20 | 10,0 | 7,843 | 1.268 saat |

| # | Karar | Gerekce |
|---|---|---|
| 3.66 | **Uretim ayari: tile 512, 4 denoising adimi.** | (a) **768 baskin sekilde kotu.** Alan 2,25 kat buyuyor, tile sayisi yalnizca 1,6 kat azaliyor; her adim sayisinda 512'den pahali. Tile boyutu bir kaldirac degilmis — olculmeden bilinemezdi. (b) **4 adim gorsel olarak 20 adimdan ayirt edilemiyor.** Maskeler kucuk ve cevre baglami cok guclu oldugu icin ornekleyici az adimda yakinsiyor. (c) Doku olcumu de ayni yonu gosteriyor: koloni ici Laplace standart sapmasi 4 adimda **26,7**, 20 adimda 21,7, gercek kolonilerde **25,3** — yani 20 adim gercekten daha *puruzsuz*, 4 adim gercege daha yakin. Uyari: bu referans arka plandaki kucuk `S.aureus` kolonilerinden, sentetikler ise buyuk `P.aeruginosa`; olcum destekleyici bir veri, tek basina karar degil. Karar gozle + bu olcumun birlikte. |

**Uretim: 961 → 232 GPU-saat.** Bugun uretim kalemi 1.590'dan 232'ye indi (%85).

### Guncel butce

| kalem | GPU-saat |
|---|---|
| egitim (61 kosu) | 375 – 623 |
| uretim (58.200 goruntu, 512/4 adim) | **232** |
| LoRA x4 (olculdu) | 1,8 |
| XAI | 0,3 |
| **GENEL** | **609 – 857 GPU-saat** |

Uretim artik toplamin %30'u degil, ~%30'undan azi — ve **olculmus** bir sayi.
Kalan belirsizligin tamami egitim tarafinda; o da ilk G100 kosusuyla kapanacak.

### ✅ Olculen: JPEG kestirmesi riski dusuk (19 Agustos)

Acik maddelerden biri suydu: gercek AGAR goruntusu **bir kez** JPEG sikistirilmis,
sentetik goruntu ise coz-uret-yeniden sikistir zincirinden geciyor. Detektor
kolonileri degil **sikistirma izini** ogrenebilir; o zaman S100 kolu gercek test
setinde coker ve bu "sentetik veri ikame etmiyor" diye okunur.

Iki sey olculdu:

| | nicemleme tablosu (toplam) | blokluk orani (duz besiyeri) |
|---|---|---|
| gercek AGAR (14618) | 369 | 0,927 |
| ayni plagin sentetik ciktidaki hali | 369 | 0,930 |
| baska bir gercek plak (14512) | 369 | 0,969 |

Nicemleme tablosu **ayni** (`cv2.imwrite` varsayilan kalitesi AGAR'inkiyle
ortusuyor). Kalan cift-sikistirma izi ise blokluk oraninda 0,003'luk bir fark
yaratiyor — **iki gercek plak arasindaki fark bunun 14 katı** (0,042). Yani iz,
dogal plaklar arasi degiskenligin altinda kaliyor.

Bu madde kapandi. Faz 4'te sifirdan olcmeye gerek yok; sadece uretim ayarlari
degisirse (ornegin PNG'ye gecilirse) tekrar bakilacak.

### 🟡 Olculen: tur gorunum sadakati (19 Agustos)

Olcum artik elle degil, repoda bir arac: `src/generate/species_check.py`
(karar 3.68). Koloni kutusunun **ic %35'lik diski** olculuyor; kenar disarida
kaliyor. "Sarilik" = R-B kanal farki, "kontrast" = koloni ici gri ile cevre
besiyerinin medyan grisi arasindaki fark, "doku" = Laplace standart sapmasi
(ksize=3).

`--level 100`, gercek: 10 plak / 387 koloni, sentetik: 4 plak / 155 koloni.

| tur | n (gercek/sentetik) | cap px | sarilik | kontrast | doku |
|---|---|---|---|---|---|
| B.subtilis | 53 / 86 | 151/150 | 34,3 → 35,2 | 35,1 → 25,9 | 9,2 → **41,3** |
| P.aeruginosa | 41 / 50 | 155/154 | 36,9 → 37,1 | 11,5 → 21,8 | 5,2 → **15,9** |
| S.aureus | 152 / 15 | 29/29 | 89,9 → **58,1** | 63,2 → 38,7 | 11,3 → **87,8** |

`E.coli` sentetik tarafta 4 koloni (< MIN_N=15) oldugu icin karsilastirmaya
girmedi, `C.albicans` uretilen 4 plagin hicbirinde yok — ikisi de **raporlaniyor**,
sessizce dusurulmuyor.

**1) Siralama testi: kanal CALISIYOR.** Sinifları her uc istatistige gore
siralayip gercek ile sentetigi karsilastirinca (Spearman) uc olcutte de
**+1,00**. Yani model turu ayirt ediyor; "metin kodlayici donuk, sinyal
gecmiyor" endisesi **curudu**. Bu, kullanicinin hipotezini (veri azligi)
destekliyor.

**2) Ayrilabilirlik testi: buyukluk cokmus.** En yakin iki sinifin merkezleri
arasindaki mesafe, sinif ici yayilim biriminde:

| | gercek | sentetik | korunan |
|---|---|---|---|
| uc eksen (sarilik+kontrast+doku) | 0,91 (CI90 0,71-2,49) | 0,98 (CI90 0,85-1,62) | %107 |
| **doku ekseni cikarilinca** | **1,65** | **0,44** | **%27** |

Bu satir kritik ve tek basina bu aracin varlik sebebi: uc eksenle bakinca
sentetik veri gercek kadar ayrilabilir gorunuyor (%107) — ama bu ayrilma
**yanlis eksenden** geliyor. Doku, modelin yanlis yaptigi istatistik (karar
3.67) ve siniflara gore farkli sekilde yanlis yapiyor, dolayisiyla
ayrilabilirligi **sahte olarak sisiriyor**. Gercegin kullandigi eksenlerde
(renk + kontrast) sentetik veri gercek ayrimin ancak **%27**'sini koruyor.

| # | Karar | Gerekce |
|---|---|---|
| 3.68 | **`species_check.py`: sinif sadakati kalici bir olcum haline getirildi.** Iki hukum veriyor — siralama (kanal canli mi?) ve ayrilabilirlik (siniflar ayirt edilebilir mi?), ikincisi **doku ekseni dahil ve haric**. Plak duzeyinde onyukleme (bootstrap) araligi veriyor. `--level` zorunlu ve gercek liste o seviyeye ait olmak zorunda (karar 3.2/3.7). `test_generate.py` 9 yeni kontrolle bunu sinar (toplam 55). | Simdiye kadarki her kapi "gercekci mi" diye soruyordu; hicbiri "**dogru mu**" diye sormuyordu. Jenerator dogru gorunumlu ama **yanlis turde** bir koloni cizerse, plak butun mevcut kapilardan gecer, sinif basina AP hicbir sey olcmez ve S100 kolu "sentetik veri ikame etmiyor" der — oysa gercek bulgu "jenerator sinifi yok saydi"dir. Bu iki iddia farklidir ve karistirilamaz. Ayrica olcumun kendisi de yanilabilir: ilk elle hesapta Laplace cekirdegi yanlis (`ksize` yerine `dst` konumuna 3 verilmis) idi ve tum doku sayilari ~3 kat dusuk cikti; bu tablodaki degerler duzeltilmis olanlar. Bir sayiyi arac haline getirmek, onu tekrar edilebilir ve **hatasi bulunabilir** kilar. |

**Faz 4 icin acik hedef:** doku oranini 2'nin altina indirmek ve doku-siz
ayrilabilirligi %27'den yukari cikarmak. Tam veride tur basina ornek sayisi
~100 kat artacak; olcum ayni araclarla tekrarlanacak ve **ilerleme sayiyla**
gosterilecek.

### Yeni acik madde

| Konu | Durum |
|---|---|
| 🟡 **Tur gorunum sadakati — kanal CALISIYOR, ama zayif** | Olculdu (asagidaki tablo). Tur kosullamasi olu degil: sarilik siralamasi gercek veriyle **birebir ayni**, ve `P.aeruginosa` her ikisinde de en dusuk kontrastli tur. Ama buyuklukler sikismis (`S.aureus` sariligi 90 yerine 58). Yani model turu ayirt ediyor, yeterince kuvvetli ayirt etmiyor — bu tam olarak "veri az" tablosu. Tam veride tur basina ornek ~100 kat artacak. Faz 4'te ayni olcum tekrarlanacak; hala sikisiksa metin kodlayici da egitilecek. |


---

## Kod taramasi ve butce duzeltmesi (17 Agustos 2026)

Deponun tamami satir satir okundu; 25 madde cikti. Tamami `KOD_HARITASI.md`'de,
oncelik siralamasiyla ve "ne yapmali" sutunuyla. Bugun **yalnizca butce kalemi**
duzeltildi cunku BİDB talebi ona bagli.

| # | Karar | Gerekce |
|---|---|---|
| 3.29 | **`budget.py` kol tanimlari `(name, total_share)` -> `(name, total_share, synth_share)`.** Sentetik ihtiyaci artik **konfigurasyon adindan turetilmiyor**, tanimda acikca yaziyor. | Eski `synth_need()` isimde `"+S"` ariyordu. Ablasyon kollarinin adlarinda (`A1_noLoRA`, `A2_background`, `A3_naive_layout`) `+S` yok — ama **ucu de bastan yeni bir sentetik kume gerektiriyor**: A1 uyarlanmamis modelle, A2 gercek arka plan olmadan, A3 naif yerlesimle uretilmis. Ayni goruntuler tekrar kullanilamaz; ablasyonun tum anlami uretimin farkli olmasi. Sonuc: sentetik pay 5.03 sayiliyordu, gercegi 7.28 — **uretim hacmi %45 eksik hesaplaniyordu.** Isimden cikarim yapmak bu hatanin kaynagiydi; artik tanim tek dogru kaynak. |
| 3.30 | **`--lora-count` argumani eklendi, varsayilan 4.** Sabit `3 *` carpani kaldirildi. | Karar 3.2 (seviye basina ayri LoRA: `lora_10`, `lora_25`, `lora_50`, `lora_100`) 14 Agustos'ta alinmisti; karar 3.9 bu parametreyi acikca istemisti. Uc gun kodda karsiligi yoktu. |
| 3.31 | **Senaryo satirlarinda yalnizca EGITIM maliyeti olcekleniyor.** `epoch 150->100` ve `imgsz` senaryolari artik `total_hours * carpan + fixed` (fixed = uretim + XAI). Ayrica "yan kollar seed 3->2" satiri ilgili kollarin secili olup olmadigini kontrol ediyor. | Onceki surum `grand_total`'i (uretim ve XAI dahil) olcekliyordu. Ama inpainting suresi epoch sayisina, egitim `imgsz`'sine bagli degil. Kisma kazanci oldugundan buyuk gorunuyordu ve `--budget` ile "SIGIYOR" isareti hak etmedigi yerde cikabilirdi. |

### Kirmizi maddelerin kapatilmasi (17 Agustos, aksam)

| # | Karar | Gerekce |
|---|---|---|
| 3.32 | **Baseline B'de `degrees: 180.0` -> `0.0`.** Rotasyon kaldirildi; `fliplr` + `flipud` + `translate` + `scale` + HSV kaliyor. | Plagin kanonik yonu gercekten yok — dondurme **mesru**. Sorun etikette: Ultralytics dondurmede kutunun dort kosesini dondurup **eksene hizali** yeni kutu ciziyor (`RandomPerspective.apply_bboxes`, kose min/max). Kare bir kutu 45°'de kenari **√2 katina**, alani iki katina cikiyor. AGAR kolonileri **yuvarlak** ve kutularin **%96.9'u kare**: yuvarlak bir nesne dondurulunce dogru kutu **hic degismez**. Yani her acili dondurme bilgi katmiyor, **etiketi bozuyor**. Baseline B sisirilmis kutularla egitilip dogru kutulu testte olculurse mAP'i haksiz duser → **difuzyon lehine sahte kazanc**. Baseline C'nin `copy_paste` riskinin aynadaki hali (karar 2.65'teki uyari). `fliplr`+`flipud` dihedral grubun 4 elemanini zaten veriyor ve kutuyu hic bozmuyor. **Makalede yazilacak:** *"Baseline B'de rotasyon uygulanmamistir; eksene hizali kutu gosteriminde acili rotasyon etiket kutusunu genisletir ve dairesel koloniler icin bilgi katmadan etiket hatasi uretir. Yatay/dikey cevirme bu bozulmayi uretmedigi icin tercih edilmistir."* |
| 3.33 | **`scripts/collect.py` yazildi.** `runs/*/run_metrics.json` + `runs/*/eval*/summary.json` taranip `results/` altina toplaniyor: `table.csv` (kosu × split, 40 sutun), `raw/<kosu>.json`, `summary.txt`. **`results/` git'e giriyor.** | `runs/` .gitignore'da ve oyle kalmali (agirliklar GB'larca). Ama icinde yalnizca agirlik yok: her kosunun suresi, tepe VRAM'i, GPU-saati, git commit'i ve butun mAP/AP/sayim sonuclari da orada. Yani **makalenin butun sonuclari surum kontrolunun disindaydi.** Bu dosyalar birkac KB ama yeniden uretilmesi 500+ GPU-saat surer; dizustu olursa ya da BİDB sunucusuna gecilince hicbir yerde yoklar. Ayrica 61 kosuluk grid tablosu zaten bir sekilde kurulacakti — iki ihtiyac tek dosyada. `collect.py` `run_metrics.json`'u olmayan kosulari da 🔴 ile bildiriyor (train.py atlanmis demektir). |
| 3.34 | **`evaluate.py` tahminleri sıraya göre değil YOLA göre eşliyor.** Sayi uyusmazsa program duruyor. | `zip(images, boxes_all)` konum tabanliydi. Tek bir goruntu atlanirsa (bozuk dosya) `zip` sessizce kisa olanda durur ve **o noktadan sonraki tum goruntuler yanlis tahminlerle eslesir**. Hata mesaji yok, program calisir, mAP saçmalar. Artik `r.path`'in stem'iyle sozluk kuruluyor; eksik/fazla/mukerrer her durumda `sys.exit`. |
| 3.35 | **`evaluate.py` yollari `KOK` tabanli.** `Path("splits")` -> `KOK / "splits"`. | `train.py` en tepede `KOK = Path(__file__).resolve().parents[1]` tanimlayip her seyi mutlak yapiyordu; `evaluate.py` goreli yol kullaniyordu. Yalnizca repo kokunden calisiyordu, ve baska bir klasorde `splits/` varsa **yanlis bolme listesini** okuyabilirdi. Iki dosya artik ayni kurali izliyor. |
| 3.36 | **`--split train` icin `--conf-thr` zorunlu.** | Eskiden `secilen = None` kalip `counting_metrics`'te `TypeError` veriyordu. Egitim kumesinde esik aramak sizinti degil ama anlamsiz (model o veriyi gordu, secim iyimser cikar). Artik net hata mesaji. |

### Testler (17 Agustos, aksam)

| # | Karar | Gerekce |
|---|---|---|
| 3.37 | **`src/generate/test_generate.py` yazildi — 36 kontrol, 15 grup.** `setup.sh`'in 5. adimina eklendi; artik iki test takimi da gecmeden kurulum bitmiyor. | `metrics.py`'nin 33 testi vardi cunku kendi AP hesabina guvenilmiyordu. `layout.py`'de en az onun kadar sinsi matematik var (ters-CDF ornekleme, Strauss ikili aramasi, lognormal kestirim) ve **bu dosyalar makalenin verisini uretecek**. `validate` alt komutu bir karsilastirma raporu, birim testi degil: yanlis bir quantile interpolasyonu medyani dogru tutup kuyrugu bozabilir ve `validate` yakalamayabilir. |
| 3.38 | **`test_metrics.py`'ye `read_yolo_txt` testi eklendi** (10. grup, 9 kontrol). Toplam 36 (pycocotools ile 42). | 33 kontrolun hepsi `ImageAnno`'yu **dogrudan piksel kutulariyla** kuruyordu, yani normalize↔piksel donusumunun yasadigi tek yeri atliyordu. `W` ile `H` yer degistirse ya da conf sutunu kaysa hicbir test yakalamazdi. Test bilerek **W ≠ H** (800×400) kullaniyor — kare goruntude takas hatasi gorunmez olurdu — ve "takas farkli kutu uretiyor mu" diye testin kendi duyarliligini da dogruluyor. |

**`test_generate.py` neyi koruyor:**

| grup | kontrol ettigi karar |
|---|---|
| ters-CDF ornekleme (5 kontrol) | 3.11 — bilinen quantile izgarasindan uretilen orneklemin quantile'lari giriyle eslesiyor mu |
| ortusme derinligi · degme orani | 3.12 — elle bilinen vakalar (tegetsel 0.0, cakisik 1.0, 2/4 degen → %50) |
| sayim siniri | 3.17 — hicbir plak min/maks disina cikmiyor, medyan `exp(ln_ort)`'a oturuyor |
| gamma uclari + monotonluk | 3.19 — `gamma=0` → %0 degme, `gamma=1` → %38.5; arada **monoton artis** (ikili arama bunu varsayiyor, dogrulanmamisti) |
| sert sinir | 3.12 — kalabalik plakta bile en derin ortusme 0.628 ≤ 0.63 |
| plak/goruntu sinirlari | 3.10 — 1500 koloni, disari tasan sifir |
| sessiz kayip | 3.22 — `uretilen + dusen = istenen`, her plakta |
| belirlenimcilik | ayni seed → ayni cikti, farkli seed → farkli |
| A3 naif mod | 3.18 — naif modda kolonilerin %18'i plagin **disinda**, boyut sinifa kosullu degil; ana hatta ikisi de yok. **Ablasyonun bos olmadigini kod duzeyinde kanitliyor.** |
| seviye kilidi | 3.7 — parametre seviyesi ≠ `--level` → `SystemExit` |
| maske alani | 3.24 — tek koloninin beyaz alani analitik `πr²`'ye %1 icinde (21.101 vs 21.083 piksel) ve `SYNTH_MARGIN == 1.00` sabiti test ediliyor: **birisi bu sabiti degistirirse test kirilir**, "etiket hatasi sifir" iddiasi sessizce kaybolmaz |
| plak kirpmasi | 3.10 — plak diski disindaki koloni maskeye hic girmiyor |
| silme payi | 3.25 — silme alani gercek koloninin `ERASE_MARGIN²` = 1.82 kati (olculen 1.91x, bulaniklastirma payiyla) |

### Kalan maddeler kapatildi (17 Agustos, gece) — tarama tamamlandi

| # | Karar | Gerekce |
|---|---|---|
| 3.39 | **Ornekli kontroller kaldirildi.** `make_splits.py` `written[:200]` ve `train.py` `lines[:50]` -> **tam liste**. Eksik etiketlerden ilk 5'i yol olarak basiliyor. | Tam veride train ~8000 goruntu; eskiden %2.5 ve %0.6'si kontrol ediliyordu. Bu iki kontrolun **tum amaci** sessiz etiket kaybini yakalamak — orneklemeyle yapilan kontrol o amaci karsilamiyor. `Path.exists()` mikrosaniye surer, 8000 tanesi bir saniye tutmaz. |
| 3.40 | **`augment` alaninda `null` yasak** (yalnizca `auto_augment` haric — Ultralytics'te None = kapali). Ihlalde program **on ucusta** duruyor, `--dry-run` ile de. | `**{k: v for k, v in cfg["augment"].items() if v is not None}` null'lari sessizce filtreliyordu → Ultralytics kendi varsayilanini kullaniyordu. Karar 2.13'un ("ortuk varsayilan = tekrarlanamazlik") tam olarak yasakladigi sey. Bugun zararsizdi, ama biri `mosaic: null` yazsa politika sessizce coker (mosaic=1.0). Test edildi: durdu. |
| 3.41 | **W&B artik config'ten yonetiliyor.** `logging.wandb` / `logging.project` okunuyor, `WANDB_MODE` ve `WANDB_PROJECT` ortam degiskenlerine yaziliyor. `--smoke` W&B'yi **gercekten** kapatiyor. | `base.yaml`'daki `logging` blogu hicbir yerde okunmuyordu ve `--smoke`'un "W&B kapali" diyen yardim metninin kodda karsiligi yoktu. `wandb login` yapildigi an Ultralytics onu kendiliginden algilayip **kendi varsayilan proje adina** loglayacakti, duman testleri de oraya karisacakti. |
| 3.42 | **`train.py --resume`**: yarim kalan kosu sarmalayicinin icinden devam ediyor; `run_metrics.json`'a `resumed: true` yaziliyor. Notebook'un resume hucresi buna cevrildi. | Notebook `YOLO('last.pt').train(resume=True)` oneriyordu — `train.py`'yi tamamen atliyor, yani `run_metrics.json` **hic yazilmiyor**: sure, tepe VRAM, git commit kayboluyor. G100'un amaci "hem ust sinir hem sure olcumu"ydu; o cagri olcum yarisini sessizce siliyordu. Colab'da kopma kural, istisna degil. |
| 3.43 | **`empty` ve `uncountable` ayri sayaclara ayrildi** (`colonies_number` ile: 0 -> empty, >0 -> uncountable, okunamiyorsa bilinmeyen). | Tek sayacta toplaniyorlardi. Makalenin veri bolumunde "kac goruntu neden elendi" ayri raporlanacak; "12.000 goruntu countable degildi" cumlesi hangisi kac tane sorusunu cevaplamiyor. |
| 3.44 | **Kirik symlink onariliyor.** `is_symlink() and not exists()` -> `unlink()` + yeniden olustur. | `exists()` symlink'i takip eder: kaynak AGAR klasoru tasinirsa link kirilir, `exists()` False doner, kod yeniden symlink atmaya calisir ve `FileExistsError` ile patlardi. |
| 3.45 | **`check_labels.py` kenardan tasan kutuyu yakaliyor.** | Onceki kontrol yalnizca merkez ve genisligin 0–1 arasinda olmasina bakiyordu; `xc=0.99, bw=0.1` gecerdi ama kutu goruntunun 1.04'une uzanir. `convert.py` kirptigi icin olmamali — bu dosyanin isi tam olarak "olmamali"yi dogrulamak. |
| 3.46 | **`ultralytics>=8.4.118`** (eski: `>=8.3.0`). Notebook'ta `==8.4.118`. Faz 5'te `pip freeze > requirements.lock`. | `base.yaml` `yolo26n.pt` ve `cutmix` istiyor, ikisi de 8.4'te geldi — eski kisit **config'i calistiramayacak** bir surume izin veriyordu. Colab'da her acilista guncel surum cekiliyordu; farkli surum → sureler karsilastirilamaz, bolum 6'nin tablosu savunulamaz. |
| 3.47 | **`setup.sh` `cu130` kuruyor** (`CUDA_CHANNEL` ile degistirilebilir, cu124'e ve varsayilana dusuyor). Batch onerisi olculen degere cekildi: 8 GB -> **8** (eskiden 4). | Tum olcumler `torch 2.13.0+cu130` ile yapildi ama script `cu124` kuruyordu; BİDB sunucusunda farkli bir yapi kurulacakti. Batch onerisi de karar 2.19'da olculmustu (batch 8 -> 6.05 GB, tavan orada), sezgisel deger altinda kaliyordu. |
| 3.48 | **`max_det` ve `nms_iou` `base.yaml`'dan okunuyor** (`evaluate.py --config`). Kodda kalanlar yalnizca yedek. `summary.json`'a da yaziliyor. | Ikisi de `evaluate.py`'de sabitti (1000 / 0.7) ve `base.yaml`'daki degerler **hicbir yerde okunmuyordu** — config'i degistirmek hicbir sey yapmiyordu. Faz 5'te "protokolu donduruyoruz" derken dondurulan sey calisan sey olmali. |
| 3.49 | **`analyze_manifest.py`'de `radius` artik PLAK YARICAPINA gore normalize** (`layout.py` ile ayni birim). Ayrica koloninin **dis kenari** ayri raporlaniyor. | Eskiden goruntunun yari genisligiyle normalize ediliyordu ve yorum *"plak kareye tam oturuyor"* diyordu — yanlis: olculen `R = 0.465 × genislik`. Gercek plak kenari o birimde **0.930**'a denk geliyordu ama kontroller 0.95 ve 1.0'daydi, yani **"plak disi" kontrolu asla atesleyemiyordu**. Ustelik iki dosya ayni kelimeyi iki farkli anlamda kullaniyordu; grafigin ekseni de ("1 = kenar") makaleye yanlis giderdi. |
| 3.50 | **`--iou-dup` esigi artik veriden kalibre ediliyor.** `analyze_manifest.py` gozlenen IoU dagiliminin p50/p90/p95/p99/p99.9/maks degerlerini basiyor ve esik p99.9'un altindaysa uyariyor. | 0.5 bir **tahmindi**. Gercek koloniler degiyor (%39, ortusme derinligi 0.63'e kadar — karar 3.12), yani yuksek IoU'lu ciftler **mesru**. Demo pakette olculdu: p99.9 = 0.436, maks = **0.450** — payi yalnizca 0.05. Tam veride kalabalik plaklar bu esigi asacak ve `duplicate_suspects.csv` gercek degen kolonilerle dolacak. |

**Tarama kapandi.** 17 Agustos'ta bulunan 25 maddenin **25'i** ele alindi.
Ayrinti ve gerekce: `KOD_HARITASI.md`.

### Guncel butce tahmini (BİDB talebi icin)

Varsayimlar: 8000 egitim goruntusu · 150 epoch · imgsz 1280 · inpainting 8 s/goruntu ·
LoRA 2 sa/adet · 4 LoRA · tek sentetik kume (seed'ler paylasiyor).

| kalem | GPU-saat |
|---|---|
| ana grid (8 konf x 5 seed = 40 kosu) | 234 – 389 |
| klasik kol (2 x 3 = 6) | 12 – 20 |
| miktar taramasi (2 x 3 = 6) | 57 – 95 |
| ablasyon (3 x 3 = 9) | 72 – 120 |
| **egitim toplami (61 kosu)** | **375 – 623** |
| uretim: 58.200 sentetik goruntu + 4 LoRA | 137 |
| XAI (egitim yok, cikarim) | 0.3 |
| **GENEL** | **512 – 761 GPU-saat** |

Onceki tahmin 470–720 idi. Fark 3.29 ve 3.30'dan geliyor.

🔴 **Iki parametre hala tahmin:** inpainting suresi (8 s/goruntu) ve LoRA egitim
suresi (2 sa). Ikisi de Faz 4 pilotunda olculecek. Uretim kalemi (137 sa) bu iki
sayiya dogrudan bagli — %50 sapma toplamda ±70 saat demek. BİDB'ye verilecek
sayida bu belirsizlik belirtilmeli.

🔴 **Egitim tarafi da G100 kosusu gelene kadar tahmin.** Aralik, duman testinin
cikarim hizindan (8.0 ms/goruntu) turetildi; egitim = 3x / 4x / 5x cikarim
varsayimlariyla uc senaryo.

### Acik kalan

| Konu | Durum |
|---|---|
| ✅ **Silme bolgesinde hayalet koloni — KAPANDI (18 Agustos, karar 3.65)** | Endise doğru cikti: difuzyonla silme yapildiginda 35 silme bolgesinin 23'u koloni olarak geri geldi (LoRA ile %66). Cozum silme gecisini klasik doldurmaya (`cv2.inpaint`) cevirmek oldu: 0/29, ve GPU maliyetinden -%39. Ayrintili olcum asagida, 18 Agustos bolumunde. Faz 4'te bir detektorle coklu plakta tekrar dogrulanacak. |
| Clark–Evans %10 yuksek | Uretilen 1.13, gercek 1.03. gamma kisa menzilde itiyor ama gercek veri daha *heterojen*: bazi ciftler cok ic ice, bazilari cok uzak. Tek parametreli Strauss bunu tam yakalayamiyor olabilir. 10 plakta karar verilemez; tam veride bakilacak. Gerekirse iki menzilli (sert cekirdek + yumusak itme) modele gecilir. |
| `lora_10`'un veri yeterliligi | ~800 goruntuluk alt kumede LoRA'nin ise yarar bir koloni gorunumu ogrenip ogrenemedigi olculmeli. Ogrenemezse G10+S kolu "sentetik veri yardim etmiyor" degil, "uretici model yetersiz veriyle uyarlanmis" sonucunu verir — ikisi farkli iddialar ve karistirilamaz. Pilotta (Faz 4) test edilecek. |
| Arka plan havuzunun buyuklugu | `train_10` icinde yeterince bos/az koloni iceren plak var mi? Yoksa ayni arka plan defalarca kullanilacak → sentetik cesitlilik duser, FID/KID bunu yakalar. Olculecek. |

---

## Faz 4 — pilot (19 Agustos 2026)

### 🔴 Ilk deney: dokunun sebebi guidance DEGIL

Hipotez: doku fazlaligi yuksek guidance'tan (7,5) geliyor olabilir; istemde
"sharp focus" var ve yuksek guidance modeli istemi asiri takip etmeye zorluyor.
Test: ayni 4 plak, guidance 1,5 / 3,0 / 5,0 / 7,5. Toplam ~5 dakika.

| guidance | B.subtilis doku | P.aeruginosa doku | S.aureus doku | dokusuz ayrilabilirlik |
|---|---|---|---|---|
| gercek | 9,2 | 5,2 | 11,3 | %100 (1,65) |
| 1,5 | 55,2 | 31,1 | 33,2 | **%6** |
| 3,0 | 48,9 | 26,7 | 58,3 | %32 |
| 5,0 | 44,4 | 22,4 | 66,7 | %25 |
| **7,5 (mevcut)** | **41,3** | **15,9** | 87,8 | %27 |

**Hipotez curudu.** Hicbir guidance degeri doku oranini 2'nin altina indirmiyor;
en iyi degerler zaten mevcut varsayilanda (7,5). Dahasi guidance'i dusurmek
**zarar veriyor**: 1,5'te `S.aureus` sariligi 58 → 39'a (gercek 89,9) ve sinif
ayrimi %27 → %6'ya cokuyor. Buyuk ve kucuk koloniler ters yonde tepki veriyor
(guidance artarken B.subtilis dokusu duser, S.aureus dokusu yukselir) — bu da
tek bir "keskinlik" acikamasinin yanlis oldugunu gosteriyor.

**Karar: guidance 7,5'te kaliyor.** Negatif sonuc ama ucuz ve kesin.

### 🔴 Ikinci deney: sebep LoRA da degil

Ayni plak, ayni tohum, ayni adim sayisi (20), yalnizca LoRA var/yok:

| | P.aeruginosa doku |
|---|---|
| gercek (plak 14512) | **4,6** |
| sentetik, **LoRA YOK** | 20,2 |
| sentetik, LoRA VAR | 18,9 |

LoRA dokuyu neredeyse hic degistirmiyor. Yani fazla doku **taban modelin**
ozelligi, uyarlamanin degil. Bu, "veri artinca duzelir" beklentisini de zayiflatiyor:
tam veri LoRA'yi iyilestirir, taban modeli degistirmez.

### 🟢 Ucuncu olcum: kusur "fazla doku" degil, "yanlis olcekte doku"

Koloni **capi** ile **doku** arasindaki sira korelasyonu (Spearman):

| | gercek | sentetik |
|---|---|---|
| B.subtilis | −0,22 | −0,02 |
| P.aeruginosa | −0,42 | +0,09 |
| C.albicans | −0,54 | (uretilmedi) |
| E.coli | −0,27 | (n<15) |
| **hepsi** | **−0,57** | **−0,18** |

Gercekte koloni **buyudukce puruzsuzlasiyor** — ic yapisi koloniyle birlikte
olcekleniyor. Sentetikte bu iliski **yok**: jenerator koloni boyutundan bagimsiz,
**sabit uzamsal frekansta** bir doku basiyor.

| # | Karar / bulgu | Gerekce |
|---|---|---|
| 3.69 | **Doku kusuru bir "miktar" sorunu degil, bir OLCEK sorunu.** Rapor edilecek ve cozulmeye calisilacak sey "doku %X fazla" degil, "uretilen dokunun uzamsal frekansi koloni boyutuyla olceklenmiyor". | Uc olcum birlikte: (a) guidance degistirmek oranı degistirmiyor, (b) LoRA degistirmek degistirmiyor, (c) gercekte cap-doku korelasyonu −0,57 iken sentetikte −0,18. Bu, sabit bir gizil (latent) izgara uzerinde uretim yapmanin beklenen sonucu: SD'nin VAE'si 8 kat asagi ornekliyor, dolayisiyla uydurulan dokunun frekansi **goruntu pikselinde sabit**, nesnenin boyutuna gore degil. Karar 3.51 (karolama) bu sorunu **plak** olceginde cozdu; **koloni** olceginde acik kaldi. Sonuc olarak 29 px'lik bir `S.aureus` kolonisi ~3,6 gizil piksele dusuyor ve puruzsuz uretilemiyor — doku orani orada en kotu (×7,8), 155 px'lik `P.aeruginosa`'da en iyi (×3,1). |

**Bu yeniden cerceveleme, cozum adaylarini da degistiriyor.** "Daha az doku"
aramak yanlis hedefti. Denenecekler, ucuzdan pahaliya:
1. **Koloni basina olcekli uretim** — karoyu, koloninin gizil ayak izi sabit
   kalacak sekilde olcekleyip uretmek, sonra geri olceklemek.
2. **Boyuta bagli son islem** — uretilen koloni icini capiyla orantili bir
   alcak geciren suzgecten gecirmek. Ucuz, ama veri uzerinde kozmetik islem;
   yapilirsa makalede **acikca** yazilmali.
3. **Daha ince gizil izgarali taban model** (SDXL vb.) veya bir super-cozunurluk
   asamasi — pahali, butceyi yeniden acar.
4. Istemi boyut bilgisiyle kosullamak / LoRA'yi boyut-kosullu egitmek.

Once (1) olculecek: kod degisikligi kucuk, maliyeti degistirmiyor, ve hipotezi
dogrudan test ediyor.

### Dorduncu deney: tuval olcekleme — KISMEN dogrulandi

`--target-diameter 128` (koloni basina tuval olcekleme) ve kontrol olarak
`--gen-scale 0.5` (duz olcekleme) ayni 4 plakta kosuldu.

**Cap-doku sira korelasyonu, SINIF ICINDE** (asil olculecek sey):

| | B.subtilis | P.aeruginosa | S.aureus |
|---|---|---|---|
| **gercek** | **−0,22** | **−0,42** | **+0,11** |
| g=7,5 (mudahale yok) | −0,02 | +0,09 | −0,06 |
| **target-diameter 128** | **−0,07** | **−0,31** | −0,20 |
| gen-scale 0,5 (kontrol) | +0,26 | −0,22 | +0,24 |

**Doku medyani:**

| | B.subtilis | P.aeruginosa | S.aureus | ayrilabilirlik (dokusuz) |
|---|---|---|---|---|
| gercek | 9,2 | 5,2 | 11,3 | %100 |
| g=7,5 | 41,3 | 15,9 | 87,8 | %27 |
| target-diameter 128 | **29,2** | 21,2 | **52,0** | **%31** |
| gen-scale 0,5 | 29,1 | 23,6 | **10,1** | %23 |

| # | Karar / bulgu | Gerekce |
|---|---|---|
| 3.71 | **Tuval olcekleme mekanizmasi gercek, ama etki yetersiz. Varsayilan KAPALI kaliyor** (`--target-diameter 0`); secenek ve olcum kayitta duruyor, tam veride tekrar denenecek. | **Kontrol grubu isini yapti.** Mudahale sinif ici korelasyonu gercege dogru **hareket ettiriyor** (P.aeruginosa +0,09 → −0,31, gercek −0,42; B.subtilis −0,02 → −0,07). Duz olcekleme ise **ters yone** goturuyor (+0,26 ve +0,24). Yani "koloni basina olcekleme" ile "sadece daha kaba uretmek" ayni sey degil ve fark olculdu. Ama doku **buyuklugu** hala 3-5 kat fazla, `P.aeruginosa` bu mudahaleyle **kotulesiyor** (15,9 → 21,2), ve dokusuz ayrilabilirlik %27 → %31 ile gurultu icinde. Kismi bir kazanc icin varsayilani degistirmek, sonraki her olcumu once/sonra diye ikiye bolerdi. |
| 3.72 | **Karar 3.69'un cercevesi DUZELTILDI: havuzlanmis −0,57 korelasyonu buyuk olcude SINIFLAR ARASI bir etki, sinif ici degil.** | Sinif ici gercek degerler −0,22 / −0,42 / +0,11; havuzlanmis −0,57 bunlarin hepsinden guclu. Sebep: kucuk siniflar (`S.aureus`, `C.albicans`) parlak ve yuksek kontrastli, buyuk sinif (`P.aeruginosa`) soluk. Yani havuzlanmis korelasyonun bir kismi **boyut degil sinif** olcuyor. Kusur hala gercek — sinif ici sentetik degerler ≈0, gercek −0,2…−0,4 — ama iddia 3.69'da yazildigi kadar guclu degil. Bir olcumu, onu uretenin kendisi asiri yorumlayabilir; sayinin ne olctugu her zaman ayrica sorulmali. |

### 🟢 Yan bulgu: 256 tuval, 92 GPU-saat

`--gen-scale 0.5` uretimi **0,305 s/karo**'ya, yani 58.200 goruntu icin
**92 GPU-saate** dusuruyor (232 yerine). Ve `S.aureus` dokusunu 10,1 yapiyor —
gercek 11,3. Kalite acisindan **yanlis** cozum (korelasyon ters yone gidiyor,
ayrilabilirlik %23'e dusuyor), ama sunu gosteriyor: **butcede daha cok yer var.**
Kalite sorunu cozuldugunde maliyet tarafinda 232 saat bir tavan degil.

### Faz 4'un durumu — 19 Agustos sonu

Dort deney, dort saat, dort sonuc:

| deney | sonuc |
|---|---|
| guidance taramasi | ❌ sebep degil, 7,5 en iyisi |
| LoRA var/yok | ❌ sebep degil, taban model |
| cap-doku korelasyonu | ✅ kusur yeniden tanimlandi (3.69) |
| tuval olcekleme + kontrol | 🟡 mekanizma gercek, etki yetersiz (3.71) |
| kendi olcumumun elestirisi | ✅ 3.69'un cercevesi duzeltildi (3.72) |

Sirada: LoRA'yi boyut-kosullu egitmek (adapt.py'de kirpma olcegini koloni
boyutuna gore degistirmek) — ama bu **tam veri gerektiriyor**, cunku demo
pakette tur basina birkac duzine ornek var. Yani Faz 4'un bu kolu da artik
veriye bagli.

---

## Faz 4 — tam veri uzerinde pilot (26 Agustos – 1 Eylul)

Demo paketi 10 plakti. Bu bolumdeki her sayi 2987 plaklik gercek train
bolunmesinden geliyor. Demo'dan tasinan tahminlerin **ucu birden yanlis cikti**
(degme orani, sayim dagilimi, karo basina sure) — hicbiri kotu niyetli degildi,
hepsi "10 ornekten kestirilen sayi, sayi degildir"in ayri ayri kanitlari.

| # | Karar / bulgu | Gerekce |
|---|---|---|
| 3.81 | **Arka plan havuzunun tur yanliligi KABUL EDILDI, esik gevsetilmeyecek.** Maksimum sapma 0,13 (kapi 0,15); `B.subtilis` havuzda **%0**. `A2_background` ablasyonu bunun onemli olup olmadigini olcecek. | `erase_ratio <= 0,06` esigi alana bakiyor, alan boyuta, boyut sinifa. `B.subtilis` 137 px medyan capla ve plak basina ~46 koloniyle esigi hicbir zaman gecemiyor. Esigi gevsetmek havuzu dengelerdi ama silinemeyen gercek kolonileri geri getirirdi: hayalet koloni sayisi 23/35'ten 0/29'a bu esikle dusmustu. **Etiketsiz nesne, dengesiz havuzdan daha pahali bir hata.** Yanlilik olculdu, yazildi, ablasyona birakildi. |
| 3.82 | **`tiles.colony_boxes` baglam dolgusunu goruntuye KIRPIYOR.** | Plak kenarindaki bir koloni icin dolgu kutuyu goruntunun disina tasiyordu: merkez x = 72 px, 367 px'lik bir `P.aeruginosa` (r = 183), pad = 24 → x0 = −135. `place_tiles`'in acabilecegi her karo zaten goruntuye kirpili, dolayisiyla −135'te baslayan bir kutuyu hicbir karo kapsayamaz; koloni "kapsanamaz" hale gelip tum kosuyu durduruyordu. **Kod dogru davraniyordu** — karar 3.22 geregi koloniyi sessizce dusurmek yerine durdu. Goruntu disindaki pikseller yok, kapsanmalari da gerekmiyor: kutu goruntuyle kesistiriliyor. Koloninin KENDISI hala butun kapsaniyor, sadece dolgu, sadece tastigi taraftan kirpiliyor. 10 plaklik demo'da kenara bu kadar yakin koloni yoktu. |
| 3.83 | **`adapt.py crops` ilerleme ve ETA basiyor. Sessiz uzun dongu basli basina bir hatadir.** | Dongu 2987 plagi okuyup ~33.000 kirpma yaziyor ve onceki surum bitene kadar HICBIR SEY basmiyordu. Iki kez askida sanilip oldurustuldu — ve `metadata.jsonl` yalnizca en sonda yazildigi icin her seferinde diskteki her sey ise yaramaz hale geldi. Bir kullanicinin "calisiyor mu, dondu mu" sorusunu cevaplayamayan program, dogru sonucu uretse bile calismiyor demektir. |
| 3.84 | **Koloni sayisi dagilimi lognormal DEGIL. Ampirik histograma gecildi. `validate` tablosuna ORTALAMA satiri eklendi. 3.17 yerini bu karara birakti.** | Ayrintili gerekce asagida. |
| 3.85 | **Bolunmus maskeler (`_synthetic` + `_erase`) HER ZAMAN yaziliyor; `inpaint.py` onlar yoksa DURUYOR.** Tek gecisli davranis yalnizca `--allow-combined-mask` ile, bilincli olarak secilebiliyor. | Bolunmus maskeler `--split-masks` bayraginin arkasindaydi ve bayrak "for visual checking" diye tanitiliyordu. Oysa karar 3.60'in iki gecisli uretimi onlarsiz calisamiyor. `inpaint.py` bir kez uyarip birlesik maskeyle devam ediyordu: kosu basarili gorunuyor, 100 plagin tamami yaziliyor ve her birinde silme bolgesine boyanmis **etiketsiz koloni** olabiliyor — kararlar 3.25 ve 3.60'in var olma sebebi. **Ana hat icin zorunlu olan bir bayragi istege bagli diye tanitmak tuzaktir**, ve 20 dakikalik logun tepesinde kaybolan bir uyari kimseyi korumaz. Ilk pilot kosusunda tuzak islevini gordu. |

### 3.84 — sayim dagilimi: neden bu kadar onemliydi

**Belirti.** `layout.py fit` medyani birebir tutturuyordu, ortalamayi tutturmuyordu:

| | medyan | ortalama | p99 | max |
|---|---|---|---|---|
| gercek (2987 plak) | 13 | **19,47** | 78 | 225 |
| uretilen | 12–14 | **27,6** | 225 | 225 |

**Kok sebep.** Uyum su iki satirdi:

```
ln_loc = median(ln n)      <- aykiri degere karsi BILEREK dayanikli
ln_std = std(ln n)         <- dayaniksiz
```

Ustlerindeki yorum satiri tehlikeyi adiyla yaziyordu ("tek bir kalabalik plak
ortalamayi etkiler") ve ciftin yalnizca yarisini savunuyordu. Lognormalde
ortalama sigmaya **ustel** baglidir (`ortalama/medyan = exp(sigma^2/2)`), yani
biraz buyuk bir sigma medyani hic bozmadan ortalamayi cok sisirir. Gozlenen imza
tam olarak buydu.

**Ilk hipotezim yanlis cikti.** "Dayanikli bir sigma tahmincisi kullanalim"
dedim; `08_count_distribution.py` bunu **oldurdu**:

| tahminci | sigma | uretilen ortalama (gercek 19,47) |
|---|---|---|
| `std(ln n)` — eski | 1,273 | 27,05 (+%38,9) |
| `IQR(ln n)/1,349` — "dayanikli" | 1,542 | 33,55 (+%72,3) |
| gercek ortalamanin ima ettigi | 0,899 | 19,47 |

Ayni parametrenin uc tahmini bu kadar ayrisiyorsa sorun tahminci degil,
**ailenin kendisidir.** `ln(n)`'in normal-kuantil tablosu ayni seyi soyluyor:
alt kuyruk normalden **sisman** (`ln(1) = 0`'da taban var, bir suru 1-5 kolonili
plak), ust kuyruk cok **basik** (p99'da z = 1,41, normal 2,33 ister). Lognormal
iki yone de serbest uzanir; bu dagilim uzanmiyor.

*(Henuz dogrulanmamis hipotez: ust kuyrugu AGAR'in kendi tanimi kesiyor —
300+ koloni "uncountable" sayilip veri setinden cikarilmis. Yani budama
biyolojik degil idari olabilir. Makale icin ilginc, ama su an sadece hipotez.)*

**Neden ampirik histogram, neden ters-CDF degil.** `radial` ve `size` kuantil
izgarasini ara deger hesaplayarak orneklemekte, ilk refleks ayni seyi yapmakti.
Olculdu: %6,7 sapiyor (ortalama 20,77). Sebep, ara deger hesabinin **surekli**
buyuklukler icin dogru arac olmasi; sayim ise **tam sayi**. p99 = 78 ile
p100 = 225 arasinda dogrusal ara deger, gercek verinin neredeyse hic kutlesi
olmayan bir araliga sahte kutle serpiyor. Histogram medyani, ortalamayi, her
kuantili ve araligi **tanim geregi** veriyor ve ayarlanacak izgarasi yok.

**Neden ortalama, medyandan onemli.** Ikame egrisi "N gercek goruntu" ile
"N sentetik goruntu"yu karsilastiriyor. Sentetik goruntu basina %42 fazla
etiketli nesne olsaydi, sentetik kol goruntu basina daha cok denetim alirdi ve
egri **sentetigin lehine** saperdi — tezin kendi iddiasi, bozuk cetvelle
olculmus olurdu.

**Ve asil ders: kapinin kendisi denetlenmemisti.** `layout.py validate` medyani
kontrol ediyordu, ortalamayi kontrol etmiyordu. Hata veriye degil, **kontrolun
kor noktasina** yerlesmisti. Uc tasarim ilkesine dorduncusu ekleniyor:

> **Bir kapinin neyi olcmedigi, neyi olctugu kadar onemlidir.**

`inpaint` calismadan once yakalandi; kayip sifir.

### gamma artik atil — 3.19'un durumu

Sayim duzeltilince degme orani `gamma = 1` (saf rastgele yerlestirme) ile %31,5
cikti; gercek %32,7, CI90 [31,5–33,9]. Yani **bastirilacak fazlalik kalmadi.**

Eski durum iki hatanin birbirini goturmesiydi: %42 fazla koloni + gamma ile
yapay bastirma (0,481). Simdi yogunluk dogru ve degme orani *uydurulmus* degil
*kendiliginden cikan* bir sayi — bir serbest parametre azaldi. Bilimsel olarak
daha temiz, ama gizlenmemesi gereken bir degisiklik.

**Hipotez (olculmedi):** eksik 1,2 puan `MAX_OVERLAP = 0,63` kirpmasindan
geliyor olabilir. Gercek veride ortusme derinligi 1,00'e (tam ic ice) kadar
cikiyor ve ic ice koloniler de "degme" sayiliyor.

### Kalan sapma: dis halka −%23

`validate`'te tek acik kalan satir. Gercekte kolonilerin %5,33'u dis %20'lik
alanda, uretilende %4,11. Muhtemel mekanizma: kenardaki buyuk kolonilerin plak
dairesine sigmayip iceri itilmesi. Mutlak fark 1,2 puan. Staj penceresinde
duzeltilmiyor, **sinirlik olarak raporlanacak.**

---

## Olculen sayilar — 1 Eylul

Hepsi tam veri uzerinde, hicbiri tahmin.

### LoRA (`lora_100`)

```
33.041 kirpma · 1500 adim · rank 16 · lr 1e-4 · batch 1
eval loss (sabit kirpma + sabit timestep)   0,02536 -> 0,02449   (-%3,4)
  adim  500  0,02468      adim 1000  0,02458      adim 1500  0,02449
sure 0,509 GPU-saat · tepe VRAM 5,654 GB
LoRA baglanma kaniti (3.63): 256 tensor birlesti, max |delta| = 0,00464
```

Ilk 500 adim toplam iyilesmenin %78'ini yapti; egri duzlesti.
**Buradan "1500 adim yeterli" sonucunu cikardim ve bu cikarim yanlisti** —
asagiya bakiniz.

### Karo ve uretim maliyeti — eski sayinin ikisi de yanlismis

| | demo varsayimi | **olcum** |
|---|---|---|
| karo / goruntu | 19,0 | **10,06** |
| saniye / karo | ~0,34 | **0,959** (p95 0,998) |
| saniye / goruntu | — | **10,41** |
| tepe VRAM | — | **2,889 GB** |
| 58.200 goruntu | 105 GPU-sa | **156 GPU-sa** |

Karo sayisi yariya indi, karo basina sure 2,8 kat cikti; net etki **+%49**.
Ara asamada yalnizca karo sayisini duzeltip "uretim %52 ucuzladi" denmisti;
bu iddia **yanlisti** ve burada duzeltiliyor. Bir carpimin bir carpanini
olcup digerini varsayimda birakmak, hic olcmemekten daha yaniltici.

**Guncel butce:**

```
①  150 epoch (ust sinir)   284,7 + 156 = 441 GPU-sa
②  86 epoch (gercekci)     163,2 + 156 = 319 GPU-sa
kiralik 4090'da ② ≈ 106 saat · ~$29 · 4 kartta ~27 saat
```

**Firsat notu:** tepe VRAM 2,889 GB. 24 GB'lik bir kartta karolar toplu
islenebilir; su an teker teker gidiyor. Faz 6'da 156 saati ciddi dusurur.

---

## Faz 4'un kapisi — 1 Eylul, ILK GERCEK OLCUM

100 sentetik plak · 2192 olculebilir koloni · `species_check.py`

### Kapi 1: doku orani < 2 — GECILEMEDI

| tur | gercek -> sentetik | oran |
|---|---|---|
| C.albicans | 24,2 -> 41,1 | ×1,70 ✓ |
| B.subtilis | 8,1 -> 21,1 | ×2,62 ✗ |
| E.coli | 5,4 -> 15,6 | ×2,90 ✗ |
| S.aureus | 11,6 -> 39,1 | ×3,38 ✗ |
| P.aeruginosa | 5,6 -> 19,2 | ×3,41 ✗ |

**Demo'da 3–8 idi, simdi 1,7–3,4.** Gercek ilerleme, ama kapi 1/5 gecti.

### Kapi 2: dokusuz ayrilabilirlik — AGIR GECILEMEDI

```
uc eksenle (renk + kontrast + doku)   gercek 0,37  -> sentetik 0,11   %28
doku ekseni CIKARILINCA               gercek 0,69  -> sentetik 0,03   %4
```

**Asil bulgu bu ve doku oranindan daha ciddi.** Uretilen koloniler tur tur
birbirinden ayrilmiyor. Siralama dogru (Spearman: sarilik +1,00, doku +1,00,
kontrast +0,90 — "E.coli S.aureus'tan saridir" biliniyor) ama aradaki mesafe
cokmus. Model her ture asagi yukari ayni koloniyi ciziyor.

Yani: **sentetik veride sinif sinyali neredeyse yalnizca doku artefaktinda
yasiyor.** Karar 3.67'nin kestirme-ipucu korkusunun olculmus hali.

*(Demo'daki %27 rakami 10 plaktan geliyordu, guven araligi cok genisti.
"Kotulesti" DENMIYOR — o sayi zaten guvenilir degildi. Bugunku %4, 100 plaktan.)*

*(Sentetik kolonilerin %21'i (584 adet) olculemeyecek kadar kucuk cikti. Bu
sayilar buyuk turlere dogru egimli.)*

### Kendi cikarimimin duzeltilmesi

Sabah "eval loss duzlesti, 1500 adim yeter" demistim. **Eval loss yeniden kurma
hatasini olcuyor, tur ayrimini olcmuyor.** Duzlesmis bir yeniden kurma egrisi,
sinif kanalinin doydugu anlamina gelmez. Iki farkli seyi ayni sayidan okumaya
calistim.

### Test edilen hipotez (1 Eylul aksami baslatildi)

```
prompt turu adiyla soyluyor      "macro photograph of S.aureus bacterial colonies..."
adapt.py ayni sablonla egitiyor   species_prompt inpaint.py'den import ediliyor
AMA CLIP "S.aureus" ile "P.aeruginosa"yi gorsel olarak tanimiyor
-> bu ayrimi UNet LoRA'sinin sifirdan ogrenmesi gerekiyor
-> LoRA'nin toplam etkisi max |delta| = 0,0046, eval loss -%3,4  = cok hafif dokunus
```

**Deney:** tek degisken, yalnizca adim sayisi. rank/lr/kirpma havuzu/seed ayni.
1500 -> 6000 adim, her 1500 adimda kontrol noktasi (karar 3.62 tam bunun icin).
Tek kosudan bir egri cikacak:

| adim | doku orani | dokusuz ayrilabilirlik |
|---|---|---|
| 1500 | ×1,7–3,4 | %4 |
| 3000 | ? | ? |
| 4500 | ? | ? |
| 6000 | ? | ? |

**Nasil okunacak:**
- ayrilabilirlik adimla **yukseliyorsa** -> LoRA az egitilmisti; makaleye
  "sinif kanali N adim ister" diye olculmus bir cumle girer
- **duz kaliyorsa** -> sorun egitim suresi degil, kosullandirmanin kendisi;
  daha guclu bir sinirlilik argumani olur

Ikisi de kullanilabilir sonuc. Bedava dogrulama: seed ayni, bu kosunun 1500.
adimi bugunku `lora_100` ile birebir ayni cikmali.

---

## Faz 4'un kapanisi — 2 Eylul

Deney bitti. Tek degisken adim sayisiydi: rank 16, lr 1e-4, kirpma havuzu ve
seed dort kosuda da ayni. Her kontrol noktasi 100 sentetik plak uzerinde
`species_check.py` ile olculdu.

| adim | dokusuz ayrilabilirlik | doku kapisi | sarilik sirasi (Spearman) | S.aureus sariligi |
|---|---|---|---|---|
| 1500 | %4 | 1/5 | +1,00 | 87,5 ← sadakat en iyi |
| 3000 | %10 | 0/5 | +0,30 | 146,8 |
| 4500 | **%23** | 1/5 | +0,80 | 145,7 ← ayrim en iyi |
| 6000 | %16 | 1/5 | +0,30 | 113,7 |
| **kapi** | **> %27** | **5/5** | — | **gercek: 88,8** |

Doku orani tur bazinda:

| tur | 1500 | 3000 | 4500 | 6000 |
|---|---|---|---|---|
| B.subtilis | ×2,62 | ×3,73 | ×3,62 | ×5,23 |
| P.aeruginosa | ×3,41 | ×4,28 | ×3,69 | ×5,90 |
| S.aureus | ×3,38 | ×2,17 | ×2,78 | ×3,56 |
| E.coli | ×2,90 | ×2,71 | **×1,43** | ×3,43 |
| C.albicans | **×1,70** | ×2,23 | ×2,20 | **×1,62** |

| # | Karar / bulgu | Gerekce |
|---|---|---|
| 3.86 | **Faz 4'un doku kapisi hicbir egitim uzunlugunda GECILMEDI. "LoRA az egitilmisti" hipotezi olcumle elendi. Faz 5'e kapi gecilmeden geciliyor; sinir makalede olculmus sinirlilik olarak raporlanacak.** Ayrica: LoRA egitim suresi bir olcek degil, bir **odunlesme parametresidir** — 1500 → 6000 adim araliginda sinif ayrimi yukselirken dagilim sadakati bozuluyor, ikisini birden veren adim sayisi yok. Ve: `species_check` kapi sayisina guven araligi vermedigi icin "4500 tepe, 6000 dusus" **iddia edilemez.** | Asagida uc baslikta. |

### Neden kapi acilmadi

Doku kapisi en iyi 1/5 gecti (1500, 4500, 6000) ve 3000'de 0/5'e dustu. Dort
noktanin hicbirinde 5/5 yok. Dokusuz ayrilabilirlik en iyi %23 (4500), kapi
%27. **Daha uzun egitim dokuyu duzeltmiyor, kotulestiriyor:** doku orani
tablosunda 6000 adim bes turun ucunde dort noktanin en kotusu.

1 Eylul aksami kurulan deneyin iki okumasi vardi. "Ayrilabilirlik adimla
yukseliyorsa LoRA az egitilmisti" kolu **kismen** dogru cikti (1500 → 4500'de
%4 → %23 gercek bir yukselis), ama kapiya ulasmadi ve bedeli sadakat oldu.
"Sorun kosullandirmanin kendisi" kolu asil gecerli okuma: CLIP tur adlarini
gorsel olarak tanimadigi icin ayrimi UNet LoRA'sinin sifirdan ogrenmesi
gerekiyor ve LoRA'nin dokunusu bunu tasiyacak kadar guclu degil. Cozum adim
sayisinda degil, kosullandirma mekanizmasinda (textual inversion, boyut-kosullu
LoRA, farkli taban model — hicbiri denenmedi, staj penceresine sigmiyor).

### Odunlesme — makaleye bulgu olarak girer

Bu deney kapiyi acmadi ama beklenmeyen bir sey olctu. LoRA sinif ayrimini
siniflar arasi mesafeyi **abartarak** aciyor: 4500 adimda dokusuz ayrilabilirlik
%4'ten %23'e cikarken S.aureus sariligi 87,5'ten 145,7'ye firliyor — gercek
degeri 88,8, yani %64 fazla. 1500 adim gercege neredeyse birebir oturuyor
(87,5 ≈ 88,8) ama siniflari ayirmiyor.

Bir uctan digerine gecerken kazanilan sey ile verilen sey **ayni eksende degil**:
ayrim kazaniliyor, sadakat veriliyor. Ikame egrisinin anlami "sentetik veri
gercege ne kadar benziyor" oldugu icin bu, Faz 6'da hangi kontrol noktasiyla
uretim yapilacagini bir **karara** donusturuyor — varsayilan degil. Karar Faz
5'te ayrica verilecek ve gerekcesiyle buraya yazilacak.

### Kapinin kendisinin kor noktasi — ilke 4, ucuncu kez

`species_check` **kapi sayisi (dokusuz ayrilabilirlik) icin guven araligi
hesaplamiyor.** Uc eksenli sayinin araligi 3 kat genis; kapi sayisi da benzer
gurultude olmali. Bu yuzden yukaridaki dort noktadan cikarilabilecek sey
sinirli:

- **soylenebilir:** 1500 → 4500 arasinda gercek bir yukselis var; dort noktanin
  hicbiri %27'ye ulasmiyor
- **soylenemez:** "4500 tepe noktasi", "6000'de dusus basliyor" — iki komsu
  nokta arasindaki %23 → %16 farkinin gurultuden buyuk oldugu gosterilmedi

Bir kapinin neyi olcmedigi, neyi olctugu kadar onemlidir (ilke 4). `validate`
ortalamayi olcmuyordu, %42'lik sayim hatasi orada gizlenmisti. Bu sefer kor
nokta hata payinda ve **hala acik** — Faz 5 acik maddeler listesinde 5. sirada.

### Faz 4'te gecen kapilar

```
✅ hayalet koloni      0/1341 silme bolgesi, 100 plak (09_ghost_check.py)
✅ layout sadakati     ortalama · medyan · radyal · degme, hepsi tutuyor
✅ etiket dogrulugu    kutular difuzyondan once doguyor, difuzyon onlari degistirmiyor
```

35 "aday" cikti, 9'una gozle bakildi: hepsi klasik inpainting'in yildiz
bulasmasi, koloni degil. Karar 3.65 bunu tahmin etmisti.

**Faz 4 kapandi. Faz 5 (protokolu dondur) basliyor.**

---

## Faz 5 — protokolu dondur (2 Eylul)

### `--resume` testi — Faz 6'nin onkosulu

Hic denenmemisti ve butun kiralik-GPU plani buna dayaniyordu: spot/kesintili
kartta kosu yarida kalirsa devam edilemezse 319 GPU-saatlik plan cope gider.

**Test tasarimi.** Ayni ayarla iki kosu, `--level 10` (299 goruntu, 6 epoch):

```
resume_ref    6 epoch kesintisiz
resume_test   3 epoch sonra kill -9 (Ctrl+C DEGIL — spot kesintisi anidir), sonra --resume
```

**Sonuc: GECTI.** Kosu dogru yerden devam etti, `RESUME` satiri dogru dosyayi
yukledi, egitim 4/6'dan 6/6'ya tamamlandi.

| epoch | resume_ref | resume_test | fark |
|---|---|---|---|
| 1 | 0,08301 | 0,08301 | — birebir |
| 2 | 0,25542 | 0,25542 | — birebir |
| 3 | 0,31424 | 0,31424 | — birebir |
| 4 | 0,31356 | 0,31308 | −0,00048 |
| 5 | 0,36054 | 0,35585 | −0,00469 |
| 6 | **0,37268** | **0,37180** | **−0,00088** |

Kesme oncesi uc epoch **birebir ayni** — `deterministic: true` calisiyor ve
test kurulumu saglam. Kesme sonrasi sapiyor: `--resume` RNG durumunu tam geri
yuklemiyor. Buyukluk son epoch'ta binde 0,9.

⚠ **Bu bir sinirlilik ve boyle raporlanacak:** ayni tohumla resume edilen bir
kosu, kesintisiz kosuyla **birebir ayni degildir.** `significant_diff_threshold`
henuz olculmedigi icin (base.yaml'da `null`, G100 × 3 tohum bekliyor) bu farkin
ihmal edilebilir olduguna **karar verilemez.** Faz 6'da kesilen kosular
`resumed: true` ile isaretli kalacak ve tohumlar arasi varyans olculdukten sonra
bu fark onunla karsilastirilacak.

### Testin ortaya cikardigi uc muhasebe hatasi

Kosu devam ediyordu ama **defter tutmuyordu.**

```
① run_metrics.json UZERINE yaziliyordu
   olculdu: gercekte ~4,3 dk suren kosu   total_min 2,32 diye yazildi
                                          epochs_actual 4 / 6
                                          peak_vram 4,047 — ilk parcanin 4,078'i kayboldu
② results.csv'nin `time` sutunu da sifirlaniyor   108,7 -> 46,7
③ epochs_actual her zaman +1
   on_fit_epoch_end final validation'da da tetikleniyor (6 planlandi, 7 sayildi)
```

①'in sebebi yapisal: `run_metrics.json` **yalnizca kosu bitince** yaziliyordu.
Kesilen segment hicbir zaman yazamiyor, `--resume` segmenti ise dosyayi kendi
kismi sayilariyla eziyor. Makalenin 6. bolumu (hesaplama maliyeti) dogrudan bu
dosyadan yazilacak; 61 kosuluk gridde spot kesintisi beklenen bir olay.

③ tek basina zararsizdi (86 epoch'ta %1,2), ama ① duzeltilip segmentler
**toplanmaya** baslayinca hata segment sayisi kadar cogaliyor: 6 epoch'luk bir
kosu 4 + 4 = 8 diye sayilirdi. Yani ①'i duzeltmek ③'u zorunlu kildi.

| # | Karar / bulgu | Gerekce |
|---|---|---|
| 3.87 | **`--resume` calisiyor, Faz 6'nin onkosulu gecildi. Ama olcum muhasebesi bozuktu: `train.py`'ye append-only `runs/<name>/segments.jsonl` eklendi — her epoch sonunda yazilip `fsync` ediliyor, `kill -9`'dan sagligiyla cikiyor. `run_metrics.json`'daki `total_sec` / `gpu_hours` / `peak_vram_gb` artik BUTUN segmentlerin birikimi; `epochs_actual` `results.csv`'den okunuyor.** Eski callback sayimi `epochs_callback_raw` olarak yaninda birakildi (ilke 3). | Asagida. |

**Neden `segments.jsonl`, neden tek bir JSON degil.** Kesintinin tanimi, programin
kapanis kodunun **calismamasi.** Kosu sonunda yazilan hicbir sey bir kesintiden
sagligiyla cikamaz; tek segmentlik bir dosyayi guncellemek de yarim yazilmis bir
dosya birakabilir. Append-only + her epoch `fsync`: en kotu durumda son satir
yarim kalir, `read_segments()` onu atlar ve onceki epoch'larin kaydi durur.

**Dogrulama** (ayni kesinti senaryosu, `resume_fix`):

```
                     eski kod olsaydi   yeni kod   gercek
total_min                    3,1           5,6      5,6
epochs_actual                  4             6        6
peak_vram_gb               4,047         4,049    4,049
segments                     yok       2 satir        —
accounting_complete            —          True        —
epochs_callback_raw            —             7        —   (3 + 4: kesilen segment
                                                           final val'e ulasmadigi
                                                           icin +1 almiyor)
```

**Yeni alanlar:** `segments` (her segmentin son hali), `resume_count`,
`segment_sec` ve `segment_peak_vram_gb` (yalnizca bu segment — denetlenebilirlik
icin birikimlinin yaninda duruyor), `accounting_complete`.

**`accounting_complete = false` ne demek.** `--resume` verilmis ama
`segments.jsonl` yok — yani kesinti bu duzeltmeden ONCE olmus. O zaman kesilen
segmentin suresi **hicbir yerde yok** ve geri getirilemez. Program bunu sessizce
gecmiyor: sari uyari basiyor ve dosyaya `false` yaziyor. Sessiz hata, gurultulu
hatadan kotudur (ilke 2).

**Kalan iki bilinen sinir:**

1. Kesilen segmentin **yarim kalan epoch'u sayilmiyor.** Segment kaydi son
   tamamlanan epoch sonunda yazildigi icin, kesintide bosa giden kismi epoch
   butceye girmiyor. Bilincli: o hesaplama bir sonuc uretmedi. Alt-sayim, epoch
   basina sureden kucuk.
2. `segments.jsonl`'deki `elapsed_sec` `t0`'dan (train cagrisindan) sayiliyor;
   `results.csv`'nin `time` sutunu Ultralytics'in kendi sayaci. Ayni sey degiller
   (olculdu: 151,2 vs 108,7) — bizimki veri tarama ve isinma dahil **duvar
   saati**, ve butce icin dogru olan bu.

### Protokolun dondurulmasi — dort ayri bulgu

Dondurma islemi, dondurulacak seylerin **gercekten var olup olmadigini** sorunca
dordu de kendini gosterdi. Hicbiri aranarak bulunmadi; hepsi "su anda kilitli
olan sey nerede yaziyor" sorusunun cevabi arandiginda ortaya cikti.

**① `requirements.lock` uretildi — ve ilk halinde kullanisiz olacakti.**

```
ortamda kurulu    torch 2.13.0+cu130
pip freeze yazdi  torch==2.13.0            <- local version etiketi dustu
```

Kiralik kartta duz `pip install -r requirements.lock` **baska bir CUDA wheel'i**
(hatta CPU surumu) kurar. Kilit dosyasi var, kilitlemiyor. Kurulum artik iki
adimli ve talimat dosyanin kendi basligina yazildi; `--index-url` ile torch once
kuruluyor, dogrulama komutu da orada. Basliga ayrica python surumu, venv yolu,
CUDA, GPU adi, platform ve commit yazildi — `pip freeze` bunlarin hicbirini
soylemiyor ve altisi da sonucu degistirebilir.

**② `conf_thr = 0,35` "protokole kilitli" deniyordu ama hicbir yerde kilitli
degildi.**

`configs/base.yaml`'daki alan `null`'di ve `evaluate.py` onu **hic okumuyordu**;
esik yalnizca komut satirindan (`--conf-thr`) geliyordu. Config'in kendi yorum
satiri niyeti dogru yaziyordu — *"chosen on VAL, **written here**, and applied to
the test that way"* — ama yazma adimi atlanmisti.

Sizinti kilidi calisiyordu (`--split test` esiksiz DURUYOR, `--split val` ariyor)
ve ona dokunulmadi: esik hala elle yaziliyor, cunku kurali cagri yerinde gorunur
tutan sey bu. Eklenen sey, **yanlis yazilan esigin sessiz gecememesi.** 61
kosuluk gridde tek bir `0.3`, o kolun sayim metriklerini kimseye haber vermeden
kaydirir.

**③ 97 otomatik kontrolun 9'u kirikti — ve bunu kimse gormuyordu.**

`pytest` ortamda **kurulu degildi.** Kapilar "kosuyor" degil, "kosturulamiyor"
durumdaydi ve `requirements.lock`'un ilk hali de icermiyordu. Kurulup
kosturulunca `test_generate.py`'nin 9 testi `layout.py`'nin 3.84 korumasina
carpti: testlerin `sample_params()` yardimcisi hala `family: "lognormal"`
uretiyordu. Yani **3.84 uygulandi, testleri guncellenmedi.**

Daha kotusu: 3.84'un korumasinin kendisi **hic test edilmiyordu.** Lognormal bir
parametre dosyasi programi durduruyor mu — bunu dogrulayan bir kontrol yoktu.
Eklendi (`test_legacy_count_rejected`), ayrica `test_count_limits` yeniden
yazildi ve artik ampirik histogramin asil ozelligini olcuyor: **histogramda
bulunmayan bir sayi uretilemez.** Ara deger hesabi olsaydi 60 ile 120 arasina
gercek verinin neredeyse hic kutlesi olmayan sahte sayilar serpilirdi — 3.84'un
ters-CDF'yi reddetme sebebi tam olarak buydu ve simdi bir kontrol olarak duruyor.

Kontroller: **30/30 gecti** (29'du; koruma testi eklendi).

**④ Faz 4'un ciktilarinin cogu git'te degildi.** `layout_10/25/50.json`,
`species_check_ck3000/4500/6000.json`, `09_ghost_check.py`, LoRA
`adapter_config.json` ve `adapt_metrics.json` dosyalari takipsizdi. 3.86'daki
dort noktali egrinin **ham cikti dosyalari** repoda yoktu; `faz5-protokol`
etiketi bunlari kapsamayacakti. 19 dosya, 11.526 satir takibe alindi
(`*.safetensors` gitignore'da kaldi, symlink'ler hangi kontrol noktasi
oldugunu belgeliyor).

| # | Karar / bulgu | Gerekce |
|---|---|---|
| 3.88 | **Protokol donduruldu: `requirements.lock` (torch `+cu130` tuzagi basliga yazili), `conf_thr = 0,35` `base.yaml`'a yazildi ve `evaluate.py` uyusmazlikta DURUYOR (`--override-conf-thr` bilincli sapma icin), 3.84 sonrasi kirik kalan 9 test duzeltildi + 3.84'un korumasina test yazildi, Faz 4 ciktilari takibe alindi. Etiket: `faz5-protokol`.** | Yukarida dort baslikta. Ortak nokta: dordu de bir kuralin **belgede** var olup **kodda** olmamasiydi. |

Ilke 1 dorduncu kez is gordu: *metodolojik kural bir yorum degil, calisma zamani
kontroludur.* Bu sefer uc ayri yerde ayni sekilde ihlal edilmisti — kilitli bir
esik null olarak, dondurulmus bir ortam eksik etiketle, uygulanmis bir karar
guncellenmemis testlerle.

**Hala acik:** `significant_diff_threshold` `base.yaml`'da `null` ve oyle
kaliyor — **olculmedi.** G100 × 3 tohum Faz 6'da kosunca doldurulacak. Simdilik
bos olmasi dogru; tahmin yazmak, olculmemis bir sayiyi protokole kilitlemek
olurdu.

### Uretim LoRA'si karari — ERTELENDI, bilincli olarak

Faz 4 bir odunlesme olctu (3.86) ve Faz 5'in gorevi bunu bir karara baglamakti:

```
lora_100         1500 adim   sadakat en iyi   S.aureus sariligi  87,5  (gercek 88,8)
                             ayrim   en kotu  dokusuz ayrilabilirlik %4
lora_100_ck4500  4500 adim   ayrim   en iyi   %23
                             sadakat bozuk    S.aureus 145,7  = %64 fazla
```

**Iki taraf da savunulabilir ve hicbiri olculmemis.**

*1500 lehine:* ikame egrisinin anlami "sentetik veri gercege ne kadar benziyor";
sadakat kaybi dogrudan sonucu bozar.

*4500 lehine:* sinif sinyali daha guclu, detektor turleri ayirt edebilir.

**Bir gozlem — kanit degil, argumanin agirligini degistiriyor.** 4500'un ayrimi
*dogru yerde degil.* Siralama korunuyor (sarilik Spearman +0,80), ama mesafeler
abartilmis: S.aureus'un sariligi gercekte 88,8 iken 145,7. Detektor bundan
"E.coli cok saridir" ogrenir ve gercek test setinde E.coli o kadar sari degildir.
1500'de siniflar ust uste biniyor — ayrim ogretmiyor, ama **yanlis bir sey de
ogretmiyor.** Yine de bu bir mekanizma tahmini; hangisinin ikame egrisini daha
iyi verdigi **olculmedi.**

| # | Karar / bulgu | Gerekce |
|---|---|---|
| 3.89 | **Uretim LoRA'si secimi ERTELENDI. Faz 6, sentetik veri gerektirmeyen 26 kosuyla baslatiliyor; secim, o kosular donerken olcumle verilecek.** | Iki argumanin da dayanagi var, ikisinin de olcumu yok. "Olculmemis sayi, sayi degildir" ilkesi bir tahmini protokole kilitlemeyi yasaklar. Erteleme bos bekleme degil: 26 kosu LoRA'yi hic beklemiyor. |

**26 kosunun sayimi** (`scripts/budget.py` icindeki `ARMS`, sentetik payi 0,00
olan kollar):

```
main_grid   G100 · G50 · G25 · G10        4 kol × 5 tohum = 20
classic     B_G25 · C_G25                 2 kol × 3 tohum =  6
                                                       toplam 26
─────────────────────────────────────────────────────────────
bekleyenler G50+S · G25+S · G10+S · S100  4 × 5 = 20
            G25+S_0,5x · G25+S_2x         2 × 3 =  6
Faz 6 toplam 52 · ablasyon (Faz 7) 9 → 61
```

**Karar nasil verilecek.** G kollari kosarken tek tohumlu sinirli bir
karsilastirma: ayni kol (S100 ya da G10+S) hem 1500 hem 4500 ile uretilip
egitilir, mAP karsilastirilir. Boylece secim bir tercih degil bir olcum olur ve
makaleye "hangi kontrol noktasi secildi ve neden" diye yazilabilir. Bu
karsilastirma da kendi basina bir bulgudur: **LoRA egitim uzunlugunun ikame
egrisine etkisi** literaturde hazir cevabi olan bir soru degil.

⚠ Erteleme, kararin kaybolmasi demek degil. Sentetik kollar baslamadan **once**
verilmek zorunda; G kollarinin bitisi bu kararin son teslim tarihidir.

---

## Kapiya kendi hata payi verildi — ve kapi hukmu kaldiramadi (3 Eylul)

Faz 5'in 5. acik maddesi: `species_check` kapi sayisi icin guven araligi
hesaplamiyordu. Kapatildi, ve sonucu 3.86'nin iki cumlesini geri aliyor.

### Once: altyapi zaten vardi, kapiya baglanmamisti

`plate_bootstrap` dosyada duruyordu ve **uc eksenli** ayrilabilirlik icin
cagriliyordu. Kapi ise **dokusuz** sayinin orani uzerinde tanimli
(`> %27`), ve o sayi bootstrap'lenmiyordu. Yani hata payi hesabi yazilmis,
karar veren sayiya uygulanmamisti.

Ayrica orana ait bir aralik hic yoktu. Iki bagimsiz araligin uc noktalarini
bolmek yanlis olurdu: en kotu gercek cekilisi en iyi sentetik cekilisiyle
eslestirir, ki yeniden orneklemenin hic uretmedigi bir kombinasyondur. Oran
icin **ayni iterasyon icinde iki tarafi birden** yeniden ornekleyen
`retention_bootstrap` yazildi.

### Ilk olcum: dordu de BELIRSIZ

```
nokta   tahmin   CI90        hukum
1500      %4     %3 – %29    belirsiz
3000     %10     %4 – %30    belirsiz
4500     %23    %10 – %91    belirsiz
6000     %16     %5 – %32    belirsiz
kapi     %27
```

Dordunun de araligi kapiyi kesiyordu. Yani **"doku kapisi gecilemedi" hukmu
bile desteklenmiyordu.**

### Aralklarin bir kismi artefaktmis — bootstrap'in kendi kusuru

Ust kuyruklar anormal genisti (4500'de %91). Sebep bulundu:

```
separability()  =  EN YAKIN CIFTIN mesafesi
usable          =  MIN_N (15) esigini gecen siniflar
sentetik tarafta B.subtilis n = 50     (esigin sadece 3,3 kati)
gercek  tarafta B.subtilis n = 2993    (orada sorun yok)
```

Plaka bazli yeniden orneklemede B.subtilis'li plakalar az secildiginde n
15'in altina dusuyor, sinif **hesaptan cikiyor**, en yakin cift degisiyor ve
deger yukari zipliyor. Yapay veriyle olculdu: uc sinifla 2,60 — bir sinif
dusunce **4,94** (+%90).

Yani aralik iki sey iceriyordu: gercek orneklem belirsizligi **ve** olculen
niceligin taniminin degismesi. Gercek/sentetik aralik asimetrisi (0,63–0,74'e
karsi 0,05–0,51) tam olarak bunun imzasiydi.

**Duzeltme:** hangi siniflar uzerinden olculdugu bir **analiz kararidir**,
orneklemim rastlantisi degil. Orijinal veride bir kez belirlenir, her yeniden
ornekleme onu miras alir. Sabit sinifin biri 2 gozlemin altina duserse o
iterasyon dusurulur ve **kac iterasyonun dustugu basilir** — sessiz kalmaz.

### Duzeltilmis olcum

```
nokta   tahmin   eski CI      yeni CI      hukum
1500      %4    %3 – %29     %3 – %25     GECEMEDI (tum aralik altta)
3000     %10    %4 – %30     %4 – %29     belirsiz
4500     %23   %10 – %91     %7 – %77     belirsiz
6000     %16    %5 – %32     %5 – %32     belirsiz
kapi     %27
```

| # | Karar / bulgu | Gerekce |
|---|---|---|
| 3.90 | **Kapi sayisina guven araligi eklendi (`retention_bootstrap`), bootstrap'te sinif kumesi sabitlendi, kapi esigi `GATE_RETENTION = 0,27` olarak koda yazildi ve hukum aralik uzerinden veriliyor (`gate_verdict`: GECTI / GECEMEDI / BELIRSIZ). Sonuc: 3.86'nin iki cumlesi GERI CEKILIYOR.** | Asagida. |

### 3.86'dan geri cekilen iki cumle

**① "Doku kapisi hicbir uzunlukta gecilmedi."** Yalnizca **1500** icin kesin.
Diger uc noktada aralik kapiyi kesiyor; nokta tahminleri kapinin altinda ama
100 sentetik plakla hukum verilemiyor.

**② "1500 → 4500 arasinda gercek bir yukselis var."** Iddia edilemez:
%3–25 ile %7–77 fazlasiyla cakisiyor. 3.86 bu cumleyi "soylenebilecek tek sey"
diye yazmisti; kapiya hata payi verilince o da dustu.

**Yerine gecen, savunulabilir ifade:**

> Dort kontrol noktasinin nokta tahmini de kapinin altindadir (%4 · %10 · %23 ·
> %16, kapi %27). 100 sentetik plak uzerinde hesaplanan %90 guven araliklari
> yalnizca 1500 adim icin kapinin tamamen altinda kalir; diger uc noktada aralik
> kapiyi kestigi icin gecilip gecilmedigi bu orneklem boyutunda ayirt
> edilememektedir.

### Kalici bir sinir: 4500 ayirt edilemez

4500'un nokta tahmini %23, kapi %27 — arada 4 puan var. Aralik daraltmak plak
sayisiyla `1/√n` gider; %7–77'yi %27'nin altina indirmek pratikte ulasilamaz bir
plak sayisi ister. **3000 ve 6000 icin ust sinirlar (%29 ve %32) kapiya cok
yakin; birkac yuz plakla bu iki nokta muhtemelen kesinlesir.** 4500 kesinlesmez,
cunku gercek degeri kapinin kendisine cok yakin.

Bu, olcumun yetersizligi degil, **kapinin ayirt gucunun sinirinin olculmesidir**
ve makalede boyle raporlanacak.

### Ne degismedi

Doku artefakti kaniti daha da guclendi: uc eksenli oran **%92**, dokusuz oran
**%15** ve iki aralik hic ortusmuyor. Siniflar, uretecin yanlis urettigi bir
eksende ayrisiyor. 3.67'nin kestirme-ipucu korkusunun en net olculmus hali.

### Ilke 4, ucuncu kez

`validate` ortalamayi olcmuyordu (%42'lik sayim hatasi orada gizlendi).
`species_check` kapi sayisina hata payi vermiyordu (bu karar). Ve bu sefer bir
adim daha derine indi: **hata payinin kendisi de yanlis olcuyordu**, cunku
bootstrap olculen niceligin tanimini iterasyondan iterasyona degistiriyordu.
Bir kapinin neyi olcmedigi, neyi olctugu kadar onemlidir — ve olcumun nasil
olculdugu de.

---

## FAZ 5 KAPANDI — 4 Eylul, stajin son gunu

Karar 3.2 seviye basina ayri LoRA sart kosuyordu (sizinti kilidi) ve Faz 5'e
girerken yalnizca seviye 100 hazirdi. Uc seviye tamamlandi:

```
seviye   kirpma   adim   eval loss           dusus    GPU-sa
  10      3.361   1500   0,01875 -> 0,01798  -%4,1    0,496
  25      8.249   1500   0,02588 -> 0,02489  -%3,8    0,501
  50     16.322   1500   0,01959 -> 0,01880  -%4,0    0,497
 100     33.041   1500   0,02536 -> 0,02449  -%3,4    0,509
                                             toplam   2,003
```

**Beklenmeyen bir dogrulama.** Kirpma havuzu 3.361'den 33.041'e **on kat**
degisiyor, uyarlamanin buyuklugu degismiyor: dort seviyede de eval loss dususu
%3,4–4,1 bandinda. Yani karar 3.2'nin dayattigi seviye basina egitim, seviyeler
arasinda **tutarli** bir uyarlama uretiyor; ikame egrisinin sentetik kolunda
seviyeler arasi fark, LoRA'nin farkli olcude ogrenmesinden gelmeyecek. Bu
istenmemis ama degerli bir kontrol — Faz 3'teki "ic ice alt kumeleme calisiyor"
dogrulamasinin LoRA tarafindaki karsiligi.

Baslangic eval loss degerleri seviyeler arasinda degisiyor (0,0188–0,0259); bu
kirpma havuzlarinin zorlugundaki dogal farktir, dususun tutarliligini
etkilemiyor.

| # | Karar / bulgu | Gerekce |
|---|---|---|
| 3.91 | **Faz 5 kapandi.** Protokol donduruldu ve etiketlendi (3.88), `--resume` test edilip gecti (3.87), kapiya kendi hata payi verildi (3.90), uretim LoRA'si karari olcume baglanarak ertelendi (3.89), ve seviye 10/25/50 icin `bg_pool` + `crops` + `lora` uretildi. **Faz 6 baslatilabilir durumda.** | Faz 5'in tanimi "degisebilecek her seyi kilitle"ydi. Kilitlendi — ve kilitleme sirasinda dort ayri yerde kuralin belgede olup kodda olmadigi gorulup duzeltildi. |

### Faz 5'in bilancosu

```
girerken                          cikarken
─────────────────────────────────────────────────────────────────
--resume hic denenmemis           test edildi, GECTI + segment muhasebesi
requirements.lock yok             var, torch +cu130 tuzagi basliginda
conf_thr "kilitli" ama null       base.yaml'da 0,35 + uyusmazlikta DURUYOR
97 kontrol kosturulamiyor         30/30 geciyor (pytest lock'ta)
Faz 4 ciktilari git'te degil      19 dosya, 11.526 satir takipte
kapi sayisinda hata payi yok      CI var; 3.86'nin iki cumlesi geri cekildi
1 seviyede LoRA                   4 seviyede LoRA
```

**Faz 5'te acilan ve kapatilamayan tek sey:** uretim LoRA'si secimi (3.89).
Bilincli, olcume bagli, ve Faz 6'nin ilk 26 kosusunu engellemiyor.

### Devredilen acik maddeler (Faz 5'i bloke etmiyor)

```
hocaya "ELR nedir" sorusu          outline'da gecen terim, EBPG olarak uygulandi
G100 detektoruyle hayalet sayimi   Faz 6'nin detektorunu bekliyor
KOD_HARITASI.md'nin 25 maddesi     bayat, koda bakilarak dogrulanmali
sentetik plak sayisini artirmak    3000/6000 noktalarinin hukmunu kesinlestirir (3.90)
```

Staj penceresi burada kapaniyor. Calisma makale olarak devam ediyor; Faz 6 ve 7
staj takvimine gore degil, dergi takvimine gore planlanacak.
