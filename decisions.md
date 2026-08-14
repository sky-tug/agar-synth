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
| 3.5 | **Yerlesim ve boyut dagilimlari da seviye basina fit edilir.** `analyze_manifest.py` cikti tablosu bolme-farkindali calisacak; `yerlesim.py` bir `--seviye` argumani almadan calismayacak (varsayilan yok, hata verir). | En sinsi kanal bu. "Koloniler plakta soyle dagiliyor, boyut medyani su" bilgisi 8000 goruntuden cikarilirsa, 800 goruntuluk seviyeye ait olmayan bir on bilgi sentetige gomulur. Varsayilan deger koymamak, bunun insan hafizasina birakilmamasini sagliyor (karar 2.4 ile ayni mantik). |
| 3.6 | **S100 ifadesi sabitlendi.** Makalede "gercek veri kullanilmadi" **yazilmayacak**. Kullanilacak ifade: *"S100 kolunda detektor yalnizca sentetik goruntulerle egitilmistir; gercek veri bu kola yalnizca uretici modelin uyarlanmasi (LoRA), arka plan havuzu ve yerlesim istatistikleri uzerinden **dolayli** olarak girmistir."* | Bu cumle makalenin durustluk cipasi. Simdi yazildi ki Faz 8'de unutulmasin. |
| 3.7 | **Isimlendirme kurali — sizintiyi yapisal olarak imkansiz kilar.** Her sentetik urun seviyesini adinda tasir: `data/synthetic/s25/`, `lora_25/`, `bg_pool_25.txt`, `yerlesim_25.json`. Farkli sayi tasiyan iki yol ayni uretim komutunda bulusursa `generate` betigi **durur**. | Insan dikkatine degil, dosya sistemine yaslanan bir kontrol. Uc hafta sonra yorgunken yanlis LoRA'yi secmek en olasi hata bicimi. |
| 3.8 | **Ablasyon A1 ("LoRA yok") tanimi:** G25 seviyesinde, uyarlanmamis base difuzyon modeliyle uretim. Arka plan ve yerlesim yine `train_25`'ten gelir — yalnizca LoRA kanali kapatilir. | Ablasyonun tek degiskenli kalmasi icin. Uc kanali birden kapatirsak A1 ile A2 ayrilamaz hale gelir. |
| 3.9 | **Butce sonucu:** LoRA egitimi 1 degil **4 kalem**. Faz 3 icin ongorulen ~95 GPU-saat bu varsayimla yeniden hesaplanacak; `scripts/butce.py`'ye `lora_seviye_sayisi` parametresi eklenecek. | ~470–720 GPU-saatlik toplam tahmin bunu icermiyordu. BİDB talebinde verilecek sayi bu duzeltmeden sonra kesinlesir. |

### Yerlesim modeli — 10 demo plak / 387 koloni uzerinden (14 Agustos)

Ayrinti ve sekil: `src/generate/YERLESIM_BULGULARI.md`. Betikler `src/generate/kesif/`.
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
| 3.16 | **Plak kompozisyonu: bir baskin tur + az sayida ikincil tur.** Uretici tek-tur varsaymaz; kombinasyon ve baskin/ikincil orani `tur_kombinasyonlari.csv` dagilimindan orneklenir. | Demo pakette 10 plagin 4'u karisik: `P.aeruginosa 15 + S.aureus 3`, `P.aeruginosa 11 + S.aureus 1`, `E.coli 31 + S.aureus 6`, `S.aureus 102 + E.coli 23`. Ikincil tur cogunlukla S.aureus. |
| 3.17 | **Koloni sayisi lognormal ailesinden.** Demo: `ln(n)` ort 3.42, std 0.69 → medyan 30, p10 13, p90 74. Parametreler tam veriden gelecek. | n = 12…125, sag kuyruklu. Aile secimi 10 plakla yapilabilir, parametre kestirimi yapilamaz. |

### 🔴 Ablasyon A3 yeniden tanimlandi (14 Agustos)

| # | Karar | Gerekce |
|---|---|---|
| 3.18 | ~~A3 = "rastgele yerlesim"~~ **A3 = "naif rastgele yerlestirme"**: plak cemberi yok sayilir (tum 2048x2048 kareye duzgun), boyut sinifa kosullanmadan tek havuzdan cekilir, degme/ortusme kisiti hic uygulanmaz. | **Bulgu: gercek yerlesim ZATEN rastgele.** Clark–Evans orani 10 plagin 9'unda 0.90–1.10 bandinda — kumelenme de yok, duzenli oruntu de yok. Eski tanimiyla A3 ana hattin kopyasi olurdu ve **3 kosu bos giderdi**; ablasyon "fark yok" derdi, bu da yerlesim modelinin gereksiz oldugu anlamina gelmezdi, sadece deneyin bos oldugu anlamina gelirdi. Yeni tanim yerlesim modelinin uc bilesenini birden kapatiyor: tek degiskenli degil, ama **"yerlesim modeli gerekli mi"** sorusunu net yanitliyor ve makalede savunulabilir bir cumle uretiyor: *"maske kontrollu yerlesim, naif rastgele yerlestirmeye kiyasla X puan getiriyor."* Hocanin "uc farkli senaryo secip kullanabilirsin" esnekligi (karar H.3) buna izin veriyor. |

### `yerlesim.py` yazildi (14 Agustos) — kod kararlari

| # | Karar | Gerekce |
|---|---|---|
| 3.19 | **Kisa menzilli itme: Strauss etkilesim parametresi `gamma`.** Bir aday konum baska bir koloniye deger halde ise `gamma` olasilikla kabul edilir, `1-gamma` ile reddedilip yeniden denenir. `gamma` elle secilmez — `uydur` icinde ikili aramayla gercek degme oranina **otomatik kalibre edilir**. | Saf rastgele yerlestirme (gamma=1) degme oranini %52 cikardi, gercek %39. Yani gercek yerlesimde **kisa menzilde zayif bir itme var** — Clark–Evans bunu goremiyor cunku ortalama-tabanli bir olcut; kisa menzil yapisina duyarsiz. Strauss sureci (Strauss 1975) bunun standart, alintilanabilir modeli. Demo veride kalibrasyon `gamma=0.317` verdi. |
| 3.20 | **Kalibrasyonda ortak rastgele sayilar.** Tarifler (sayim / sinif / cap) bir kez uretilir, her `gamma` denemesinde aynen kullanilir; degisen tek sey yerlestirmedir. | Ilk surumde gamma reddi rastgele akisi kaydiriyordu; ayni gamma farkli tarifler uretiyor, ikili arama gurultuye oturuyordu. Fonksiyon `tarif()` ve `bir_plak()` olarak ayrildi. |
| 3.21 | **Sayim icin lognormal konumu `ln(n)`'in ORTALAMASI degil MEDYANI.** | Ortalama, tek bir kalabalik plaktan (demo'da n=125) asiri etkileniyor; uretilen medyan gercegin %15 altinda kaliyordu. Medyanla fark %6'ya indi. Saglam kestirim. |
| 3.22 | **Koloni sessizce dusurulmez.** Denemeler tukendiginde once gamma'nin reddettigi gecerli konum, o da yoksa en az ihlal eden konum kabul edilir; her iki durum da sayilir ve raporlanir. Hicbiri yoksa `🔴` ile ekrana basilir. | Ilk surumde gamma reddi sonrasi yedek aday tutulmuyordu ve koloni **sessizce dusuyordu** — uretilen sayim dagilimi asagi kayiyordu, hicbir uyari yoktu. Faz 2'deki `make_splits` hatasiyla ayni sinif: sessiz, olumcul, ve ancak metrik tuhaf gelince fark edilir. |
| 3.23 | **`dogrula` alt komutu**: uretilen yerlesim gercekle 10 olcutte karsilastirilir (sayim, r/R, dis halka, degme orani, ortusme derinligi, Clark–Evans). | "Gozle bakip iyi gorunuyor" bir kabul kriteri degil. Uretim hattinin her degisikliginden sonra bu tablo calistirilacak. |

### 🔴 Demo veride kalibrasyon anlamli degil — mekanizma dogru, sayi gurultu

Gercek degme orani %39.0, ama **plak duzeyinde onyukleme (bootstrap) %90 guven
araligi: %26.7 – %48.2** (10 plak). 22 puan genislik.

Sebep: degme orani plak icinde son derece korele — seyrek plak ~%0, kalabalik plak
~%90. **Etkin ornek buyuklugu koloni sayisi (387) degil, plak sayisi (10).**
Uretimde farkli seed'ler %31.5 ile %36.0 arasinda degisiyor; ikisi de bu araligin icinde.

Sonuc: `gamma = 0.317` **gecici**. `uydur` bunu `"gecici": true` alaniyla JSON'a
yaziyor ve ekrana kirmizi uyari basiyor. Tam veri gelince seviye basina yeniden
uydurulacak (karar 3.5). **Bu sayiya dayanan hicbir uretim baslatilmayacak.**

### `maske.py` yazildi (14 Agustos) — maske ve arka plan kararlari

| # | Karar | Gerekce |
|---|---|---|
| 3.24 | **Sentetik maske = etiketlenen diskin TA KENDISI. Pay yok, genisletme yok (`SENTETIK_PAY = 1.00`).** | Difuzyon yalnizca maskenin **icini** boyayabilir. Maskeyi buyutursek uretilen koloni etiket kutusunu tasabilir ve **"etiket hatasi sifir" iddiasi "yaklasik sifir"a doner** — bu makalenin ana iddiasi, pazarlik konusu degil. Bedeli: koloninin dogal golgesi/halesi kutu disina cikamaz, kenarda dikis izi riski var. Bu bir **uretim kalitesi** sorunu; difuzyon parametreleriyle cozulecek, maskeyi buyuterek degil. Parametre kodda acikca duruyor ki degistiren kisi neyi kaybettigini gorsun. |
| 3.25 | **Arka plandaki GERCEK koloniler silme maskesine alinir** (`SILME_PAY = 1.35`, comert). Maske = sentetik diskler ∪ silme bolgeleri. | 🔴 **REHBER'in "bos/az koloni iceren plaklar toplanir" varsayimi yanlis: AGAR'da bos plak yok.** Demo pakette en temiz plakta bile 32 koloni var. Arka plani oldugu gibi kullanirsak goruntude **etiketsiz gercek koloniler** kalir → model "koloni = arka plan" ogrenir. Bu, sentetik veriye sessizce **yanlis negatif enjekte etmektir** ve karar 3.12'deki tuzagin aynasi: sentetik verinin aleyhine calisir, "difuzyon ise yaramiyor" diye okunur. |
| 3.26 | **Arka plan havuzu `silme_orani` (silinecek alan ÷ plak diski) esigiyle secilir**, varsayilan 0.06. Havuz < 20 plak ise kirmizi uyari. | Silinecek alan ne kadar buyukse difuzyon o kadar cok "yeni plak uretiyor", o kadar az "gercek arka plan kullaniyor" — ve A2 ablasyonunun ("gercek arka plan yok") anlami zayifliyor. Esik, bu odunlesmeyi tek bir gorunur sayiya baglar. |
| 3.27 | **🔴 `havuz` komutu sinif yanliligi kontrolu yapar** ve %15'i asarsa uyarir. | Demo veride ortaya cikti: esik 0.06'da havuzun %100'u kucuk-koloni turlerinden (S.aureus, C.albicans) geldi. **Sebep zincirleme:** esik silinecek ALANA bakar → alan koloni BOYUTUYLA belirlenir → boyut da SINIFLA (karar 3.14: iki ayri rejim, 27 px vs 140 px). Yani masum gorunen bir esik, arka plan havuzunu sessizce tek bir tur ailesine daraltiyor. Plak ortami, aydinlatma ve besiyeri rengi ture gore degisiyorsa sentetik verinin tamami tek bir gorsel ailenin uzerine kuruluyor demektir. Demo'da 10 plak var, tam veride bu daha yumusak cikabilir — **ama kontrol edilmeden gecilmeyecek.** |
| 3.28 | **Her uretilen plagin yaninda `kayit/<ad>.json` var:** hangi arka plan, hangi seviyeden, hangi listeden, kac gercek koloni silindi. | Sizinti denetim izi (karar 3.4). Hakem "G25'in arka planlari nereden geldi?" diye sorarsa cevap dosyada; hafizada degil. |

### Demo veride uctan uca calisti

`400 plak · havuz 6 arka plan · en cok kullanilan arka plan uretimin %19.5'i ·
sentetik disklerin %17.1'i silme bolgesiyle cakisiyor · tasan etiket 0`

Uretilen kolonilerin dis kenari: p95 = 0.952·R, maks = 1.000·R.
Gercekte: p95 = 0.927·R, maks = 1.003·R. Kenara yakin koloni orani gercekte %3.1,
uretilende %5.2. Fark, yaricapin koloni boyutundan **bagimsiz** orneklenmesinden
geliyor (gercekte buyuk koloniler kenara pek gitmiyor). Kucuk etki, 10 plakta
duzeltilmeye degmez; tam veride bakilacak.

### Acik kalan

| Konu | Durum |
|---|---|
| 🔴 **Silme bolgesinde hayalet koloni** | Sentetik diskle ortulmeyen silme bolgelerinde difuzyonun **duz besiyeri** uretmesi gerekiyor. Onun yerine oraya bir koloni uydurursa, goruntude **etiketsiz bir nesne** olusur — tam da 3.25'in onlemeye calistigi sey, bu sefer difuzyonun kendi eliyle. Karar 3.24 etiketin **fazla buyuk** olmamasini garantiliyor; bunun aynasi olan "etiketsiz nesne yok" garantisi **yok** ve kodla saglanamaz. Faz 4 pilotunda olculecek: uretilen goruntude silme bolgelerine bir detektor koyup kac yanlis pozitif ciktigina bakilacak. Cikarsa cozum: silme bolgelerini sentetik kolonilerle kapatmayi zorunlu kilmak, veya negatif istem (negative prompt). |
| Clark–Evans %10 yuksek | Uretilen 1.13, gercek 1.03. gamma kisa menzilde itiyor ama gercek veri daha *heterojen*: bazi ciftler cok ic ice, bazilari cok uzak. Tek parametreli Strauss bunu tam yakalayamiyor olabilir. 10 plakta karar verilemez; tam veride bakilacak. Gerekirse iki menzilli (sert cekirdek + yumusak itme) modele gecilir. |
| `lora_10`'un veri yeterliligi | ~800 goruntuluk alt kumede LoRA'nin ise yarar bir koloni gorunumu ogrenip ogrenemedigi olculmeli. Ogrenemezse G10+S kolu "sentetik veri yardim etmiyor" degil, "uretici model yetersiz veriyle uyarlanmis" sonucunu verir — ikisi farkli iddialar ve karistirilamaz. Pilotta (Faz 4) test edilecek. |
| Arka plan havuzunun buyuklugu | `train_10` icinde yeterince bos/az koloni iceren plak var mi? Yoksa ayni arka plan defalarca kullanilacak → sentetik cesitlilik duser, FID/KID bunu yakalar. Olculecek. |
