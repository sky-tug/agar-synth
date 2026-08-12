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

## Faz 2 — Olcum altyapisi (12 Agustos 2026)

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

## Faz 3 — Uretim hatti

*(bos)*
