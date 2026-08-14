# Rehber — kodu anlama + Faz 3 girişi

> 13 Ağustos 2026 akşamı yazıldı. İki bölüm:
> **A)** Şu ana kadar yazılanı hangi sırayla okuyacağın
> **B)** Faz 3 nedir, yarın nereden başlıyoruz

---

# A) Kod rehberi

Sırayla oku. Her dosya bir öncekinin çıktısını kullanıyor, o yüzden bu sıra
verinin izlediği yolun aynısı.

```
AGAR JSON  →  convert.py  →  manifest.csv + YOLO etiketleri
                                    ↓
                          analyze_manifest.py  →  dağılım istatistikleri
                                    ↓
                            make_splits.py  →  splits/ listeleri
                                    ↓
                     train.py + base.yaml  →  runs/<ad>/weights/best.pt
                                    ↓
                            evaluate.py  →  ozet.json
                                    ↓
                             butce.py  →  "sığıyor mu"
```

---

### 1. `src/convert.py` — AGAR JSON'unu YOLO'ya çevirir

**Ne yapar:** Her JSON'u okur, filtreleri uygular, YOLO formatında `.txt`
etiketi yazar, `manifest.csv` üretir.

**Anlamanı istediğim üç şey:**

- **Sınıf sırası sabit** (`S.aureus`=0 … `C.albicans`=4). Bu sıra proje boyunca
  değişmeyecek. Değişirse yazılmış tüm etiketler sessizce yanlış sınıfa kayar.
- **`defects`/`contamination` içeren görüntü tamamen atılıyor.** Sadece kutuyu
  silip görüntüyü tutsaydık, modele "burada nesne yok" diye öğretmiş olurduk —
  oysa orada bir şey var, biz onu etiketlemiyoruz. Sinsi bir zehirlenme.
- **Görüntü boyutu dosyadan okunuyor**, JSON'dan değil. AGAR JSON'unda boyut
  alanı yok; 2048×2048 varsaysaydık ve bir görüntü farklı boyutta olsaydı,
  o görüntünün tüm kutuları kayardı ve kimse fark etmezdi.

**Çalıştır:** `python src/convert.py --src data/AGAR_representative --out data/processed`

---

### 2. `src/analyze_manifest.py` — betimsel analiz

**Ne yapar:** manifest'ten sınıf başına kutu sayısı, boyut dağılımı, görüntü
başına koloni sayısı, tür kombinasyonları ve kalite kontrolleri çıkarır.

**Neden önemli:** İki işi birden görüyor.

1. Makalenin **veri tablosu** buradan yazılacak.
2. **Faz 3'ün girdisi burası.** Sentetik görüntü üretirken "bir plakta kaç
   koloni olsun, ne büyüklükte, nereye" sorularının cevabı bu dağılımlardan
   örnekleniyor. Uydurulmuyor.

**Bakman gereken çıktı:** `data/processed/reports/`. Özellikle
`sinif_ozeti.csv` — C.albicans medyanı 27.5 px, S.aureus 29 px. Bu iki sayı
`imgsz=1280` kararının tüm gerekçesi.

---

### 3. `src/make_splits.py` — bölme ve alt örneklemler

**Ne yapar:** %70/15/15 böler, sonra train içinden iç içe alt örneklemler
(%50 ⊃ %25 ⊃ %10) çıkarır.

**İki tasarım kararı:**

- **Tabakalı bölme.** Tabaka = tür kombinasyonu. Rastgele bölseydik %10
  alt kümesinde C.albicans hiç kalmayabilirdi; o zaman "az veriyle başarım
  düştü" mü yoksa "bir sınıf kayboldu" mu olduğunu ayıramazdık.
- **İç içelik.** %10 ⊂ %25 ⊂ %50 ⊂ %100. Değilse seviyeler arası farkın veri
  *miktarından* mı yoksa *hangi görüntülerin seçildiğinden* mi geldiği belirsiz kalır.

**Buradaki hata bugün iki kez ısırdı — anlamaya değer:**

`convert.py`, `data/processed/images/` içine ham dosyalara **symlink** atıyor.
Ultralytics ise etiketi şöyle buluyor: görüntü yolundaki son `/images/` parçasını
`/labels/` ile değiştir, `.txt` uzantısı ekle, o dosyayı oku.

İlk sürüm liste yazarken `.resolve()` kullanıyordu. `.resolve()` symlink'i
takip eder → yol `data/AGAR_representative/...` olur → içinde `/images/` yok →
**hiçbir etiket bulunamaz.** Ultralytics bu durumda durmaz, uyarır ve geçer;
model her görüntüyü boş sanar, mAP sıfır çıkar.

Daha sinsisi: kontrol fonksiyonu "0 eksik" diyordu — çünkü *değişkendeki* değeri
test ediyordu, *dosyaya yazılan* değeri değil. **Kontrol yanlış şeye bakıyordu.**
Şimdi yazılan satırları geri okuyup doğruluyor ve örnek bir satırı ekrana basıyor.

> Ders: bir kontrol, kontrol ettiğini iddia ettiği şeye bakmıyorsa,
> kontrol olmaktan çıkıp yanlış güven kaynağına dönüşür.

---

### 4. `configs/base.yaml` — protokol

**Bu bir ayar dosyası değil, bir sözleşme.** Faz 5'te dondurulacak. Sonuçları
gördükten sonra buradaki bir sayıyı değiştirmek "cherry-picking" olarak okunur.

**İki karar öne çıkıyor:**

- **`imgsz: 1280`.** Bugün ampirik olarak doğrulandı: 27.5 px'lik C.albicans
  AP50 0.963 aldı. 640'ta o koloni 8.6 piksele inerdi — tespit başının stride'ı
  8, yani tek piksel. **VRAM yetmezse batch düşür, buna dokunma.**
- **Ana gridde minimum augmentation.** Ultralytics varsayılan `mosaic=1.0` ile
  gelir. Hiçbir şey yapmasaydık kontrol grupların (G50/G25/G10) *zaten* klasik
  veri artırma kullanıyor olurdu ve "difüzyon vs klasik" karşılaştırması baştan
  bulanırdı. Açık olanlar: flip (plağın kanonik yönü yok) ve HSV (aydınlatma
  gerçek bir varyans kaynağı). Kapalı olanların her birinin gerekçesi dosyada.

---

### 5. `src/eval/metrics.py` — ölçüm çekirdeği

**Projenin en dikkatli yazılmış dosyası.** Ultralytics'ten bağımsız.

**Neden kendi kodumuz var:** Ultralytics boyut bazlı AP (small/medium/large)
vermiyor — oysa makalenin ana argümanı küçük koloniler üzerinde. Sayım metrikleri
(MAE/sMAPE) hiç yok. Ve ileride ikinci bir detektör denenecekse, ölçümün
detektörden bağımsız olması şart; yoksa karşılaştırma geçersiz.

**Güvenilirliğinin kanıtı:** `test_metrics.py`, 33 kontrol. Altısı sonuçları
**pycocotools ile 1e-4 hassasiyetinde** karşılaştırıyor. Hakem "kendi metriğinizi
mi yazdınız?" diye sorduğunda cevap bu test.

**Bir tasarım detayı:** sayım metrikleri bir güven eşiği gerektiriyor. O eşik
**val kümesinde** seçilir, teste öyle uygulanır. `evaluate.py --split test`
eşik verilmeden **çalışmaz** — insan hafızasına bırakılmadı, kod zorunlu tutuyor.
Test kümesinde eşik aramak, test kümesini ayara dahil etmektir.

---

### 6. `scripts/train.py` — eğitim sarmalayıcısı

**Neden düz `yolo train` değil:** Süre, GPU-saat ve tepe VRAM **ilk koşudan
itibaren** kaydedilmeli. Makalenin "hesaplama maliyeti" bölümü bu dosyalardan
yazılacak ve sonradan geri dönüp ölçmek mümkün değil.

Ayrıca augmentation politikasını komut satırından değil config'den alıyor —
"hangi koşuda ne açıktı" sorusu üç ay sonra cevaplanabilir olsun diye.

**Çıktı:** `runs/<ad>/olcum.json`. `butce.py` bunu okuyor.

---

### 7. `scripts/butce.py` — bütçe hesabı

Tek ölçülmüş koşudan tüm gridi ölçekler. Model: süre ≈ görüntü sayısı × epoch ×
imgsz². Bugün verdiği cevap: **61 koşu bu dizüstünde ~470–720 GPU-saat**, yani
3–4 hafta kesintisiz. Sığmıyor.

---

# B) Faz 3 girişi — üretim hattı

## Ne yapıyoruz

Şimdiye kadar gerçek veriyle çalıştık. Faz 3'te **sentetik görüntü üretmeye**
başlıyoruz. Projenin cevaplamaya çalıştığı soru burada doğuyor.

**Temel fikir bir kez daha:** Difüzyonla nesne tespiti için veri üretmenin klasik
sorunu şu — ürettiğin görüntüde nesnelerin *nerede* olduğunu bilmezsin, etiketi
yine elle çıkarman gerekir. Etiketleme maliyeti yok olmaz, yer değiştirir.

Çözüm ters yönden çalışmak: **önce nereye ne koyacağına karar ver, sonra
difüzyona sadece o bölgeleri doldurt.** Kutu koordinatları zaten elinde olduğu
için etiket dosyası kendiliğinden oluşur. Etiket hatası sıfır — insan
etiketlemesinden bile temiz.

## Beş parça

| # | Parça | Ne yapar | Veri gerekir mi |
|---|---|---|---|
| 1 | **Arka plan havuzu** | Boş/az koloni içeren gerçek plak görüntüleri toplanır. Sentetik koloniler bunların üstüne çizilecek | 🟡 demo yeter (test için) |
| 2 | **Yerleşim istatistikleri** | Gerçek plaklarda koloniler nereye, kaç tane, ne büyüklükte düşüyor? `analyze_manifest.py` çıktısından modellenir | ✅ demo yeter |
| 3 | **Maske üretici** | İstatistiklerden örnekleyip "şu koordinatlarda, şu boyutlarda, şu sınıftan koloniler olacak" diyen maske + etiket üretir | ✅ **veri gerekmiyor** |
| 4 | **LoRA** | Difüzyon modeline "agar plağı kolonisi nasıl görünür" öğretilir | 🔴 gerçek veri + GPU |
| 5 | **Inpainting** | Maskelenmiş bölgeler LoRA'lı modelle doldurulur | 🔴 GPU |

**Yarın 2 ve 3'ten başlıyoruz** — ikisi de tam veri beklemiyor, ve 3 numara
projenin en özgün parçası.

## Faz 3'ün kapısı

> *Ürettiğin görüntülere baktığında "bu gerçek olabilir" diyor musun?*

Demiyorsan sırayla: parametre ayarı → ControlNet ile konum kontrolü → farklı
base model. Bu döngü günler alabilir. **Faz 3'ün riskli olmasının sebebi bu.**

## 🔴 Baştan bilmen gereken tuzak: LoRA sızıntısı

Bu, projenin en önemli metodolojik riski ve **outline'da bile yok.**

G10+S deneyinde iddia şu: "sadece %10 gerçek veri kullandım, gerisini sentetikle
tamamladım." Ama LoRA'yı %100 gerçek veriyle eğitirsen, sentetik görüntüleri
üreten model dışarıda bıraktığın %90'ı **zaten görmüş** olur. O bilgi sentetik
görüntülerin içine sızar. İddian çöker, hakem de bunu bulur.

**Çözüm:** her gerçek veri seviyesi için **ayrı LoRA**, yalnızca o seviyenin
verisiyle eğitilmiş. Yani 3 ayrı LoRA eğitimi (%50, %25, %10). Ve S100 için
"gerçek veri hiç kullanılmadı" denemez — "gerçek veri yalnızca LoRA üzerinden
dolaylı olarak kullanıldı" denir.

Bunu Faz 3'ün en başında kurmak lazım, sonradan düzeltmek tüm üretimi
tekrarlamak demek.

## Yarın nereden başlıyoruz

1. `analyze_manifest.py` çıktısına birlikte bakıp yerleşim dağılımını çıkarmak
   (koloniler plak merkezine mi toplanıyor, kenara mı; yoğunluk nasıl dağılıyor;
   koloniler birbirine değiyor mu)
2. `src/generate/yerlesim.py` — o dağılımdan örnekleyen kod
3. `src/generate/maske.py` — maske + etiket üretici
4. Gözle kontrol: ürettiğimiz maskeler gerçek plakların yerleşimine benziyor mu

Difüzyona daha girmiyoruz. Önce "nereye ne koyacağız" sorusunu çözüyoruz.
