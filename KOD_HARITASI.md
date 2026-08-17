# Kod Haritası — agar-synth

> 17 Ağustos 2026'da, deponun tamamı satır satır okunarak yazıldı.
> Kapsam: `src/`, `scripts/`, `configs/`, `notebooks/` altındaki **her** kod dosyası.
>
> `REHBER.md` "hangi sırayla okuyayım" sorusuna cevap verir.
> Bu dosya **"bu satır neden böyle yazılmış"** sorusuna cevap verir.
> Kararların gerekçesi `decisions.md`'de; buradaki atıflar oraya işaret eder.

---

## İçindekiler

1. [Zincir — veri nereden nereye akıyor](#1-zincir)
2. [Veri hattı](#2-veri-hattı) — `convert` · `check_labels` · `analyze_manifest` · `make_splits`
3. [Ölçüm](#3-ölçüm) — `metrics` · `test_metrics` · `evaluate`
4. [Eğitim](#4-eğitim) — `train.py` · `configs/*.yaml`
5. [Üretim (Faz 3)](#5-üretim-faz-3) — `yerlesim` · `maske`
6. [Yardımcı](#6-yardımcı) — `butce` · `kurulum.sh` · notebook · `requirements` · `.gitignore`
7. [Tekrar eden desenler](#7-tekrar-eden-desenler)
8. [Bulunan 25 madde](#8-bulunan-25-madde)

---

## 1. Zincir

```
ham AGAR (jpg + json)
   │
   │  src/convert.py            filtreler, koordinat dönüşümü
   ▼
data/processed/
   ├── labels/*.txt             YOLO etiketi
   ├── images/*.jpg             symlink (kopya değil)
   ├── classes.txt              sabit sınıf sırası
   └── manifest.csv             ── her görüntü için bir satır
         │                          │
         │ src/make_splits.py       │ src/analyze_manifest.py
         ▼                          ▼
   lists/train · train_50 · 25 · 10 · val · test      reports/*.csv + veri_ozeti.png
         │                                                  │
         │ scripts/train.py  (+ configs/*.yaml)              └─→ Faz 3'ün girdisi
         ▼                                                        │
   runs/<koşu>/                                                   ▼
   ├── weights/best.pt                              src/generate/yerlesim.py
   └── olcum.json   ── süre, VRAM, GPU-saat, git commit           │
         │                                                        ▼
         │ src/eval/evaluate.py → src/eval/metrics.py    src/generate/maske.py
         ▼                                                        │
   runs/<koşu>/eval/                                              ▼
   ├── ozet.json    ── 61 koşuluk grid tablosu bundan kurulacak   difüzyon (henüz yok)
   ├── sinif_ap.csv · boyut_ap.csv · sayim.csv
   └── conf_egrisi.csv
         │
         │ scripts/butce.py
         ▼
   GPU-saat tahmini
```

**Bir cümlelik özet:** `convert` veriyi standartlaştırır, `make_splits` onu böler,
`train` eğitir, `evaluate` ölçer, `yerlesim`+`maske` sentetik veri için koordinat
üretir, `butce` hepsinin maliyetini hesaplar.

---

## 2. Veri hattı

### `src/convert.py` — 247 satır

AGAR JSON → YOLO. Zincirin başı.

**Çıktısı dört şey:** `labels/<id>.txt`, `images/<id>.jpg` (symlink),
`manifest.csv`, `classes.txt`.

**Sabit sınıf sırası (satır 29)** — proje boyunca değişmeyecek:

```python
CLASS_ORDER = ["S.aureus", "B.subtilis", "P.aeruginosa", "E.coli", "C.albicans"]
```

Değişirse tüm etiket dosyaları sessizce bozulur: `0` artık S.aureus değil başka
bir şey olur, ama dosyalar aynı görünür. Aynı liste `metrics.py`, `make_splits.py`
ve `base.yaml`'da tekrar ediyor — dördü tutmak zorunda.

**Üç filtre:**

| # | Filtre | Gerekçe |
|---|---|---|
| 1 | yalnızca `lower-resolution` | outline 3.2 kapsamı |
| 2 | `labels` boş olanlar elenir (countable değil) | koloni seviyesinde etiket yalnızca countable'da var |
| 3 | `defects`/`contamination` içeren görüntü **tamamen** atılır | kutuyu silip görüntüyü tutmak modele "burada nesne yok" öğretir (karar 1.2) |

**Görüntü boyutu dosyadan okunuyor (satır 128)** — AGAR JSON'unda boyut alanı yok
(karar 1.3). 2048×2048 varsaymak sessiz normalize hatası üretirdi.

**Koordinat dönüşümü (satır 148–164)** — asıl iş:

```python
x, y = float(l["x"]), float(l["y"])          # AGAR: SOL ÜST köşe, piksel
w, h = float(l["width"]), float(l["height"])
x1, y1 = min(float(W), x + w), min(float(H), y + h)   # görüntüye kırp
if bw <= 1 or bh <= 1:                                # bozuk kutu at
    atilan_kutu += 1
    continue
xc = (x0 + bw / 2) / W                       # YOLO: MERKEZ, normalize
yc = (y0 + bh / 2) / H
```

Buradaki tek işaret hatası her kutuyu kaydırır, eğitim çalışır, mAP düşük çıkar
ve sen "difüzyon işe yaramıyor" sanırsın. `check_labels.py` tam olarak bunun için var.

**Eleme özeti (satır 218–240)** — her elenen görüntünün sebebi ayrı sayaçta.
Bu tablo makalenin veri bölümüne doğrudan girecek: hakem "18.000 görüntüden nasıl
12.271'e düştünüz" diye soracak.

---

### `src/check_labels.py` — 152 satır

Etiketleri görüntü üzerine çizer. Docstring projenin en dürüst cümlesi:

> *"Kutu koordinatlarındaki tek bir hata eğitimde SESSİZCE ilerler ve sonra
> 'modelim neden öğrenmiyor' olarak geri döner."*

**Akıllı örnekleme (satır 56–64)** — rastgele 30 görüntü seçmiyor:

```python
pay = max(1, args.n // (len(CLASS_ORDER) * 2))
for c in CLASS_ORDER:
    aday = [r for r in rows if int(r.get(f"n_{c}", 0) or 0) > 0 ...]
```

Önce her sınıftan birkaç tane garantiliyor, sonra kalanı rastgele dolduruyor.
C.albicans nadir olduğu için 30 rastgele görüntüde hiç çıkmayabilir — ve o
sınıfın dönüşümü bozuk olsa fark edilmezdi.

**Otomatik kontroller (satır 88–108):** manifest boyutu dosyayla tutuyor mu,
normalize değerler 0–1 arasında mı, 4 pikselden küçük kutu var mı.

---

### `src/analyze_manifest.py` — 226 satır

Üç iş: makalenin betimsel tablosu, Faz 3'ün yerleşim girdisi, kalite kontrolü.
`reports/` altına 4 CSV + 4 panelli PNG.

**Mükerrer kutu taraması (satır 157–175)** vektörize: görüntü içi tüm çiftlerin
IoU'su tek matris işleminde.

```python
inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
union = area[:, None] + area[None, :] - inter
m = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
ii, jj = np.where(np.triu(m, k=1) >= args.iou_dup)
```

🔴 **`radius` burada 0.5·genişlik ile normalize** (satır 56), `yerlesim.py`'de ise
plak yarıçapı (0.465·genişlik) ile. Aynı kelime iki dosyada iki farklı şey.
Bkz. [bulgu 9](#8-bulunan-25-madde).

---

### `src/make_splits.py` — 246 satır — **en kritik dosya**

Dört bölüm: böl → yaz → kontrol et → raporla.

**Tabakalama (satır 33):**

```python
for _, grp in df.groupby("classes", sort=True):
```

Tabaka = tür kombinasyonu. `["E.coli"]` bir tabaka, `["E.coli","S.aureus"]` ayrı
bir tabaka. Her tabaka kendi içinde %70/15/15 bölünüyor (karar 1.5).

**Train asla boş kalmaz (satır 40–44):**

```python
while n - n_test - n_val < 1 and (n_test + n_val) > 0:
    if n_test >= n_val: n_test -= 1
    else:               n_val -= 1
```

**İç içelik — tek satır (satır 73):**

```python
havuz = secilen    # sonraki seviye BUNUN alt kümesi olacak
```

`LEVELS = [50, 25, 10]`, `oran = lvl / onceki_oran` → 0.5, sonra 0.5, sonra 0.4.
Zincirleme daralma, %10 ⊂ %25 ⊂ %50 ⊂ %100 (karar 1.6).

`k = max(1, int(round(len(items) * oran)))` — her tabakadan en az 1 görüntü.
Demo pakette `train_50 = train_25 = train_10` çıkmasının sebebi bu; kod bunu
hata değil **uyarı** olarak basıyor.

**İki yara izi.** Satır 154:

```python
# DIKKAT: .resolve() KULLANMA. data/processed/images/ altındakiler ham AGAR'a
# SYMLINK; resolve() onları takip eder, '/images/' parçası kaybolur,
# Ultralytics etiketi bulamaz.
```

Satır 171 — ve bu daha öğretici:

```python
# Kontrol, DOSYAYA YAZILAN satırların AYNISI üzerinden yapılmalı.
# (Önceki sürüm path_of'u kontrol ediyordu ama dosyaya resolve() edilmiş hali
#  yazılıyordu; kontrol geçti, eğitim patladı.)
```

İlk hata koddaydı. İkinci hata **kontroldeydi** — kontrol, kontrol edilmesi
gereken şeye bakmıyordu. Ders: *bir kontrol yazdığında, gerçekten kullanılacak
olan şeyi mi kontrol ediyorsun?*

**Üç otomatik kontrol (satır 167–202):** etiket çözümleme · iç içelik · sızıntı.
Hepsi `tamam &= ...` ile birikiyor, sonda tek satır: `Durum: TAMAM` / `SORUN VAR`.

---

## 3. Ölçüm

### `src/eval/metrics.py` — 357 satır

Ultralytics'ten **bağımsız** AP hesabı. Dört gerekçe docstring'de, en önemli ikisi:

> 1. Ultralytics boyut bazlı AP vermiyor — makalenin ana argümanı küçük koloniler.
> 3. İkinci detektör kontrolü için AYNI ölçüm kodunun çalışması gerekiyor.
>    Detektöre bağlı metrik, detektörler arası karşılaştırmayı geçersiz kılar.

Yani: **modeli değiştirdiğinde cetvel değişmemeli.**

**IoU (satır 104–116)** — her şeyin dayandığı kavram. Kesişim ÷ birleşim.

**`_match_one()` (satır 129–182)** — COCO'nun `cocoeval.evaluateImg`'iyle birebir.

Tespitler güvene göre sıralanıyor:

```python
order_d = np.argsort(-dt_conf, kind="stable")
```

Ve dosyanın en kurnaz kısmı, **`ignore` mekanizması**:

```python
gt_ig = (ga < area_rng[0]) | (ga > area_rng[1])
order_g = np.argsort(gt_ig, kind="stable")
```

"Sadece küçük nesnelerde AP" hesaplarken büyük koloniler **silinmiyor**, "yok say"
diye işaretleniyor. Silinseydi, model büyük koloniyi doğru bulduğunda bu yanlış
pozitif sayılır ve küçük-nesne skoru haksız düşerdi. İşaretlenince ne ödül var ne
ceza. `argsort` de yok-sayılmayanları öne alıyor.

**`evaluate_detection()` (satır 189–282)** — bir satır yanlış olsa metrik şişerdi:

```python
order = np.argsort(-conf, kind="stable")   # TÜM görüntüler boyunca
```

PR eğrisi tüm test kümesi boyunca tek sıralı liste üzerinden çiziliyor, görüntü
başına değil. Kendi AP'sini yazanların en sık hatası budur.

Precision zarfı (satır 256–259):

```python
pr = np.concatenate([pr, [0.0]])
for i in range(len(pr) - 2, -1, -1):
    pr[i] = max(pr[i], pr[i + 1])
```

Ham PR eğrisi tırtıklıdır; COCO sağdan sola yürüyüp her noktayı "bundan sonra
ulaşılan en iyi precision" yapıyor. Sonra 101 sabit recall noktasında örnekleyip
ortalıyor — AP bu. 10 IoU eşiğinde tekrarlanıp ortalanınca `mAP50-95` çıkıyor.

**`counting_metrics()` (satır 289–339)** — yorumu projenin özeti:

```python
# Koloni sayimi -- mikrobiyolojide ASIL is bu, mAP vekil olcut
```

`ME` (işaretli ortalama hata) var çünkü MAE yanlılığı gizler:

```python
"ME": float(err.mean()),   # + = fazla sayiyor
```

Mikrobiyoloğa "model sistematik olarak az sayıyor" demek, "MAE 4.2" demekten
daha kullanışlı (karar 2.6).

**`select_conf_threshold()` (satır 342–356)** — sızıntı kilidi:

> *"conf_thr TEST kümesinde seçilemez. VAL'da seçilip donduruldu, teste öyle
> uygulanır. evaluate.py bunu zorunlu tutuyor."*

---

### `src/eval/test_metrics.py` — 294 satır, 33 kontrol

Kural: *"kendi yazdığın metriğe, doğru cevabını ELLE bildiğin vakalarla güvenmeden
önce inanma."*

**Test 4 — en zarifi.** Kutu 10 px kaydırılıyor:

```python
# GT 100x100, tahmin 10px kaymis -> IoU = 90*100 / (2*10000 - 9000) = 0.818
kontrol("AP50-95 = 7/10", abs(r["ap"]["all"][0] - 0.7) < 1e-6)
```

IoU 0.818, 10 eşikten yedisinde (0.50–0.80) eşleşir, üçünde eşleşmez.
AP50-95 **tam olarak 0.7** olmak zorunda. Yaklaşık değil, kesin.

**Test 5 — sıralamanın çalıştığını kanıtlıyor.** Aynı iki kutu, güven skorları
yer değişince:

```
dt_box=d               -> AP50 = 1.0    (yanlış alarm düşük skorlu)
dt_box=d[::-1].copy()  -> AP50 < 0.6    (yanlış alarm yüksek skorlu)
```

**Test 6** boyut kırılımını, **Test 10** `pycocotools`'a karşı 1e-4 hassasiyette
çapraz doğrulamayı yapıyor. Hakemin "kendi metriğinizi mi yazdınız?" sorusunun
cevabı test 10.

---

### `src/eval/evaluate.py` — 247 satır

`metrics.py`'nin CLI'ı. Her koşu için 5 dosya üretir; `ozet.json` en önemlisi —
61 koşunun grid tablosu onlardan kurulacak.

**İki çalışma biçimi:**

```
A) --weights   Ultralytics modelini yükler, tahminleri kendisi üretir
B) --pred-dir  hazır YOLO-format tahmin dosyalarından okur
```

B, karar 2.1'in somut hâli: ikinci bir detektör denemek istersen ölçüm kodu hiç
değişmez.

**Sızıntı kilidi (satır 141–144):**

```python
if args.split == "test" and args.conf_thr is None:
    sys.exit("HATA: test kumesinde conf esigi ARANAMAZ.\n"
             "  once: --split val   (esigi secer)\n"
             "  sonra: --split test --conf-thr <secilen deger>")
```

**`MIN_CONF = 0.001` (satır 48)** — mAP hesabı tüm tespitleri ister; PR eğrisinin
kuyruğu düşük güvenli tahminlerden çizilir. Sayım ise ayrı eşik kullanıyor.
İki metrik, iki eşik, karışmıyor.

**Alan aralıkları orijinal pikselde** (satır 75–77) — `small` < 32², `large` > 96²
hesabı `imgsz`'de değil 2048'de yapılmalı, yoksa boyut kırılımı kayar:

```python
with Image.open(ip) as im:
    W, H = im.size          # gerçek dosyadan
gc, gb, _ = read_yolo_txt(..., W, H)
```

---

## 4. Eğitim

### `scripts/train.py` — 327 satır

Neden doğrudan `yolo train` değil — dört gerekçe, en önemli ikisi:

> 1. Süre, GPU-saat ve tepe VRAM İLK koşudan itibaren kayıt altında olmalı.
>    Sonradan geri dönüp ölçülemez.
> 4. Faz 5'te protokol dondurulduğunda, dondurulan şey bu dosya + config olacak.

**Ön uçuş kontrolü (satır 101–124)** — `make_splits.py`'deki kontrolün aynısını
bir kez daha yapıyor. Aynı hataya karşı üçüncü savunma hattı; o hata iki kez
sessizce vurdu.

**Dürüstlük kilidi (satır 190–199):**

```python
if val_bos and args.duman:
    print("! val kumesi bos -- duman testi icin val=train kullaniliyor.")
    print("  Bu kosunun mAP degeri ANLAMSIZ.")
```

Yalnızca `--duman` ile mümkün. Gerçek koşuda val boşsa program **durur**. Yani
"yanlışlıkla kendi eğitim verinde test etmek" kod seviyesinde imkânsız.

**Ölçüm kaydı (satır 271–299)** — `olcum.json`:

```python
"git_commit": git_commit(),      # hangi kod sürümüyle koştu
"toplam_sure_s", "epoch_basina_s", "gpu_saat", "tepe_vram_gb",
"goruntu_saniye", "ortam": {python, platform, kart, vram_gb, cuda_surum}
```

`git_commit` özellikle iyi: üç ay sonra "bu sonuç hangi kodla çıktı" sorusunun
cevabı dosyanın içinde.

---

### `configs/base.yaml` — 170 satır

Kod yok, sadece sayı ve gerekçe.

**`imgsz: 1280` — projenin en kritik tek hiperparametresi.** Yorumdaki hesap:

```
C.albicans medyan 27.5 px (2048'de)
  imgsz=640  -> 8.6 px    P5 başının stride'ı 8, tespit imkânsıza yakın
  imgsz=1280 -> 17.2 px   <- seçilen
Bedeli: VRAM ve süre ~ imgsz². 1280, 640'ın ~4 katı.
Karar: VRAM yetmezse batch'i düşür, imgsz'yi DÜŞÜRME.
```

Bu artık tahmin değil ölçüm: 13 Ağustos ezber testinde C.albicans 0.963,
S.aureus 0.951 AP aldı — en küçük iki sınıf, en büyüklerle aynı bantta (karar 2.18).

**Augmentation politikası (satır 68–97).** Sorun: Ultralytics varsayılan olarak
`mosaic=1.0` ile gelir. Hiçbir şey yapmazsan G50/G25/G10 kontrol grupların
**zaten** klasik veri artırma kullanır ve "difüzyon vs klasik" karşılaştırması
baştan bulanır.

```yaml
fliplr: 0.5      flipud: 0.5      hsv_*: açık      # plak fiziğine uygun
scale: 0.0       # koloni boyutu bir SINIF ipucu — oynatmak sinyali bozar
mosaic: 0.0      # 4 plağı tek karede birleştirir, sayım işi için zararlı
copy_paste: 0.0  # bu tam olarak Baseline C'nin kolu
```

Sıfır olanlar bile açıkça yazılı (karar 2.13) — örtük varsayılan, sürüm değişince
sessizce değişir.

---

### `configs/aug_b_klasik.yaml` · `aug_c_kopyala.yaml`

Difüzyonun rakipleri. `base.yaml`'ın üzerine biniyorlar; `train.py`'deki
`derin_birlestir` iç içe sözlükleri birleştirdiği için overlay'de yazmadığın
alanlar base'den korunuyor (`cutmix: 0.0` aug_b'de yazmasa da geçerli).

**Baseline B** — klasik geometrik + renk. `degrees: 180`, `translate: 0.1`,
`scale: 0.5`. Dosyanın kendi uyarısı: *"koloni boyutu sınıf ipucu; bu kol o ipucunu
bilerek bozuyor."*

🔴 Ama `degrees: 180`'in etikete yaptığı, dosyada yazmıyor. Bkz. [bulgu 1](#8-bulunan-25-madde).

**Baseline C** — copy-paste/mosaic/mixup, difüzyonun asıl rakibi. En dürüst uyarı:

> *"Ultralytics'in copy_paste'i segmentasyon maskesi ister. Detect görevinde
> sessizce etkisiz kalabilir. ... Sessizce etkisiz bir baseline, difüzyon lehine
> sahte bir kazanç üretir — en tehlikeli hata bu."*

Yedek plan `offline_copy_paste` bloğunda hazır, `kaynak: train` notuyla
(val/test'ten kırpmak sızıntı olur).

---

## 5. Üretim (Faz 3)

### `src/generate/yerlesim.py`

Üç alt komut: `uydur` (gerçek etiketlerden dağılım çıkar) → `ornekle` (sentetik
koordinat üret) → `dogrula` (üretileni gerçekle 10 ölçütte karşılaştır).

**Sızıntı karşıtı: `--seviye` zorunlu, varsayılanı yok** (karar 3.5). Parametre
dosyasının seviyesiyle `--seviye` uyuşmazsa program durur (karar 3.7).

**Yerleşim modeli:**

| Bileşen | Nasıl |
|---|---|
| plak çemberi | sabit: merkez = görüntü merkezi, `R = 0.465 × genişlik` (3.10) |
| açı | `θ ~ U(0, 2π)` — KS p=0.44, düzgünlük reddedilemedi |
| yarıçap | gözlenen `r_norm` dağılımının ters-CDF'si (3.11) — düzgün **değil**, son halka %20 yerine %8.5 |
| boyut | sınıf başına ayrı (3.14) — iki rejim: ~28 px ve ~140 px, arada hiçbir şey yok |
| sayım | lognormal, konum parametresi `ln(n)`'in **medyanı** (3.21) |
| kompozisyon | bir baskın tür + az sayıda ikincil (3.16) |
| değme | Strauss `gamma`, gerçek değme oranına otomatik kalibre (3.19) |

**Strauss kalibrasyonu (karar 3.19)** — projenin en ince parçası. Saf rastgele
yerleştirme değme oranını %52 veriyor, gerçek %39. Yani gerçek yerleşimde kısa
menzilli zayıf bir itme var; Clark–Evans bunu göremiyor çünkü ortalama tabanlı
bir ölçüt. `gamma` elle seçilmiyor, ikili aramayla oturuyor:

```python
if enb > MAKS_ORTUSME:                    # sert sınır (3.12)
    ...
if enb == 0.0 or rng.random() < gamma:    # Strauss kabulü (3.19)
    yerlesik.append(...)
```

**Ortak rastgele sayılar (3.20)** — tarifler bir kez üretilip her `gamma`
denemesinde aynen kullanılıyor; yoksa rastgele akış kayar ve arama gürültüye oturur.

**Koloni sessizce düşürülmez (3.22)** — denemeler tükendiğinde yedek konum
kabul edilir ve sayılır. İlk sürümde düşüyordu ve hiçbir uyarı yoktu.

---

### `src/generate/maske.py`

İki alt komut: `havuz` (seviyenin arka plan havuzunu seç) → `uret` (maske + arka
plan + etiket).

**Maskede iki bölge var:**

```
SENTETİK  koloni diskleri — difüzyon buraya yeni koloni çizecek
SİLME     arka plandaki GERÇEK kolonilerin üzeri — boyanıp yok edilecek
```

İkincisi kritik. Plan "boş plakları arka plan olarak kullan" diyordu ama
**AGAR'da boş plak yok** — en temiz plakta bile 32 koloni var (karar 3.25).
Arka planı olduğu gibi kullanırsak görüntüde etiketsiz gerçek koloniler kalır,
model "koloni = arka plan" öğrenir.

**Sentetik maske = etiket diskinin ta kendisi (karar 3.24):**

```python
SENTETIK_PAY = 1.00   # genişletme YOK
```

Difüzyon yalnızca maskenin içini boyayabildiği için üretilen koloni etiket
kutusunu **taşıyamaz**. "Etiket hatası sıfır" iddiası buna dayanıyor. Bedeli:
koloninin doğal gölgesi kutu dışına çıkamaz, kenarda dikiş izi riski var — bu
bir üretim kalitesi sorunu, difüzyon parametreleriyle çözülecek, maskeyi
büyüterek değil.

**Sınıf yanlılığı kontrolü (karar 3.27)** — havuz eşiği silinecek **alana** bakar,
alan koloni **boyutuyla** belirlenir, boyut da **sınıfla**. Yani masum görünen bir
eşik, arka plan havuzunu sessizce tek bir tür ailesine daraltıyor. Kod bunu ölçüp
%15'i aşarsa kırmızı uyarı basıyor.

---

## 6. Yardımcı

### `scripts/butce.py` — 309 satır

Maliyet modeli (satır 158–160):

```python
s_per_img_epoch = epoch_s / n_olculen
imgsz_carpani   = (imgsz / olculen_imgsz) ** 2
tam_kosu_s      = s_per_img_epoch * n_tam * args.epochs * imgsz_carpani
```

Görüntü sayısı doğrusal, epoch doğrusal, görüntü boyutu **karesel**. Her kol için
`saat = pay * seed * tam_kosu_s / 3600`.

Duman testine karşı uyarı da var — kısa koşularda epoch başına süre ısınma
yüzünden yanıltıcı olur.

🔴 İki hata: LoRA sabit 3 (4 olmalı), ablasyonun üretim maliyeti hiç sayılmıyor.
Bkz. [bulgu 2 ve 3](#8-bulunan-25-madde).

### `scripts/kurulum.sh` — 5 adım

GPU → ortam → paketler → doğrulama → **33 sanity testi**. Test geçmezse `exit 1`.

İçinde yaşanmış bir hatanın izi var:

```bash
# DIKKAT: Zaten etkin bir venv varsa conda activate onu EZEMEZ; paketler
# venv'e kurulur ve sonda "conda activate agar" demek yanıltıcı olur
```

### `notebooks/faz2_colab.ipynb` — 26 hücre

Faz 2'nin koşu kılavuzu: GPU → kurulum → repo/veri → sanity testi → veri hattı →
duman testi → G100 → değerlendirme → bütçe → kapı.

Hücre 8 eğitimden **önce** 33 testi koşuyor. Hücre 18–21 sızıntı sırasını
zorunlu kılıyor: önce VAL (eşik seçilir), sonra TEST (o eşikle).

Kapı: baseline mAP50-95 ~0.55–0.65 bandında mı (AGAR makalesi ~0.594), ve grid
bütçeye sığıyor mu.

### `requirements.txt`

PyTorch bilinçli olarak yok — CUDA sürümüne göre `kurulum.sh` kuruyor.
Faz 3 bağımlılıkları (diffusers, peft, clean-fid, lpips) yorumda bekliyor;
**artık Faz 3'tesin, sıra onları açmaya geldi.**

### `.gitignore`

`data/` `runs/` `wandb/` `*.pt` `.venv/` dışarıda; `splits/` içeride (karar 1.7).

🔴 `runs/` makalenin bütün sonuçlarını da dışarıda bırakıyor.
Bkz. [bulgu 5](#8-bulunan-25-madde).

---

## 7. Tekrar eden desenler

Depoyu baştan sona okuyunca üç desen belirgin çıkıyor. Yeni kod yazarken bunlara
uy — proje bu üç fikir üzerine kurulu.

### Desen 1 — Metodolojik kural = çalışma zamanı kontrolü

Bir kural yorumda kalırsa unutulur. Bu depoda kurallar `sys.exit`'te:

| Kural | Nerede |
|---|---|
| test kümesinde eşik aranamaz | `evaluate.py:141` |
| val boşsa gerçek koşu yapılamaz | `train.py:200` |
| seviye belirtilmeden yerleşim üretilemez | `yerlesim.py` argparse `required=True` |
| havuz seviyesi plan seviyesiyle uyuşmalı | `maske.py:168` |
| iç içelik ve sızıntı doğrulanmadan devam edilmez | `make_splits.py:187–202` |
| sanity testi geçmeden kurulum bitmez | `kurulum.sh` son adım |

### Desen 2 — Sessiz hata, gürültülü hatadan tehlikelidir

Projenin bugüne kadar aldığı her ciddi yara sessizdi:

- `make_splits` yanlış yol yazdı → Ultralytics uyardı ama durmadı → model "nesne yok" öğrendi
- kontrol yanlış şeye baktı → yeşil yandı, eğitim patladı
- `max_det` config'de yazıyordu ama geçmiyordu → sessizce 300 kullanıldı
- `yerlesim.py` gamma reddi sonrası koloni düşürüyordu → sayım dağılımı kaydı

Bu yüzden kod her yerde **saydırıyor ve basıyor**: kaç kutu kırpıldı, kaç görüntü
neden elendi, kaç koloni yer bulamadı, havuz sınıf bakımından yanlı mı.
Yeni kod yazarken: *bu işlem sessizce yanlış sonuç üretebilir mi? Üretebiliyorsa
sayacını koy.*

### Desen 3 — Ölçüm, tahminin yerini alır

`imgsz=1280` bir tahmindi → ezber testiyle ölçüldü (2.18).
Değme oranı bir tahmindi → 387 koloni üzerinde ölçüldü (3.12).
`gamma` elle ayarlanabilirdi → kalibre edildi (3.19).
Batch 8 tavanı bir tahmindi → VRAM ölçüldü (2.19).

Ve ölçümün belirsizliği de raporlanıyor: `gamma` için plak düzeyinde önyükleme
güven aralığı hesaplanıp *"mekanizma doğru, sayı gürültü"* diye işaretlendi.

---

## 8. Bulunan 25 madde

17 Ağustos taramasında çıktı. Öncelik sırasıyla.

### 🔴 Sonucu doğrudan bozabilecekler

| # | Dosya | Sorun | Ne yapmalı |
|---|---|---|---|
| 1 | `aug_b_klasik.yaml` | `degrees: 180` — Ultralytics döndürmede kutunun köşelerini döndürüp eksene hizalı yeni kutu çiziyor. Kare kutu 45°'de kenarı **√2 katına** çıkıyor. Oysa koloniler yuvarlak: doğru kutu döndürmede **hiç değişmemeli**. Baseline B şişirilmiş etiketlerle eğitiliyor, doğru etiketli testte ölçülüyor → sakat baseline → difüzyon lehine sahte kazanç. Baseline C'nin `copy_paste` riskinin aynadaki hâli. | `degrees: 0.0` + flip'lere güven, ya da 90°'nin katlarıyla offline döndürme. Hangisi olursa `decisions.md`'ye gerekçesiyle yaz. |
| 2 | `butce.py:209` | `sentetik_ihtiyaci()` isimde `+S` arıyor; ablasyon kolları (`A1_LoRAsiz`, `A2_arka_plan`, `A3_rastgele_yerlesim`) 0 sentetik sayılıyor. Oysa üçü de baştan yeni sentetik küme gerektiriyor. Sayılan 5.03, gerçek 7.28 → **üretim hacmi ~%45 eksik**. | Sentetik payını isimden değil `KOLLAR` tanımından oku. |
| 3 | `butce.py:231` | `+ 3 * args.lora_saat` — karar 3.2'ye göre **4** LoRA. Karar 3.9 zaten `lora_seviye_sayisi` parametresi istemişti, eklenmedi. | `--lora-sayisi` argümanı, varsayılan 4. |
| 4 | `evaluate.py:106` | `zip(images, boxes_all)` — tahminler görüntülere **sıraya göre** eşleniyor. Bir görüntü atlanırsa `zip` sessizce kısa olanda durur ve o noktadan sonraki tüm eşleşmeler kayar. Hata mesajı yok, mAP saçmalar. | `r.path` ile eşle; en azından `assert len(boxes_all) == len(images)`. |
| 5 | `.gitignore` | `runs/` — altında her koşunun `olcum.json` (süre/VRAM/GPU-saat/commit) ve `eval/ozet.json` (mAP, AP, sayım) var. **Makalenin bütün sonuçları sürüm kontrolü dışında.** Birkaç KB, ama 500 GPU-saat sürmeden yeniden üretilemez. | `scripts/topla.py`: tüm `olcum.json` + `ozet.json`'ları izlenen bir `sonuclar/tablo.csv`'ye topla. Grid tablosunu zaten kurman gerekiyor. |

### 🟡 Sessiz, şimdilik etkisiz

| # | Dosya | Sorun |
|---|---|---|
| 6 | `train.py:258` | `**{k: v for k, v in cfg["augment"].items() if v is not None}` — `null` değerler filtrelenip Ultralytics varsayılanı devreye giriyor. Şu an sadece `auto_augment` etkileniyor (detect'te kullanılmıyor, zararsız). Ama biri `mosaic: null` yazarsa augmentation politikası sessizce çöker — karar 2.13'ün tam olarak yasakladığı şey. |
| 7 | `train.py` | `--duman` yardım metni "W&B kapalı" diyor; kodda `wandb` kelimesi geçmiyor. `base.yaml`'daki `logging` bloğu (`wandb: true`, `project: agar-synth`) hiç okunmuyor. `wandb login` yapıldığı an Ultralytics kendi varsayılan proje adına loglar ve duman testleri de oraya karışır. |
| 8 | `base.yaml:66` | `nms_iou: 0.7` hiçbir yerde okunmuyor; `evaluate.py:100` `iou=0.7`'yi sabit yazmış. Config'deki değeri değiştirsen hiçbir şey olmaz. |
| 9 | `analyze_manifest.py:56` | `radius` 0.5·genişlik ile normalize, `yerlesim.py`'de 0.465 ile. Yorum *"plak kareye tam oturuyor"* diyor — yanlış. Gerçek plak kenarı bu birimde **0.930**; kontroller 0.95 ve 1.0'da. Ölçüldü: maks radius 0.876, yani **"plak dışı" kontrolü asla ateşlenemiyor**. Grafiğin ekseni de ("1 = kenar") makaleye yanlış gider. |
| 10 | `analyze_manifest.py` | `--iou-dup 0.5` — demo veride gerçek kutu çiftlerinin **en yüksek IoU'su 0.450**. Pay yalnızca 0.05. Tam veride kalabalık plaklar bu eşiği aşacak ve `mukerrer_supheli.csv` gerçek değen kolonilerle dolacak. Eşiği tahminle değil, gerçek IoU dağılımının p99.9'undan koy. |
| 11 | `make_splits.py:177` | `yazilan[:200]` — etiket varlık kontrolü ilk 200 satırla sınırlı. Tam veride train ~8000 görüntü → **%2.5'i kontrol ediliyor**. `Path.exists()` mikrosaniye sürer, sınırın kalması için sebep yok. |
| 12 | `train.py:112` | `satir[:50]` — aynı desen, liste doğrulamasında. |
| 13 | `convert.py:181` | `if not dst_img.exists(): dst_img.symlink_to(...)` — `exists()` symlink'i takip eder. Kaynak taşınırsa link kırılır, `exists()` False döner, yeniden symlink denenir, `FileExistsError`. Düzeltme: `if not dst_img.is_symlink() and not dst_img.exists()`. |

### 🔵 Doğruluk ve raporlama

| # | Dosya | Sorun |
|---|---|---|
| 14 | `convert.py:102` | `elendi_countable_degil` sayacı **empty** ve **uncountable** görüntüleri karıştırıyor. Makalede ayrı raporlanmalı; `colonies_number` alanı ikisini ayırabilir. |
| 15 | `check_labels.py:105` | Aralık kontrolü merkez ve genişliğin 0–1 arasında olmasına bakıyor; `xc=0.99, bw=0.1` geçer ama kutu 1.04'e uzanır. Ek satır: `if xc - bw/2 < -1e-6 or xc + bw/2 > 1 + 1e-6`. |
| 16 | `evaluate.py:53` | `Path("splits")` sabit göreli yol — yalnızca repo kökünden çalışıyor. `train.py` `KOK = Path(__file__).resolve().parents[1]` kullanıyor; ikisi aynı kuralı izlemeli. |
| 17 | `evaluate.py:189` | `--split train` + eşiksiz → `secilen = None` → `counting_metrics`'te `TypeError`. Ya val gibi eşik seçmeli ya da başta net hata vermeli. |
| 18 | `butce.py:291` | Senaryo satırları (`epoch 150->100`, `imgsz 1280->1024`) `genel`'i ölçekliyor — ama `genel` üretim ve XAI'yi de içeriyor, onlar epoch/imgsz'ye bağlı değil. Kısma kazancı şişik görünüyor; `--butce` ile "sığar" işareti hak etmediği yerde çıkabilir. Satır 288'deki "yan kollar" satırı da `"klasik" in secili` kontrolü yapmıyor. |
| 19 | `test_metrics.py` | `read_yolo_txt` **hiç test edilmiyor**. 33 kontrolün hepsi `ImageAnno`'yu doğrudan piksel kutularıyla kuruyor, dosya okuyucuyu atlıyor. Oysa normalize→piksel dönüşümünün yaşadığı tek yer orası; `W`/`H` yer değiştirse hiçbir test yakalamaz. |
| 20 | `src/generate/` | **Hiç test yok.** `yerlesim.py`'de ters-CDF örnekleme, Strauss ikili araması, lognormal kestirim, önyükleme güven aralığı var. `dogrula` bir karşılaştırma raporu, birim testi değil: yanlış bir quantile interpolasyonu medyanı doğru tutup kuyruğu bozabilir. Yazılabilecekler: bilinen dağılımdan ters-CDF → quantile'lar eşleşiyor mu · `gamma=0` → hiç değme yok · seviye uyuşmazlığı → program duruyor mu · maske beyaz piksel alanı → analitik disk alanına eşit mi. |
| 21 | `requirements.txt` | `ultralytics>=8.3.0` — ama `base.yaml` `yolo26n.pt` ve `cutmix` istiyor, ikisi de **8.4**'te geldi. Kısıt, config'i çalıştıramayacak sürüme izin veriyor. `>=8.4.118` olmalı; Faz 5'te `pip freeze > requirements.lock`. |
| 22 | `kurulum.sh` | `--index-url .../cu124` kuruyor, ama ölçümler `torch 2.13.0+cu130` ile yapıldı. BİDB sunucusunda farklı bir CUDA yapısı kurulur; bölüm 6'nın tablosu iki makineden gelecekse karşılaştırılamaz. Ayrıca batch önerisi (`gb < 10 → 4`) ölçülen değerin altında: karar 2.19'da batch 8 tavan olarak ölçüldü. |
| 23 | `faz2_colab.ipynb` hücre 17 | Resume hücresi `YOLO(...).train(resume=True)` ile `train.py`'yi **tamamen atlıyor** → `olcum.json` hiç yazılmaz: süre yok, VRAM yok, commit yok. Notebook'un kendi başlığı G100 için "hem üst sınır hem süre ölçümü" diyor; bu hücre ölçüm yarısını siliyor. Colab'da kopma kural, istisna değil. `train.py`'ye `--devam` bayrağı eklenmeli. |
| 24 | `faz2_colab.ipynb` | Başlıkta ve 10. bölümde hâlâ **"~85 koşu"** yazıyor; kapsam 13 Ağustos'ta **61**'e indi (karar H.6). |
| 25 | `faz2_colab.ipynb` hücre 4 | `!pip install -q ultralytics ...` — sürüm sabitlenmemiş. Colab her açılışta güncelini çeker; dizüstünde 8.4.118 ile ölçülen süreler karşılaştırılamaz hâle gelir. |

---

## Kapanış notu

Tarama bir şeyi net gösterdi: **14 Ağustos'ta verilen kararlar koda inmemiş.**
Karar 3.2 (4 LoRA), 3.9 (`lora_seviye_sayisi`), 3.18 (A3'ün yeni tanımı) —
üçü de `decisions.md`'de yazılı, üçü de `butce.py` ve notebook'ta eski hâliyle
duruyor.

Bir günlük gecikmede zararsız. Faz 5'te protokolü dondururken ölümcül: dondurduğun
şey gerçekte çalışan şey olmaz.

Öneri: Faz 5 öncesi bir **karar–kod tutarlılık kontrolü**. `decisions.md`'deki her
karar için "bu koda indi mi" sütunu, ve `train.py` başlangıcında config'de olup
koda geçmeyen alanları uyaran bir kontrol.
