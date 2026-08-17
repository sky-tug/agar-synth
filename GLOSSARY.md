# Türkçe → İngilizce sözlük

> 17 Ağustos 2026. Kod tamamen İngilizceye geçiriliyor (public repo).
> `decisions.md` **Türkçe kalıyor** — laboratuvar defteri, makale de Türkçe.
> Bu dosya çeviri boyunca tek doğru kaynak; yeni bir ad gerekirse **önce buraya** yaz.

## Dosya adları

| eski | yeni |
|---|---|
| `src/generate/yerlesim.py` | `src/generate/layout.py` |
| `src/generate/maske.py` | `src/generate/mask.py` |
| `src/generate/test_uretim.py` | `src/generate/test_generate.py` |
| `src/generate/kesif/` | `src/generate/exploration/` |
| `scripts/butce.py` | `scripts/budget.py` |
| `scripts/topla.py` | `scripts/collect.py` |
| `scripts/kurulum.sh` | `scripts/setup.sh` |
| `KOD_HARITASI.md` | `ARCHITECTURE.md` (İngilizce) + Türkçesi kalır |
| `decisions.md` | **değişmiyor**, Türkçe kalıyor |

Değişmeyenler (zaten İngilizce): `convert.py` · `check_labels.py` ·
`analyze_manifest.py` · `make_splits.py` · `eval/metrics.py` · `eval/evaluate.py` ·
`eval/test_metrics.py` · `scripts/train.py`

## Alt komutlar

| eski | yeni |
|---|---|
| `layout uydur` | `layout fit` |
| `layout ornekle` | `layout sample` |
| `layout dogrula` | `layout validate` |
| `mask havuz` | `mask pool` |
| `mask uret` | `mask build` |

## CLI argümanları

| eski | yeni |
|---|---|
| `--seviye` | `--level` |
| `--siniflar` | `--classes` |
| `--liste` | `--list` |
| `--etiketler` | `--labels` |
| `--goruntuler` | `--images` |
| `--cikti` | `--out` |
| `--parametre` | `--params` |
| `--adet` | `--count` |
| `--naif` | `--naive` |
| `--havuz` | `--pool` |
| `--planlar` | `--plans` |
| `--uretilen` | `--generated` |
| `--esik` | `--threshold` |
| `--silme-pay` | `--erase-margin` |
| `--maske-ayri` | `--split-masks` |
| `--duman` | `--smoke` |
| `--duman-dahil` | `--include-smoke` |
| `--kuru` | `--dry-run` |
| `--devam` | `--resume` |
| `--etiket` | `--tag` |
| `--olcum` | `--metrics` |
| `--butce` | `--budget` |
| `--senaryo` | `--scenarios` |
| `--kollar` | `--arms` |
| `--tam-veri` | `--full-size` |
| `--olculen-imgsz` | `--measured-imgsz` |
| `--uretim-s-goruntu` | `--gen-sec-per-image` |
| `--lora-saat` | `--lora-hours` |
| `--lora-sayisi` | `--lora-count` |
| `--sentetik-seed-basina` | `--synth-per-seed` |
| `--xai-s-goruntu` | `--xai-sec-per-image` |

## Çıktı dosyaları ve klasörler

| eski | yeni |
|---|---|
| `runs/<r>/olcum.json` | `runs/<r>/run_metrics.json` |
| `runs/<r>/eval/ozet.json` | `runs/<r>/eval/summary.json` |
| `sinif_ap.csv` | `class_ap.csv` |
| `boyut_ap.csv` | `size_ap.csv` |
| `sayim.csv` | `counting.csv` |
| `conf_egrisi.csv` | `conf_curve.csv` |
| `splits/dagilim.csv` | `splits/distribution.csv` |
| `splits/distribution.csv` sutunlari: `goruntu` / `kutu` | `images` / `boxes` |
| `<data>/viz/<stem>_kutulu.jpg` | `<data>/viz/<stem>_boxed.jpg` |
| `reports/sinif_ozeti.csv` | `reports/class_summary.csv` |
| `reports/boyut_kirilimi.csv` | `reports/size_breakdown.csv` |
| `reports/goruntu_basina_koloni.csv` | `reports/colonies_per_image.csv` |
| `reports/tur_kombinasyonlari.csv` | `reports/species_combinations.csv` |
| `reports/mukerrer_supheli.csv` | `reports/duplicate_suspects.csv` |
| `reports/veri_ozeti.png` | `reports/data_summary.png` |
| `yerlesim_<lvl>.json` | `layout_<lvl>.json` |
| `bg_havuz_<lvl>.json` | `bg_pool_<lvl>.json` |
| `sonuclar/` | `results/` |
| `sonuclar/tablo.csv` | `results/table.csv` |
| `sonuclar/ham/` | `results/raw/` |
| `sonuclar/ozet.txt` | `results/summary.txt` |
| `<out>/planlar/` | `<out>/plans/` |
| `<out>/maske/` | `<out>/masks/` |
| `<out>/arkaplan/` | `<out>/backgrounds/` |
| `<out>/kayit/` | `<out>/provenance/` |

## Fonksiyon ve değişken adları

### `layout.py` (eski `yerlesim.py`)

| eski | yeni |
|---|---|
| `uydur` | `fit` |
| `ornekle` | `sample` |
| `dogrula` | `validate` |
| `tarif` | `recipe` |
| `bir_plak` | `one_plate` |
| `_ters_cdf` | `_inverse_cdf` |
| `_ortusme` | `_overlap_depth` |
| `_degen_oran` | `_touching_rate` |
| `_degen_gb` | `_touching_ci` |
| `_kalibre_gamma` | `_calibrate_gamma` |
| `_etiket_oku` | `_read_labels` |
| `_istatistik` | `_stats` |
| `PLAK_R_ORANI` | `PLATE_R_RATIO` |
| `MAKS_ORTUSME` | `MAX_OVERLAP` |
| `ELIPS_JITTER` | `ELLIPSE_JITTER` |
| `DENEME_SAYISI` | `PLACEMENT_TRIES` |
| `KALIBRE_PLAK` | `CALIB_PLATES` |
| `KDEMET` | `QUANTILES` |

### `mask.py` (eski `maske.py`)

| eski | yeni |
|---|---|
| `havuz` | `pool` |
| `uret` | `build` |
| `maske_uret` | `build_masks` |
| `_plak_maskesi` | `_plate_mask` |
| `_sinif_dagilimi` | `_species_distribution` |
| `SENTETIK_PAY` | `SYNTH_MARGIN` |
| `SILME_PAY` | `ERASE_MARGIN` |
| `SILME_BULANIK` | `ERASE_BLUR` |

### `train.py`

| eski | yeni |
|---|---|
| `derin_birlestir` | `deep_merge` |
| `gpu_bilgisi` | `gpu_info` |
| `liste_yolu` | `list_path` |
| `liste_dogrula` | `validate_list` |
| `dataset_yaml_yaz` | `write_dataset_yaml` |
| `AUGMENT_NULL_IZINLI` | `AUGMENT_NULL_ALLOWED` |
| `KOK` | `ROOT` |

### `budget.py` · `collect.py`

| eski | yeni |
|---|---|
| `biçim` | `fmt_hours` |
| `ANA_GRID` | `MAIN_GRID` |
| `KLASIK` / `KLASIK_GENIS` | `CLASSIC` / `CLASSIC_WIDE` |
| `MIKTAR` / `MIKTAR_GENIS` | `AMOUNT` / `AMOUNT_WIDE` |
| `ABLASYON` | `ABLATION` |
| `IKINCI_DETEKTOR` | `SECOND_DETECTOR` |
| `KOLLAR` / `VARSAYILAN_KOLLAR` | `ARMS` / `DEFAULT_ARMS` |
| `topla` | `collect` |
| `satirlastir` | `to_rows` |
| `_oku` | `_read_json` |
| `SUTUNLAR` | `COLUMNS` |

### Test yardımcıları

| eski | yeni |
|---|---|
| `kontrol` | `check` |
| `kutu` | `box` |
| `GECTI` / `KALDI` | `PASSED` / `FAILED` |
| `ornek_par` | `sample_params` |

## Config anahtarları (`base.yaml`)

| eski | yeni |
|---|---|
| `eval.ana_metrik` | `eval.primary_metric` |
| `eval.anlamli_fark_esigi` | `eval.significant_diff_threshold` |
| `eval.boyut_araliklari` | `eval.area_ranges` |
| `eval.raporlama.grafik` | `eval.reporting.plot` |
| `eval.raporlama.tablo` | `eval.reporting.table` |
| `seeds.ana_grid` | `seeds.main_grid` |
| `seeds.yan_kollar` | `seeds.side_arms` |
| `xai.aktif` | `xai.enabled` |
| `xai.yontem` | `xai.methods` |
| `xai.konfigurasyonlar` | `xai.configs` |
| `xai.n_goruntu` | `xai.n_images` |
| `xai.metrikler` | `xai.metrics` |
| `logging.olculecek` | `logging.measured_fields` |

Değişmeyenler: `model` · `data` · `train` · `augment` · `imgsz` · `epochs` ·
`patience` · `batch` · `optimizer` · `max_det` · `nms_iou` · tüm augment alanları.

## Değişmeyecek olanlar

- **Sınıf adları veri:** `S.aureus`, `B.subtilis`, `P.aeruginosa`, `E.coli`,
  `C.albicans` — AGAR'ın JSON'undan geliyor, dokunulmaz.
- `CLASS_ORDER` sırası (karar 1.1) — ad İngilizce, sıra sabit.
- `splits/`, `data/processed/`, `runs/`, `configs/` — zaten İngilizce.
- `decisions.md` — Türkçe kalıyor.

## Kurallar

1. **Yarım bırakma yok.** Bir dosya çevrildiyse içinde tek Türkçe tanımlayıcı
   kalmaz.
2. **Her modülden sonra testler koşulur.** `test_metrics.py` + `test_generate.py`
   yeşil değilse devam edilmez.
3. `decisions.md` içindeki dosya/komut adları güncellenir (metin Türkçe kalır) —
   yoksa kararlar var olmayan dosyalara atıf yapar.
4. Yorumlar İngilizce ama **gerekçeyi kaybetmeden**. "silent failure" uyarıları
   bu deponun en değerli kısmı; kısaltılmaz.
