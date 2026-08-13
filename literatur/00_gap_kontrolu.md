# Research Gap Doğrulaması — ilk tarama

> 13 Ağustos 2026. Hoca gap cümlesini yazmış ama "literatür taramasıyla
> doğrulandıktan sonra" notu düşmüş. Bu dosya o kontrolün ilk turu.
>
> **Sonuç: birebir aynı çalışma bulunamadı.** Ama üç makale okunmadan gap
> cümlesi yazılmamalı — aşağıda işaretli.

---

## Bizim iddia ettiğimiz gap

Üç bileşenin **birlikte** yapılmamış olması:

1. **Alan:** agar plağı üzerinde koloni **tespiti** (segmentasyon değil, sayım odaklı)
2. **Yöntem:** maske kontrollü difüzyon inpainting → etiket üretimden önce elde, maliyeti sıfır
3. **Soru:** sentetik verinin gerçek veriyi **ne kadar ikame ettiğinin** nicel ölçümü —
   iç içe alt örneklemler, kontrol grupları, ikame eğrisi

Tek tek her biri literatürde var. Üçünün kesişimi bulunamadı.

---

## 🔴 Okunmadan gap cümlesi yazılmayacak — 3 makale

| # | Çalışma | Neden tehdit | Ne kontrol edilecek |
|---|---|---|---|
| 1 | **Denoising diffusion probabilistic models for generation of realistic fully-annotated microscopy image datasets** — PLOS Comput Biol, 2024 | Metodolojik olarak en yakın: difüzyon + **tam etiketli** mikroskopi veri seti üretimi | Etiketi nasıl üretiyor (bizim gibi maske önce mi)? İkame oranı ölçüyor mu, yoksa sadece "sentetik veri işe yarıyor" mu diyor? Tespit mi segmentasyon mu? |
| 2 | **Deep generative modeling of annotated bacterial biofilm images** — npj Biofilms and Microbiomes, 2025 | **Bakteri** + **etiketli** + üretken model. Alan olarak en yakın | Biyofilm mi koloni mi (farklı görüntüleme, farklı problem)? Difüzyon mu GAN mı? Aşağı akış tespit görevi var mı? |
| 3 | **Diffusion-Based Synthetic Brightfield Microscopy Images for Enhanced Single Cell Detection** — arXiv:2512.00078 | Difüzyon + mikroskopi + **tespit**. Üç bileşenden ikisi | Tek hücre vs koloni. İkame eğrisi çiziyor mu? Kontrol grubu var mı? |

Bu üçü gap'i **kapatmıyor** görünüyor (hiçbiri agar plağı + ikame eğrisi
kombinasyonunda değil) ama okunmadan emin olunamaz.

---

## 🟡 Doğrudan öncül — makalede mutlaka konumlanılacak

**Generation of microbial colonies dataset with deep learning style transfer**
— Scientific Reports, 2022 (Nature)

AGAR ekibinin **kendi** çalışması. Aynı veri seti, aynı motivasyon, ama
**stil transferi** ile. Öneri dokümanında "bu veri setinde denenmiş" diye
reddedilen alternatif tam olarak bu.

**Bu makale bizim için hem tehdit hem fırsat:**
- Tehdit: "bu zaten yapıldı" itirazının kaynağı
- Fırsat: **doğrudan karşılaştırma noktası.** Onlar stil transferi, biz difüzyon.
  Onlar etiketi nasıl çıkarmış? İkame oranı ölçmüşler mi? Ölçmemişlerse
  bizim katkımız net: *"aynı veri setinde, aynı soruyu nicel olarak soran ilk çalışma."*

**Giriş bölümünün omurgası bu makaleye karşı konumlanmak olacak.**

---

## 🟢 Yöntem atası — sorun değil, alıntılanacak

| Çalışma | İlişki |
|---|---|
| **Outline-Guided Object Inpainting with Diffusion Models** — arXiv:2402.16421 | Bizim yöntemin genel-alan atası: nesnenin dış hattı önceden verilip inpainting yapılıyor → etiket bedava. Farkımız: mikrobiyoloji alanı + LoRA alan uyarlaması + ikame eğrisi |
| **SmartBrush: Text and Shape Guided Object Inpainting** — CVPR 2023 | Şekil kontrollü inpainting, temel referans |
| **Exploring the potential of synthetic data to replace real data** — arXiv:2408.14559 | İkame sorusunun genel-alan versiyonu. Bizim sorunun literatürdeki karşılığı |
| **The Impact of Synthetic Data on Object Detection Model Performance** — arXiv:2510.12208 | Sentetik/gerçek karşılaştırma protokolü için referans |

---

## 🟢 Aynı veri seti, çakışma yok

| Çalışma | Neden çakışmıyor |
|---|---|
| **AGAR: a microbial colony dataset for deep learning detection** — arXiv:2108.01234 | Veri setinin kendisi. Baseline referansımız (lower-res mAP %59,4) |
| **Colony Grounded SAM2** (2025) | AGAR + Grounding DINO/SAM2, **sentetik üretim yok**, tamamen gerçek veri |
| **Assessing microbial colony counting: a deep learning approach with the AGAR image dataset** — Neurocomputing 2025 | Sayım odaklı, gerçek veri. Sayım metriklerimiz (MAE/sMAPE) için referans |
| **Bacterial Colony Counting and Classification System Based on Deep Learning** — Applied Sciences 2026 | Güncel sayım çalışması, referans |

---

## Sıradaki adımlar

1. Zotero kur, yukarıdaki 12 kaydı içe aktar → kütüphanenin çekirdeği
2. 🔴 işaretli üç makaleyi oku, bu dosyanın altına birer paragraf not düş
3. Sci Rep 2022 (stil transferi) makalesini **dikkatle** oku — giriş bölümü buna
   karşı yazılacak
4. Gap cümlesini yaz, hocaya onaya gönder
5. Hedef 30+ kaynak: outline bölüm 2'nin dört alt başlığına dağıt
   (difüzyon · sentetik veri ile eğitim · mikrobiyal koloni tespiti · az etiketli öğrenme)

---

## Not

Bu tarama arama motoru sonuçlarına dayanıyor, makalelerin tam metinleri
okunmadı. **Gap cümlesi bu dosyaya dayanarak yazılamaz** — 🔴 üçlüsü
okunduktan sonra yazılır.
