# Türkçe LLM — Colab + Kaggle Relay Planı

> Durum: PLAN. Kod yazılmadı. Önce bu belge onaylanacak.

---

## 0. Önce iki yanlış varsayımı düzeltelim

**1) Kaggle'daki "2x T4 32GB" tek havuz DEĞİL.**
İki ayrı 16GB kart. Model paralelliği yazmadığın sürece (bu ölçekte yazmaya değmez)
model **16GB'a sığmak zorunda**. İkinci kart sana *hız* verir (DDP = veri paralelliği),
*kapasite* vermez.

İyi haber: bu yüzden Colab'da çalışan config Kaggle'da da çalışır. Relay bu sayede
mümkün — tek config, iki platform.

**2) "450M sığar mı" ile "450M'i eğitmeye gücün yeter mi" ayrı sorular.**
- Sığar mı? **Evet.** 440M + AdamW-8bit + grad checkpointing + tied embeddings
  → ~6-7GB. 16GB'da rahat.
- Yetişir mi? **Chinchilla'da hayır.** Aşağıdaki tabloya bak.

---

## 1. Colab vs Kaggle — net süreler

| | **Kaggle** | **Colab (free)** |
|---|---|---|
| Haftalık kota | **30 saat, garanti, sayaç görünür** | ~15-30 saat, **dinamik, yayınlanmıyor**, kullandıkça kısılır |
| Session max | 12 saat | 12 saat (ama free'de tipik 3-6) |
| Idle kill | yok (commit modunda) | **90 dk etkileşim yoksa atar** |
| Tarayıcı kapatabilir misin? | **EVET** — "Save & Run All (Commit)" arka planda çalışır | **HAYIR** — sekme kapanırsa gider |
| GPU | 2x T4 (16GB+16GB) veya P100 | 1x T4 16GB, garanti değil |

**Sonuç:** Kaggle ana at, Colab yedek/yan koşu.
Kaggle'ın commit modu (tarayıcı kapalı, 12 saat arka plan) Colab'da yok — bu tek
başına Kaggle'ı 2 kat değerli yapıyor.

**Ve evet: "durdur → kaydet → devam et" zaten mecburi tasarım.** Seçenek değil.
12 saatte kesin ölecek, Colab'da 3. saatte de ölebilir. Sistem buna göre kurulacak.

---

## 2. Acı tablo — ne kadar sürer

Varsayımlar: T4 fp16 peak 65 TFLOPS, gerçekçi %20-30 MFU → **~10 TFLOPS efektif**,
grad checkpointing dahil. FLOP = 6·N·D. Chinchilla = 20 token/param.
Kaggle 30h × 2 GPU × 0.85 (PCIe all-reduce vergisi, NVLink yok) + Colab ~15h
= **~66 efektif T4-saat / hafta**.

| Model | Chinchilla token | Gereken T4-saat | **Süre** |
|---|---|---|---|
| 135M | 2.2B | ~40 | **< 1 hafta** |
| **290M** | 5.8B | ~280 | **~4 hafta** |
| 440M | 8.8B | ~640 | **~10 hafta** |

> ⚠️ Bu sayılar **tahmini MFU'ya** dayanıyor. İlk iş 30 dakikalık benchmark
> çalıştırıp gerçek tok/s'i ölçmek. Boyut kararı ölçümden **sonra** verilecek.

**Muon optimizer** bu ölçekte (130M-300M) AdamW'ye göre ~1.3-2x sample efficiency
veriyor. Yani 290M ~3 haftaya, 440M ~6-7 haftaya inebilir. Ücretsiz kaldıraç.

---

## 3. Asıl sorun: "dil öğreniyor, bilgi öğrenmiyor"

Bu **parametre sayısı sorunu değil.** 100M → 450M seni kurtarmaz. Sebepler:

### a) Kapasite — DÜZELTME (önceki hesabım fazla karamsardı)

Önce yanlışı düzeltelim: "110MB bilgi vs 1GB Wikipedia metni" karşılaştırması
elma-armut. Ham metin çok fazla tekrar içerir; metnin *bilgi* içeriği byte
sayısından kat kat azdır.

Makalenin kendi kalibrasyonu: **7B model = 14 Gbit ≈ İngilizce Wikipedia +
ders kitaplarının toplamından fazla.** Aşağı ölçekleyelim:

| Model | Bilgi kapasitesi | Wikipedia+kitap'a oranı | Kaba gerçek sayısı |
|---|---|---|---|
| 100M | 200 Mbit | ~%1.4 | ~2 milyon |
| **290M** | **580 Mbit** | **~%4** | **~5 milyon** |
| 440M | 880 Mbit | ~%6 | ~8 milyon |

**Yani 290M'e milyonlarca gerçek sığar. Kapasite bağlayıcı kısıt DEĞİL.**
"Türkiye'nin başkenti Ankara" için 290M fazlasıyla yeterli — 100M bile yeter.

**Bağlayıcı kısıt: token bütçesi ve maruz kalma sayısı.** Parametre değil.

### b) Maruz kalma sayısı
Bir bilginin oturması için ~**1000 exposure** gerekiyor. Ham web verisinde bir
gerçek 1-2 kez geçer. Modelin bilgi öğrenmemesinin **1 numaralı sebebi bu.**

### c) En büyük kaldıraç: paraphrase augmentation
Aynı gerçeği farklı cümlelerle tekrar tekrar yazmak, biyografik gerçek
ezberlemeyi **%9.7 → %96.6** çıkarıyor. Model boyutu değil — **veri işleme.**

### d) QA formatını pretrain'e karıştır
Yoksa bilgi ağırlıklarda durur ama model onu *çıkaramaz*. Fine-tune'da eklemek geç.

### e) Sadece sentetik veri tuzağı
Cosmopedia/phi tarzı saf sentetik ile eğitilen modeller **bilgi
benchmarklarında (TriviaQA) düşük** kalıyor. Karışım gerekli:
gerçek ansiklopedik metin **+** onun paraphrase/QA augmentasyonu.

---

## 4. Mimari — "Qwen3.5 0.8B ne kullanıyor?"

**Cevap: hibrit lineer attention.** Qwen3.5-0.8B:
- 24 katman, hidden 1024, FFN 3584, RoPE dim 64, **262K context**
- Yerleşim: `6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`
- GDN: 16 QK head + 16 V head, head_dim 128
- Gated Attention: 8 Q head / 2 KV head, head_dim 256

**Ama bunu kopyalama. Üç sebep:**

1. **Hibrit attention bir uzun-context/inference optimizasyonu, kalite hilesi değil.**
   Raschka'nın özeti net: *kısa girdilerde yoğun attention'ın avantajı beklenmiyor;
   lineer ve hibrit modeller kısa-context işlerde transformer ile denk veya çok az
   iyi.* Sen 1024-2048 seq ile eğiteceksin → GDN'nin kazandıracağı şey **yok**.
   262K context için var o mekanizma; senin öyle bir derdin yok.

2. **T4'te kernel yok.** GDN, `flash-linear-attention` (Triton) kernellerine bağlı.
   SM75 (T4) desteği **deneysel**, kerneller B200 için tune edilmiş. Kaggle
   kotanı debug'a yakarsın.

3. **0.8B'yi iyi yapan şey mimari değil, 36T+ token.** Sen ~4B vereceksin.
   **9000 kat az.** Mimariyle kapatılacak bir fark değil bu.

### Alacaklarımız (hepsi ucuz, T4'te sorunsuz, gerçek fayda)
| Ne | Neden |
|---|---|
| RMSNorm + pre-norm | standart, stabil |
| RoPE | standart |
| SwiGLU (8/3·d) | standart |
| GQA (kv = q/4) | eğitimde az, inference'de büyük kazanç |
| **QK-Norm** | Qwen3'ün eklediği; **fp16 stabilitesi için kritik**, bedavaya yakın |
| **Tied embeddings** | 32k×1024 = 33M param geri kazanır — bu boyutta şart |
| **z-loss** (küçük katsayı) | fp16'da logit patlamasını engeller |
| **Muon optimizer** | 130M-300M'de en büyük tek kazanç, ~1.3-2x sample efficiency |
| **WSD scheduler** (cosine değil) | ⭐ aşağıya bak |
| value residual, LayerNorm scaling | IMU-1'den, ucuz, opsiyonel |

**WSD (Warmup-Stable-Decay) neden kritik:** stable fazda checkpoint alıp
**sonradan devam edebiliyorsun**, en sonda decay yapıyorsun. Cosine'de toplam adım
sayısını en baştan sabitlemek zorundasın — kotan belirsizken bu felaket.
WSD tam senin kesintili/belirsiz düzenine göre.

### Almayacaklarımız
- ❌ Gated DeltaNet / hibrit — T4'te kernel yok, kısa contextte fayda yok
- ❌ MoE — bu ölçekte veri açlığı yapar, VRAM mantığı tutmaz
- ❌ MTP (multi-token prediction) — karmaşık, marjinal
- ⚠️ `torch.compile` — T4'te genelde kazandırmıyor, **ölç** sonra karar ver

### Referans nokta: IMU-1 (Şubat 2026)
**430M parametre, 72B token** ile 4T token'la eğitilmiş modellere denk sonuç.
Tam senin boyutun. Reçetesi: QK-norm, per-head gating, value residuals,
LayerNorm scaling, NorMuon + cautious weight decay, muP, 3 aşamalı schedule + EMA.

Dürüst not: **72B token senin bütçende ~2 yıl.** SOTA reçete bile compute
problemini çözmüyor. Mimari hileler kaliteyi artırır, bütçeyi büyütmez.

---

## 5. Tokenizer — gizli 2x

GPT-2/Llama vocab'ı Türkçeyi paramparça ediyor (sondan eklemeli dil:
"evlerinizden" → 5-6 token). **Kendi 32k Türkçe BPE'n**:
- token başına daha çok anlam → veri bütçen efektif **1.5-2x** büyür
- vocab 151k yerine 32k → embedding'de ~120M param tasarrufu

Bu, Türkçe-only gitmenin **en güçlü argümanı**. İki dil = daha büyük vocab +
bölünmüş kapasite.

---

## 6. Türkçe mi, Türkçe+İngilizce mi

**Veri sıkıntın yok.** Mevcut Türkçe pretraining verisi:
- FineWeb2 `tur_Latn` + CulturaX `tr` → birlikte **96-130B token**
- Senin ihtiyacın: **4-6B token**

Yani "veri yetmez" diye İngilizce eklemene **gerek yok**.

**Öneri: %90 Türkçe + %10 İngilizce** (teknik terim, kod, formül tutarlılığı için).
Saf Türkçe de tamamen geçerli.

---

## 7. Sistem mimarisi (relay)

```
   [ Yerel / 3060 Ti ]          [ HuggingFace Hub ]         [ Kaggle 2xT4 ]
   tokenizer eğit        ──►    dataset repo         ──►    torchrun DDP
   veri hazırla (CPU)           (.bin memmap)               12h commit run
   sentetik augment                  ▲  │                        │
                                     │  │                        ▼
                                     │  └──────────────►  ckpt push (15dk)
                                     │                           │
                              [ Colab 1xT4 ] ◄──────── ckpt pull ┘
                              boşluk doldurma
```

**Checkpoint bus = HuggingFace Hub.** Google Drive DEĞİL:
- Drive 15GB, Colab'da persistent disk yok
- her devirde GB'larca indir/yükle → session süreni yer
- Hub her iki platformda da hızlı, versiyonlu, ücretsiz

**Checkpoint kuralları:**
- Her **15 dakikada** atomik kayıt (tmp yaz → rename), son 2'yi tut
  (Colab 12. saatte değil, 3. saatte de ölebilir)
- Kayıt içeriği: model + optimizer + **scheduler step** + **GradScaler state**
  + RNG + **veri pozisyonu**

### ⚠️ En sinsi bug — şimdiden söylüyorum
Resume'da veri loader baştan başlarsa, model ilk tokenları **20 kez** görür ve
corpus'un sonunu **hiç görmez**. Ve **loss normal görünür.** Aylar sonra fark edersin.

**Çözüm:** batch'i step numarasından deterministik üret
(`seed = base_seed + step`). Resume = adım numarasını geri koy, veri sırası
otomatik doğru yere gelir. nanoGPT paterni.

### Kaggle özel
- Script'i dosyaya yaz, `!torchrun --nproc_per_node=2 train.py` ile başlat
  (notebook hücresinde DDP spawn etme, patlar)
- `gradient_as_bucket_view=True`, `find_unused_parameters=False`
- "Save Version → Save & Run All" ile commit → tarayıcı kapat

---

## 8. Yol haritası

| Aşama | İş | Süre | GPU kotası |
|---|---|---|---|
| **0** | Tokenizer eğit (32k TR BPE) + veri indir/temizle/tokenize → .bin | 2-4 gün | **0** (CPU işi, 3060 Ti'de veya laptop'ta) |
| **0.5** | **Benchmark: gerçek tok/s ölç, 3 boyut için süre projeksiyonu** | 30 dk | ~1 saat |
| **1** | **Boyut kararı** — ölçüme göre | — | — |
| **2** | 135M pilot koşu: relay sistemini test et, bug'ları avla | 3-4 gün | ~10 saat |
| **3** | Ana koşu (WSD stable faz) | 3-6 hafta | tüm kota |
| **4** | Bilgi-yoğun faz: ansiklopedik + paraphrase + QA, 3-4 epoch | 1 hafta | 15 saat |
| **5** | WSD decay + EMA | 2-3 gün | 8 saat |
| **6** | SFT (Türkçe instruction) | 1 gün | 3 saat |

**Aşama 2'yi atlama.** Relay sisteminin bug'ını 135M'de 3 günde bulursun,
440M'de 5. haftada bulursun.

---

## 9. Dürüst alternatif — C rotası

Amacın *"bilen bir model"* ise doğru cevap sıfırdan eğitmek değil:

**Qwen3.5-0.8B veya Qwen3-0.6B üzerine Türkçe continued pretraining + LoRA.**
- T4'e rahat sığar, **gün**ler sürer, ay değil
- Zaten 36T token'lık dünya bilgisi var — sen sadece Türkçeyi güçlendirirsin
- Sonuç: gerçekten işe yarayan bir model

Amacın *"sıfırdan LLM eğitmeyi öğrenmek"* ise → **A rotası** (bu plan). O da
tamamen meşru, ama çıktının ne olacağı konusunda beklenti net olsun.

**Benim önerim: ikisini birden yap.**
- A rotasını **290M** ile yap → öğrenmek + kendi modelin
- C rotasını paralel yap → günlük kullanacağın model
- İkisi aynı tokenizer/veri hattını paylaşmaz ama aynı veri hazırlığını paylaşır

---

## 10. Karar verilmesi gerekenler (senden)

1. **Boyut**: 135M / **290M (önerilen)** / 440M — *(benchmark'tan sonra kesinleşir)*
2. **Dil**: saf Türkçe / **%90 TR + %10 EN (önerilen)**
3. **Bilgi alanı**: 110MB tavanı var — hangi alan? (genel kültür / tarih / hukuk /
   sağlık / yazılım / …) Bu, veri karışımını belirler.
4. **Rota**: sadece A / **A + C (önerilen)**
5. HF hesabı var mı? (checkpoint bus için lazım)

---

## Kaynaklar
- Kaggle kotası: https://www.kaggle.com/general/108481
- Colab limitleri: https://research.google.com/colaboratory/faq.html
- Knowledge Capacity Scaling Laws: https://arxiv.org/abs/2404.05405
- IMU-1 (430M / 72B token): https://arxiv.org/abs/2602.02522
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Qwen3.5-0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Hybrid Attention (Raschka): https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/
- Muon: https://kellerjordan.github.io/posts/muon/
- FineWeb2: https://arxiv.org/abs/2506.20920
- Cosmopedia: https://github.com/huggingface/blog/blob/main/cosmopedia.md
