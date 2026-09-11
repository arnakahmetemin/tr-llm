# Yarın Sabah Çalıştırma Kılavuzu

## Kurulum (3060 Ti PC'sinde, bir kere)
```bash
pip install torch numpy huggingface_hub      # CUDA'lı torch: pytorch.org/get-started
git clone <bu klasör>  &&  cd AItrain
```

---

## ADIM 0 — TOKENIZER (en baştaki ve tek yönlü karar)

> Tokenizer'ı değiştirirsen bütün `.bin`'ler çöp olur. Bu yüzden **ilk iş bu.**
> Altın kural: tokenizer'ı, eğiteceğin veriyle **aynı oranlarda** bir örnek
> üzerinde eğit.

```bash
# 1) Her kaynaktan küçük bir örnek topla (birkaç yüz MB yeter)
python prepare_data.py --source wiki    --target 3e7 --out /tmp/s_wiki.bin ...  # veya ham .txt
# 2) Örnekleri final karışım oranlarıyla ver
python train_tokenizer.py --out tokenizer.json --vocab 32768 --budget_mb 500 \
    --input "ornek_web.txt:0.55,ornek_wiki.txt:0.25,kendi.txt:0.12,ornek_kod.txt:0.08"
# 3) Kontrol et
python check_tokenizer.py tokenizer.json
```

**Ölçülmüş sonuç** (16k vocab, gerçek tr.wikipedia + gerçek Python kaynağı):

| tokenizer | Türkçe | Kod |
|---|---|---|
| %100 Türkçe | 4.07 byte/token | 2.11 byte/token |
| **%92 Türkçe + %8 kod** | **4.04** (−%0.9) | **2.87** (+%36) |

Kod eklemek Türkçeye **%0.9**'a mal oluyor, koda **%36** kazandırıyor.
Neredeyse bedava — kod kullanacaksan tokenizer örneğine mutlaka koy.

Not: iki tokenizer da kodu ve `İstanbul, ığüşöç`'ü **tam geri çözüyor**.
ByteLevel BPE'de "bozulma" diye bir şey yok, sadece verimlilik farkı var.

## ADIM 1 — Ortam sağlam mı? (30 saniye)
```bash
python tests/smoke_test.py
```
17 test: model, Muon, QK-Norm, z-loss, chunked loss, loader determinizmi,
WSD, checkpoint/resume. **Hepsi geçmeden ilerleme.**

## ADIM 2 — Mevcut .bin'ini tanı (1 dakika)
```bash
python data.py data/train.bin uint16
```
Çıktıdan **iki şeye** bak:
- `vocab en az N` → bu sayıyı ADIM 4'te `--vocab_size` olarak vereceksin
- `token: X` → toplam token sayın

Tokenizer kalitesini de ölçmek istersen `data.py` içindeki `inspect()`'e
tokenizer'ını geçir; **byte/token 3.5-4.5 arası iyi, 3.0'ın altı zayıf.**

## ADIM 2.5 — Veriyi doyur (FineWeb-2 + Wikipedia)
Dataset adları **doğrulandı** (2026-09 itibarıyla çalışıyor):
`wikimedia/wikipedia::20231101.tr` ve `HuggingFaceFW/fineweb-2::tur_Latn`,
ikisinde de metin kolonu `text`.

```bash
pip install datasets tokenizers

# Kendi .txt'in (kod + doclar + notlar)
python prepare_data.py --source local --input kendi_veri.txt \
    --target 1e8 --out data/kendi.bin --tokenizer tokenizer.json --eos_id <EOS>

# Wikipedia ÖNCE -- küçük, bilgi çekirdeği, hızlı biter
python prepare_data.py --source wiki --target 1e9 \
    --out data/wiki.bin --tokenizer <tokenizer.json> --eos_id <EOS>

# FineWeb-2 tr -- büyük, akıcılık için. Saatler sürer, KESİLEBİLİR.
python prepare_data.py --source fineweb --target 7e9 \
    --out data/fineweb.bin --tokenizer <tokenizer.json> --eos_id <EOS>
```
Yarıda kesersen aynı komutla kaldığı yerden devam eder (`.progress.json`).
İndirme yok, **streaming** -- 100GB'lık dataset diske inmez.

⚠️ Türkçe Wikipedia'da **663k madde** var; gerçekçi verim **~400-500M token**.
1e9 hedefi koy, kaynak tükenince kendisi durur ve gerçek sayıyı basar.

### Karışım oranı
```bash
--train_bin "data/fineweb.bin:0.6,data/wiki.bin:0.3,data/train.bin:0.1"
```
Ağırlık = token PAYI, kaynağın boyutu değil. Wikipedia küçük ama %30 pay
alınca **çok daha fazla epoch görür** -- "1000 exposure" kuralı pratikte bu.

## ADIM 3 — ÖLÇ, sonra karar ver (2 dakika)
```bash
python train.py --preset m290 --train_bin data/train.bin \
                --vocab_size <ADIM2'DEKİ> --bench
```
60 adım koşar, gerçek `tok/s` ve MFU'yu basar, sonra **her preset için**
kaç GPU-saat ve kaç hafta süreceğini hesaplar. Tahmine göre değil, **ölçüme
göre** boyut seç.

## ADIM 4 — Başlat
```bash
# RTX 3060 Ti (8GB)
python train.py --preset m290 --train_bin data/train.bin \
  --vocab_size <N> --micro_bs 4 --grad_accum 32 \
  --total_tokens 5.9e9 --out_dir checkpoints --hub_repo kullanici/tr-290m
```
`micro_bs × grad_accum` çarpımını sabit tut (=128). OOM alırsan
`--micro_bs 2 --grad_accum 64`.

---

## Gürültü çözümü (ablan için) 🔇
```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 130          # 200W -> 130W
```
Ampere'de %65 güç limiti tipik olarak **performansın ~%90'ını korur**, fan
devri ve gürültü belirgin şekilde düşer. Windows'ta MSI Afterburner ile
power limit %65 + custom fan curve aynı işi görür.

---

## Kaggle 2xT4 (30 saat/hafta, arka planda)
```python
# notebook hücresi -- DDP'yi hücrede spawn ETME, torchrun kullan
!torchrun --nproc_per_node=2 train.py --preset m290 \
   --train_bin /kaggle/input/tr-corpus/train.bin --vocab_size <N> \
   --micro_bs 12 --grad_accum 6 --total_tokens 5.9e9 \
   --out_dir /kaggle/working/ckpt --hub_repo kullanici/tr-290m
```
Sonra **Save Version → Save & Run All (Commit)** → tarayıcıyı kapat,
12 saat arka planda çalışır.

## Colab (1xT4, yedek)
Aynı komut, `--micro_bs 12 --grad_accum 11`, tek GPU.
⚠️ 90 dakika etkileşim olmazsa atar — sekmeyi açık tut.

---

## Relay: makineler arası geçiş
`--hub_repo` verdiysen checkpoint saatte bir HF Hub'a gidiyor.
Başka makinede devam:
```bash
huggingface-cli download kullanici/tr-290m ckpt.pt --local-dir checkpoints
python train.py --preset m290 ... --resume auto
```
`--resume auto` = `checkpoints/ckpt.pt` varsa oradan devam.
**Veri sırası adım numarasından türediği için** makine ve GPU sayısı
değişse bile model aynı veriyi iki kez görmez.

---

## Disk ihtiyacı
| | 290M |
|---|---|
| checkpoint (model+Muon+AdamW) | ~2.4 GB |
| `keep_last=2` | ~5 GB |
| 5.9B token .bin (uint16) | ~11.8 GB |
| **toplam** | **~17 GB** |

## Sorun giderme
| Belirti | Çözüm |
|---|---|
| CUDA OOM | `--micro_bs` yarıya, `--grad_accum` iki katına |
| loss NaN (T4) | QK-Norm ve z-loss açık mı? `--micro_bs` düşür |
| loss düşmüyor | `|g|` normuna bak; sürekli clip'e çarpıyorsa `muon_lr` düşür |
| çok yavaş | `--bench` ile MFU'ya bak; %15'in altıysa `--no_grad_ckpt` dene |

## Erken bitirmek istersen (WSD'nin asıl faydası)
Hedefi 9B koydun ama uzadı mı? **Baştan başlamana gerek yok** -- daha küçük
bir hedefle resume et, scheduler decay fazına kendiliğinden girer:
```bash
python train.py ... --total_tokens 6e9 --resume auto
```
Cosine kullansaydık bu mümkün olmazdı; toplam adımı en baştan sabitlemen
gerekirdi. WSD'yi bu yüzden seçtik.
