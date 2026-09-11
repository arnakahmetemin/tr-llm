"""Model + eğitim konfigürasyonu.

Donanım gerçeği:
  RTX 3060 Ti : 8GB,  Ampere SM86 -> bf16 + FlashAttention-2, KOTA YOK  <- ANA AT
  Kaggle 2xT4 : 2x16GB (tek havuz DEĞİL), Turing SM75 -> fp16, 30sa/hafta
  Colab 1xT4  : 16GB, ~15sa/hafta, 90dk idle-kill                        <- yedek

Bağlayıcı kısıt 3060 Ti'ın 8GB'ı. Her şey ona göre ayarlı; 16GB'lık T4'te
micro_bs'i büyütüp grad_accum'u küçültmek yeterli (efektif batch sabit kalsın).
"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 32768        # <-- KENDİ TOKENIZER'INA GÖRE DEĞİŞTİR
    n_layer: int = 22
    n_head: int = 16
    n_kv_head: int = 8             # GQA 2:1
    d_model: int = 1024
    ffn_mult: float = 8 / 3
    ffn_multiple_of: int = 128
    seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    qk_norm: bool = True           # fp16 stabilitesi -- T4'te kapatma
    z_loss: float = 1e-4           # logit patlamasına karşı
    loss_chunk: int = 256          # chunked CE (8GB için kritik); 0 = kapalı

    @property
    def d_head(self): return self.d_model // self.n_head

    @property
    def d_ffn(self):
        h = int(self.ffn_mult * self.d_model)
        m = self.ffn_multiple_of
        return m * ((h + m - 1) // m)

    def n_params(self, embedding=True):
        d, L, kv = self.d_model, self.n_layer, self.n_kv_head * self.d_head
        attn = 2 * d * d + 2 * d * kv
        qkn = 2 * self.d_head if self.qk_norm else 0
        per = attn + 3 * d * self.d_ffn + 2 * d + qkn
        tot = L * per + d
        if embedding:
            tot += self.vocab_size * d * (1 if self.tie_embeddings else 2)
        return tot

    def vram_estimate_gb(self, micro_bs=4, muon=True):
        """Kaba VRAM tahmini (GB). Grad checkpointing + chunked loss varsayar."""
        N = self.n_params()
        states = 2 if muon else 3          # Muon: 1 momentum; AdamW: 2 moment
        core = N * 4 * (2 + states) / 1e9  # fp32 param + grad + optim state
        act = micro_bs * self.seq_len * self.d_model * self.n_layer * 2 * 2 / 1e9
        logits = micro_bs * (self.loss_chunk or self.seq_len) * self.vocab_size * 4 * 3 / 1e9
        return core + act + logits + 0.6   # +0.6 cuda context/fragmentasyon


@dataclass
class TrainConfig:
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    bin_dtype: str = "uint16"

    micro_bs: int = 4              # 3060 Ti/8GB icin 4. T4/16GB icin 12-16.
    grad_accum: int = 32           # efektif batch = micro*accum*world*seq token
    seq_len: int = 1024

    # Muon + AdamW (bkz. muon.py)
    muon_lr: float = 0.02
    adam_lr: float = 3e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # WSD scheduler -- cosine DEĞİL.
    # Sebep: stable fazda checkpoint alip SONRADAN devam edebiliyorsun.
    # Kotan belirsizken toplam adimi bastan sabitlemek zorunda degilsin.
    warmup_steps: int = 300
    decay_frac: float = 0.15       # son %15'te 1-sqrt ile sondur
    min_lr_ratio: float = 0.02
    total_tokens: int = 5_000_000_000

    grad_checkpoint: bool = True
    compile: bool = False          # T4'te genelde kazandirmiyor; 3060 Ti'da OLC

    out_dir: str = "checkpoints"
    ckpt_minutes: int = 15         # Colab 3. saatte de olebilir
    keep_last: int = 2
    hub_repo: str = ""             # "kullanici/model" -> checkpoint bus
    hub_every_min: int = 60

    log_every: int = 10
    eval_every: int = 500
    eval_iters: int = 40
    seed: int = 1337


PRESETS = {
    "s135": ModelConfig(n_layer=12, n_head=12, n_kv_head=6, d_model=768),
    "m290": ModelConfig(n_layer=22, n_head=16, n_kv_head=8, d_model=1024),
    "l440": ModelConfig(n_layer=22, n_head=20, n_kv_head=10, d_model=1280),
}


if __name__ == "__main__":
    print(f"{'preset':>7} {'toplam':>10} {'emb-siz':>10} {'d_ffn':>6} "
          f"{'chinchilla':>11} {'VRAM@bs4':>10} {'VRAM@bs12':>10}")
    for n, c in PRESETS.items():
        print(f"{n:>7} {c.n_params()/1e6:>9.0f}M {c.n_params(False)/1e6:>9.0f}M "
              f"{c.d_ffn:>6} {c.n_params()*20/1e9:>10.1f}B "
              f"{c.vram_estimate_gb(4):>9.1f}G {c.vram_estimate_gb(12):>9.1f}G")
