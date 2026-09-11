"""Türkçe LLM -- dense decoder-only transformer.

Mimari kararlar (PLAN.md bölüm 4):
  RMSNorm pre-norm, RoPE, SwiGLU, GQA, QK-Norm, tied embeddings, z-loss.
  Gated DeltaNet / hibrit attention YOK -- kısa context'te faydası yok,
  T4'te kernel yok.

Donanım:
  3060 Ti (Ampere/SM86) -> bf16 + FlashAttention-2 (SDPA otomatik seçer)
  T4 (Turing/SM75)      -> fp16 + GradScaler, mem-efficient SDPA
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dt) * self.weight


def build_rope_cache(seq_len, d_head, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(seq_len, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)          # fp32 tut, apply'da cast et


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    T = x.shape[-2]
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.nh, self.nkv, self.dh = c.n_head, c.n_kv_head, c.d_head
        self.rep = c.n_head // c.n_kv_head
        self.wq = nn.Linear(c.d_model, c.n_head * c.d_head, bias=False)
        self.wk = nn.Linear(c.d_model, c.n_kv_head * c.d_head, bias=False)
        self.wv = nn.Linear(c.d_model, c.n_kv_head * c.d_head, bias=False)
        self.wo = nn.Linear(c.n_head * c.d_head, c.d_model, bias=False)
        # QK-Norm: Qwen3'ün eklediği. fp16'da attention logit patlamasını önler.
        # T4'te fp16 zorunlu olduğu için bu bizde OPSİYONEL DEĞİL.
        self.qk_norm = c.qk_norm
        if c.qk_norm:
            self.q_norm = RMSNorm(c.d_head)
            self.k_norm = RMSNorm(c.d_head)
        self.dropout = c.dropout

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.dh)
        k = self.wk(x).view(B, T, self.nkv, self.dh)
        v = self.wv(x).view(B, T, self.nkv, self.dh)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))     # -> (B, H, T, D)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if self.rep > 1:                                      # GQA
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(c.d_model, c.d_ffn, bias=False)
        self.w_up = nn.Linear(c.d_model, c.d_ffn, bias=False)
        self.w_down = nn.Linear(c.d_ffn, c.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn = RMSNorm(c.d_model), Attention(c)
        self.n2, self.mlp = RMSNorm(c.d_model), SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.norm_f = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.grad_checkpoint = False
        self._rope_key = None

        self.apply(self._init)
        for n, p in self.named_parameters():          # residual çıkışlarını küçült
            if n.endswith("wo.weight") or n.endswith("w_down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def _rope(self, device):
        if self._rope_key != device:
            cos, sin = build_rope_cache(self.cfg.seq_len, self.cfg.d_head,
                                        self.cfg.rope_theta, device)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self._rope_key = device
        return self.rope_cos, self.rope_sin

    def _loss(self, h, targets):
        """Chunked cross-entropy.

        Neden: logits tensörü (B,T,32k) bellekte devasa.
        B=8,T=1024,V=32768 fp32 -> 1GB. 8GB'lık kartta tek başına öldürür.
        Sequence'i parçalara bölüp her parçanın loss'unu ayrı hesaplıyoruz.
        """
        B, T, _ = h.shape
        chunk = self.cfg.loss_chunk or T
        tot_loss = h.new_zeros((), dtype=torch.float32)
        tot_z = h.new_zeros((), dtype=torch.float32)
        n = 0
        for i in range(0, T, chunk):
            hs = h[:, i:i + chunk]
            ts = targets[:, i:i + chunk]
            logits = self.lm_head(hs).float()
            tot_loss = tot_loss + F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), ts.reshape(-1),
                ignore_index=-1, reduction="sum")
            if self.cfg.z_loss > 0:
                tot_z = tot_z + torch.logsumexp(logits, dim=-1).pow(2).sum()
            n += (ts != -1).sum()
        n = n.clamp(min=1)
        loss = tot_loss / n
        if self.cfg.z_loss > 0:
            # z-loss: logit'lerin büyümesini cezalandırır, fp16'da şart
            loss = loss + self.cfg.z_loss * (tot_z / n)
        return loss

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        cos, sin = self._rope(x.device)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = ckpt_fn(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        x = self.norm_f(x)

        if targets is None:
            return self.lm_head(x), None
        return None, self._loss(x, targets)

    # ---- yardımcılar ----
    def num_params(self, embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if not embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def flops_per_token(self):
        c = self.cfg
        return 6 * self.num_params(False) + 12 * c.n_layer * c.d_model * c.seq_len

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.seq_len:])
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, -1), 1)), dim=1)
        return idx
