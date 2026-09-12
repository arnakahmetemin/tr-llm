"""Muon optimizer -- 130M-300M ölçeğinde AdamW'ye göre ~1.3-2x sample efficiency.

Mantık: gradyan matrisini ortogonalize et (Newton-Schulz), sonra adım at.
Sadece 2D "gizli katman" matrisleri için. Embedding / lm_head / norm / bias
AdamW'de kalır -- Muon onlarda çalışmaz.

Bonus: Muon momentum buffer'ı TEK tensör tutar (AdamW 2 tutar).
290M modelde ~1.2GB VRAM tasarrufu -> 8GB'lık 3060 Ti'da bu kritik.

Kaynak: Keller Jordan, https://kellerjordan.github.io/posts/muon/
"""
import torch


def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """G'yi (yaklaşık) en yakın ortogonal matrise çevirir.

    bf16 varsa bf16'da çalışır (Ampere/3060 Ti). T4'te bf16 yok -> fp32.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315

    dev = G.device.type
    if dev == "xla":                      # TPU: bf16 yerli, tercih edilir
        work_dtype = torch.bfloat16
    elif dev == "cuda" and torch.cuda.is_bf16_supported():
        work_dtype = torch.bfloat16
    else:                                 # T4 (Turing) ve CPU: bf16 yok
        work_dtype = torch.float32
    X = G.to(work_dtype)
    X = X / (X.norm() + eps)

    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """2D gizli katman matrisleri için. Diğer her şey AdamW'ye gitmeli."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        super().__init__(list(params), dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, f"Muon sadece 2D ister, {g.ndim}D geldi"

                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if group["nesterov"] else buf

                g = _newton_schulz(g, steps=group["ns_steps"])

                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                # dikdörtgen matrislerde adım boyunu dengele
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-lr * scale)
        return loss


def build_optimizers(model, muon_lr=0.02, adam_lr=3e-3, weight_decay=0.1,
                     betas=(0.9, 0.95)):
    """Parametreleri doğru optimizer'a böl.

    Muon  <- transformer bloklarındaki 2D matrisler (attn + mlp)
    AdamW <- embedding, lm_head, tüm 1D (norm ağırlıkları)
    """
    muon_p, adam_decay, adam_nodecay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = "tok_emb" in name or "lm_head" in name
        if p.ndim == 2 and not is_embed:
            muon_p.append(p)
        elif p.ndim >= 2:
            adam_decay.append(p)
        else:
            adam_nodecay.append(p)

    opts = []
    if muon_p:
        opts.append(Muon(muon_p, lr=muon_lr, weight_decay=weight_decay))
    opts.append(torch.optim.AdamW(
        [{"params": adam_decay, "weight_decay": weight_decay},
         {"params": adam_nodecay, "weight_decay": 0.0}],
        lr=adam_lr, betas=betas, eps=1e-8))

    n_muon = sum(p.numel() for p in muon_p)
    n_adam = sum(p.numel() for p in adam_decay + adam_nodecay)
    print(f"[optim] Muon: {len(muon_p)} tensör / {n_muon/1e6:.1f}M param | "
          f"AdamW: {len(adam_decay)+len(adam_nodecay)} tensör / {n_adam/1e6:.1f}M param")
    return opts
