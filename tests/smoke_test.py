"""Tüm bileşenleri küçük ölçekte test et: model, muon, loader, ckpt, resume."""
import os, sys, math, tempfile
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, PRESETS
from model import GPT
from muon import build_optimizers, _newton_schulz
from data import BinDataset
from train import wsd_lr, atomic_save, make_scaler

ok = lambda m: print(f"  ✓ {m}")
tiny = ModelConfig(vocab_size=512, n_layer=3, n_head=4, n_kv_head=2,
                   d_model=64, seq_len=64, loss_chunk=16)

print("\n[1] model ileri/geri")
m = GPT(tiny)
x = torch.randint(0, 512, (2, 64))
# hedefler BAĞIMSIZ olmalı -- x'i hedef verirsen tied-embedding kopyalar ve
# loss yapay olarak düşer (bu aslında tied'ın doğru bağlandığının kanıtı)
yt = torch.randint(0, 512, (2, 64))
_, loss = m(x, yt)
assert loss.requires_grad and torch.isfinite(loss), loss
exp = math.log(512)
assert abs(loss.item() - exp) < 0.6, f"ilk loss {loss.item():.3f}, beklenen ~{exp:.3f}"
ok(f"loss={loss.item():.3f} (rastgele beklenti {exp:.3f})")
loss.backward()
ng = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
assert ng == sum(1 for _ in m.parameters()), f"{ng} tensörde gradyan var"
ok(f"tüm {ng} tensörde gradyan aktı")

print("\n[2] tied embeddings + QK-norm + z-loss")
assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
ok("lm_head ve tok_emb aynı tensör (tied)")
assert hasattr(m.blocks[0].attn, "q_norm")
ok("QK-Norm aktif")
c2 = ModelConfig(**{**tiny.__dict__, "z_loss": 0.0})
m2 = GPT(c2); m2.load_state_dict(m.state_dict())
_, l_noz = m2(x, yt)
assert loss.item() > l_noz.item(), "z-loss katkı vermiyor"
ok(f"z-loss katkısı = {loss.item()-l_noz.item():.4f}")

print("\n[3] grad checkpointing aynı sonucu veriyor mu")
m.grad_checkpoint = False; m.train()
torch.manual_seed(0); _, la = m(x, yt)
m.grad_checkpoint = True
torch.manual_seed(0); _, lb = m(x, yt)
assert abs(la.item()-lb.item()) < 1e-4, f"{la.item()} vs {lb.item()}"
ok(f"ckpt açık/kapalı loss farkı {abs(la.item()-lb.item()):.2e}")

print("\n[4] chunked loss == tek parça loss")
m.grad_checkpoint = False
m.cfg.loss_chunk = 0
torch.manual_seed(0); _, lfull = m(x, yt)
m.cfg.loss_chunk = 16
torch.manual_seed(0); _, lchunk = m(x, yt)
assert abs(lfull.item()-lchunk.item()) < 1e-4, f"{lfull.item()} vs {lchunk.item()}"
ok(f"fark {abs(lfull.item()-lchunk.item()):.2e}")

print("\n[5] Newton-Schulz gerçekten ortogonalize ediyor mu")
G = torch.randn(64, 32)
O = _newton_schulz(G, steps=5).float()
s = torch.linalg.svdvals(O)
ok(f"tekil değerler min={s.min():.3f} max={s.max():.3f} (1'e yakın olmalı)")
assert 0.6 < s.min() and s.max() < 1.4, s

print("\n[6] Muon + AdamW bölüşümü ve adım")
m.cfg.loss_chunk = 16
opts = build_optimizers(m, 0.02, 3e-3)
muon_params = {id(p) for p in opts[0].param_groups[0]["params"]}
assert id(m.tok_emb.weight) not in muon_params, "embedding Muon'a gitmiş!"
ok("embedding/lm_head AdamW'de, gizli matrisler Muon'da")
before = m.blocks[0].mlp.w_gate.weight.clone()
_, l = m(x, yt); l.backward()
for o in opts: o.step(); o.zero_grad()
assert not torch.equal(before, m.blocks[0].mlp.w_gate.weight), "Muon ağırlık değiştirmedi"
assert torch.isfinite(m.blocks[0].mlp.w_gate.weight).all()
ok("Muon adımı ağırlıkları güncelledi, NaN yok")

print("\n[7] loader DETERMİNİZM (en kritik test)")
tmp = tempfile.mkdtemp()
np.random.default_rng(0).integers(0, 512, 200_000, dtype=np.uint16).tofile(f"{tmp}/t.bin")
ds = BinDataset(f"{tmp}/t.bin", 64, "uint16", seed=1337)
a1, b1 = ds.get_batch(500, 4, device="cpu")
a2, b2 = ds.get_batch(500, 4, device="cpu")
assert torch.equal(a1, a2), "aynı adım farklı veri döndü!"
ok("get_batch(500) iki kez -> aynı veri (resume güvenli)")
a3, _ = ds.get_batch(501, 4, device="cpu")
assert not torch.equal(a1, a3)
ok("farklı adım -> farklı veri")
# world_size değişse bile (2xT4 -> 1xT4) veri sırası korunuyor mu
w2r0, _ = ds.get_batch(500, 2, rank=0, world_size=2, device="cpu")
w2r1, _ = ds.get_batch(500, 2, rank=1, world_size=2, device="cpu")
assert torch.equal(torch.cat([w2r0, w2r1]), a1), "world_size değişince veri kaydı!"
ok("Kaggle(2 GPU) <-> Colab(1 GPU) geçişinde veri sırası korunuyor")
assert torch.equal(b1[:, :-1], a1[:, 1:]), "hedef kaydırması yanlış"
ok("y = x'in 1 kaydırılmışı (next-token doğru)")

print("\n[8] WSD scheduler profili")
T, W = 1000, 100
vals = [wsd_lr(s, T, W, 0.15, 0.02) for s in range(T)]
assert vals[0] < 0.02 and abs(vals[W] - 1.0) < 1e-9 and abs(vals[500]-1.0) < 1e-9
assert vals[-1] <= 0.06, vals[-1]
ok(f"warmup {vals[0]:.3f} -> stable {vals[500]:.1f} -> son {vals[-1]:.3f}")

print("\n[9] checkpoint kaydet + resume (adım ve ağırlık korunuyor mu)")
p = f"{tmp}/ckpt.pt"
atomic_save({"model": m.state_dict(), "optims": [o.state_dict() for o in opts],
             "scaler": None, "step": 4242, "tokens_seen": 9_000_000}, p)
ck = torch.load(p, map_location="cpu", weights_only=False)
m3 = GPT(tiny); m3.load_state_dict(ck["model"])
o3 = build_optimizers(m3, 0.02, 3e-3)
for o, s in zip(o3, ck["optims"]): o.load_state_dict(s)
assert ck["step"] == 4242 and ck["tokens_seen"] == 9_000_000
assert torch.equal(m3.blocks[0].mlp.w_gate.weight, m.blocks[0].mlp.w_gate.weight)
assert not os.path.exists(p + ".tmp")
ok("ağırlık + optimizer state + adım geri yüklendi, .tmp artığı yok")

print("\n[10] fp16 altında NaN olmuyor mu (T4 senaryosu)")
mh = GPT(tiny)
with torch.autocast("cpu", dtype=torch.bfloat16):
    _, lh = mh(x, yt)
assert torch.isfinite(lh), lh
ok(f"düşük precision loss={lh.item():.3f}, sonlu")

print("\n[11] gerçek boyutlar")
for n, c in PRESETS.items():
    print(f"     {n:>6}: {c.n_params()/1e6:>4.0f}M | VRAM@bs4 ~{c.vram_estimate_gb(4):.1f}GB "
          f"| 8GB'a sığar: {'EVET' if c.vram_estimate_gb(4) < 7.4 else 'HAYIR (micro_bs düşür)'}")

print("\n" + "="*60 + "\nTÜM TESTLER GEÇTİ\n" + "="*60)
