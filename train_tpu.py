"""TPU (torch-xla) egitim dongusu -- train.py'nin XLA karsiligi.

AYNI checkpoint formatini kullanir: CUDA'da baslayip TPU'da devam edebilir,
tersi de gecerli. Veri sirasi yine adim numarasindan turedigi icin
world_size 2 (Kaggle T4) -> 8 (TPU v5e-8) gecisinde de bozulmaz.

Kaggle TPU: v5e-8, 20 saat/hafta, oturum 9 saat, gunde 9 saat.
    python train_tpu.py --preset m290 --vocab_size 32768 \
        --micro_bs 16 --grad_accum 1 --max_hours 8.5 ...

    (8 core x micro_bs 16 x accum 1 = 128 -> Kaggle GPU'daki 2x(8x8) ile AYNI)

XLA notlari:
  * Statik sekil sart -- modelimiz zaten statik (sabit seq_len, chunked loss
    sabit parca, Newton-Schulz sabit 5 iterasyon).
  * xm.mark_step() grafi bosaltir; her optimizer adiminda bir kez cagriliyor.
  * .item() senkronizasyon yaptirir -> sadece log adimlarinda cagriliyor.
  * GradScaler YOK: TPU'da bf16 yerli, loss scaling gereksiz.
  * Grad checkpointing KAPALI: 293M model v5e'nin 16GB HBM'ine rahat sigiyor,
    acmanin tek etkisi %30 yavaslama olurdu.
"""
import argparse, os, time
from dataclasses import asdict

import torch

import torch_xla
import torch_xla.core.xla_model as xm

try:                                    # torch-xla 2.x
    import torch_xla.runtime as xr
    _world = xr.world_size
    _ordinal = xr.global_ordinal
except (ImportError, AttributeError):   # eski surumler
    _world = xm.xrt_world_size
    _ordinal = xm.get_ordinal

from config import PRESETS, ModelConfig, TrainConfig
from data import parse_mix
from model import GPT
from muon import build_optimizers
from train import wsd_lr, atomic_save


def _mp_fn(index, args):
    device = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
    world, rank = _world(), _ordinal()
    master = rank == 0

    def log(*a):
        if master:
            print(*a, flush=True)

    mc: ModelConfig = PRESETS[args.preset]
    tc = TrainConfig()
    if args.vocab_size:   mc.vocab_size = args.vocab_size
    if args.seq_len:      mc.seq_len = tc.seq_len = args.seq_len
    if args.micro_bs:     tc.micro_bs = args.micro_bs
    if args.grad_accum:   tc.grad_accum = args.grad_accum
    if args.total_tokens: tc.total_tokens = int(args.total_tokens)
    tc.out_dir, tc.hub_repo = args.out_dir, args.hub_repo
    tc.grad_checkpoint = False          # TPU'da gereksiz

    torch.manual_seed(tc.seed)

    log(f"[hw] TPU {world} core | bf16 | grad_ckpt=False")

    train_ds = parse_mix(args.train_bin, mc.seq_len, args.bin_dtype, tc.seed)
    val_ds = (parse_mix(args.val_bin, mc.seq_len, args.bin_dtype, tc.seed + 1)
              if args.val_bin and os.path.exists(args.val_bin) else None)
    log(f"[veri] {train_ds}")
    if master and hasattr(train_ds, "check_epochs"):
        train_ds.check_epochs(tc.total_tokens)

    tokens_per_step = tc.micro_bs * tc.grad_accum * world * mc.seq_len
    total_steps = max(1, tc.total_tokens // tokens_per_step)
    log(f"[plan] {tokens_per_step:,} token/adım | {total_steps:,} adım | "
        f"{tc.total_tokens/1e9:.2f}B token")

    model = GPT(mc).to(device)
    log(f"[model] {args.preset}: {model.num_params()/1e6:.0f}M")

    opts = build_optimizers(model, tc.muon_lr, tc.adam_lr, tc.weight_decay) \
        if master else build_optimizers(model, tc.muon_lr, tc.adam_lr, tc.weight_decay)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

    # ---- resume: CUDA'da uretilmis checkpoint de yuklenir ----
    os.makedirs(tc.out_dir, exist_ok=True)
    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")
    start_step, tokens_seen = 0, 0
    if args.resume != "none" and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model = model.to(device)
        for o, s in zip(opts, ck["optims"]):
            o.load_state_dict(s)
        start_step, tokens_seen = ck["step"], ck["tokens_seen"]
        log(f"[resume] adım {start_step:,}, {tokens_seen/1e9:.3f}B token")

    model.train()
    t0 = t_last = time.time()
    tok_last, t_ckpt, t_hub = tokens_seen, time.time(), time.time()

    for step in range(start_step, total_steps):
        lr_mult = wsd_lr(step, total_steps, tc.warmup_steps,
                         tc.decay_frac, tc.min_lr_ratio)
        for o, bl in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, bl):
                g["lr"] = b * lr_mult

        for micro in range(tc.grad_accum):
            x, y = train_ds.get_batch(step * tc.grad_accum + micro,
                                      tc.micro_bs, rank, world, device="cpu")
            x, y = x.to(device), y.to(device)
            with torch.autocast("xla", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / tc.grad_accum).backward()

        for o in opts:
            xm.reduce_gradients(o)          # core'lar arasi gradyan toplama
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        xm.mark_step()                       # XLA grafini bosalt

        tokens_seen += tokens_per_step

        if step % tc.log_every == 0 or step == total_steps - 1:
            l = loss.item()                  # senkronizasyon: seyrek yap
            dt = time.time() - t_last
            tps = (tokens_seen - tok_last) / max(dt, 1e-6)
            t_last, tok_last = time.time(), tokens_seen
            rem = (tc.total_tokens - tokens_seen) / max(tps, 1) / 3600
            log(f"adım {step:>6,}/{total_steps:,} | loss {l:.4f} | lr x{lr_mult:.3f} "
                f"| |g| {gnorm.item():.2f} | {tps:,.0f} tok/s | "
                f"{tokens_seen/1e9:.3f}B | kalan ~{rem:.0f}sa")

        # ---- checkpoint ----
        sure_doldu = args.max_hours and (time.time() - t0) > args.max_hours * 3600
        if (time.time() - t_ckpt > tc.ckpt_minutes * 60 or sure_doldu
                or step == total_steps - 1):
            t_ckpt = time.time()
            xm.mark_step()
            cpu_sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
            if master:
                atomic_save({
                    "model": cpu_sd,
                    "optims": [o.state_dict() for o in opts],
                    "scaler": None,
                    "step": step + 1, "tokens_seen": tokens_seen,
                    "model_cfg": asdict(mc), "train_cfg": asdict(tc),
                }, ckpt_path)
                log(f"  [ckpt] adım {step+1:,} "
                    f"({os.path.getsize(ckpt_path)/1e9:.2f}GB)")
                if tc.hub_repo and (sure_doldu
                                    or time.time() - t_hub > tc.hub_every_min * 60):
                    t_hub = time.time()
                    try:
                        from huggingface_hub import HfApi
                        HfApi().upload_file(path_or_fileobj=ckpt_path,
                                            path_in_repo="ckpt.pt",
                                            repo_id=tc.hub_repo, repo_type="model")
                        log(f"  [hub] {tc.hub_repo} güncellendi")
                    except Exception as e:
                        log(f"  [hub] HATA (eğitim devam): {e}")
            xm.rendezvous("ckpt")            # tum core'lar kaydi bekler

        if sure_doldu:
            log(f"\n[süre] {args.max_hours} saat doldu, temiz çıkılıyor "
                f"(adım {step+1:,})")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="m290", choices=list(PRESETS))
    ap.add_argument("--train_bin", required=True)
    ap.add_argument("--val_bin", default="")
    ap.add_argument("--bin_dtype", default="uint16")
    ap.add_argument("--vocab_size", type=int, default=0)
    ap.add_argument("--micro_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=0)
    ap.add_argument("--total_tokens", type=float, default=0)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--hub_repo", default="")
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--max_hours", type=float, default=8.5,
                    help="Kaggle TPU oturumu 9 saat -> 8.5 ver, temiz çıksın")
    args = ap.parse_args()

    if hasattr(torch_xla, "launch"):
        torch_xla.launch(_mp_fn, args=(args,))
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp_fn, args=(args,), nprocs=None)


if __name__ == "__main__":
    main()
