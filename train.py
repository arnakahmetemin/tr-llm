"""Eğitim döngüsü -- kesintiye dayanıklı, çok platformlu.

Tek GPU (3060 Ti / Colab):
    python train.py --preset m290 --train_bin data/train.bin

Kaggle 2xT4 (notebook hücresinde DDP spawn ETME, torchrun kullan):
    !torchrun --nproc_per_node=2 train.py --preset m290 --micro_bs 12 --grad_accum 11

Önce ölç, sonra karar ver:
    python train.py --preset m290 --train_bin data/train.bin --bench
"""
import argparse, json, math, os, shutil, time
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import PRESETS, ModelConfig, TrainConfig
from data import BinDataset, parse_mix
from model import GPT
from muon import build_optimizers

PEAK_TFLOPS = {"T4": 65, "3060 Ti": 65, "P100": 19, "V100": 125, "A100": 312}


def make_scaler(enabled):
    """torch 2.4+ -> torch.amp.GradScaler, öncesi -> torch.cuda.amp.GradScaler."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------- utils
def is_master():
    return int(os.environ.get("RANK", 0)) == 0


def log(*a):
    if is_master():
        print(*a, flush=True)


def peak_flops(name):
    for k, v in PEAK_TFLOPS.items():
        if k.lower() in name.lower():
            return v * 1e12
    return 50e12


def wsd_lr(step, total, warmup, decay_frac, min_ratio):
    """Warmup-Stable-Decay. Cosine değil -- stable fazda durup devam edebilesin."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return 1.0
    p = (step - decay_start) / max(1, total - decay_start)
    return max(min_ratio, 1.0 - math.sqrt(p))


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)          # aynı fs'te atomik -- yarım checkpoint olmaz


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="m290", choices=list(PRESETS))
    ap.add_argument("--train_bin", default="data/train.bin",
                    help="tek .bin, veya karışım: 'fineweb.bin:0.6,wiki.bin:0.3,eski.bin:0.1'")
    ap.add_argument("--val_bin", default="")
    ap.add_argument("--bin_dtype", default="uint16")
    ap.add_argument("--vocab_size", type=int, default=0, help="0 = preset'ten al")
    ap.add_argument("--micro_bs", type=int, default=0)
    ap.add_argument("--grad_accum", type=int, default=0)
    ap.add_argument("--seq_len", type=int, default=0)
    ap.add_argument("--total_tokens", type=float, default=0)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--resume", default="auto", help="auto | yol | none")
    ap.add_argument("--hub_repo", default="")
    ap.add_argument("--bench", action="store_true", help="60 adım ölç, süre tahmin et, çık")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    args = ap.parse_args()

    mc: ModelConfig = PRESETS[args.preset]
    tc = TrainConfig()
    if args.vocab_size:   mc.vocab_size = args.vocab_size
    if args.seq_len:      mc.seq_len = tc.seq_len = args.seq_len
    if args.micro_bs:     tc.micro_bs = args.micro_bs
    if args.grad_accum:   tc.grad_accum = args.grad_accum
    if args.total_tokens: tc.total_tokens = int(args.total_tokens)
    if args.out_dir:      tc.out_dir = args.out_dir
    if args.hub_repo:     tc.hub_repo = args.hub_repo
    if args.no_grad_ckpt: tc.grad_checkpoint = False
    tc.train_bin, tc.bin_dtype = args.train_bin, args.bin_dtype
    tc.val_bin = args.val_bin

    # ---- dağıtık kurulum ----
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world, local_rank = 0, 1, 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(tc.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- precision: Ampere+ -> bf16 (GradScaler gereksiz), Turing -> fp16 ----
    gpu_name = torch.cuda.get_device_name(local_rank) if device != "cpu" else "cpu"
    use_bf16 = device != "cpu" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = make_scaler(amp_dtype == torch.float16)

    log(f"[hw] {gpu_name} x{world} | amp={amp_dtype} | "
        f"grad_ckpt={tc.grad_checkpoint} | flash={'evet' if use_bf16 else 'hayır (T4)'}")

    # ---- veri ----
    train_ds = parse_mix(tc.train_bin, mc.seq_len, tc.bin_dtype, tc.seed)
    val_ds = BinDataset(tc.val_bin, mc.seq_len, tc.bin_dtype, tc.seed + 1) \
        if tc.val_bin and os.path.exists(tc.val_bin) else None
    log(f"[veri] {train_ds}")
    if hasattr(train_ds, "check_epochs") and is_master():
        train_ds.check_epochs(tc.total_tokens)

    tokens_per_step = tc.micro_bs * tc.grad_accum * world * mc.seq_len
    total_steps = max(1, tc.total_tokens // tokens_per_step)
    log(f"[plan] {tokens_per_step:,} token/adım | {total_steps:,} adım | "
        f"{tc.total_tokens/1e9:.2f}B token | "
        f"{tc.total_tokens/train_ds.n_tokens:.2f} epoch")

    # ---- model ----
    model = GPT(mc).to(device)
    model.grad_checkpoint = tc.grad_checkpoint
    log(f"[model] {args.preset}: {model.num_params()/1e6:.0f}M toplam, "
        f"{model.num_params(False)/1e6:.0f}M emb-siz, "
        f"{model.flops_per_token()/1e6:.0f} MFLOP/token")

    opts = build_optimizers(model, tc.muon_lr, tc.adam_lr, tc.weight_decay)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

    # ---- resume ----
    os.makedirs(tc.out_dir, exist_ok=True)
    start_step, tokens_seen = 0, 0
    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")
    resume_from = ckpt_path if args.resume == "auto" and os.path.exists(ckpt_path) \
        else (args.resume if args.resume not in ("auto", "none") else None)
    if resume_from:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        for o, s in zip(opts, ck["optims"]):
            o.load_state_dict(s)
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_step, tokens_seen = ck["step"], ck["tokens_seen"]
        log(f"[resume] {resume_from} -> adım {start_step:,}, "
            f"{tokens_seen/1e9:.3f}B token görülmüş")

    if ddp:
        model = DDP(model, device_ids=[local_rank],
                    gradient_as_bucket_view=True, find_unused_parameters=False)
    raw = model.module if ddp else model

    if tc.compile:
        model = torch.compile(model)

    # ---- eval ----
    @torch.no_grad()
    def evaluate(ds, iters):
        model.eval()
        tot = 0.0
        for i in range(iters):
            x, y = ds.get_batch(10**9 + i, tc.micro_bs, rank, world, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device != "cpu"):
                _, loss = model(x, y)
            tot += loss.item()
        model.train()
        return tot / iters

    # ---- döngü ----
    model.train()
    pf = peak_flops(gpu_name)
    fpt = raw.flops_per_token()
    t_last, tok_last = time.time(), tokens_seen
    t_ckpt, t_hub, t0 = time.time(), time.time(), time.time()
    n_steps = 60 if args.bench else total_steps

    for step in range(start_step, n_steps):
        lr_mult = wsd_lr(step, total_steps, tc.warmup_steps,
                         tc.decay_frac, tc.min_lr_ratio)
        for o, bl in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, bl):
                g["lr"] = b * lr_mult

        for micro in range(tc.grad_accum):
            x, y = train_ds.get_batch(step * tc.grad_accum + micro,
                                      tc.micro_bs, rank, world, device)
            if ddp:
                model.require_backward_grad_sync = (micro == tc.grad_accum - 1)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device != "cpu"):
                _, loss = model(x, y)
            scaler.scale(loss / tc.grad_accum).backward()

        for o in opts:
            scaler.unscale_(o)
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), tc.grad_clip)
        for o in opts:
            scaler.step(o)
        scaler.update()
        for o in opts:
            o.zero_grad(set_to_none=True)

        tokens_seen += tokens_per_step

        # ---- log ----
        if step % tc.log_every == 0 or step == n_steps - 1:
            dt = time.time() - t_last
            tps = (tokens_seen - tok_last) / max(dt, 1e-6)
            t_last, tok_last = time.time(), tokens_seen
            mfu = fpt * tps / (pf * world) * 100
            vram = (torch.cuda.max_memory_allocated() / 1e9
                    if device != "cpu" else 0.0)
            rem = (tc.total_tokens - tokens_seen) / max(tps, 1) / 3600
            log(f"adım {step:>6,}/{total_steps:,} | loss {loss.item():.4f} | "
                f"lr x{lr_mult:.3f} | |g| {gnorm:.2f} | {tps:,.0f} tok/s | "
                f"MFU ~%{mfu:.0f} | VRAM {vram:.1f}G | {tokens_seen/1e9:.3f}B | "
                f"kalan ~{rem:.0f}sa")

        if val_ds is not None and step > 0 and step % tc.eval_every == 0:
            log(f"  >> val loss {evaluate(val_ds, tc.eval_iters):.4f}")

        # ---- süreli checkpoint (Colab 3. saatte de ölebilir) ----
        if is_master() and not args.bench and \
                (time.time() - t_ckpt > tc.ckpt_minutes * 60 or step == n_steps - 1):
            t_ckpt = time.time()
            atomic_save({
                "model": raw.state_dict(),
                "optims": [o.state_dict() for o in opts],
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "step": step + 1, "tokens_seen": tokens_seen,
                "model_cfg": asdict(mc), "train_cfg": asdict(tc),
            }, ckpt_path)
            shutil.copyfile(ckpt_path, os.path.join(tc.out_dir, f"ckpt_{step+1}.pt")) \
                if (step + 1) % (tc.eval_every * 4) == 0 else None
            log(f"  [ckpt] adım {step+1:,} kaydedildi "
                f"({os.path.getsize(ckpt_path)/1e9:.2f}GB)")

            if tc.hub_repo and time.time() - t_hub > tc.hub_every_min * 60:
                t_hub = time.time()
                try:
                    from huggingface_hub import HfApi
                    HfApi().upload_file(path_or_fileobj=ckpt_path,
                                        path_in_repo="ckpt.pt",
                                        repo_id=tc.hub_repo, repo_type="model")
                    log(f"  [hub] {tc.hub_repo} güncellendi")
                except Exception as e:
                    log(f"  [hub] HATA (eğitim devam ediyor): {e}")

    # ---- bench özeti ----
    if args.bench and is_master():
        el = time.time() - t0
        tps = (tokens_seen - (start_step * tokens_per_step)) / el
        print("\n" + "=" * 64)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"ÖLÇÜM: {gpu_name} x{world}")
        print(f"  VRAM: {peak:.1f}G kullanılan / {reserved:.1f}G ayrılan / "
              f"{total_vram:.1f}G kart")
        if reserved > total_vram * 0.92:
            print(f"  ⚠️  KARTA SIĞMIYOR -> Windows fazlasını sistem RAM'ine")
            print(f"     taşıyor (sysmem fallback). PCIe üzerinden çalışıyor,")
            print(f"     MFU bu yüzden düşük. --micro_bs DÜŞÜR.")
        print(f"  {tps:,.0f} token/saniye | MFU ~%{fpt*tps/(pf*world)*100:.0f}")
        print(f"  saatte {tps*3600/1e6:,.0f}M token")
        print("=" * 64)
        for name, c in PRESETS.items():
            scale = raw.flops_per_token() / (6 * c.n_params(False) +
                                             12 * c.n_layer * c.d_model * c.seq_len)
            t = c.n_params() * 20 / (tps * scale) / 3600
            print(f"  {name:>6} ({c.n_params()/1e6:>3.0f}M, chinchilla "
                  f"{c.n_params()*20/1e9:.1f}B token): {t:>6.0f} GPU-saat"
                  f"  ~{t/(70+60):.1f} hafta (3060Ti 70sa + Kaggle 60sa/hafta)")
        print("=" * 64)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
