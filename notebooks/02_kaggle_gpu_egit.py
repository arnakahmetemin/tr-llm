# ============================================================
#  NOTEBOOK 2 - KAGGLE GPU: 2x T4 ile egit
#  Accelerator: GPU T4 x2   |   30 saat/hafta, oturum 12 saat
#
#  ONCE KAGGLE'DA:
#   1) Add-ons -> Secrets -> HF_TOKEN (write yetkili)
#   2) Settings -> Internet: ON
#   3) Accelerator: GPU T4 x2
#
#  BASLATIRKEN: "Save Version -> Save & Run All (Commit)"
#  -> tarayiciyi kapatabilirsin, 12 saat arka planda calisir.
# ============================================================
import os, subprocess, sys

HF_USER   = "Ahmetemiiii2"
DATA_REPO = f"{HF_USER}/tr-corpus"
CKPT_REPO = f"{HF_USER}/tr-290m"
D = "/kaggle/working/data"

subprocess.run("pip install -q huggingface_hub", shell=True)
subprocess.run("git clone -q https://github.com/arnakahmetemin/tr-llm.git", shell=True)
os.chdir("/kaggle/working/tr-llm")

from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
from huggingface_hub import hf_hub_download, create_repo
create_repo(CKPT_REPO, exist_ok=True, private=True)

# --- veriyi indir (oturum basina bir kez, ~15GB, Google<->HF hizli) ---
os.makedirs(D, exist_ok=True)
for f in ("fineweb.bin", "wiki.bin", "val.bin"):
    if not os.path.exists(f"{D}/{f}"):
        print(f">> {f} indiriliyor...", flush=True)
        p = hf_hub_download(DATA_REPO, f, repo_type="dataset", local_dir=D)
        print(f"   {os.path.getsize(p)/1e9:.1f} GB", flush=True)

# --- onceki vardiyanin checkpoint'ini al (varsa) ---
os.makedirs("checkpoints", exist_ok=True)
try:
    hf_hub_download(CKPT_REPO, "ckpt.pt", local_dir="checkpoints")
    print(">> onceki checkpoint alindi, devam edilecek", flush=True)
except Exception:
    print(">> checkpoint yok, SIFIRDAN baslaniyor", flush=True)

# --- egit ---
# 2 GPU x micro_bs 8 x accum 8 = 128 (tek GPU'daki 4x32 ile AYNI efektif batch)
# Efektif batch'i sabit tutmak sart: vardiyalar arasi degisirse egitim bozulur.
cmd = f"""torchrun --nproc_per_node=2 train.py \
  --preset m290 --vocab_size 32768 \
  --micro_bs 8 --grad_accum 8 \
  --train_bin "{D}/fineweb.bin:0.82,{D}/wiki.bin:0.18" \
  --val_bin {D}/val.bin \
  --total_tokens 5e9 \
  --out_dir checkpoints \
  --max_hours 11.5 \
  --hub_repo {CKPT_REPO}"""
print(">>", cmd, flush=True)
subprocess.run(cmd, shell=True)
