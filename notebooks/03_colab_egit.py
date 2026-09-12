# ============================================================
#  NOTEBOOK 3 - COLAB: 1x T4 ile egit (bosluk doldurucu vardiya)
#  Runtime -> Change runtime type -> T4 GPU
#
#  ⚠️ COLAB UYARILARI:
#   * 90 dakika etkilesim yoksa atar -> SEKMEYI ACIK TUT
#   * Haftalik kota dinamik (~15 saat), yayinlanmiyor
#   * Checkpoint 15 dakikada bir Hub'a gidiyor, kopsa da kayip 15 dk
# ============================================================
import os, subprocess

HF_USER   = "Ahmetemiiii2"
DATA_REPO = f"{HF_USER}/tr-corpus"
CKPT_REPO = f"{HF_USER}/tr-290m"
D = "/content/data"

subprocess.run("pip install -q huggingface_hub", shell=True)
subprocess.run("git clone -q https://github.com/arnakahmetemin/tr-llm.git /content/tr-llm",
               shell=True)
os.chdir("/content/tr-llm")

# Colab: sol paneldeki anahtar ikonundan HF_TOKEN secret'i ekle
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from huggingface_hub import hf_hub_download

os.makedirs(D, exist_ok=True)
for f in ("fineweb.bin", "wiki.bin", "val.bin"):
    if not os.path.exists(f"{D}/{f}"):
        print(f">> {f} indiriliyor...", flush=True)
        hf_hub_download(DATA_REPO, f, repo_type="dataset", local_dir=D)

os.makedirs("checkpoints", exist_ok=True)
try:
    hf_hub_download(CKPT_REPO, "ckpt.pt", local_dir="checkpoints")
    print(">> onceki checkpoint alindi", flush=True)
except Exception:
    print(">> checkpoint yok, sifirdan", flush=True)

# tek GPU: 4 x 32 = 128 -- Kaggle'daki 2x(8x8) ile AYNI efektif batch
cmd = f"""python train.py \
  --preset m290 --vocab_size 32768 \
  --micro_bs 4 --grad_accum 32 \
  --train_bin "{D}/fineweb.bin:0.82,{D}/wiki.bin:0.18" \
  --val_bin {D}/val.bin \
  --total_tokens 5e9 \
  --out_dir checkpoints \
  --max_hours 11 \
  --hub_repo {CKPT_REPO}"""
print(">>", cmd, flush=True)
subprocess.run(cmd, shell=True)
