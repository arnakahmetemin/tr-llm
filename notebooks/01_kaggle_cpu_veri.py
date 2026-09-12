# ============================================================
#  NOTEBOOK 1 - KAGGLE CPU: veriyi uret, HF Hub'a pushla
#  Accelerator: NONE (CPU)  <-- GPU/TPU kotasindan HIC yemez
#  Kaggle CPU oturumlari kotasiz; 12 saate kadar calisir.
#
#  ONCE KAGGLE'DA:
#   1) Add-ons -> Secrets -> HF_TOKEN ekle (huggingface.co/settings/tokens,
#      "write" yetkili)
#   2) tokenizer.json: IKI YOLDAN BIRI
#      (a) TAVSIYE: huggingface.co/new-dataset ile Ahmetemiiii2/tr-corpus repo'sunu
#          ac, tokenizer.json'u web arayuzunden surukleyip birak. Notebook
#          otomatik oradan ceker. Colab da ayni yerden alir, yedegi de olur.
#      (b) Ya da Kaggle Dataset olarak yukle (adi "tr-tokenizer") ve bu
#          notebook'a input olarak ekle.
#   3) Settings -> Internet: ON
# ============================================================
import os, subprocess, sys

HF_USER  = "Ahmetemiiii2"
DATA_REPO = f"{HF_USER}/tr-corpus"
TOK = None      # asagida otomatik bulunuyor
WORK = "/kaggle/working"

# --- bu notebook'ta hangisini uretecegiz? ---
# /kaggle/working 20GB sinirli, fineweb 14GB.
# Bu yuzden IKI AYRI CALISTIRMA yap: once "wiki", sonra "fineweb".
HEDEF = "wiki"          # <-- ikinci calistirmada "fineweb" yap

subprocess.run("pip install -q tokenizers datasets huggingface_hub", shell=True)
subprocess.run("git clone -q https://github.com/arnakahmetemin/tr-llm.git", shell=True)
os.chdir("/kaggle/working/tr-llm")

from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
from huggingface_hub import HfApi, create_repo, hf_hub_download
api = HfApi()
create_repo(DATA_REPO, repo_type="dataset", exist_ok=True, private=True)

# --- tokenizer'i bul: once HF, sonra Kaggle Dataset ---
try:
    TOK = hf_hub_download(DATA_REPO, "tokenizer.json", repo_type="dataset",
                          local_dir=WORK)
    print(f">> tokenizer HF'ten alindi: {TOK}")
except Exception as e:
    import glob
    hit = glob.glob("/kaggle/input/*/tokenizer.json")
    if not hit:
        raise SystemExit(
            "tokenizer.json bulunamadi!\n"
            f"  * HF'e yukle: huggingface.co/datasets/{DATA_REPO} -> Files -> Add file\n"
            "  * ya da Kaggle Dataset olarak yukleyip bu notebook'a input ekle")
    TOK = hit[0]
    print(f">> tokenizer Kaggle input'tan alindi: {TOK}")


def uret(source, target, out, extra=""):
    cmd = (f"{sys.executable} prepare_data.py --source {source} --target {target} "
           f"--out {out} --tokenizer {TOK} --eos_id 0 {extra}")
    print(">>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def pushla(path, isim):
    gb = os.path.getsize(path) / 1e9
    print(f">> HF'e yukleniyor: {isim} ({gb:.1f} GB)", flush=True)
    api.upload_file(path_or_fileobj=path, path_in_repo=isim,
                    repo_id=DATA_REPO, repo_type="dataset")
    os.remove(path)                      # yer ac
    print(f">> {isim} yuklendi ve yerelden silindi", flush=True)


if HEDEF == "wiki":
    uret("wiki", "1e9", f"{WORK}/wiki.bin")

    # val setini wiki'nin SONUNDAN kes -- egitim verisiyle ortusmesin
    p, n = f"{WORK}/wiki.bin", 10_000_000
    sz = os.path.getsize(p)
    with open(p, "rb") as f:
        f.seek(sz - n); tail = f.read(n)
    open(f"{WORK}/val.bin", "wb").write(tail)
    with open(p, "r+b") as f:
        f.truncate(sz - n)
    # progress dosyasini da duzelt, yoksa resume dosyayi sifirla sisirir
    import json
    pj = p + ".progress.json"
    if os.path.exists(pj):
        d = json.load(open(pj)); d["tokens"] = (sz - n) // 2
        json.dump(d, open(pj, "w"))
    print(f"val ayrildi: {n//2:,} token")

    pushla(f"{WORK}/val.bin", "val.bin")
    pushla(f"{WORK}/wiki.bin", "wiki.bin")

elif HEDEF == "fineweb":
    uret("fineweb", "7e9", f"{WORK}/fineweb.bin")
    pushla(f"{WORK}/fineweb.bin", "fineweb.bin")

print("\nBITTI ->", f"https://huggingface.co/datasets/{DATA_REPO}")
