"""FineWeb-2 (tr) + Türkçe Wikipedia -> uint16 .bin

Özellikler:
  * STREAMING -- 100GB'lık dataset'i diske indirmez, akıtarak tokenize eder
  * KESİLEBİLİR -- yarıda kesersen kaldığı yerden devam eder (.progress.json)
  * çok çekirdekli tokenizasyon
  * hedef token sayısına ulaşınca durur

Kullanım:
    python prepare_data.py --source wiki    --target 1e9 --out data/wiki.bin \
        --tokenizer tokenizer.json
    python prepare_data.py --source fineweb --target 7e9 --out data/fineweb.bin \
        --tokenizer tokenizer.json
"""
import argparse, json, os, re, time
from multiprocessing import Pool

import numpy as np

SOURCES = {
    "fineweb": dict(path="HuggingFaceFW/fineweb-2", name="tur_Latn",
                    split="train", field="text"),
    "wiki":    dict(path="wikimedia/wikipedia", name="20231101.tr",
                    split="train", field="text"),
    "culturax": dict(path="uonlp/CulturaX", name="tr", split="train",
                     field="text"),          # gated: HF'de şartları kabul et
    # --- kod ---
    # NOT: kod dataset'lerinin çoğu "gated" -- HF sayfasında şartları kabul et.
    # "-ids" ekli Stack v2 varyantları İÇERİK TAŞIMAZ (sadece S3 işaretçisi),
    # onları kullanma. İçerik taşıyanlar aşağıda.
    "code": dict(path="bigcode/the-stack-smol", name=None, split="train",
                 field="content"),           # küçük, hızlı, denemelik
    "code_big": dict(path="bigcode/starcoderdata", name=None, split="train",
                     field="content", data_dir="python"),   # --data_dir ile dil seç
}

_TOK = None      # worker başına global


def local_docs(path, doc_sep="\n\n\n"):
    """Kendi .txt'in -> belge akışı. Dosya ya da klasör olabilir.

    Klasör verirsen her dosya bir belge. Tek dosya verirsen doc_sep ile
    bölünür; ayırıcı yoksa dosyanın tamamı tek belge sayılır.
    """
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fn in sorted(files):
                if fn.startswith("."):
                    continue
                fp = os.path.join(root, fn)
                try:
                    txt = open(fp, encoding="utf-8", errors="ignore").read()
                except (OSError, UnicodeError):
                    continue
                if txt.strip():
                    yield {"text": txt}
    else:
        txt = open(path, encoding="utf-8", errors="ignore").read()
        parts = txt.split(doc_sep) if doc_sep in txt else [txt]
        for p in parts:
            if p.strip():
                yield {"text": p}


# --------------------------------------------------------------------- tokenizer
def load_tokenizer(path):
    """tokenizers .json / transformers klasörü / sentencepiece .model -- hepsi olur."""
    if path.endswith(".json"):
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(path)
        return lambda s: t.encode(s).ids, t.get_vocab_size()
    if path.endswith(".model"):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=path)
        return lambda s: sp.encode(s), sp.get_piece_size()
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(path)
    return lambda s: t(s, add_special_tokens=False)["input_ids"], len(t)


def _init(tok_path):
    global _TOK
    _TOK, _ = load_tokenizer(tok_path)


# --------------------------------------------------------------------- temizlik
_WIKI_TAIL = re.compile(
    r"\n=+\s*(Kaynakça|Kaynaklar|Dış bağlantılar|Ayrıca bakınız|Notlar|"
    r"References|External links)\s*=+.*", re.S | re.I)
_MULTISPACE = re.compile(r"\n{3,}")


def clean(text, source):
    if source == "wiki":
        text = _WIKI_TAIL.sub("", text)          # kaynakça/dış bağlantı kırp
    return _MULTISPACE.sub("\n\n", text).strip()


def _encode(args):
    text, source, eos = args
    text = clean(text, source)
    min_len = 80 if source.startswith("code") else 200
    if len(text) < min_len:                      # taslak/çöp ele
        return None
    ids = _TOK(text)
    if eos is not None:
        ids.append(eos)                          # belge sınırı
    return np.asarray(ids, dtype=np.uint16)


# --------------------------------------------------------------------- ana
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=list(SOURCES) + ["local"])
    ap.add_argument("--out", default="", help="--dump_text ile gerekmez")
    ap.add_argument("--tokenizer", default="",
                    help="--dump_text ile gerekmez (tavuk-yumurta: tokenizer henüz yok)")
    ap.add_argument("--target", type=float, default=0, help="hedef token, ör 7e9")
    ap.add_argument("--dump_text", default="",
                    help="TOKENIZE ETME, ham metni bu dosyaya yaz (tokenizer örneği için)")
    ap.add_argument("--target_mb", type=float, default=200,
                    help="--dump_text için hedef boyut (MB)")
    ap.add_argument("--eos_id", type=int, default=-1, help="-1 = belge ayırıcı yok")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--dataset", default="", help="SOURCES'ı ezmek için")
    ap.add_argument("--config", default="")
    ap.add_argument("--field", default="", help="metin kolonunun adı (ör. content)")
    ap.add_argument("--data_dir", default="", help="ör. starcoderdata icin 'python'")
    ap.add_argument("--input", default="", help="--source local için dosya/klasör")
    ap.add_argument("--doc_sep", default="\n\n\n", help="--source local belge ayırıcı")
    args = ap.parse_args()

    if not args.out and not args.dump_text:
        raise SystemExit("--out gerekli (ya da --dump_text kullan)")
    if args.source == "local" and not args.input:
        raise SystemExit("--source local için --input <dosya|klasör> gerekli")
    src = dict(SOURCES.get(args.source, {"field": "text", "path": "local",
                                         "name": None, "split": "train"}))
    if args.dataset: src["path"] = args.dataset
    if args.config:  src["name"] = args.config
    if args.field:   src["field"] = args.field
    if args.data_dir: src["data_dir"] = args.data_dir
    eos = None if args.eos_id < 0 else args.eos_id

    # ---- ham metin dökümü: tokenizer'ı EĞİTMEK için örnek toplar ----
    if args.dump_text:
        want = int(args.target_mb * 1_000_000)
        if args.source == "local":
            rows = local_docs(args.input, args.doc_sep)
        else:
            from datasets import load_dataset
            kw = {"data_dir": src["data_dir"]} if src.get("data_dir") else {}
            rows = load_dataset(src["path"], name=src.get("name"),
                                split=src["split"], streaming=True, **kw)
        got, n = 0, 0
        os.makedirs(os.path.dirname(args.dump_text) or ".", exist_ok=True)
        with open(args.dump_text, "w", encoding="utf-8") as out:
            for r in rows:
                t = clean(r[src["field"]], args.source)
                if len(t) < (80 if args.source.startswith("code") else 200):
                    continue
                out.write(t + "\n\n\n")
                got += len(t.encode("utf-8")); n += 1
                if got >= want:
                    break
                if n % 2000 == 0:
                    print(f"  {got/1e6:.1f} / {args.target_mb:.0f} MB  ({n:,} belge)",
                          flush=True)
        print(f"[bitti] {args.dump_text}: {got/1e6:.1f} MB, {n:,} belge")
        print(f"  -> bunu train_tokenizer.py --input içine ver")
        return

    if not args.tokenizer or not args.target:
        raise SystemExit("--tokenizer ve --target gerekli (ya da --dump_text kullan)")
    target = int(args.target)

    _, vocab = load_tokenizer(args.tokenizer)
    if vocab > 65535:
        raise SystemExit(f"vocab {vocab} > 65535 -- uint16'ya sığmaz, "
                         f"uint32'ye geçmen gerekir (dosya 2x büyür)")
    print(f"[tokenizer] vocab={vocab}  (uint16 OK)")

    # ---- kesilmişse devam et ----
    prog_path = args.out + ".progress.json"
    docs_done, tokens_done = 0, 0
    if os.path.exists(prog_path) and os.path.exists(args.out):
        p = json.load(open(prog_path))
        docs_done, tokens_done = p["docs"], p["tokens"]
        on_disk = os.path.getsize(args.out) // 2
        if on_disk < tokens_done:
            # progress dosyanın önünde: elektrik kesintisi ya da dış müdahale
            # (ör. val kesme). ASLA truncate ile şişirme -- sahte token üretir.
            print(f"[uyarı] progress {tokens_done:,} token diyor ama dosyada "
                  f"{on_disk:,} var -> dosyaya güveniliyor")
            tokens_done = on_disk
        else:
            with open(args.out, "r+b") as f:
                f.truncate(tokens_done * 2)   # yarım yazılmış kuyruğu at
        print(f"[devam] {docs_done:,} belge / {tokens_done/1e9:.3f}B token atlanıyor")

    if tokens_done >= target:
        print(f"[bitti] hedef zaten dolu: {tokens_done/1e9:.3f}B")
        return

    if args.source == "local":
        print(f"[yerel] {args.input}")
        ds = local_docs(args.input, args.doc_sep)
        if docs_done:
            ds = (d for i, d in enumerate(ds) if i >= docs_done)
    else:
        from datasets import load_dataset
        print(f"[akış] {src['path']} :: {src.get('name')} ...")
        kw = {}
        if src.get("data_dir"):
            kw["data_dir"] = src["data_dir"]
        try:
            ds = load_dataset(src["path"], name=src.get("name"),
                              split=src["split"], streaming=True, **kw)
        except Exception as e:
            raise SystemExit(
            f"[hata] {src['path']} açılamadı: {e}\n"
            f"  * gated dataset ise HF sayfasında şartları kabul et + "
            f"`huggingface-cli login`\n"
            f"  * config/field adı değişmiş olabilir: "
            f"--dataset/--config/--field/--data_dir ile ezebilirsin")
    if docs_done:
        ds = ds.skip(docs_done)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out, "ab" if tokens_done else "wb")
    t0, last = time.time(), time.time()
    docs, tokens, skipped = docs_done, tokens_done, 0

    def batches():
        buf = []
        for row in ds:
            buf.append((row[src["field"]], args.source, eos))
            if len(buf) >= args.batch:
                yield buf; buf = []
        if buf:
            yield buf

    try:
        with Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
            for batch in batches():
                for arr in pool.imap_unordered(_encode, batch, chunksize=25):
                    docs += 1
                    if arr is None:
                        skipped += 1
                        continue
                    f.write(arr.tobytes())
                    tokens += len(arr)
                if time.time() - last > 20:
                    last = time.time()
                    el = time.time() - t0
                    rate = (tokens - tokens_done) / max(el, 1)
                    eta = (target - tokens) / max(rate, 1) / 3600
                    print(f"  {tokens/1e9:.3f}B / {target/1e9:.1f}B token | "
                          f"{docs:,} belge ({skipped:,} elendi) | "
                          f"{rate/1e6:.2f}M tok/s | ETA {eta:.1f} sa", flush=True)
                    f.flush(); os.fsync(f.fileno())   # gerçekten diske in
                    json.dump({"docs": docs, "tokens": tokens}, open(prog_path, "w"))
                if tokens >= target:
                    print("[hedef doldu]")
                    break
    except KeyboardInterrupt:
        print("\n[kesildi] ilerleme kaydedildi, aynı komutla devam edebilirsin")
    finally:
        f.close()
        json.dump({"docs": docs, "tokens": tokens}, open(prog_path, "w"))

    got = os.path.getsize(args.out) // 2
    print(f"\n[bitti] {args.out}")
    print(f"  {got:,} token ({got/1e9:.3f}B) | {os.path.getsize(args.out)/1e9:.1f} GB")
    print(f"  {docs:,} belge işlendi, {skipped:,} elendi (<200 karakter)")
    if got < target * 0.95:
        print(f"  ⚠️  hedefin altında -- kaynak tükendi. Bu, o kaynağın "
              f"gerçek büyüklüğü.")


if __name__ == "__main__":
    main()
