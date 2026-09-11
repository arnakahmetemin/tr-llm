"""32k Türkçe BPE tokenizer eğit -- FİNAL VERİ KARIŞIMININ örneği üzerinde.

Altın kural: tokenizer'ı, eğiteceğin veriyle AYNI oranlarda bir örnek
üzerinde eğit. Sadece Türkçe metinle eğitip sonra %8 kod beslersen,
kod token başına 2 byte'a düşer -- bozulmaz ama pahalıya gelir.

Türkçeye özel iki tuzak, ikisi de burada kapalı:
  * LOWERCASE YAPMA. Türkçede I/ı ve İ/i ayrı harfler; küçültme "İstanbul"u
    "i̇stanbul" yapar ve bilgi kaybolur.
  * ByteLevel kullan -> hiçbir girdi <unk> olamaz. "Bozulma" diye bir
    ihtimal kalmaz, sadece verimlilik farkı kalır.

Kullanım:
    python train_tokenizer.py --out tokenizer.json --vocab 32768 \
        --input "tr_ornek.txt:0.80,kod_ornek.txt:0.08,kendi.txt:0.12"
"""
import argparse, os, random, unicodedata


def sample_lines(path, share, budget_bytes, rng):
    """Dosyadan payına düşen kadar byte oku (baştan değil, serpiştirerek)."""
    want = int(budget_bytes * share)
    size = os.path.getsize(path)
    out, got = [], 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        if size <= want:                       # küçük dosya: tamamını al
            return [f.read()], size
        # büyük dosya: rastgele noktalardan bloklar çek
        while got < want:
            f.seek(rng.randrange(0, max(1, size - 200_000)))
            f.readline()                       # yarım satırı at
            block = f.read(200_000)
            if not block:
                continue
            out.append(block)
            got += len(block.encode("utf-8", "ignore"))
    return out, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="'a.txt:0.8,b.txt:0.2' -- paylar final karışımınla aynı olsun")
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--budget_mb", type=int, default=500,
                    help="tokenizer eğitimi için toplam örnek boyutu")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if args.vocab > 65535:
        raise SystemExit("vocab > 65535 -> uint16 .bin kullanamazsın")

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

    rng = random.Random(args.seed)
    specs = []
    for chunk in args.input.split(","):
        p, _, w = chunk.partition(":")
        specs.append((p.strip(), float(w) if w else 1.0))
    tot = sum(w for _, w in specs)
    specs = [(p, w / tot) for p, w in specs]

    print("örnekleniyor (paylar final karışımınla aynı olmalı):")
    corpus = []
    for p, w in specs:
        if not os.path.exists(p):
            raise SystemExit(f"yok: {p}")
        blocks, got = sample_lines(p, w, args.budget_mb * 1_000_000, rng)
        corpus.extend(blocks)
        print(f"  {w:5.1%}  {p:<32} {got/1e6:7.1f} MB")

    tok = Tokenizer(models.BPE(unk_token=None))
    # NFC: Türkçe aksanlı harfler tek kod noktasına toplansın. Lowercase YOK.
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        # rakamları tek tek ayır -> sayılar ve aritmetik çok daha tutarlı
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # 256 byte -> <unk> imkansız
        min_frequency=2,
        show_progress=True,
    )
    print(f"\nBPE eğitiliyor (vocab={args.vocab}) ...")
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(args.out)

    eos = tok.token_to_id("<|endoftext|>")
    print(f"\n[bitti] {args.out}")
    print(f"  vocab   : {tok.get_vocab_size()}")
    print(f"  EOS id  : {eos}     <- prepare_data.py'ye --eos_id {eos} olarak ver")
    print(f"\nŞimdi kontrol et:  python check_tokenizer.py {args.out}")


if __name__ == "__main__":
    main()
