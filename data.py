"""Veri yükleyici -- ADIM-İNDEKSLİ, deterministik.

⚠️ Bu dosyanın tek amacı şu bug'ı önlemek:

  Resume'da loader baştan başlarsa model ilk token'ları 20 kez görür ve
  corpus'un sonunu HİÇ görmez. Ve loss tamamen normal görünür.
  Aylar sonra fark edersin.

Çözüm: batch tamamen adım numarasının fonksiyonu.
  get_batch(step=5000) her zaman aynı veriyi döner -- hangi makinede,
  kaçıncı yeniden başlatmada olduğun fark etmez.
Resume = adım numarasını geri koy. Loader durumu diye bir şey yok.
"""
import os

import numpy as np
import torch


class BinDataset:
    def __init__(self, path, seq_len, dtype="uint16", seed=1337):
        self.path, self.seq_len, self.seed = path, seq_len, seed
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(path, dtype=self.dtype, mode="r")
        self.n_tokens = len(self.data)
        if self.n_tokens < seq_len + 1:
            raise ValueError(f"{path}: {self.n_tokens} token, seq_len={seq_len} icin az")

    def __repr__(self):
        return (f"BinDataset({self.path}, {self.n_tokens/1e9:.3f}B token, "
                f"{self.dtype}, {self.n_tokens*self.dtype.itemsize/1e9:.1f}GB)")

    def get_batch(self, step, batch_size, rank=0, world_size=1, device="cuda"):
        """step + rank -> deterministik batch. Tüm dünya için tek seed."""
        g = np.random.default_rng(self.seed + step)
        # önce dünyanın tamamı için çek, sonra rank'ine düşeni al ->
        # world_size değişse bile (2xT4 -> 1xT4 geçişi) veri sırası bozulmaz
        total = batch_size * world_size
        ix = g.integers(0, self.n_tokens - self.seq_len - 1, size=total)
        ix = ix[rank * batch_size:(rank + 1) * batch_size]

        x = np.stack([self.data[i:i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def epochs_at(self, tokens_seen):
        return tokens_seen / self.n_tokens


class MixtureDataset:
    """Ağırlıklı karışım -- bilgi için Wikipedia'yı FineWeb'den fazla tekrarlar.

    Ağırlık = o kaynaktan gelen token PAYI (kaynağın boyutu değil).
    Küçük ama bilgi-yoğun bir kaynağa yüksek ağırlık verirsen o kaynak
    daha çok epoch görür -- "1000 exposure" kuralının pratikteki karşılığı bu.

    Kaynak seçimi de adım numarasından deterministik: resume güvenli.
    """

    def __init__(self, specs, seq_len, dtype="uint16", seed=1337):
        self.seed = seed
        self.sets = [BinDataset(p, seq_len, dtype, seed + 1000 * i)
                     for i, (p, _) in enumerate(specs)]
        w = np.array([w for _, w in specs], dtype=np.float64)
        self.w = w / w.sum()
        self.n_tokens = sum(d.n_tokens for d in self.sets)
        self.warned = False

    def check_epochs(self, total_tokens, max_epochs=6.0):
        """Bir kaynak kaç kez tekrarlanacak? Aşırı tekrar = ezberleme.

        Küçük bir dosyaya büyük ağırlık vermek kolay bir hata: 45K token'lık
        bir kaynağa %12 pay verirsen model onu 24.000 kez görür ve ezberler.
        Tekrarlanan veri ~4 epoch'a kadar taze veriye denk; sonrası hızla bozulur.
        """
        bad = []
        print("[karışım] hedef token sayısında beklenen epoch:")
        for w, d in zip(self.w, self.sets):
            ep = total_tokens * w / d.n_tokens
            flag = ""
            if ep > max_epochs:
                flag = f"  <-- ÇOK YÜKSEK, ağırlığı {w * max_epochs / ep:.4f}'e düşür"
                bad.append((d.path, ep, w))
            print(f"  {w:6.1%}  {os.path.basename(d.path):<20} "
                  f"{d.n_tokens/1e9:7.3f}B token  ->  {ep:7.2f} epoch{flag}")
        if bad:
            print("\n  ⚠️  Yukarıdaki kaynak(lar) ezberlenecek kadar tekrarlanıyor.")
            print("     Küçük bir dosyaya büyük pay vermek eğitimi bozar --")
            print("     ya ağırlığı düşür ya o kaynağı karışımdan çıkar.")
        return bad

    def __repr__(self):
        return "MixtureDataset(\n" + "\n".join(
            f"    {w:5.1%}  {d}" for w, d in zip(self.w, self.sets)) + "\n)"

    def get_batch(self, step, batch_size, rank=0, world_size=1, device="cuda"):
        # her micro-batch kendi kaynağını çeker -> optimizer adımı içinde karışım
        g = np.random.default_rng(self.seed ^ (step * 2654435761 + 12345))
        i = int(g.choice(len(self.sets), p=self.w))
        return self.sets[i].get_batch(step, batch_size, rank, world_size, device)

    def report(self, tokens_seen):
        out = []
        for w, d in zip(self.w, self.sets):
            out.append(f"{os.path.basename(d.path)}: "
                       f"{tokens_seen * w / d.n_tokens:.2f} epoch")
        return " | ".join(out)


def parse_mix(spec, seq_len, dtype="uint16", seed=1337):
    """'a.bin:0.7,b.bin:0.3' veya tek dosya yolu -> dataset"""
    if ":" not in spec and "," not in spec:
        return BinDataset(spec, seq_len, dtype, seed)
    parts = []
    for chunk in spec.split(","):
        path, _, w = chunk.partition(":")
        parts.append((path.strip(), float(w) if w else 1.0))
    return MixtureDataset(parts, seq_len, dtype, seed)


def inspect(path, dtype="uint16", tokenizer=None):
    """Mevcut .bin'ini kontrol et: token sayısı, vocab tavanı, byte/token."""
    d = np.dtype(dtype)
    arr = np.memmap(path, dtype=d, mode="r")
    n = len(arr)
    sample = np.asarray(arr[:min(n, 20_000_000)])
    print(f"dosya      : {path}")
    print(f"token      : {n:,} ({n/1e9:.3f}B)")
    print(f"dtype      : {d} ({n*d.itemsize/1e9:.2f} GB dosya)")
    print(f"min/max id : {sample.min()} / {sample.max()}  -> vocab en az {sample.max()+1}")
    uniq = len(np.unique(sample))
    print(f"kullanılan : ~{uniq} farklı token (ilk 20M'de)")
    if tokenizer is not None:
        txt = tokenizer.decode(sample[:2000].tolist())
        bpt = len(txt.encode("utf-8")) / 2000
        print(f"byte/token : {bpt:.2f}  <- Türkçe 32k BPE'de 3.5-4.5 iyi, "
              f"<3.0 ise tokenizer zayıf")
        print(f"örnek      : {txt[:300]!r}")
    return n


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("kullanım: python data.py <train.bin> [dtype]")
        raise SystemExit(1)
    inspect(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "uint16")
