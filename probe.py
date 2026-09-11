"""Mevcut 100M modelinde BİLGİ var mı, yoksa ÇIKARILAMIYOR mu?

Bunu bilmeden 290M eğitmek körlemesine. 10 dakikalık test, her şeyi belirler.

Kullanım: aşağıdaki load_model() içini kendi modeline göre doldur, sonra:
    python probe.py
"""
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# BURAYI KENDİ MODELİNE GÖRE DOLDUR
# ---------------------------------------------------------------------------
def load_model():
    # HF formatındaysa:
    #   from transformers import AutoModelForCausalLM, AutoTokenizer
    #   tok = AutoTokenizer.from_pretrained("yol/model")
    #   model = AutoModelForCausalLM.from_pretrained("yol/model")
    #   return model.eval(), tok
    #
    # Kendi nanoGPT-vari checkpoint'insa:
    #   ckpt = torch.load("ckpt.pt", map_location="cpu")
    #   model = SeninGPT(ckpt["config"]); model.load_state_dict(ckpt["model"])
    #   tok = senin_tokenizer
    #   return model.eval(), tok
    raise NotImplementedError("load_model() doldur")


# ---------------------------------------------------------------------------
# Test setleri
# ---------------------------------------------------------------------------
# A) TAMAMLAMA formatı -- base model'in DOĞAL formatı. Asıl test bu.
COMPLETION = [
    ("Türkiye'nin başkenti", "Ankara"),
    ("Fransa'nın başkenti", "Paris"),
    ("Türkiye'nin en kalabalık şehri", "İstanbul"),
    ("Mustafa Kemal Atatürk", "1881"),        # doğum yılı gelirse harika
    ("Dünyanın en büyük okyanusu", "Pasifik"),
    ("Su molekülünün formülü", "H2O"),
]

# B) SORU formatı -- base model'den bunu beklemek HATA. Karşılaştırma için.
QUESTION = [
    ("Türkiye'nin başkenti neresidir?", "Ankara"),
    ("Soru: Türkiye'nin başkenti neresidir?\nCevap:", "Ankara"),
]


@torch.no_grad()
def next_token_ranking(model, tok, prompt, target, k=10):
    """Hedef kelimenin ilk token'ı, sıralamada kaçıncı sırada?"""
    ids = tok.encode(prompt, return_tensors="pt") if hasattr(tok, "return_tensors") \
        else torch.tensor([tok.encode(prompt)])
    out = model(ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    probs = F.softmax(logits[0, -1].float(), dim=-1)

    # hedefin ilk token'ı (baştaki boşlukla -- BPE'de fark eder)
    cand_ids = set()
    for variant in (" " + target, target):
        try:
            e = tok.encode(variant)
            e = e["input_ids"] if isinstance(e, dict) else e
            if len(e):
                cand_ids.add(int(e[0]))
        except Exception:
            pass

    order = torch.argsort(probs, descending=True)
    rank, p = None, 0.0
    for cid in cand_ids:
        r = (order == cid).nonzero()
        if len(r):
            r = int(r[0, 0]) + 1
            if rank is None or r < rank:
                rank, p = r, float(probs[cid])

    topk = [(tok.decode([int(i)]), float(probs[i])) for i in order[:k]]
    return rank, p, topk


def verdict(rank):
    if rank is None:        return "❌ ilk 1000'de yok"
    if rank == 1:           return "✅ 1. sırada -- BİLİYOR"
    if rank <= 5:           return "🟡 ilk 5 -- BİLGİ VAR, çıkarım zayıf"
    if rank <= 50:          return "🟠 ilk 50 -- zayıf iz var"
    return "❌ çok gerilerde"


def main():
    model, tok = load_model()
    for title, tests in (("A) TAMAMLAMA (asıl test)", COMPLETION),
                         ("B) SORU (base model'de çalışmaması NORMAL)", QUESTION)):
        print(f"\n{'='*70}\n{title}\n{'='*70}")
        for prompt, target in tests:
            rank, p, topk = next_token_ranking(model, tok, prompt, target)
            print(f"\n  '{prompt}' -> beklenen '{target}'")
            print(f"  {verdict(rank)}   (rank={rank}, p={p:.4f})")
            print("  top-10:", ", ".join(f"{w!r}:{pr:.3f}" for w, pr in topk[:10]))

    print(f"""
{'='*70}
NASIL OKUNUR
{'='*70}
A'da rank<=5 ama B'de kötü
    -> Bilgi ZATEN İÇERİDE, model onu ÇIKARAMIYOR.
       Çözüm: parametre değil, pretrain'e QA formatı karıştırmak.
       290M'e geçmek bunu ÇÖZMEZ.

A'da da B'de de yok
    -> Bilgi gerçekten yok. Token bütçesi + augmentation problemi.
       Çözüm: daha çok token, ansiklopedik veriyi tekrarla, paraphrase.

A'da rank=1
    -> 100M modelin zaten biliyor. Test yöntemin yanlışmış.
       290M'de çok daha iyisi gelir.
""")


if __name__ == "__main__":
    main()
