"""Tokenizer verimliliğini ölç -- byte/token ("fertility").

NEDEN ŞİMDİ: tokenizer'ı DEĞİŞTİRİRSEN tüm .bin'leri yeniden üretmen gerekir.
7B token üretmeden önceki son geri dönülebilir karar bu.

byte/token ne kadar YÜKSEK, o kadar iyi -- aynı VRAM ve aynı adımda daha çok
gerçek metin işliyorsun. Efektif veri bütçen doğrudan bununla çarpılıyor.

Kullanım: python check_tokenizer.py tokenizer.json
"""
import sys
from prepare_data import load_tokenizer

TESTS = {
    "türkçe (düz)": """Türkiye'nin başkenti Ankara'dır. İstanbul ise ülkenin en
kalabalık şehri olup, Marmara Bölgesi'nde yer almaktadır. Cumhuriyet 29 Ekim
1923 tarihinde ilan edilmiştir.""",

    "türkçe (eklemeli)": """Evlerinizden çıkarılamayacaklarmış gibi davranıyorlardı.
Kitaplaştırılabilirliğini sorgulayanlardanmışsınız. Çalıştırılamayacaklarından
bahsedilebileceğini düşünmüyorum.""",

    "türkçe (ansiklopedik)": """Mustafa Kemal Atatürk (1881, Selanik - 10 Kasım 1938,
İstanbul), Türk asker, devlet adamı ve Türkiye Cumhuriyeti'nin kurucusu.
Kurtuluş Savaşı'nı yöneten Atatürk, 1923-1938 arasında cumhurbaşkanlığı yaptı.""",

    "kod (python)": '''def merge_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    return _merge(left, right)
''',

    "kod (js/json)": '''const config = { maxRetries: 3, timeoutMs: 5000 };
export async function fetchUser(userId) {
  const res = await fetch(`/api/users/${userId}`, { headers });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}''',

    "ingilizce": """The capital of Turkey is Ankara. Istanbul is the most populous
city in the country and serves as its economic and cultural center.""",
}

# byte/token eşikleri (32k civarı vocab için)
def grade(bpt, kind):
    if "kod" in kind:
        lo, hi = 2.2, 3.0
    elif "ingilizce" in kind:
        lo, hi = 3.0, 4.0
    else:
        lo, hi = 3.2, 4.2
    return "✅ iyi" if bpt >= hi else ("🟡 idare eder" if bpt >= lo else "❌ zayıf")


def main(path):
    enc, vocab = load_tokenizer(path)
    print(f"tokenizer : {path}")
    print(f"vocab     : {vocab}"
          f"{'  ⚠️ >65535, uint16 .bin kullanamazsın' if vocab > 65535 else '  (uint16 OK)'}\n")
    print(f"{'içerik':<24}{'byte':>7}{'token':>8}{'byte/token':>12}   değerlendirme")
    print("-" * 72)
    tr = []
    for kind, text in TESTS.items():
        text = " ".join(text.split())
        nb = len(text.encode("utf-8"))
        nt = len(enc(text))
        bpt = nb / max(nt, 1)
        if "türkçe" in kind:
            tr.append(bpt)
        print(f"{kind:<24}{nb:>7}{nt:>8}{bpt:>12.2f}   {grade(bpt, kind)}")

    avg = sum(tr) / len(tr)
    print("-" * 72)
    print(f"\nTürkçe ortalama: {avg:.2f} byte/token")
    if avg >= 3.6:
        print("  ✅ Tokenizer'ın iyi. Değiştirme, mevcut .bin'lerin geçerli kalsın.")
    elif avg >= 3.0:
        print("  🟡 İdare eder. Yeniden eğitmenin kazancı ~%15-20; ")
        print("     mevcut .bin'i yeniden üretme maliyetine değer mi, sana kalmış.")
    else:
        print("  ❌ Zayıf. Yeniden eğitmeye DEĞER -- efektif veri bütçen "
              f"{3.8/avg:.1f}x büyür.")
        print("     Ama şimdi yap: 7B token ürettikten sonra dönüşü çok pahalı.")
    print("\n  Referans: iyi bir 32k Türkçe BPE ~3.5-4.5 byte/token verir.")
    print("  GPT-2/Llama'nın İngilizce-merkezli vocab'ı Türkçede ~2.0-2.5'e düşer.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("kullanım: python check_tokenizer.py <tokenizer.json|klasör|model>")
        raise SystemExit(1)
    main(sys.argv[1])
