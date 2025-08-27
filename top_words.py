import matplotlib.pyplot as plt
from collections import Counter
from util import *
import string


def plot_top_words_vertical(
    pairs,
    lang1_name="Inggris",
    lang2_name="Indonesia",
    top_n=20,
    remove_punct=True,
    fontsize=20
):
    """
    pairs: list of (lang1_tokens, lang2_tokens)
    """

    # Flatten token list per language
    lang1_tokens = [tok for src, _ in pairs for tok in src]
    lang2_tokens = [tok for _, tgt in pairs for tok in tgt]

    if remove_punct:
        # buang token yang hanya tanda baca/angka (opsional)
        punct = set(string.punctuation)
        lang1_tokens = [t for t in lang1_tokens if any(ch.isalpha() for ch in t) and t not in punct]
        lang2_tokens = [t for t in lang2_tokens if any(ch.isalpha() for ch in t) and t not in punct]

    # Hitung frekuensi
    c1 = Counter(lang1_tokens).most_common(top_n)
    c2 = Counter(lang2_tokens).most_common(top_n)

    words1, counts1 = zip(*c1) if c1 else ([], [])
    words2, counts2 = zip(*c2) if c2 else ([], [])

    # --- PERBAIKAN: Ubah tata letak menjadi 2 baris, 1 kolom & sesuaikan figsize ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 20)) # Lebih tinggi dari lebar

    # --- Language 1 (atas) ---
    axes[0].barh(words1, counts1, color='skyblue')
    axes[0].invert_yaxis()  # frekuensi tertinggi di paling atas
    axes[0].set_title(f"Top {top_n} kata-kata bahasa {lang1_name}", fontsize=fontsize+2)
    axes[0].set_xlabel("Jumlah", fontsize=fontsize)
    axes[0].tick_params(axis='both', labelsize=fontsize)

    # --- Language 2 (bawah) ---
    axes[1].barh(words2, counts2, color='salmon')
    axes[1].invert_yaxis()
    axes[1].set_title(f"Top {top_n} kata-kata bahasa {lang2_name}", fontsize=fontsize+2)
    axes[1].set_xlabel("Jumlah", fontsize=fontsize)
    axes[1].tick_params(axis='both', labelsize=fontsize)

    plt.tight_layout(pad=3.0) # Beri sedikit padding agar judul tidak tumpang tindih
    plt.show()

# Example usage
# Muat dataset Inggris-Indonesia Anda.
# Ganti "data/ind.txt" dengan path file yang benar.
pairs = load_pairs("data/ind.txt", max_len=20)
# Sesuaikan nama bahasa: lang1=Source (Inggris), lang2=Target (Indonesia)
plot_top_words_vertical(pairs, lang1_name="Inggris", lang2_name="Indonesia", top_n=20, fontsize=16)