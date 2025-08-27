# File: util.py

# File ini berisi fungsi-fungsi utilitas untuk:
# 1. Membaca dataset (bilingual: eng - in).
# 2. Membersihkan teks (opsional, misalnya hapus spasi ganda).
# 3. Membuat model tokenisasi subword (SentencePiece).
# 4. Menyimpan dan memuat tokenizer.
# 5. Membagi dataset ke train / valid / test.

import unicodedata
import re
import numpy as np
import random
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch
import sentencepiece as spm
import sacrebleu
import string 
def train_sentencepiece(input_file, model_prefix, vocab_size=8000, character_coverage=1.0, model_type="bpe"):
    """
    Latih SentencePiece model untuk subword tokenization.
    Hasil: spm.model + spm.vocab
    """
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type
    )

def load_sentencepiece(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

def load_pairs_sp(path, sp_src, sp_tgt, max_len=100, max_pairs=None):
    """
    Load pasangan kalimat dan tokenize pakai SentencePiece.
    sp_src: SentencePiece tokenizer untuk source
    sp_tgt: SentencePiece tokenizer untuk target
    """
    pairs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue

            src, tgt = cols[0], cols[1]
            src_ids = [BOS] + sp_src.encode(src, out_type=int) + [EOS]
            tgt_ids = [BOS] + sp_tgt.encode(tgt, out_type=int) + [EOS]

            if 1 <= len(src_ids) <= max_len and 1 <= len(tgt_ids) <= max_len:
                pairs.append((src_ids, tgt_ids))

            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)

def clean_sp_ids(ids_list, eos_id):
    """Memotong list token ID setelah token EOS pertama."""
    try:
        eos_index = ids_list.index(eos_id)
        # Kembalikan list dari setelah <bos> hingga sebelum <eos>
        return ids_list[1:eos_index] 
    except ValueError:
        # Jika EOS tidak ditemukan, kembalikan semua kecuali <bos>
        return ids_list[1:]

def collate_batch(batch):
    """
    batch: list of (src_ids[T1], trg_ids[T2]).
    Returns:
      src_pad: [Tsrc, B]
      trg_pad: [Ttrg, B]
      src_lens, trg_lens (optional if you need)
    """
    src_seqs, trg_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    trg_lens = [len(t) for t in trg_seqs]

    max_src = max(src_lens)
    max_trg = max(trg_lens)

    padded_src = torch.full((len(batch), max(len(s) for s in src_seqs)), PAD, dtype=torch.long)
    padded_trg = torch.full((len(batch), max(len(t) for t in trg_seqs)), PAD, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, trg_seqs)):
        padded_src[i, : len(s)] = s
        padded_trg[i, : len(t)] = t

     # Transpose ke [T, B] agar sesuai dengan ekspektasi model RNN
    return padded_src.t().contiguous(), padded_trg.t().contiguous()

def decode_ids(ids, itos, src=None, src_itos=None, return_tokens=False):
    tokens = []
    for i, tok_id in enumerate(ids):
        tok = tok_id.item()
        if tok == EOS:
            break
        if tok == PAD or tok == BOS:
            continue
        if tok == UNK and src is not None and src_itos is not None:
            if i < len(src):
                tokens.append(src_itos.get(src[i].item(), "<src-unk>"))
            else:
                tokens.append("<unk>")
        else:
            tokens.append(itos.get(tok, "<unk>"))
    return tokens if return_tokens else " ".join(tokens)

def evaluate_sacrebleu(model, loader, trg_itos=None, sp_trg=None, max_len=40):
    """
    Evaluasi dengan SacreBLEU, kompatibel dengan word-level dan SentencePiece.
    """
    model.eval()
    refs, hyps = [], []

    with torch.no_grad():
        for src, trg in loader:
            src = src.to(model.device)   # Shape: [B, Tsrc]
            trg = trg.to(model.device)   # Shape: [B, Ttrg]
            pred_ids, _ = model.greedy_decode(src, max_len=max_len) # Shape: [Tout, B]

            B = src.size(1)
            for b in range(B):
                # Slicing kolom [:, b] sudah benar untuk mengambil satu kalimat
                pred_ids_list = pred_ids[:, b].tolist()
                trg_ids_list  = trg[:, b].tolist()
                
                if sp_trg:
                    cleaned_pred_ids = clean_sp_ids(pred_ids_list, EOS)
                    cleaned_trg_ids  = clean_sp_ids(trg_ids_list, EOS)
                    hyp_str = sp_trg.decode(cleaned_pred_ids)
                    ref_str = sp_trg.decode(cleaned_trg_ids)
                else:
                    ref_tokens = decode_ids(trg[:, b], trg_itos, return_tokens=True)
                    hyp_tokens = decode_ids(pred_ids[:, b], trg_itos, return_tokens=True)
                    ref_str = " ".join(ref_tokens)
                    hyp_str = " ".join(hyp_tokens)

                refs.append(ref_str)
                hyps.append(hyp_str)

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score

def save_vocab(vocab, path):
    import json
    with open(path, "w") as f:
        json.dump(vocab, f)

def load_vocab(path):
    import json
    with open(path) as f:
        vocab = json.load(f)
    vocab = {k: int(v) for k, v in vocab.items()}
    itos = {v: k for k, v in vocab.items()}
    return vocab, itos

def load_pairs(path, max_len=20, max_pairs=None):
    pairs = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            cols = line.rstrip("\n").split("\t")

            if len(cols) < 2:
                continue

            src, tgt = cols[0], cols[1]
            # --- PERBAIKAN 1: Tentukan bahasa saat memanggil fungsi normalisasi ---
            src_token = normalize_and_tokenize(src, lang="en") 
            tgt_token = normalize_and_tokenize(tgt, lang="id") 

            if 1 <= len(src_token) <= max_len and 1 <= len(tgt_token) <= max_len:
                pairs.append((src_token, tgt_token))

            if max_pairs and len(pairs) >= max_pairs:
                break

    return pairs
def normalize_and_tokenize(text, lang="en"):
    """
    Normalize and tokenize text.
    - English: lower, strip, filter punctuation
    - Indonesian: lower, strip, filter punctuation
    """
    text = text.strip()
    if lang == "en":
        # normalisasi english
        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
        # Ganti regex yang tidak valid dengan yang valid ---
        # Pola lama (error): text = re.sub(r"[^a-z0-9\s\p{P}]", "", text)
        # Pola baru:
        allowed_chars = "a-z0-9\s" + re.escape(string.punctuation)
        text = re.sub(f"[^{allowed_chars}]", "", text)

    elif lang == "id":
        # normalisasi untuk Bahasa Indonesia
        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
        # Pola regex yang sama dengan bahasa Inggris cocok untuk bahasa Indonesia
        allowed_chars = "a-z0-9\s" + re.escape(string.punctuation)
        text = re.sub(f"[^{allowed_chars}]", "", text)
    else:
        # fallback umum
        text = unicodedata.normalize("NFKC", text)

    tokens = text.split()
    return tokens

def to_ids(tokens, vocab, unk_log=None):
    ids = [BOS]
    for tok in tokens:
        tok_id = vocab.get(tok, UNK)
        if tok_id == UNK and unk_log is not None:
            unk_log.append(tok)
        ids.append(tok_id)
    ids.append(EOS)
    return ids

def pad_batch(batch, pad_id=PAD):
    max_len = max(len(x) for x in batch)
    return [seq + [pad_id] * (max_len - len(seq)) for seq in batch]

def split_pairs(pairs, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = pairs[:n_train]
    val = pairs[n_train : n_train + n_val]
    test = pairs[n_train + n_val :]
    return train, val, test