import unicodedata
from collections import Counter
from pathlib import Path
import argparse
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
import csv
import sentencepiece as spm

from util import *
from encoder import BahdanauEncoder
from decoder import BahdanauDecoder
from attention import BahdanauAttentionQKV
from seq2seq import BahdanauSeq2Seq
from transformer import TransformerNMT 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP = 1.0  # clip grad norm


# ---- Noam/Warmup scheduler  ----
class NoamWarmup:
    """
    Scheduler gaya Noam: lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """
    def __init__(self, optimizer, d_model=256, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.d_model = d_model
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.factor * (self.d_model ** -0.5) * min(
            self._step ** -0.5, self._step * (self.warmup_steps ** -1.5)
        )
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    @property
    def step_num(self):
        return self._step

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/ind.txt', help='Path to txt data')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--tf', type=float, default=0.5, help='Teacher Forcing')
parser.add_argument('--dropout', type=float, default=0.15, help='dropout')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--max_vocab', type=int, default=None)
parser.add_argument('--target_lang', type=str, default='ID', help='Bahasa tujuan')
parser.add_argument('--checkpoint', type=str, default='cat_dog_checkpoint.pth', help='Path to save model checkpoint')
parser.add_argument('--tokenizer', type=str, default='word', choices=['word','sp'],
                    help='Tokenisasi: word-level atau subword (sentencepiece)')
parser.add_argument('--sp_src_model', type=str, default=None, help='Path SentencePiece model source')
parser.add_argument('--sp_trg_model', type=str, default=None, help='Path SentencePiece model target')

parser.add_argument('--model', type=str, default='rnn', choices=['rnn','transformer'],
                    help='Pilih arsitektur: rnn (Bahdanau) atau transformer')  
parser.add_argument('--d_model', type=int, default=256)      
parser.add_argument('--nhead', type=int, default=8)         
parser.add_argument('--enc_layers', type=int, default=4)     
parser.add_argument('--dec_layers', type=int, default=4)       
parser.add_argument('--ff_dim', type=int, default=1024)        
parser.add_argument('--pe_dropout', type=float, default=0.1)    
parser.add_argument('--label_smoothing', type=float, default=0.0)  
parser.add_argument('--warmup_steps', type=int, default=0)      
parser.add_argument('--exp_name', type=str, default='exp')      
args = parser.parse_args()


SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)


def build_vocab(token_lists, min_freq=1, max_size=None, specials=["<pad>", "<bos>", "<eos>", "<unk>"]):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)

    # Filter berdasarkan min_freq lebih dulu
    filtered = [(w, c) for w, c in counter.items() if c >= min_freq]

    # Sort berdasarkan frekuensi
    filtered.sort(key=lambda x: (-x[1], x[0]))  # descending freq, then lexicographically

    # Batasi ukuran jika max_size diberikan
    if max_size is not None:
        filtered = filtered[:max(0, max_size - len(specials))]

    # Inisialisasi vocab dengan SPECIALS
    vocab = {sp: i for i, sp in enumerate(specials)}

    for w, _ in filtered:
        if w not in vocab:
            vocab[w] = len(vocab)

    itos = {i: w for w, i in vocab.items()}
    return vocab, itos


class NMTDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.data = [(to_ids(src, src_vocab), to_ids(trg, trg_vocab)) for src, trg in pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, trg_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)



data_file = Path(args.data_path)

id_itos = None
sp_trg = None

if args.tokenizer == "word":
    pairs = load_pairs(data_file, max_len=20, max_pairs=None)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, 0.8, 0.1)

    # Build vocab: en (source), id (target)
    en_vocab, en_itos = build_vocab([src for src, _ in train_pairs], max_size=args.max_vocab)
    id_vocab, id_itos = build_vocab([tgt for _, tgt in train_pairs], max_size=args.max_vocab)

    with open("en_vocab.json","w") as f: json.dump(en_vocab,f)
    with open("id_vocab.json","w") as f: json.dump(id_vocab,f)

    print(f"EN vocab size: {len(en_vocab)} | ID vocab size: {len(id_vocab)}")

    class NMTDataset(Dataset):
        def __init__(self, pairs, src_vocab, trg_vocab):
            self.data = [(to_ids(src, src_vocab), to_ids(trg, trg_vocab)) for src, trg in pairs]
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            src_ids, trg_ids = self.data[idx]
            return torch.tensor(src_ids), torch.tensor(trg_ids)

    train_ds = NMTDataset(train_pairs, en_vocab, id_vocab)
    val_ds   = NMTDataset(val_pairs,   en_vocab, id_vocab)
    test_ds  = NMTDataset(test_pairs,  en_vocab, id_vocab)

    input_dim, output_dim = len(en_vocab), len(id_vocab)

else:  # sentencepiece
    sp_src = spm.SentencePieceProcessor()
    sp_src.Load(args.sp_src_model)

    sp_trg = spm.SentencePieceProcessor()
    sp_trg.Load(args.sp_trg_model)
    pairs = load_pairs_sp(data_file, sp_src, sp_trg, max_len=100)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, 0.8, 0.1)

    class NMTDatasetSP(Dataset):
        def __init__(self, pairs): self.data = pairs
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            src_ids, trg_ids = self.data[idx]
            return torch.tensor(src_ids), torch.tensor(trg_ids)

    train_ds = NMTDatasetSP(train_pairs)
    val_ds   = NMTDatasetSP(val_pairs)
    test_ds  = NMTDatasetSP(test_pairs)

    input_dim  = sp_src.get_piece_size()
    output_dim = sp_trg.get_piece_size()


BATCH_SIZE = args.batch_size
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# ==== konstruksi model ====
if args.model == 'rnn':
    ENCODER_HIDDEN_SIZE = 512
    DECODER_HIDDEN_SIZE = 256
    ENCODER_EMBEDDING_DIM = 256
    DECODER_EMBEDDING_DIM = 256

    encoder = BahdanauEncoder(input_dim, ENCODER_EMBEDDING_DIM,
                              ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, dropout_p=args.dropout)
    attn = BahdanauAttentionQKV(
        hidden_size=DECODER_HIDDEN_SIZE,
        query_size=DECODER_HIDDEN_SIZE,
        key_size=2 * ENCODER_HIDDEN_SIZE,
        dropout_p=0.0
    )
    decoder = BahdanauDecoder(output_dim, DECODER_EMBEDDING_DIM,
                              ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE,
                              attn, dropout_p=args.dropout)

    seq2seq = BahdanauSeq2Seq(encoder, decoder, device,
                              pad_id=PAD, bos_id=BOS, eos_id=EOS).to(device)
    is_transformer = False

else:  # transformer
    seq2seq = TransformerNMT(
        src_vocab_size=input_dim,
        trg_vocab_size=output_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.pe_dropout,
        pad_id=PAD, bos_id=BOS, eos_id=EOS,
        device=device
    ).to(device)
    is_transformer = True

# ========= Criterion + Optimizer =========
if args.label_smoothing > 0 and is_transformer:
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=args.label_smoothing)
else:
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

optimizer = torch.optim.Adam(seq2seq.parameters(), lr=args.lr)
if args.warmup_steps > 0 and is_transformer:
    scheduler = NoamWarmup(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
else:
    scheduler = None

def epoch_run(model, loader, train=True, teacher_forcing=0.5):
    """
    Untuk Transformer: teacher_forcing diabaikan oleh model.
    Untuk RNN: teacher_forcing digunakan.
    """
    model.train() if train else model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.set_grad_enabled(train):
        for src, trg in tqdm(loader):
            src = src.to(device)  # [B, Tsrc]
            trg = trg.to(device)  # [B, Ttrg]

            # Forward (model unify: outputs [B, Ttrg, V] untuk transformer)
            outputs, _att = model(src, trg, teacher_forcing_ratio=(teacher_forcing if train else 0.0))

            # Untuk Transformer, outputnya [Time, Batch, Vocab]
            # Untuk RNN, outputnya [Time, Batch, Vocab]
            # Logika loss menjadi sama untuk keduanya
            output_for_loss = outputs[:-1, :, :].contiguous()
            target_for_loss = trg[1:, :].contiguous()

            # Reshape untuk CrossEntropyLoss
            logits = output_for_loss.view(-1, outputs.size(-1))
            target = target_for_loss.view(-1)
            loss = criterion(logits, target)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n_tokens = (target != PAD).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


def plot_curves(history, save_prefix="run", fontsize=14):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train")
    plt.plot(epochs, history["val_loss"],   marker="o", label="Val")
    plt.title("Cross-Entropy Loss per Epoch", fontsize=fontsize+2)
    plt.xlabel("Epoch", fontsize=fontsize); plt.ylabel("Loss", fontsize=fontsize)
    plt.grid(True, alpha=0.3); plt.legend(fontsize=fontsize); plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png", dpi=180); plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_ppl"], marker="o", label="Train")
    plt.plot(epochs, history["val_ppl"],   marker="o", label="Val")
    plt.title("Perplexity (PPL) per Epoch", fontsize=fontsize+2)
    plt.xlabel("Epoch", fontsize=fontsize); plt.ylabel("PPL", fontsize=fontsize)
    plt.grid(True, alpha=0.3); plt.legend(fontsize=fontsize); plt.tight_layout()
    plt.savefig(f"{save_prefix}_ppl.png", dpi=180); plt.show()

# -----------------------
# Train loop
# -----------------------

history = { "train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": [], "val_bleu": [] }
best_val = float("inf")

print(f"Running on: {device} | Model: {args.model} | Tokenizer: {args.tokenizer}")

for epoch in range(1, args.epochs + 1):
    # TF schedule untuk RNN (Transformer diabaikan)
    tf = max(0.3, 0.7 - 0.04 * (epoch - 1)) if not is_transformer else 0.0

    train_loss, train_ppl = epoch_run(seq2seq, train_loader, train=True,  teacher_forcing=tf)
    val_loss,   val_ppl   = epoch_run(seq2seq, val_loader,   train=False, teacher_forcing=0.0)
    # BLEU valid
    # (untuk SP: en_itos tidak dipakai di evaluate_sacrebleu, fungsi build string sendiri)
    val_bleu = evaluate_sacrebleu(seq2seq, val_loader, trg_itos=id_itos, sp_trg=sp_trg)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_ppl"].append(train_ppl)
    history["val_ppl"].append(val_ppl)
    history["val_bleu"].append(val_bleu)

    print(f"Epoch {epoch:02d} | TF={tf:.2f} | "
          f"Train Loss {train_loss:.4f} PPL {train_ppl:.2f} | "
          f"Val Loss {val_loss:.4f} PPL {val_ppl:.2f} | "
          f"Val BLEU {val_bleu:.2f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(seq2seq.state_dict(), args.checkpoint)
        print(f"✓ Saved best to {args.checkpoint}")

# -------------------------------
# Save history CSV
# -------------------------------
hist_csv = f"{args.exp_name}_history.csv"
with open(hist_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch","train_loss","val_loss","train_ppl","val_ppl","val_bleu"])
    for i in range(len(history["train_loss"])):
        w.writerow([
            i+1,
            history["train_loss"][i], history["val_loss"][i],
            history["train_ppl"][i],  history["val_ppl"][i],
            history["val_bleu"][i]
        ])
print(f"History saved to {hist_csv}")

# plot_curves(history, save_prefix=args.exp_name, fontsize=14)

# Evaluate on test (best ckpt)
# -------------------------------
seq2seq.load_state_dict(torch.load(args.checkpoint, map_location=device))
test_loss, test_ppl = epoch_run(seq2seq, test_loader, train=False, teacher_forcing=0.0)

# Helper function to stop decoding at <eos> for SentencePiece
def clean_sp_ids(ids_list, eos_id):
    """Memotong list token ID setelah token EOS pertama."""
    try:
        # Cari indeks dari token EOS pertama
        eos_index = ids_list.index(eos_id)
        # Kembalikan list hanya sampai sebelum token EOS
        return ids_list[1:eos_index] # Kita skip <bos> di awal (indeks 0)
    except ValueError:
        # Jika EOS tidak ditemukan (misal: prediksi mencapai max_len),
        # kembalikan list seperti semula (tanpa <bos>)
        return ids_list[1:]

# a few greedy samples + BLEU test
seq2seq.eval()
with torch.no_grad():
    n_show = 5
    shown = 0
    for src, trg in test_loader:
        src = src.to(device); trg = trg.to(device)   # Shape: [B, T]
        ys, _ = seq2seq.greedy_decode(src, max_len=40) # Shape: [Tout, B]
        
        B = src.size(1) # <-- SEKARANG SUDAH BENAR, karena src adalah [T, B]
        for b in range(min(B, n_show - shown)):
            if args.tokenizer == "sp":
                pred_ids_list = ys[:, b].tolist()
                # src dan trg sekarang [T, B], 
                src_ids_list  = src[:, b].tolist() 
                trg_ids_list  = trg[:, b].tolist()

                # 2. Bersihkan list dari token setelah <eos> menggunakan helper function
                cleaned_pred_ids = clean_sp_ids(pred_ids_list, EOS)
                cleaned_src_ids  = clean_sp_ids(src_ids_list, EOS)
                cleaned_trg_ids  = clean_sp_ids(trg_ids_list, EOS)

                # 3. Decode hanya list yang sudah bersih
                pred_txt = sp_trg.decode(cleaned_pred_ids)
                src_txt  = sp_src.decode(cleaned_src_ids)
                trg_txt  = sp_trg.decode(cleaned_trg_ids)
            else:
                pred_txt = decode_ids(ys[:, b], id_itos)
                # src dan trg sekarang [T, B]
                src_txt  = decode_ids(src[:, b], en_itos)
                trg_txt  = decode_ids(trg[:, b], id_itos)

            print("-" * 60)
            print("SRC :", src_txt)
            print("TRG :", trg_txt)
            print("PRED:", pred_txt)
            shown += 1
        if shown >= n_show:
            break

test_bleu = evaluate_sacrebleu(
    seq2seq, 
    test_loader, 
    trg_itos=id_itos if args.tokenizer == 'word' else None,
    sp_trg=sp_trg if args.tokenizer == 'sp' else None
)
print(f"TEST | Loss {test_loss:.4f} | PPL {test_ppl:.2f} | SacreBLEU {test_bleu:.2f}")


