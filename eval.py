from util import *
from encoder import BahdanauEncoder
from decoder import BahdanauDecoder
from attention import BahdanauAttentionQKV
from seq2seq import BahdanauSeq2Seq
import json, torch, argparse
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
from pathlib import Path
import sacrebleu

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/ind.txt')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--sp_src_model', type=str, default=None)
parser.add_argument('--sp_trg_model', type=str, default=None)
parser.add_argument('--tokenizer', type=str, default='word', choices=['word','sp'])
parser.add_argument('--checkpoint', type=str, default='bahdanau_best.pt')
parser.add_argument('--model', type=str, default='rnn', choices=['rnn','transformer'])
args = parser.parse_args()

PAD, BOS, EOS, UNK = 0, 1, 2, 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Dataset & Vocab =====
data_file = Path(args.data_path)

if args.tokenizer == "word":
    with open("en_vocab.json") as f: en_vocab = {k:int(v) for k,v in json.load(f).items()}
    with open("id_vocab.json") as f: id_vocab = {k:int(v) for k,v in json.load(f).items()}
    en_itos = {v:k for k,v in en_vocab.items()}
    id_itos = {v:k for k,v in id_vocab.items()}

    pairs = load_pairs(data_file, max_len=20)
    _, _, test_pairs = split_pairs(pairs, 0.8, 0.1)
    test_ds = [(to_ids(src, en_vocab), to_ids(trg, id_vocab)) for src,trg in test_pairs]
    input_dim, output_dim = len(en_vocab), len(id_vocab)

else:  # sentencepiece
    sp_src = load_sentencepiece(args.sp_src_model)
    sp_trg = load_sentencepiece(args.sp_trg_model)
    pairs = load_pairs_sp(data_file, sp_src, sp_trg, max_len=100)
    _, _, test_pairs = split_pairs(pairs, 0.8, 0.1)
    test_ds = test_pairs
    max_src_id = max(max(seq) for seq,_ in test_pairs)
    max_trg_id = max(max(seq) for _,seq in test_pairs)
    input_dim  = max(max_src_id, EOS) + 1
    output_dim = max(max_trg_id, EOS) + 1

test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

# ===== Model =====
if args.model == "rnn":
    encoder = BahdanauEncoder(input_dim, 256, 512, 256, dropout_p=0.3)
    attn = BahdanauAttentionQKV(256, 256, 1024)
    decoder = BahdanauDecoder(output_dim, 256, 512, 256, attn, dropout_p=0.3)
    seq2seq = BahdanauSeq2Seq(encoder, decoder, device, pad_id=PAD, bos_id=BOS, eos_id=EOS).to(device)
else:
    from transformer import TransformerNMT
    seq2seq = TransformerNMT(
        src_vocab_size=input_dim,
        trg_vocab_size=output_dim,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pad_id=PAD, bos_id=BOS, eos_id=EOS,
        device=device
    ).to(device)

seq2seq.load_state_dict(torch.load(args.checkpoint, map_location=device))
seq2seq.eval()

# ===== Evaluation =====
references, hypotheses = [], []
with torch.no_grad():
    for src, trg in tqdm(test_loader):
        src, trg = src.to(device), trg.to(device)
        ys, _ = seq2seq.greedy_decode(src, max_len=40)

        B = src.size(1)
        for b in range(B):
            if args.tokenizer == "sp":
                pred_txt = sp_trg.decode_ids(ys[:,b].tolist())
                trg_txt  = sp_trg.decode_ids(trg[:,b].tolist())
                ref = trg_txt.split()
                hyp = pred_txt.split()
            else:
                ref = decode_ids(trg[:,b], id_itos, return_tokens=True)
                hyp = decode_ids(ys[:,b], id_itos, return_tokens=True)
                ref = [w for w in ref if w not in {"<pad>","<bos>","<eos>"}]
                hyp = [w for w in hyp if w not in {"<pad>","<bos>","<eos>"}]

            references.append([ref])
            hypotheses.append(hyp)

# ===== NLTK BLEU =====
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
smoothie = SmoothingFunction().method4
bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)*100
print(f"NLTK BLEU: {bleu:.2f}")

# ===== SacreBLEU =====
import sacrebleu
sacre = sacrebleu.corpus_bleu(hypotheses, [references])
print(f"SacreBLEU: {sacre.score:.2f}")

