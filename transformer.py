# transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        #Ubah shape pe agar bisa di-broadcast ke [Time, Batch, Dim]
        pe = pe.unsqueeze(1) # Shape menjadi [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [Time, Batch, d_model]
        """
        # Sesuaikan dengan format [Time, Batch, Dim]
        # self.pe[:x.size(0)] akan mengambil slice [Time, 1, d_model]
        # yang bisa ditambahkan ke x via broadcasting
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz: int, device):
    mask = torch.triu(torch.full((sz, sz), True, dtype=torch.bool, device=device), diagonal=1)
    return mask

class TransformerNMT(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 trg_vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 pad_id: int = 0,
                 bos_id: int = 1,
                 eos_id: int = 2,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.trg_embed = nn.Embedding(trg_vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # PASTIKAN batch_first=False
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=False, norm_first=False
        )
        self.generator = nn.Linear(d_model, trg_vocab_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=d_model**-0.5)

    @torch.no_grad()
    def make_pad_mask(self, seq):
        # seq inputnya [Time, Batch] -> return [Batch, Time]
        return (seq == self.pad_id).t().contiguous()

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.0):
        """
        src: [Tsrc, B], trg: [Ttrg, B]
        """
        src_key_padding_mask = self.make_pad_mask(src)
        tgt_key_padding_mask = self.make_pad_mask(trg)

        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.trg_embed(trg) * math.sqrt(self.d_model))

        ## PERBAIKAN: Gunakan size(0) untuk panjang sekuens, bukan size(1)
        tgt_mask = generate_square_subsequent_mask(trg.size(0), src.device)

        mem = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        out = self.transformer.decoder(
            tgt=tgt_emb, memory=mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        ) # out: [Ttrg, B, C]

        logits = self.generator(out)  # [Ttrg, B, V]
        return logits, None

    @torch.no_grad()
    def greedy_decode(self, src, max_len=50):
        """
        src: [Tsrc, B]
        returns: ys: [Tout, B]
        """
        # Gunakan size(1) untuk batch size
        B = src.size(1) 
        src_key_padding_mask = self.make_pad_mask(src)
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        # Bangun `ys` dalam format [Time, Batch] dari awal
        ys = torch.full((1, B), self.bos_id, dtype=torch.long, device=src.device)

        for _ in range(1, max_len):
            tgt_emb = self.pos_enc(self.trg_embed(ys) * math.sqrt(self.d_model))
            # Gunakan size(0) untuk panjang sekuens
            tgt_mask = generate_square_subsequent_mask(ys.size(0), src.device)
            out = self.transformer.decoder(
                tgt=tgt_emb, memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=self.make_pad_mask(ys),
                memory_key_padding_mask=src_key_padding_mask
            ) # out: [T_current, B, C]
            
            # Ambil output dari token terakhir (dimensi waktu)
            logits = self.generator(out[-1, :, :])  # out[-1, :, :] -> [B, C]
            next_tok = torch.argmax(logits, dim=-1) # [B]

            # Concat di dimensi waktu (dim=0)
            ys = torch.cat([ys, next_tok.unsqueeze(0)], dim=0)
            
            if (next_tok == self.eos_id).all():
                break
        
        # ys sudah dalam format [Tout, B], tidak perlu transpose
        attn_stub = torch.zeros(ys.size(0), B, src.size(0), device=src.device)
        return ys, attn_stub