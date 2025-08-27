import sentencepiece as spm

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path ke file teks training (satu per baris)")
    p.add_argument("--prefix", required=True, help="Prefix nama file model (mis. spm_en_id)")
    p.add_argument("--vocab_size", type=int, default=8000)
    args = p.parse_args()

    spm.SentencePieceTrainer.Train(
        f"--input={args.input} --model_prefix={args.prefix} "
        f"--vocab_size={args.vocab_size} --character_coverage=1.0 --model_type=bpe"
    )

    print(f"Tokenizer saved: {args.prefix}.model / {args.prefix}.vocab")
