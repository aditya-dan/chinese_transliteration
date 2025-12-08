import os
import json
import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from transformers import (
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelForMaskedLM,
)

# -----------------------------
# Dataset: line offsets (doesn't load all text into RAM)
# -----------------------------

class LineOffsetDataset(Dataset):
    def __init__(self, text_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int = 128):
        self.text_path = text_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Build offsets for random access
        self.offsets = []
        with open(text_path, "rb") as f:
            off = f.tell()
            line = f.readline()
            while line:
                if line.strip():
                    self.offsets.append(off)
                off = f.tell()
                line = f.readline()

        if not self.offsets:
            raise RuntimeError(f"No non-empty lines found in {text_path}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        off = self.offsets[idx]
        with open(self.text_path, "rb") as f:
            f.seek(off)
            line = f.readline().decode("utf-8", errors="ignore").strip()

        enc = self.tokenizer(
            line,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# -----------------------------
# Model adaptation
# -----------------------------

def initialize_adapted_model(chinese_bert_path: str, pinyin_vocab: Dict[str, int], tokenizer: PreTrainedTokenizerFast):
    # Load pretrained MLM model + its tokenizer (so we can map token->id)
    old_tok = AutoTokenizer.from_pretrained(chinese_bert_path, use_fast=True)
    old_mlm = AutoModelForMaskedLM.from_pretrained(chinese_bert_path)  # BertForMaskedLM under the hood

    # Build new config (same as old, but new vocab size + correct special ids)
    config = old_mlm.config
    old_vocab_size = config.vocab_size
    config.vocab_size = len(pinyin_vocab)

    config.pad_token_id  = tokenizer.pad_token_id
    config.unk_token_id  = tokenizer.unk_token_id
    config.cls_token_id  = tokenizer.cls_token_id
    config.sep_token_id  = tokenizer.sep_token_id
    config.mask_token_id = tokenizer.mask_token_id

    print(f"[Model] vocab: {old_vocab_size} -> {config.vocab_size}")

    # Create new MLM model with resized vocab
    new_mlm = type(old_mlm)(config)

    # --- (A) Copy everything we can from the pretrained model, EXCEPT word embeddings / decoder heads ---
    old_sd = old_mlm.state_dict()
    filtered = {}
    skip_prefixes = (
        "bert.embeddings.word_embeddings.",   # word embedding matrix depends on vocab size
        "cls.predictions.decoder.",           # LM head decoder depends on vocab size (often tied anyway)
        "cls.predictions.bias",               # vocab-sized bias
    )
    for k, v in old_sd.items():
        if k.startswith(skip_prefixes):
            continue
        filtered[k] = v

    missing, unexpected = new_mlm.load_state_dict(filtered, strict=False)
    # missing/unexpected are expected due to vocab change
    # print("[Load] missing:", missing)
    # print("[Load] unexpected:", unexpected)

    # --- (B) Init new word embeddings + LM head bias ---
    # Init word embeddings like BERT does
    new_embed = new_mlm.bert.embeddings.word_embeddings.weight.data
    new_embed.normal_(mean=0.0, std=config.initializer_range)

    # Init prediction bias (size = new vocab)
    if hasattr(new_mlm.cls.predictions, "bias") and new_mlm.cls.predictions.bias is not None:
        new_mlm.cls.predictions.bias.data.zero_()

    # Tie decoder weights to embeddings (standard BERT MLM behavior)
    new_mlm.tie_weights()

    # --- (C) Copy overlapping tokens (special tokens + any shared ASCII tokens) ---
    old_vocab = old_tok.get_vocab()          # token -> old_id
    new_vocab = pinyin_vocab                 # token -> new_id

    shared = set(old_vocab.keys()) & set(new_vocab.keys())

    # Always ensure special tokens are included if names match
    specials = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
    shared |= (specials & set(new_vocab.keys()) & set(old_vocab.keys()))

    # Copy embedding rows and LM bias for shared tokens
    old_embed = old_mlm.bert.embeddings.word_embeddings.weight.data
    copied = 0
    for tok in shared:
        old_id = old_vocab[tok]
        new_id = new_vocab[tok]
        new_mlm.bert.embeddings.word_embeddings.weight.data[new_id].copy_(old_embed[old_id])
        if hasattr(old_mlm.cls.predictions, "bias") and old_mlm.cls.predictions.bias is not None:
            new_mlm.cls.predictions.bias.data[new_id].copy_(old_mlm.cls.predictions.bias.data[old_id])
        copied += 1

    new_mlm.tie_weights()  # re-tie after copying

    print(f"[Init] copied {copied} shared token embeddings (includes specials + shared ASCII if present).")
    print(f"[Init] example shared tokens: {list(sorted(shared))[:15]}")
    return new_mlm


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train BERT MLM on pinyin tokens")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed data (from load.py)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--chinese_bert",
        type=str,
        default="hfl/chinese-bert-wwm-ext",
        help="Pretrained Chinese BERT model to adapt"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data paths
    corpus_path = data_dir / "pinyin_tokens.txt"
    tokenizer_dir = data_dir / "tokenizer"
    vocab_path = data_dir / "vocab.json"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}. Run load.py first!")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_dir}. Run load.py first!")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}. Run load.py first!")

    print("=" * 60)
    print("BERT MLM Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Corpus: {corpus_path}")
    print(f"Base model: {args.chinese_bert}")
    print("=" * 60)

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"[Vocab] Loaded vocabulary with {len(vocab):,} tokens")

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    print(f"[Tokenizer] Loaded from {tokenizer_dir}")

    # Initialize model
    print(f"[Model] Initializing adapted model from {args.chinese_bert}...")
    model = initialize_adapted_model(args.chinese_bert, vocab, tokenizer)

    # Create dataset
    print(f"[Dataset] Loading corpus from {corpus_path}...")
    dataset = LineOffsetDataset(str(corpus_path), tokenizer, max_length=args.max_length)
    print(f"[Dataset] Loaded {len(dataset):,} sentences")

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=2000,
        logging_steps=200,
        learning_rate=args.learning_rate,
        warmup_steps=1000,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("\n" + "=" * 60)
    print("[Train] Starting training...")
    print("=" * 60)
    trainer.train()

    # Save final model
    final_dir = out_dir / "final_model"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    print("\n" + "=" * 60)
    print(f"[Done] Model saved to {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()