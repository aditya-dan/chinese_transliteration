import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Iterator, Optional, List

import torch
from torch.utils.data import Dataset

import jieba
from pypinyin import lazy_pinyin, Style

from transformers import (
    BertModel,
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


# -----------------------------
# Helpers: read wiki JSON/JSONL
# -----------------------------

def iter_files(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        name = p.name.lower()
        suf = p.suffix.lower()

        # accept .json/.jsonl and also wikipedia dump shards like "wiki_00"
        if suf in {".json", ".jsonl"} or name.startswith("wiki_"):
            yield p

def iter_json_objects(fp: Path) -> Iterator[dict]:
    """Supports: single JSON dict, JSON array, or JSON lines."""
    raw = fp.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return

    # Try full json first
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            yield obj
            return
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    yield it
            return
    except Exception:
        pass

    # Fallback: JSON lines
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        except Exception:
            continue

_HTML_RE = re.compile(r"<[^>]+>")
_WIKI_TABLE_RE = re.compile(r"^\s*[!|{|}]=*")  # crude: table markup-ish

def clean_wiki_text(text: str) -> str:
    text = text.replace("\r", "")
    text = _HTML_RE.sub(" ", text)               # remove simple html tags
    # drop obvious table/meta heavy lines
    lines = []
    for ln in text.split("\n"):
        if not ln.strip():
            continue
        if _WIKI_TABLE_RE.match(ln):
            continue
        lines.append(ln.strip())
    return " ".join(lines)


# -----------------------------
# Pinyin word-tokenization
# -----------------------------

_ASCII_RE = re.compile(r"^[A-Za-z0-9_./:+-]+$")

def word_to_token(word: str) -> Optional[str]:
    w = word.strip()
    if not w:
        return None

    # Keep simple ASCII tokens (optional, but often useful for dates/names/urls)
    if _ASCII_RE.match(w):
        return w.lower()

    # Convert to pinyin (concatenate syllables) for Chinese-ish words
    pys = lazy_pinyin(w, style=Style.NORMAL, errors="ignore")
    tok = "".join([p for p in pys if p])
    return tok if tok else None

def text_to_pinyin_tokens(text: str) -> List[str]:
    words = jieba.lcut(text)
    toks = []
    for w in words:
        tok = word_to_token(w)
        if tok:
            toks.append(tok)
    return toks


# -----------------------------
# Build pinyin token corpus + vocab
# -----------------------------

def build_pinyin_corpus_and_vocab(
    wiki_root: str,
    out_corpus_path: str,
    min_freq: int,
    max_docs: Optional[int] = None,
) -> Dict[str, int]:
    root = Path(wiki_root)
    outp = Path(out_corpus_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    counter = Counter()
    n_docs = 0
    n_written = 0

    with outp.open("w", encoding="utf-8") as fout:
        for fp in iter_files(root):
            for obj in iter_json_objects(fp):
                txt = obj.get("text")
                if not isinstance(txt, str) or not txt.strip():
                    continue
                txt = clean_wiki_text(txt)
                if not txt:
                    continue

                toks = text_to_pinyin_tokens(txt)
                if not toks:
                    continue

                fout.write(" ".join(toks) + "\n")
                counter.update(toks)
                n_written += 1

                n_docs += 1
                if max_docs is not None and n_docs >= max_docs:
                    break
            if max_docs is not None and n_docs >= max_docs:
                break

    # Build vocab
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for tok, c in counter.items():
        if c >= min_freq:
            vocab.append(tok)

    vocab_dict = {w: i for i, w in enumerate(vocab)}
    print(f"[Corpus] wrote lines: {n_written} -> {outp}")
    print(f"[Vocab] min_freq={min_freq} | size={len(vocab_dict)}")
    return vocab_dict


# -----------------------------
# Tokenizer: WordLevel + whitespace
# -----------------------------

def create_wordlevel_tokenizer(pinyin_vocab: Dict[str, int], save_dir: str) -> PreTrainedTokenizerFast:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # tokenizers expects vocab dict[token->id]
    tok = Tokenizer(WordLevel(vocab=pinyin_vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    fast.save_pretrained(str(save_path))
    return fast


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki_root", type=str, required=True, help="e.g. /xxx/xxx/wiki_zh")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--chinese_bert", type=str, default="hfl/chinese-bert-wwm-ext")
    ap.add_argument("--min_freq", type=int, default=30)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_docs", type=int, default=None, help="optional: limit docs for quick test")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = str(out_dir / "pinyin_tokens_wordlevel.txt")
    tokenizer_dir = str(out_dir / "tokenizer")

    # 1) Build corpus + vocab
    vocab = build_pinyin_corpus_and_vocab(
        wiki_root=args.wiki_root,
        out_corpus_path=corpus_path,
        min_freq=args.min_freq,
        max_docs=args.max_docs,
    )

    # 2) Tokenizer
    tokenizer = create_wordlevel_tokenizer(vocab, tokenizer_dir)

    # 3) Model
    model = initialize_adapted_model(args.chinese_bert, vocab, tokenizer)

    # 4) Dataset + MLM
    dataset = LineOffsetDataset(corpus_path, tokenizer, max_length=args.max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir / "runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=2000,
        logging_steps=200,
        learning_rate=5e-5,
        warmup_steps=1000,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        gradient_accumulation_steps=8,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("[Train] starting...")
    trainer.train()

    final_dir = out_dir / "final_model"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[Done] saved -> {final_dir}")

if __name__ == "__main__":
    main()