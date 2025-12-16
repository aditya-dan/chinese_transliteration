import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Iterator, Optional, List

import jieba
from pypinyin import lazy_pinyin, Style

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


# -----------------------------
# Helpers: read wiki JSON/JSONL
# -----------------------------

def iter_files(root: Path) -> Iterator[Path]:
    """Iterate through all JSON/JSONL files or single files"""
    root = Path(root)
    if root.is_file():
        # If root itself is a file, yield it
        yield root
    else:
        # If root is a directory, yield all files inside recursively
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            name = p.name.lower()
            suf = p.suffix.lower()
            # accept .json/.jsonl or files named like wiki_*
            if suf in {".json", ".jsonl"} or name.startswith("wiki_"):
                yield p



def iter_json_objects(fp: Path) -> Iterator[dict]:
    """Stream line-by-line (JSONL). Works for big wiki_* shards."""
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] != "{":
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue

# -----------------------------
# Text cleaning
# -----------------------------

_HTML_RE = re.compile(r"<[^>]+>")
_WIKI_TABLE_RE = re.compile(r"^\s*[!|{|}]=*")  # crude: table markup-ish


def clean_wiki_text(text: str) -> List[str]:
    """Remove HTML tags and wiki table markup, return list of sentences"""
    text = text.replace("\r", "")
    text = _HTML_RE.sub(" ", text)  # remove simple html tags
    
    # drop obvious table/meta heavy lines
    lines = []
    for ln in text.split("\n"):
        if not ln.strip():
            continue
        if _WIKI_TABLE_RE.match(ln):
            continue
        lines.append(ln.strip())
    
    full_text = " ".join(lines)
    
    # Split into sentences using common Chinese sentence delimiters
    import re
    sentence_delimiters = r'[。！？；\.\!\?;]+'
    sentences = re.split(sentence_delimiters, full_text)
    
    # Filter out empty sentences and very short ones
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


# -----------------------------
# Pinyin tokenization (for BERT)
# -----------------------------

from functools import lru_cache

@lru_cache(maxsize=200000)
def word_to_pinyin_token(word: str) -> Optional[str]:
    """Convert a word to pinyin token"""
    w = word.strip()
    if not w:
        return None

    # Keep simple ASCII tokens (dates/names/urls)
    _ASCII_RE = re.compile(r"^[A-Za-z0-9_./:+-]+$")
    if _ASCII_RE.match(w):
        return w.lower()

    # Convert to pinyin (concatenate syllables) for Chinese words
    pys = lazy_pinyin(w, style=Style.NORMAL, errors="ignore")
    tok = "".join([p for p in pys if p])
    return tok if tok else None


def text_to_pinyin_tokens(text: str) -> List[str]:
    """Convert text to list of pinyin tokens"""
    words = jieba.lcut(text)
    toks = []
    for w in words:
        tok = word_to_pinyin_token(w)
        if tok:
            toks.append(tok)
    return toks


def text_to_hanzi_tokens(text: str) -> List[str]:
    """Convert text to list of Chinese word tokens (for Word2Vec)
    
    Filters tokens the same way as pinyin tokenization to ensure
    both corpora have identical structure (no punctuation, etc.)
    """
    words = jieba.lcut(text)
    toks = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        # Apply the same filtering as pinyin: keep if it would produce a valid pinyin token
        if word_to_pinyin_token(w) is not None:
            toks.append(w)
    return toks


# -----------------------------
# Main processing functions
# -----------------------------

def process_wiki_data(
    wiki_root: str,
    out_dir: str,
    min_freq: int = 30,
    max_docs: Optional[int] = None,
):
    """
    Process Wikipedia data and create:
    1. Pinyin token corpus for BERT (saved as pinyin_tokens.txt)
    2. Hanzi token corpus for Word2Vec (saved as hanzi_tokens.txt)
    3. Pinyin vocabulary dictionary
    """
    root = Path(wiki_root)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pinyin_corpus_path = out_path / "pinyin_tokens.txt"
    hanzi_corpus_path = out_path / "hanzi_tokens.txt"

    pinyin_counter = Counter()
    n_docs = 0
    n_pinyin_lines = 0
    n_hanzi_lines = 0

    print("[Processing] Reading Wikipedia data...")
    
    with pinyin_corpus_path.open("w", encoding="utf-8") as pinyin_out, \
         hanzi_corpus_path.open("w", encoding="utf-8") as hanzi_out:
        
        for fp in iter_files(root):
            print(f"[Processing] File: {fp.name}")
            
            for obj in iter_json_objects(fp):
                txt = obj.get("text")
                if not isinstance(txt, str) or not txt.strip():
                    continue
                
                sentences = clean_wiki_text(txt)
                if not sentences:
                    continue

                # Process each sentence separately
                for sentence in sentences:
                    # Process for BERT (pinyin)
                    pinyin_toks = text_to_pinyin_tokens(sentence)
                    if pinyin_toks:
                        pinyin_out.write(" ".join(pinyin_toks) + "\n")
                        pinyin_counter.update(pinyin_toks)
                        n_pinyin_lines += 1

                    # Process for Word2Vec (hanzi)
                    hanzi_toks = text_to_hanzi_tokens(sentence)
                    if hanzi_toks:
                        hanzi_out.write(" ".join(hanzi_toks) + "\n")
                        n_hanzi_lines += 1

                n_docs += 1
                if n_docs % 1000 == 0:
                    print(f"[Progress] docs={n_docs:,} | pinyin_lines={n_pinyin_lines:,} | "
                          f"hanzi_lines={n_hanzi_lines:,} | vocab_seen={len(pinyin_counter):,}")

                if max_docs is not None and n_docs >= max_docs:
                    break
            
            if max_docs is not None and n_docs >= max_docs:
                break

    # Build pinyin vocabulary (for BERT tokenizer)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for tok, c in pinyin_counter.items():
        if c >= min_freq:
            vocab.append(tok)

    vocab_dict = {w: i for i, w in enumerate(vocab)}
    
    print(f"\n[Corpus Summary]")
    print(f"  Total documents processed: {n_docs:,}")
    print(f"  Pinyin corpus lines: {n_pinyin_lines:,} -> {pinyin_corpus_path}")
    print(f"  Hanzi corpus lines: {n_hanzi_lines:,} -> {hanzi_corpus_path}")
    print(f"  Pinyin vocab size (min_freq={min_freq}): {len(vocab_dict):,}")
    
    return vocab_dict, str(pinyin_corpus_path), str(hanzi_corpus_path)


def create_pinyin_tokenizer(vocab_dict: Dict[str, int], save_dir: str) -> PreTrainedTokenizerFast:
    """Create and save WordLevel tokenizer for pinyin tokens"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create tokenizer with vocab
    tok = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()

    # Wrap in HuggingFace tokenizer
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    fast.save_pretrained(str(save_path))
    print(f"[Tokenizer] Saved to {save_path}")
    
    return fast


def save_vocab_json(vocab_dict: Dict[str, int], save_path: str):
    """Save vocabulary dictionary as JSON"""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    print(f"[Vocab] Saved vocabulary to {save_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load Wikipedia data and create corpora for BERT and Word2Vec"
    )
    parser.add_argument(
        "--wiki_root",
        type=str,
        required=True,
        help="Path to Wikipedia JSON/JSONL data directory (e.g., /path/to/wiki_zh)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=30,
        help="Minimum frequency for pinyin tokens to include in vocab (default: 30)"
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Optional: limit number of documents for testing (default: process all)"
    )
    
    args = parser.parse_args()

    print("=" * 60)
    print("Wikipedia Data Loader and Preprocessor")
    print("=" * 60)
    print(f"Wiki root: {args.wiki_root}")
    print(f"Output dir: {args.out_dir}")
    print(f"Min frequency: {args.min_freq}")
    print(f"Max docs: {args.max_docs if args.max_docs else 'all'}")
    print("=" * 60)

    # Process Wikipedia data
    vocab_dict, pinyin_corpus, hanzi_corpus = process_wiki_data(
        wiki_root=args.wiki_root,
        out_dir=args.out_dir,
        min_freq=args.min_freq,
        max_docs=args.max_docs,
    )

    # Create and save tokenizer
    tokenizer_dir = os.path.join(args.out_dir, "tokenizer")
    create_pinyin_tokenizer(vocab_dict, tokenizer_dir)

    # Save vocabulary as JSON
    vocab_json_path = os.path.join(args.out_dir, "vocab.json")
    save_vocab_json(vocab_dict, vocab_json_path)

    print("\n" + "=" * 60)
    print("[Done] Data processing complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  1. {pinyin_corpus} - Pinyin tokens for BERT")
    print(f"  2. {hanzi_corpus} - Hanzi tokens for Word2Vec")
    print(f"  3. {tokenizer_dir}/ - BERT tokenizer")
    print(f"  4. {vocab_json_path} - Vocabulary dictionary")
    print("\nNext steps:")
    print("  - Use pinyin_tokens.txt for BERT MLM training")
    print("  - Use hanzi_tokens.txt for Word2Vec/SGNS training")
    print("=" * 60)


if __name__ == "__main__":
    main()