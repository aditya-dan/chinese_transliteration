import argparse
import json
from pathlib import Path
from typing import List, Tuple
from load import word_to_pinyin_token

import torch
import joblib
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, PreTrainedTokenizerFast
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression
from pypinyin import lazy_pinyin, Style

def load_training_sentences(corpus_path: str, max_sentences: int = None) -> List[List[str]]:
    """Load hanzi sentences from corpus for alignment training"""
    sentences = []
    print(f"[Corpus] Loading sentences from {corpus_path}...")
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            words = line.split()
            if words:
                sentences.append(words)
            
            if max_sentences and len(sentences) >= max_sentences:
                break
            
            if line_num % 10000 == 0:
                print(f"[Progress] Loaded {len(sentences):,} sentences")
    
    print(f"[Corpus] Loaded {len(sentences):,} sentences")
    return sentences

def extract_aligned_embeddings(
    hanzi_sentences: List[List[str]],
    bert_model,
    bert_tokenizer: PreTrainedTokenizerFast,
    sgns_model: Word2Vec,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract aligned pinyin and hanzi embeddings from sentences.
    
    Returns:
        pinyin_embeddings: (N, bert_dim) array
        hanzi_embeddings: (N, sgns_dim) array
    """
    bert_model.to(device)
    bert_model.eval()
    
    all_pinyin_embs = []
    all_hanzi_embs = []
    
    print(f"[Extraction] Processing {len(hanzi_sentences):,} sentences...")
    
    # Process sentences in batches
    for batch_start in tqdm(range(0, len(hanzi_sentences), batch_size), desc="Extracting embeddings"):
        batch_sentences = hanzi_sentences[batch_start:batch_start + batch_size]
        
        for hanzi_tokens in batch_sentences:
            # Convert hanzi tokens to pinyin
            pinyin_tokens = []
            valid_hanzi_tokens = []
            
            for hanzi_token in hanzi_tokens:
                py_token = word_to_pinyin_token(hanzi_token)
                if py_token and hanzi_token in sgns_model.wv:
                    pinyin_tokens.append(py_token)
                    valid_hanzi_tokens.append(hanzi_token)
            
            if not pinyin_tokens:
                continue
            
            # Tokenize pinyin sentence for BERT
            pinyin_text = " ".join(pinyin_tokens)
            encoded = bert_tokenizer(
                pinyin_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=128,
            )
            
            offsets = encoded.pop("offset_mapping")[0]
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = bert_model(**encoded, output_hidden_states=True)
            
            # Use last hidden state
            bert_embeddings = outputs.hidden_states[-1][0].cpu()  # (seq_len, hidden_dim)
            
            # Align pinyin tokens to BERT subword tokens
            pinyin_spans = []
            char_pos = 0
            for py in pinyin_tokens:
                start = char_pos
                end = char_pos + len(py)
                pinyin_spans.append((start, end))
                char_pos = end + 1  # +1 for space
            
            # Find matching BERT token for each pinyin token
            for i, (p_start, p_end) in enumerate(pinyin_spans):
                matched_index = None
                
                for j, (b_start, b_end) in enumerate(offsets.tolist()):
                    # Skip special tokens
                    if b_start == 0 and b_end == 0:
                        continue
                    
                    # Check for overlap
                    if not (b_end <= p_start or b_start >= p_end):
                        matched_index = j
                        break
                
                if matched_index is None:
                    continue
                
                hanzi_token = valid_hanzi_tokens[i]
                
                # Get embeddings
                pinyin_emb = bert_embeddings[matched_index].numpy()
                hanzi_emb = sgns_model.wv[hanzi_token]
                
                all_pinyin_embs.append(pinyin_emb)
                all_hanzi_embs.append(hanzi_emb)
    
    print(f"[Extraction] Extracted {len(all_pinyin_embs):,} aligned embedding pairs")
    
    return np.array(all_pinyin_embs), np.array(all_hanzi_embs)


def train_linear_regression(
    pinyin_embeddings: np.ndarray,
    hanzi_embeddings: np.ndarray,
    save_path: str,
):
    """
    Train linear regression to map pinyin embeddings to hanzi embeddings.
    
    The regression learns: hanzi_emb ≈ W @ pinyin_emb
    where W is a (sgns_dim, bert_dim) matrix
    """
    print("\n" + "=" * 60)
    print("Training Linear Regression")
    print("=" * 60)
    print(f"Pinyin embeddings shape: {pinyin_embeddings.shape}")
    print(f"Hanzi embeddings shape: {hanzi_embeddings.shape}")
    print(f"Number of training pairs: {len(pinyin_embeddings):,}")
    print("=" * 60 + "\n")
    
    # Train regression (no intercept for embedding space mapping)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(pinyin_embeddings, hanzi_embeddings)
    
    # Compute training R² score
    score = reg.score(pinyin_embeddings, hanzi_embeddings)
    
    W = reg.coef_
    print(f"[Regression] Training complete!")
    print(f"[Regression] Weight matrix shape: {W.shape}")
    print(f"[Regression] R² score on training data: {score:.4f}")
    
    # Save model
    joblib.dump(reg, save_path)
    print(f"[Regression] Saved to {save_path}")
    
    return reg


def main():
    parser = argparse.ArgumentParser(
        description="Train linear regression to map BERT pinyin embeddings to Word2Vec hanzi embeddings"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed data (from load.py)"
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        required=True,
        help="Path to trained BERT model directory"
    )
    parser.add_argument(
        "--sgns_model",
        type=str,
        required=True,
        help="Path to trained Word2Vec SGNS model file"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="pinyin_to_hanzi_regression.joblib",
        help="Output path for regression model"
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=100000,
        help="Maximum number of sentences to use for training (default: 100000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for BERT inference (default: 32)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    hanzi_corpus_path = data_dir / "hanzi_tokens.txt"
    
    if not hanzi_corpus_path.exists():
        raise FileNotFoundError(f"Hanzi corpus not found: {hanzi_corpus_path}")
    
    print("=" * 60)
    print("Linear Regression Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"BERT model: {args.bert_model}")
    print(f"SGNS model: {args.sgns_model}")
    print(f"Output: {args.out_path}")
    print(f"Max sentences: {args.max_sentences:,}")
    print("=" * 60 + "\n")
    
    # Load models
    print("[Models] Loading BERT model...")
    bert_tokenizer = PreTrainedTokenizerFast.from_pretrained(args.bert_model)
    bert_model = AutoModelForMaskedLM.from_pretrained(args.bert_model)
    
    print("[Models] Loading Word2Vec SGNS model...")
    sgns_model = Word2Vec.load(args.sgns_model)
    print(f"[Models] SGNS vocabulary size: {len(sgns_model.wv):,}")
    
    # Load training sentences
    hanzi_sentences = load_training_sentences(
        str(hanzi_corpus_path),
        max_sentences=args.max_sentences
    )
    
    # Extract aligned embeddings
    pinyin_embeddings, hanzi_embeddings = extract_aligned_embeddings(
        hanzi_sentences=hanzi_sentences,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        sgns_model=sgns_model,
        batch_size=args.batch_size,
    )
    
    if len(pinyin_embeddings) == 0:
        raise RuntimeError("No aligned embeddings extracted! Check your data and models.")
    
    # Train regression
    reg = train_linear_regression(
        pinyin_embeddings=pinyin_embeddings,
        hanzi_embeddings=hanzi_embeddings,
        save_path=args.out_path,
    )
    
    print("\n" + "=" * 60)
    print("[Done] Regression training complete!")
    print("=" * 60)
    print(f"\nSaved model: {args.out_path}")
    print("\nUsage example:")
    print("  import joblib")
    print("  reg = joblib.load('pinyin_to_hanzi_regression.joblib')")
    print("  hanzi_emb = reg.predict(pinyin_emb.reshape(1, -1))[0]")
    print("=" * 60)


if __name__ == "__main__":
    main()