import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import joblib
import jieba
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskedLM, PreTrainedTokenizerFast
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from pypinyin import lazy_pinyin, Style
import re
import jieba
from functools import lru_cache
import csv

# Import consistent tokenization from load.py
_ASCII_RE = re.compile(r"^[A-Za-z0-9_./:+-]+$")

@lru_cache(maxsize=200000)
def word_to_pinyin_token(word: str) -> str:
    """Convert a word to pinyin token (same as in load.py)"""
    w = word.strip()
    if not w:
        return None
    if _ASCII_RE.match(w):
        return w.lower()
    pys = lazy_pinyin(w, style=Style.NORMAL, errors="ignore")
    tok = "".join([p for p in pys if p])
    return tok if tok else None


def text_to_hanzi_tokens(text: str) -> List[str]:
    """
    Convert text to list of Chinese word tokens (for evaluation).
    
    Uses the SAME filtering as load.py to ensure tokens match training data:
    - Segments with jieba
    - Filters out punctuation and tokens that wouldn't produce valid pinyin
    """
    words = jieba.lcut(text)
    toks = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        # Apply the same filtering as pinyin: keep only if it produces a valid pinyin token
        if word_to_pinyin_token(w) is not None:
            toks.append(w)
    return toks


def evaluate_mapping(
    text: str,
    bert_model,
    bert_tokenizer: PreTrainedTokenizerFast,
    sgns_model: Word2Vec,
    regression_matrix: np.ndarray,
    pinyin_hanzi_dict: Dict[str, List[str]],
    result_file_path: str = "result.txt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Evaluate the pinyin-to-hanzi mapping quality.
    
    For each hanzi token in the text:
    1. Convert to pinyin
    2. Get BERT embedding for pinyin
    3. Transform to predicted hanzi embedding using regression
    4. Compare with actual hanzi embedding from SGNS
    """
    bert_model.to(device)
    bert_model.eval()
    
    cosine_similarities = []
    not_closest_counter = 0
    not_in_closest_5 = 0
    word_counter = 0
    results = []
    
    result_file = open(result_file_path,  "w", newline="", encoding="utf-8")
    
    # Filter text using the same logic as load.py (removes punctuation, keeps valid tokens)
    hanzi_tokens = text_to_hanzi_tokens(text)
    
    print(f"[Processing] Original text length: {len(text)} chars")
    print(f"[Processing] After filtering: {len(hanzi_tokens)} tokens")
    print(f"[Processing] Sample tokens: {hanzi_tokens[:10]}")
    
    # Convert hanzi tokens to pinyin
    pinyin_tokens = []
    valid_indices = []
    
    for i, hanzi in enumerate(hanzi_tokens):
        py = word_to_pinyin_token(hanzi)
        if py and hanzi in sgns_model.wv:
            pinyin_tokens.append(py)
            valid_indices.append(i)
    
    if not pinyin_tokens:
        print("[Warning] No valid tokens found in text")
        result_file.close()
        return np.array([])
    
    # Tokenize pinyin for BERT
    pinyin_text = " ".join(pinyin_tokens)
    encoded = bert_tokenizer(
        pinyin_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    
    offsets = encoded.pop("offset_mapping")[0]
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = bert_model(**encoded, output_hidden_states=True)
    
    bert_embeddings = outputs.hidden_states[-1][0].cpu()  # (seq_len, hidden_dim)
    
    # Align pinyin tokens to BERT subword tokens
    pinyin_spans = []
    char_pos = 0
    for py in pinyin_tokens:
        start = char_pos
        end = char_pos + len(py)
        pinyin_spans.append((start, end))
        char_pos = end + 1  # +1 for space
    
    alignment = []
    for (p_start, p_end) in pinyin_spans:
        matched_index = None
        for i, (b_start, b_end) in enumerate(offsets.tolist()):
            if b_start == 0 and b_end == 0:
                continue
            if not (b_end <= p_start or b_start >= p_end):
                matched_index = i
                break
        alignment.append(matched_index)
    
    # Evaluate each token
    for idx, hanzi_idx in enumerate(valid_indices):
        word_counter += 1
        
        hanzi_token = hanzi_tokens[hanzi_idx]
        pinyin_token = pinyin_tokens[idx]
        bert_index = alignment[idx]
        
        if bert_index is None:
            continue
        
        # BERT pinyin embedding
        pinyin_embedding = bert_embeddings[bert_index].numpy()
        
        # Predicted hanzi embedding via regression
        predicted_hanzi_embedding = regression_matrix @ pinyin_embedding
        
        # Actual hanzi embedding from SGNS
        actual_hanzi_embedding = sgns_model.wv[hanzi_token]
        
        # Cosine similarity between actual and predicted
        true_hanzi_similarity = cosine_similarity(
            actual_hanzi_embedding.reshape(1, -1),
            predicted_hanzi_embedding.reshape(1, -1)
        )[0, 0]
        
        cosine_similarities.append(true_hanzi_similarity)
        
        results.append({
            "pinyin": pinyin_token,
            "hanzi": hanzi_token,
            "true_sim": true_hanzi_similarity,
            "is_top5": is_correct_top_k(predicted_hanzi_embedding, hanzi_token, sgns_model),
            "confusions": []  # store higher-sim hanzi with same pinyin
        })
        
        # Check if true hanzi is in top 5 neighbors
        if not is_correct_top_k(predicted_hanzi_embedding, hanzi_token, sgns_model):
            not_in_closest_5 += 1
        
        # Check if any other hanzi with same pinyin has higher similarity
        if pinyin_token in pinyin_hanzi_dict:
            for possible_hanzi in pinyin_hanzi_dict[pinyin_token]:
                if possible_hanzi == hanzi_token:
                    continue
                
                # Skip if not in SGNS vocabulary
                if possible_hanzi not in sgns_model.wv:
                    continue
                
                possible_hanzi_embedding = sgns_model.wv[possible_hanzi]
                
                possible_hanzi_similarity = cosine_similarity(
                    possible_hanzi_embedding.reshape(1, -1),
                    predicted_hanzi_embedding.reshape(1, -1)
                )[0, 0]
                
                if true_hanzi_similarity < possible_hanzi_similarity:
                    results[-1]["confusions"].append({
                        "hanzi": possible_hanzi,
                        "sim": possible_hanzi_similarity
                    })
                    not_closest_counter += 1
                    break
                
    results.sort(key=lambda x: x["true_sim"], reverse=True)
    
    writer = csv.writer(result_file)
    writer.writerow([
        "pinyin",
        "true_hanzi",
        "true_sim",
        "confused_hanzi",
        "confused_sim"
    ])

    for r in results:
        # If there are no confusions, still write the true row
        if not r["confusions"]:
            writer.writerow([
                r["pinyin"],
                r["hanzi"],
                f"{r['true_sim']:.6f}",
                "",
                ""
            ])
        else:
            # One row per confusion
            for c in r["confusions"]:
                writer.writerow([
                    r["pinyin"],
                    r["hanzi"],
                    f"{r['true_sim']:.6f}",
                    c["hanzi"],
                    f"{c['sim']:.6f}"
                ])
    
    result_file.close()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total words evaluated: {word_counter}")
    print(f"Words where true hanzi has lower similarity than incorrect hanzi: {not_closest_counter}")
    print(f"Words where true hanzi is not in top 5 neighbors: {not_in_closest_5}")
    
    if cosine_similarities:
        print(f"\nCosine similarity statistics:")
        print(f"  Mean: {np.mean(cosine_similarities):.4f}")
        print(f"  Median: {np.median(cosine_similarities):.4f}")
        print(f"  Std: {np.std(cosine_similarities):.4f}")
        print(f"  Min: {np.min(cosine_similarities):.4f}")
        print(f"  Max: {np.max(cosine_similarities):.4f}")
    
    print(f"\nDetailed results written to: {result_file_path}")
    print("=" * 60)
    
    return np.array(cosine_similarities)


def is_correct_top_k(predicted_vec: np.ndarray, correct_char: str, sgns_model: Word2Vec, k: int = 5) -> bool:
    """Check if the correct character is within top-k nearest neighbors"""
    neighbors = sgns_model.wv.similar_by_vector(predicted_vec, topn=k)
    return correct_char in [w for w, _ in neighbors]


def plot_cosine_similarities(cosine_similarities: np.ndarray, save_path: str = "cosine_similarity_histogram.png"):
    """Plot histogram of cosine similarities"""
    if len(cosine_similarities) == 0:
        print("[Warning] No similarities to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(cosine_similarities, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Cosine Similarities\n(Actual vs Predicted Hanzi Embeddings)", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add vertical line for mean
    mean_sim = np.mean(cosine_similarities)
    plt.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved histogram to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pinyin-to-hanzi embedding mapping quality"
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
        "--regression_model",
        type=str,
        required=True,
        help="Path to trained regression model (.joblib)"
    )
    parser.add_argument(
        "--pinyin_hanzi_dict",
        type=str,
        required=True,
        help="Path to pinyin-hanzi dictionary JSON file"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="我是一名学生。我喜欢数学。",
        help="Test text to evaluate (default: '我是一名学生。我喜欢数学。')"
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Optional: Path to text file to evaluate (overrides --text)"
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="result.txt",
        help="Output file for detailed results (default: result.txt)"
    )
    parser.add_argument(
        "--plot_file",
        type=str,
        default="cosine_similarity_histogram.png",
        help="Output file for histogram plot (default: cosine_similarity_histogram.png)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Pinyin-to-Hanzi Mapping Evaluation")
    print("=" * 60)
    print(f"BERT model: {args.bert_model}")
    print(f"SGNS model: {args.sgns_model}")
    print(f"Regression model: {args.regression_model}")
    print(f"Pinyin-Hanzi dict: {args.pinyin_hanzi_dict}")
    print("=" * 60 + "\n")
    
    # Load models
    print("[Models] Loading BERT model...")
    bert_tokenizer = PreTrainedTokenizerFast.from_pretrained(args.bert_model)
    bert_model = AutoModelForMaskedLM.from_pretrained(args.bert_model)
    
    print("[Models] Loading Word2Vec SGNS model...")
    sgns_model = Word2Vec.load(args.sgns_model)
    print(f"[Models] SGNS vocabulary size: {len(sgns_model.wv):,}")
    
    print("[Models] Loading regression model...")
    reg = joblib.load(args.regression_model)
    W = reg.coef_
    print(f"[Models] Regression matrix shape: {W.shape}")
    
    print("[Data] Loading pinyin-hanzi dictionary...")
    with open(args.pinyin_hanzi_dict, 'r', encoding='utf-8') as f:
        pinyin_hanzi_dict = json.load(f)
    print(f"[Data] Dictionary has {len(pinyin_hanzi_dict)} pinyin entries")
    
    # Get test text
    if args.text_file:
        print(f"[Data] Loading text from {args.text_file}...")
        with open(args.text_file, 'r', encoding='utf-8') as f:
            test_text = f.read()
    else:
        test_text = args.text
    
    print(f"[Data] Evaluating on {len(test_text)} characters")
    print(f"[Data] Text preview: {test_text[:100]}...")
    
    # Evaluate mapping
    cosine_similarities = evaluate_mapping(
        text=test_text,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        sgns_model=sgns_model,
        regression_matrix=W,
        pinyin_hanzi_dict=pinyin_hanzi_dict,
        result_file_path=args.result_file,
    )
    
    # Plot results
    if len(cosine_similarities) > 0:
        plot_cosine_similarities(cosine_similarities, save_path=args.plot_file)
    
    print("\n" + "=" * 60)
    print("[Done] Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()