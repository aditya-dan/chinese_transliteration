import argparse
from pathlib import Path
from typing import List

from gensim.models import Word2Vec
from tqdm import trange

def load_corpus(corpus_path: str) -> List[List[str]]:
    """Load tokenized corpus where each line is a sentence"""
    corpus = []
    print(f"[Corpus] Loading from {corpus_path}...")
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            words = line.split()
            if words:
                corpus.append(words)
            
            if line_num % 100000 == 0:
                print(f"[Progress] Loaded {line_num:,} sentences, {len(corpus):,} non-empty")
    
    print(f"[Corpus] Loaded {len(corpus):,} sentences")
    return corpus

def train_sgns_model(
    corpus: List[List[str]],
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    workers: int,
    save_path: str,
):
    """Train Word2Vec Skip-Gram with Negative Sampling (SGNS) model"""
    print("\n" + "=" * 60)
    print("Training Word2Vec SGNS Model")
    print("=" * 60)
    print(f"Vector size: {vector_size}")
    print(f"Window size: {window}")
    print(f"Min count: {min_count}")
    print(f"Epochs: {epochs}")
    print(f"Workers: {workers}")
    print("=" * 60 + "\n")
    
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # Skip-gram (1 = skip-gram, 0 = CBOW)
        workers=workers
    )

    # Build vocab
    model.build_vocab(corpus)

    print(f"Training on {model.corpus_count:,} sentences")

    last_loss = 0.0

    # tqdm progress bar over epochs
    for epoch in trange(epochs, desc="Training SGNS"):
        model.train(
            corpus,
            total_examples=model.corpus_count,
            epochs=1,
            compute_loss=True
        )

        # Get per-epoch loss (not cumulative)
        cumulative_loss = model.get_latest_training_loss()
        epoch_loss = cumulative_loss - last_loss
        last_loss = cumulative_loss

        # Update tqdm postfix with live loss
        trange.set_postfix if False else None  # (ignore; keeps tqdm happy)

        print(f"Epoch {epoch + 1} | Loss: {epoch_loss:.2f}")

    print(f"\n[Model] Training complete!")
    print(f"[Model] Vocabulary size: {len(model.wv):,}")
    print(f"[Model] Total training examples: {model.corpus_total_words:,}")
    
    # Save model
    model.save(str(save_path))
    print(f"[Model] Saved to {save_path}")
    
    # Also save just the word vectors (smaller file, faster loading)
    wv_path = save_path.with_suffix(".wordvectors")
    model.wv.save(str(wv_path))
    print(f"[Vectors] Saved word vectors to {wv_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Word2Vec SGNS on hanzi tokens")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed data (from load.py)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="hanzi_sgns.model",
        help="Output path for trained model (default: hanzi_sgns.model)"
    )
    parser.add_argument(
        "--vector_size",
        type=int,
        default=768,
        help="Dimensionality of word vectors (default: 768, matching BERT)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Maximum distance between current and predicted word (default: 5)"
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Minimum word frequency to include in vocabulary (default: 5)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)"
    )
    
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    corpus_path = data_dir / "hanzi_tokens.txt"
    out_path = out_dir / "hanzi_sgns.model"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}. Run load.py first!")

    print("=" * 60)
    print("Word2Vec SGNS Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Corpus: {corpus_path}")
    print(f"Output: {out_path}")
    print("=" * 60 + "\n")

    # Load corpus
    corpus = load_corpus(str(corpus_path))

    # Train model
    model = train_sgns_model(
        corpus=corpus,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
        save_path=out_path,
    )

    # Show some example similar words
    print("\n" + "=" * 60)
    print("Example similar words (if vocabulary is large enough):")
    print("=" * 60)
    
    test_words = ["中国", "北京", "大学", "经济", "技术"]
    for word in test_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            print(f"\n'{word}' similar words:")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.4f}")
        else:
            print(f"\n'{word}' not in vocabulary")
    
    print("\n" + "=" * 60)
    print("[Done] Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()