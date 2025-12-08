
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec
import torch

from pypinyin import lazy_pinyin, Style
import jieba
import numpy as np

model_path = "/home/jelyn/School/LING282/final/data/outputs/final_model"

# Load your trained Pinyin BERT
pinyin_tokenizer = AutoTokenizer.from_pretrained(model_path)
pinyin_bert = AutoModel.from_pretrained(model_path)

# Load your Hanzi SGNS
hanzi_model = Word2Vec.load("/home/jelyn/School/LING282/final/experiments/sgns/models/hanzi_spacy.model")

# Move BERT to eval mode
pinyin_bert.eval()

def get_pinyin_bert_embedding(pinyin_text: str, tokenizer, model):
    """
    Get BERT embedding for a pinyin token/sentence
    pinyin_text: e.g., "zhongguo" or "zhongguo renmin"
    """
    inputs = tokenizer(
        pinyin_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding (represents whole sequence)
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        # OR use mean pooling:
        # mask = inputs['attention_mask'].unsqueeze(-1)
        # embedding = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
    
    return embedding.squeeze(0)

def create_aligned_pairs(hanzi_corpus, hanzi_model, pinyin_tokenizer, pinyin_bert):
    """
    Create aligned pairs: BERT(Pinyin) -> SGNS(Hanzi)
    """
    pinyin_embeddings = []
    hanzi_embeddings = []
    pairs_info = []
    
    for hanzi_sent in hanzi_corpus:
        for hanzi_word in hanzi_sent:
            # Skip if not in Hanzi vocab
            if hanzi_word not in hanzi_model.wv:
                continue
            
            # Convert Hanzi word to Pinyin token (same way BERT was trained)
            # This matches your word_to_token function
            pys = lazy_pinyin(hanzi_word, style=Style.NORMAL, errors="ignore")
            pinyin_token = "".join([p for p in pys if p])
            
            if not pinyin_token:
                continue
            
            # Get BERT embedding for this Pinyin token
            pinyin_emb = get_pinyin_bert_embedding(
                pinyin_token, 
                pinyin_tokenizer, 
                pinyin_bert
            )
            
            # Get Hanzi SGNS embedding
            hanzi_emb = hanzi_model.wv[hanzi_word]
            
            pinyin_embeddings.append(pinyin_emb.numpy())
            hanzi_embeddings.append(hanzi_emb)
            pairs_info.append((pinyin_token, hanzi_word))
    
    return (
        torch.tensor(np.array(pinyin_embeddings), dtype=torch.float32),
        torch.tensor(np.array(hanzi_embeddings), dtype=torch.float32),
        pairs_info
    )

# Create pairs
print("Creating aligned pairs...")
pinyin_embs, hanzi_embs, pairs_info = create_aligned_pairs(
    hanzi_corpus,
    hanzi_model,
    pinyin_tokenizer,
    pinyin_bert
)

print(f"Created {len(pinyin_embs)} aligned pairs")
print(f"Sample pairs: {pairs_info[:10]}")
print(f"Pinyin embedding shape: {pinyin_embs[0].shape}")  # Should be (768,) for BERT-base
print(f"Hanzi embedding shape: {hanzi_embs[0].shape}")    # Should be (100,) from your SGNS

from torch.utils.data import Dataset, DataLoader, random_split

class PinyinHanziDataset(Dataset):
    def __init__(self, pinyin_embeddings, hanzi_embeddings):
        self.pinyin = pinyin_embeddings
        self.hanzi = hanzi_embeddings
    
    def __len__(self):
        return len(self.pinyin)
    
    def __getitem__(self, idx):
        return self.pinyin[idx], self.hanzi[idx]

# Create dataset
dataset = PinyinHanziDataset(pinyin_embs, hanzi_embs)

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

import torch.nn as nn

class EmbeddingMapper(nn.Module):
    def __init__(self, pinyin_dim=768, hanzi_dim=100, hidden_dim=512):
        super().__init__()
        self.mapper = nn.Sequential(
            nn.Linear(pinyin_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hanzi_dim)
        )
    
    def forward(self, x):
        return self.mapper(x)

# Initialize mapper
# BERT-base outputs 768-dim, your Hanzi SGNS is 100-dim
mapper = EmbeddingMapper(pinyin_dim=768, hanzi_dim=100)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mapper = mapper.to(device)

def predict_hanzi_from_pinyin(pinyin_text, mapper, pinyin_tokenizer, pinyin_bert, hanzi_model, k=5, device='cpu'):
    """
    Given Pinyin text, predict most likely Hanzi
    pinyin_text: e.g., "zhongguo" or "zhongguo renmin"
    """
    # Get BERT embedding for Pinyin
    pinyin_emb = get_pinyin_bert_embedding(pinyin_text, pinyin_tokenizer, pinyin_bert)
    pinyin_emb = pinyin_emb.to(device)
    
    # Map to Hanzi space
    mapper.eval()
    with torch.no_grad():
        mapped_emb = mapper(pinyin_emb.unsqueeze(0)).squeeze(0)
    
    # Get all Hanzi embeddings
    hanzi_vocab = list(hanzi_model.wv.index_to_key)
    hanzi_embeddings = torch.tensor(hanzi_model.wv.vectors, dtype=torch.float32).to(device)
    
    # Compute cosine similarity
    similarities = torch.nn.functional.cosine_similarity(
        mapped_emb.unsqueeze(0),
        hanzi_embeddings,
        dim=1
    )
    
    # Get top-k
    top_k_scores, top_k_indices = similarities.topk(k)
    
    predictions = [
        (hanzi_vocab[idx.item()], score.item())
        for idx, score in zip(top_k_indices, top_k_scores)
    ]
    
    return predictions

# Load best model
mapper.load_state_dict(torch.load('best_mapper.pth'))
mapper = mapper.to(device)

# Test predictions
test_cases = [
    "zhongguo",   # 中国
    "renmin",     # 人民
    "gongheguo",  # 共和国
    "beijing",    # 北京
]

print("\n=== Testing Predictions ===")
for pinyin in test_cases:
    predictions = predict_hanzi_from_pinyin(
        pinyin, 
        mapper, 
        pinyin_tokenizer, 
        pinyin_bert, 
        hanzi_model, 
        k=5,
        device=device
    )
    
    print(f"\nPinyin: {pinyin}")
    print("Top 5 predictions:")
    for i, (hanzi, score) in enumerate(predictions, 1):
        print(f"  {i}. {hanzi} (score: {score:.4f})")