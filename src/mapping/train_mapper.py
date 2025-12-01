import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import numpy as np

model_path = "ShannonAI/ChineseBERT-base"


class SGNSMapper(nn.Module):
    """
    Projects BERT embeddings to SGNS embedding space
    """
    def __init__(self, bert_dim, sgns_dim):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, sgns_dim),
            nn.LayerNorm(sgns_dim)
        )
    
    def forward(self, bert_embeddings):
        """
        Args:
            bert_embeddings: (batch_size, bert_dim) or (batch_size, seq_len, bert_dim)
        Returns:
            projected: same shape but last dim is sgns_dim
        """
        return self.projection(bert_embeddings)


class PinyinHanziDataset(Dataset):
    """
    Dataset for training the mapper
    """
    def __init__(self, pinyin_hanzi_pairs, bert_model, tokenizer, sgns_embeddings, device):
        """
        Args:
            pinyin_hanzi_pairs: list of (pinyin, hanzi) tuples (should include all words used in SGNS, can generate directly from)
            bert_model: loaded BERT model
            tokenizer: BERT tokenizer
            sgns_embeddings: dict mapping hanzi -> numpy array
            device: torch device
        """
        self.pairs = [(p, h) for p, h in pinyin_hanzi_pairs if h in sgns_embeddings]
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.sgns_embeddings = sgns_embeddings
        self.device = device
        
        print(f"Dataset size: {len(self.pairs)} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pinyin, hanzi = self.pairs[idx]
        
        # get BERT embedding for pinyin
        inputs = self.tokenizer(
            pinyin,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
            
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
            mask[:, 0, :] = 0  # Exclude [CLS]
            mask[:, -1, :] = 0  # Exclude [SEP]
            
            sum_embeddings = torch.sum(hidden_states * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            bert_embedding = (sum_embeddings / sum_mask).squeeze(0)  # (hidden_dim,)
        
        # get target SGNS embedding
        target_embedding = torch.tensor(
            self.sgns_embeddings[hanzi],
            dtype=torch.float32,
            device=self.device
        )
        
        return {
            'bert_embedding': bert_embedding,
            'target_embedding': target_embedding,
            'pinyin': pinyin,
            'hanzi': hanzi
        }


def train_mapper(
    pinyin_hanzi_pairs,
    sgns_embeddings,
    bert_model_path=model_path,
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    device='cuda'
):
    """
    Train the BERT -> SGNS mapper
    
    Args:
        pinyin_hanzi_pairs: list of (pinyin, hanzi) tuples
        sgns_embeddings: dict mapping hanzi -> numpy array
        bert_model_path: path to BERT model
        epochs: number of training epochs
        batch_size: batch size
        learning_rate: learning rate
        device: 'cuda' or 'cpu'
    
    Returns:
        trained mapper model
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # load fine-tuned BERT
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    bert_model = BertModel.from_pretrained(bert_model_path)
    bert_model.to(device)
    bert_model.eval()  # Freeze BERT
    
    # dimensions
    bert_dim = bert_model.config.hidden_size
    sgns_dim = next(iter(sgns_embeddings.values())).shape[0]
    print(f"BERT dimension: {bert_dim}, SGNS dimension: {sgns_dim}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = PinyinHanziDataset(
        pinyin_hanzi_pairs,
        bert_model,
        tokenizer,
        sgns_embeddings,
        device
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # initialize mapper
    mapper = SGNSMapper(bert_dim, sgns_dim).to(device)
    
    optimizer = torch.optim.Adam(mapper.parameters(), lr=learning_rate)
    
    def cosine_loss(pred, target):
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        cosine_sim = torch.sum(pred_norm * target_norm, dim=1)
        return -cosine_sim.mean()  # Negative because we want to maximize
    
    print("Training mapper...")
    mapper.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            bert_embeddings = batch['bert_embedding']  # (batch_size, bert_dim)
            target_embeddings = batch['target_embedding']  # (batch_size, sgns_dim)
            
            # Forward pass
            projected = mapper(bert_embeddings)
            
            # Compute loss
            loss = cosine_loss(projected, target_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("Training complete!")
    return mapper


def save_mapper(mapper, save_path):
    """Save the trained mapper"""
    torch.save(mapper.state_dict(), save_path)
    print(f"Mapper saved to {save_path}")


def load_mapper(bert_dim, sgns_dim, load_path, device='cuda'):
    """Load a trained mapper"""
    mapper = SGNSMapper(bert_dim, sgns_dim)
    mapper.load_state_dict(torch.load(load_path, map_location=device))
    mapper.to(device)
    mapper.eval()
    print(f"Mapper loaded from {load_path}")
    return mapper