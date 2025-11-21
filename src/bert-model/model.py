# use chinese-bert
from transformers import BertModel, BertTokenizer
import torch

model_path = "ShannonAI/ChineseBERT-base"

class ChineseBert:
    def __init__(self, model_path=model_path):
        print("Loading ChineseBERT model from:", model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert_model = BertModel.from_pretrained(model_path)
        self.bert_model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)

    def get_embedding(self, text):
        """
        Given a pinyin text sequence, return its contextual embeddings
        """
        text = " ".join(text)  # join characters with spaces

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use last hidden state
            embeddings = outputs.last_hidden_state[0]

        return embeddings.cpu().numpy()