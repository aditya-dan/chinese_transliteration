from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import spacy
from pypinyin import pinyin, Style
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression

nlp = spacy.load("zh_core_web_sm")

bert_model_path = "pinyin_bert"
sgns_model_path = "hanzi_sgns_model/hanzi_sgns.model"

tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
model = AutoModelForMaskedLM.from_pretrained(bert_model_path)

sgns_model = Word2Vec.load(sgns_model_path)

hanzi_text = "我是一名学生。我喜欢数学。"

doc = nlp(hanzi_text)

hanzi_tokens = [token.text for token in doc]
pinyin_tokens = []

for token in hanzi_tokens:
    pinyin_token = "".join(item[0] for item in pinyin(token, style=Style.NORMAL, heteronym=False))
    pinyin_tokens.append(pinyin_token)

pinyin_text = " ".join(pinyin_tokens)

inputs = tokenizer(pinyin_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

token_embeddings = outputs.hidden_states[-1][0]

hanzi_embeddings = []
pinyin_embeddings = []

for index, token in enumerate(hanzi_tokens):
    hanzi_embeddings.append(sgns_model.wv[token])
    pinyin_embeddings.append(token_embeddings[index].numpy())

reg = LinearRegression(fit_intercept=False)
reg.fit(hanzi_embeddings, pinyin_embeddings)

W = reg.coef_
