from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import spacy
from pypinyin import pinyin, Style
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression
import joblib


nlp = spacy.load("zh_core_web_sm")

bert_model_path = "pinyin_bert"
sgns_model_path = "hanzi_sgns_model/hanzi_sgns.model"

tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
model = AutoModelForMaskedLM.from_pretrained(bert_model_path)

sgns_model = Word2Vec.load(sgns_model_path)

with open("hanzi_train_reg.txt", "r") as rf:
    text = rf.read()

doc = nlp(text)

all_pinyin_embeddings = []
all_hanzi_embeddings = []


# use pypinyin to convert hanzi tokens to pinyin
def tokens_to_pinyin(tokens):
    pinyin_tokens = []
    for t in tokens:
        py = "".join(item[0] for item in pinyin(t, style=Style.NORMAL, heteronym=False))
        pinyin_tokens.append(py)
    return pinyin_tokens


for sent in doc.sents:
    hanzi_tokens = [t.text for t in sent]
    pinyin_tokens = tokens_to_pinyin(hanzi_tokens)
    pinyin_text = " ".join(pinyin_tokens)

    encoded = tokenizer(
        pinyin_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
    )

    offsets = encoded.pop("offset_mapping")[0]

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    bert_embeddings = outputs.hidden_states[-1][0]

    pinyin_spans = []
    char_pos = 0
    for py in pinyin_tokens:
        start = char_pos
        end = char_pos + len(py)
        pinyin_spans.append((start, end))
        char_pos = end + 1

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

    for i, hanzi_token in enumerate(hanzi_tokens):

        bert_index = alignment[i]

        if bert_index is None:
            continue
        if hanzi_token not in sgns_model.wv:
            continue  # skip tokens that are missing in SGNS vocab

        hanzi_vec = sgns_model.wv[hanzi_token]
        pinyin_vec = bert_embeddings[bert_index].numpy()

        all_hanzi_embeddings.append(hanzi_vec)
        all_pinyin_embeddings.append(pinyin_vec)

reg = LinearRegression(fit_intercept=False)
reg.fit(all_pinyin_embeddings, all_hanzi_embeddings)

W = reg.coef_
print("Regression weight matrix shape:", W.shape)
joblib.dump(reg, "pinyin_to_hanzi_regression.joblib")
