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

with open("hanzi_train_reg.txt", "r") as rf:
    text = rf.read()

doc = nlp(text)

all_pinyin_embeddings = []
all_hanzi_embeddings = []


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

    # Request offsets from the tokenizer
    encoded = tokenizer(
        pinyin_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
    )

    # pull out offsets (and convert to Python list if you like)
    offsets = encoded.pop("offset_mapping")[0]   # remove from dict so model won't see it
    # offsets now can be used for alignment; offsets is a tensor of shape (seq_len, 2)

    # Now safe to call model with only tensor inputs
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    bert_embeddings = outputs.hidden_states[-1][0]

    pinyin_spans = []
    char_pos = 0
    for py in pinyin_tokens:
        start = char_pos
        end = char_pos + len(py)
        pinyin_spans.append((start, end))
        char_pos = end + 1  # +1 for space

    # Align each pinyin token → corresponding BERT token index
    alignment = []
    for (p_start, p_end) in pinyin_spans:
        matched_index = None
        for i, (b_start, b_end) in enumerate(offsets.tolist()):
            if b_start == 0 and b_end == 0:
                continue  # special tokens like [CLS], [SEP]

            # Overlap check
            if not (b_end <= p_start or b_start >= p_end):
                matched_index = i
                break

        alignment.append(matched_index)

    for i, hanzi_token in enumerate(hanzi_tokens):

        bert_index = alignment[i]

        if bert_index is None:
            continue  # no valid BERT alignment

        if hanzi_token not in sgns_model.wv:
            continue  # skip tokens missing in SGNS vocab

        hanzi_vec = sgns_model.wv[hanzi_token]
        pinyin_vec = bert_embeddings[bert_index].numpy()

        all_hanzi_embeddings.append(hanzi_vec)
        all_pinyin_embeddings.append(pinyin_vec)


reg = LinearRegression(fit_intercept=False)
reg.fit(all_pinyin_embeddings, all_hanzi_embeddings)

W = reg.coef_
print("Regression weight matrix shape:", W.shape)
