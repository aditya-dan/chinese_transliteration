from gensim.models import Word2Vec
import spacy
import itertools
import json
from pypinyin import pinyin, Style

nlp = spacy.load("zh_core_web_sm")

pinyin_hanzi_dict = {}

def tokenize(file: str):
    with open(file, "r") as file:
        text = file.read()

    doc = nlp(text)
    corpus = []

    for sentence in doc.sents:
        words = [token.text for token in sentence]
        corpus.append(words)

    return corpus


def build_pinyin_hanzi_dictionary(corpus: list):
    for word in set(list(itertools.chain.from_iterable(corpus))):
        pinyin_word = "".join(item[0] for item in pinyin(word, style=Style.NORMAL, heteronym=False))
        if not (pinyin_word in pinyin_hanzi_dict.keys()):
            pinyin_hanzi_dict[pinyin_word] = []
        pinyin_hanzi_dict[pinyin_word].append(word)

    with open('pinyin_hanzi_dictionary.json', 'w') as json_file:
        # Use json.dump() to write the dictionary to the file
        json.dump(pinyin_hanzi_dict, json_file, indent=4)


def train_sgns_model(corpus: list, vector_size: int, window: int, min_count: int, epochs: int, save_as: str):

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        epochs=epochs
    )

    model.save(save_as)


hanzi_corpus = tokenize("hanzi.txt")
build_pinyin_hanzi_dictionary(hanzi_corpus)
# train_sgns_model(hanzi_corpus, 768, 3, 1, 100, "hanzi_sgns_model/hanzi_sgns.model")
