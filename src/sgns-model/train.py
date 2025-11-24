from gensim.models import Word2Vec
from pypinyin import pinyin, Style
import spacy

nlp = spacy.load("zh_core_web_sm")

def tokenize(file: str):
    with open(file, "r") as file:
        text = file.read()

    doc = nlp(text)

    hanzi_corpus = []
    pinyin_corpus = []

    for sentence in doc.sents:
        words = [token.text for token in sentence]
        hanzi_corpus.append(words)
        pinyin_words = []
        for word in words:
            pinyin_word = "".join(item[0] for item in pinyin(word, style=Style.NORMAL, heteronym=False))
            pinyin_words.append(pinyin_word)
        pinyin_corpus.append(pinyin_words)

    return (hanzi_corpus, pinyin_corpus)

def train_sgns_model(corpus: str, vector_size: int, window: int, min_count: int, epochs: int, save_as: str):

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        epochs=epochs
    )

    model.save(save_as)

hanzi_corpus, pinyin_corpus = tokenize("cat.txt")
train_sgns_model(hanzi_corpus, 100, 3, 1, 100, "hanzi.model")
train_sgns_model(pinyin_corpus, 100, 3, 1, 100, "pinyin.model")