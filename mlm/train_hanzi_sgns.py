from gensim.models import Word2Vec
import spacy

nlp = spacy.load("zh_core_web_sm")


def tokenize(file: str):
    with open(file, "r") as file:
        text = file.read()

    doc = nlp(text)
    corpus = []

    for sentence in doc.sents:
        words = [token.text for token in sentence]
        corpus.append(words)

    return corpus


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
train_sgns_model(hanzi_corpus, 768, 3, 1, 100, "hanzi_sgns_model/hanzi_sgns.model")
