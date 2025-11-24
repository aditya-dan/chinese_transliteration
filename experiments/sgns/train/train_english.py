from gensim.models import Word2Vec
import spacy

nlp = spacy.load("en_core_web_sm")

def tokenize(file: str):
    with open(file, "r") as file:
        text = file.read()

    doc = nlp(text)

    lower_corpus = []
    upper_corpus = []

    for sentence in doc.sents:
        words = [token.text for token in sentence]
        lower_corpus.append(words)
        upper_words = []
        for word in words:
            upper_words.append(word.upper())
        upper_corpus.append(upper_words)

    return (lower_corpus, upper_corpus)

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

lower_corpus, upper_corpus = tokenize("../texts/psalm_23.txt")
train_sgns_model(upper_corpus, 100, 3, 1, 100, "../models/upper.model")
train_sgns_model(lower_corpus, 100, 3, 1, 100, "../models/lower.model")