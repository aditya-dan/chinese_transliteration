from spacy.lang.kn import Kannada
from kn_to_latin import kn_to_latin
from gensim.models import Word2Vec

nlp = Kannada()
nlp.add_pipe('sentencizer')

def tokenize(file: str):
    with open("kn_small.txt", "r") as f:
        text = f.read()

    doc = nlp(text)
    kn_corpus = []
    latin_corpus = []

    for sentence in doc.sents:
        words = [token.text for token in sentence]
        kn_corpus.append(words)
        latin_words = []
        for word in words:
            latin_word = kn_to_latin(word)
            latin_words.append(latin_word)
        latin_corpus.append(latin_words)

    return (kn_corpus, latin_corpus)

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

kn_corpus, latin_corpus = tokenize("training_data.txt")
train_sgns_model(kn_corpus, 100, 10, 1, 100, "kn.model")
train_sgns_model(latin_corpus, 100, 10, 1, 100, "latin.model")