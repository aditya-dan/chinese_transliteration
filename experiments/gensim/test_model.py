import spacy
from gensim.models import Word2Vec
from pypinyin import lazy_pinyin

nlp = spacy.load("zh_core_web_sm")

with open("cat.txt", "r") as file:
    text = file.read()

doc = nlp(text)

hanzi_corpus = []
pinyin_corpus = []

for sentence in doc.sents:
    words = [token.text for token in sentence]
    hanzi_corpus.append(words)
    pinyin_words = []
    for word in words:
        pinyin_word = "".join(lazy_pinyin(word))
        pinyin_words.append(pinyin_word)
    pinyin_corpus.append(pinyin_words)

hanzi_model = Word2Vec(
    sentences=hanzi_corpus,  # list of tokenized sentences
    vector_size=50,    # embedding dimension
    window=3,          # context window
    min_count=1,       # ignore words with freq < 1
    sg=1,              # use skip-gram (1) instead of CBOW (0)
    epochs=100,        # training iterations
)

pinyin_model = Word2Vec(
    sentences=pinyin_corpus,  # list of tokenized sentences
    vector_size=50,    # embedding dimension
    window=3,          # context window
    min_count=1,       # ignore words with freq < 1
    sg=1,              # use skip-gram (1) instead of CBOW (0)
    epochs=100,        # training iterations
)

pinyin_model.save("pinyin.model")
hanzi_model.save("hanzi.model")

