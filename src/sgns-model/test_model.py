import jieba
from gensim.models import Word2Vec
from pypinyin import pinyin, Style

with open("cat.txt", "r") as file:
    text = file.read()

words = list(jieba.cut(text))

hanzi_corpus = []
pinyin_corpus = []

# for sentence in doc.sents:
#     words = [token.text for token in sentence]
#     hanzi_corpus.append(words)
#     pinyin_words = []
#     for word in words:
#         pinyin_word = "".join(lazy_pinyin(word))
#         pinyin_words.append(pinyin_word)
#     pinyin_corpus.append(pinyin_words)

hanzi_corpus.append(words)
pinyin_words = []
for word in words:
    py_str = "".join(item[0] for item in pinyin(word, style=Style.NORMAL, heteronym=False))
    pinyin_words.append(py_str)
pinyin_corpus.append(pinyin_words)

print(hanzi_corpus)
print(pinyin_corpus)

hanzi_model = Word2Vec(
    sentences=hanzi_corpus,  # list of tokenized sentences
    vector_size=128,    # embedding dimension
    window=3,          # context window size
    min_count=1,       # ignore words with frequency below min_count
    sg=1,              # SGNS is 1, CBOW is 0
    epochs=100,        # number of epochs
)

pinyin_model = Word2Vec(
    sentences=pinyin_corpus,  # list of tokenized sentences
    vector_size=128,    # embedding dimension
    window=3,          # context window size
    min_count=1,       # ignore words with frequency below min_count
    sg=1,              # SGNS is 1, CBOW is 0
    epochs=100,        # number of epochs
)

pinyin_model.save("pinyin.model")
hanzi_model.save("hanzi.model")

