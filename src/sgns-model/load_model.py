from gensim.models import Word2Vec
from pypinyin import pinyin, Style
from gensim.matutils import cossim, any2sparse

hanzi_model = Word2Vec.load("hanzi.model")
pinyin_model = Word2Vec.load("pinyin.model")

below_eight = []
below_nine = []
over_nine = []

for hanzi_word in hanzi_model.wv.index_to_key:

    pinyin_word = "".join(item[0] for item in pinyin(hanzi_word, style=Style.NORMAL, heteronym=False))

    hanzi_vector = hanzi_model.wv[hanzi_word]
    pinyin_vector = pinyin_model.wv[pinyin_word]

    sim = cossim(any2sparse(hanzi_vector), any2sparse(pinyin_vector))

    if sim < 0.8:
        below_eight.append((hanzi_word, pinyin_word, sim))

    elif sim < 0.9:
        below_nine.append((hanzi_word, pinyin_word, sim))

    else:
        over_nine.append((hanzi_word, pinyin_word, sim))

print("\nBelow 0.8:\n")
for i in below_eight:
    print(i)

print("\nBetween 0.8 and 0.9:\n")
for i in below_nine:
    print(i)

print("\nOver 0.9:\n")
for i in over_nine:
    print(i)
