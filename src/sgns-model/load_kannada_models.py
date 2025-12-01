from gensim.models import Word2Vec
from gensim.matutils import cossim, any2sparse
from kn_to_latin import kn_to_latin

kn_model = Word2Vec.load("kn.model")
latin_model = Word2Vec.load("latin.model")

below_eight = []
below_nine = []
over_nine = []

for kn_word in kn_model.wv.index_to_key:

    latin_word = kn_to_latin(kn_word)

    kn_vector = kn_model.wv[kn_word]
    latin_vector = latin_model.wv[latin_word]

    sim = cossim(any2sparse(kn_vector), any2sparse(latin_vector))

    if sim < 0.8:
        below_eight.append((kn_word, latin_word, sim))

    elif sim < 0.9:
        below_nine.append((kn_word, latin_word, sim))

    else:
        over_nine.append((kn_word, latin_word, sim))

print("\nBelow 0.8:\n")
for i in below_eight:
    print(i)

print("\nBetween 0.8 and 0.9:\n")
for i in below_nine:
    print(i)

print("\nOver 0.9:\n")
for i in over_nine:
    print(i)
