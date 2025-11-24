from gensim.models import Word2Vec
from gensim.matutils import cossim, any2sparse

lower_model = Word2Vec.load("../models/lower.model")
upper_model = Word2Vec.load("../models/upper.model")

below_eight = []
below_nine = []
over_nine = []

for lower_word in lower_model.wv.index_to_key:

    upper_word = lower_word.upper()

    lower_vector = lower_model.wv[lower_word]
    upper_vector = upper_model.wv[upper_word]

    sim = cossim(any2sparse(lower_vector), any2sparse(upper_vector))

    if sim < 0.8:
        below_eight.append((lower_word, upper_word, sim))

    elif sim < 0.9:
        below_nine.append((lower_word, upper_word, sim))

    else:
        over_nine.append((lower_word, upper_word, sim))

print("\nBelow 0.8:\n")
for i in below_eight:
    print(i)

print("\nBetween 0.8 and 0.9:\n")
for i in below_nine:
    print(i)

print("\nOver 0.9:\n")
for i in over_nine:
    print(i)
