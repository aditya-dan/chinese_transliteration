from gensim.models import Word2Vec

corpus = [["hello", "world"], ["goodbye", "earth"], ["the", "quick", "brown", "fox", "jumps"],
          ["the", "yellow", "sun", "rises"], ["the", "moon", "shines"], ["the", "happy", "prince", "smiles"],
          ["the", "princess", "sleeps"], ["the", "boy", "runs"], ["the", "girl", "walks"]]

model = Word2Vec(
    sentences=corpus,  # list of tokenized sentences
    vector_size=50,    # embedding dimension
    window=3,          # context window
    min_count=1,       # ignore words with freq < 1
    sg=1,              # use skip-gram (1) instead of CBOW (0)
    epochs=100,        # training iterations
)

model.save("hello_world_word2vec.model")
