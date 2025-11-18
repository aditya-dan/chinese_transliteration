from gensim.models import Word2Vec
model = Word2Vec.load("cat_zh_wiki.model")
for word in model.wv.index_to_key:
    vector = model.wv[word]
    print(f"Word: '{word}', Vector: {vector}")
