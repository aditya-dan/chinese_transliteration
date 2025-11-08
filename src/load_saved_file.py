from gensim.models import Word2Vec
model = Word2Vec.load("hello_world_word2vec.model")
# print(model.wv['hello'])
print(model.wv.most_similar(positive=['girl', 'prince'], negative=['boy']))
