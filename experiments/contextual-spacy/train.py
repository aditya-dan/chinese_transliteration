import spacy

nlp = spacy.load("en_core_web_trf")
doc = nlp("The bank is located near the river bank.")

token_vec = doc._.trf_data.tensors[0]

print(embeddings.shape)
