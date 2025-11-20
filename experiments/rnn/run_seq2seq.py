import spacy
from Decoder import Decoder
from Encoder import Encoder
from Seq2Seq import Seq2Seq, translate
from Vocab import Vocab
from pypinyin import lazy_pinyin


encoder = Encoder(input_size=335, emb_size=100, hidden_size=100)
decoder = Decoder(output_size=311, emb_size=100, hidden_size=100)

model = Seq2Seq(encoder, decoder, device="cpu")

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

hanzi_token_list = []
pinyin_token_list = []

for sentence in hanzi_corpus:
    for token in sentence:
        if token not in hanzi_token_list:
            hanzi_token_list.append(token)

for sentence in pinyin_corpus:
    for token in sentence:
        if token not in pinyin_token_list:
            pinyin_token_list.append(token)

special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

hanzi_token_list = special_tokens + hanzi_token_list
pinyin_token_list = special_tokens + pinyin_token_list

print(len(pinyin_token_list))
print(len(hanzi_token_list))

hanzi_stoi = {tok: i for i, tok in enumerate(hanzi_token_list)}
pinyin_stoi = {tok: i for i, tok in enumerate(pinyin_token_list)}

src_vocab = Vocab(hanzi_stoi)
tgt_vocab = Vocab(pinyin_stoi)

sentence = "一般以家貓從古到今都保存著的畏寒特點，所以貓的祖先產於溫暖地帶。"

translation = translate(
    model,
    sentence,
    src_tokenizer=lambda s: s.split(),
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab,
    device="cpu"
)

print(translation)
