import spacy
from pypinyin import lazy_pinyin
nlp = spacy.load("zh_core_web_sm")

text = "猫是小型哺乳动物，以啮齿动物、鸟类和爬行动物为食。"

doc = nlp(text)

for sentence in doc.sents:
    words = [token.text for token in sentence]
    pinyin_words = []
    for word in words:
        pinyin_word = "".join(lazy_pinyin(word))
        pinyin_words.append(pinyin_word)

print(pinyin_words)