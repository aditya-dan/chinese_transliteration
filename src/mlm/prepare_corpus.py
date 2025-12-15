from pypinyin import pinyin, Style
import spacy

nlp = spacy.load("zh_core_web_sm")
nlp.max_length = 40000000

hanzi_file = "hanzi.txt"
pinyin_file = "pinyin.txt"

def convert_and_write_to_file(read_file: str, write_file: str):
    with open(read_file, "r") as rf:
        text = rf.read()

    doc = nlp(text)

    for sentence in doc.sents:
        words = [token.text for token in sentence]
        pinyin_words = []
        for word in words:
            pinyin_word = "".join(item[0] for item in pinyin(word, style=Style.NORMAL, heteronym=False))
            pinyin_words.append(pinyin_word)
        with open(write_file, 'a') as wf:
            wf.write(" ".join(pinyin_words))

def write_corpus_to_file(corpus: list, file: str):
    with open(file, 'w') as f:
        for sentence in corpus:
            sentence_text = " ".join(sentence)
            f.write(sentence_text)



convert_and_write_to_file(hanzi_file, pinyin_file)
