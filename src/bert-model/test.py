char_string = "早上好中国现在我有冰淇淋。我很喜欢冰淇淋。"

# convert to pinyin list
from pypinyin import pinyin, Style
import jieba

words = list(jieba.cut(char_string))
pinyin_list = []
for word in words:
    py_str = "".join(item[0] for item in pinyin(word, style=Style.NORMAL, heteronym=False))
    pinyin_list.append(py_str)
print(pinyin_list)

from model import ChineseBert
chinese_bert = ChineseBert()
embeddings = chinese_bert.get_embedding(pinyin_list)
print(embeddings)
print(embeddings.shape)