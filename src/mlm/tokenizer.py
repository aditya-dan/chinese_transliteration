from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer()
tokenizer.train(files="pinyin.txt", vocab_size=17851, min_frequency=1)
tokenizer.save_model("bert_tokenizer")