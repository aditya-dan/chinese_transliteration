from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert_tokenizer")
print(tokenizer.model_max_length)