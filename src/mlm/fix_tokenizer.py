from transformers import BertTokenizerFast

# Load your existing tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert_tokenizer")

# Set the maximum sequence length
tokenizer.model_max_length = 512

# Save the tokenizer back to the folder
tokenizer.save_pretrained("bert_tokenizer")