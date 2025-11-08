# chinese_transliteration
LING482 project

The objective of this project is to train an RNN-based encoder-decoder model and an SGNS-based encoder-only model to perform a transliteration from Pinyin to Hanzi. The main research question is: can the same accuracy be achieved using an SGNS-based model for a lower computational cost? As of now, we plan to use Gensim for SGNS (that trains on a Hanzi corpus), and Torch for the RNN. We will also fine tune a BERT model trained on a Pinyin corpus to give us embeddings for Pinyin words, and map the embeddings it produces with those from the SGNS.

## Setup Instructions
### 1. Clone the repository

```
git clone https://github.com/aditya-dan/pinyin-sgns.git
cd pinyin-ime
```

### 2. Create and activate the conda environment

```
conda env create -f pinyin-sgns.yml
conda activate pinyin-sgns
```

