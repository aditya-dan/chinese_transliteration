# chinese_transliteration

LING482 project

The objective of this project is to train an RNN-based encoder-decoder model and an SGNS-based encoder-only model to perform a transliteration from Pinyin to Hanzi. The main research question is: can the same accuracy be achieved using an SGNS-based model for a lower computational cost? As of now, we plan to use Gensim for SGNS (that trains on a Hanzi corpus), and Torch for the RNN. We will also fine tune a BERT model trained on a Pinyin corpus to give us embeddings for Pinyin words, and map the embeddings it produces with those from the SGNS.

## File Structure

```
pinyin-sgns/
│
├── data/
│   ├── pinyin/         # pinyin corpora
│   └── hanzi/          # hanzi corpora
│
├── src/
│   ├── bert_model/
│   ├── sgns_model/
│   ├── mapping/        # mapping bert to sgns vectors
│   └── inference/      # converting the pinyin to hanzi
│
├── scripts/            # data collection scripts
├── configs/            # experiment configs
├── outputs/            # experiment outputs
├── README.md
└── environment.yml     # conda env
```

## Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/aditya-dan/pinyin-sgns.git
cd pinyin-ime
```

### 2. Create and activate the conda environment

```
conda env create --name pinyin-sgns -f environment.yml
conda activate pinyin-sgns
```

### 3. Download and unzip the dataset

We use the wiki2019zh dataset:

[Download via Google Drive](https://drive.google.com/file/d/1EdHUZIDpgcBoSqbjlfNKJ3b1t0XIUjbt/view?usp=sharing)

Create a data/ folder, then unzip the `wiki_zh_2019.zip` file into that directory.

### 4. Load and train models

Load the model tokenizer and corpus data.

```bash
# load the wiki section (We used AA)
python scripts/load.py --wiki_root data/wiki_zh/[YOUR_WIKI_ROOT]/ --out_dir data/output/

# train the models
python scripts/train_bert.py --data-dir data/output/ --out_dir src/bert/
python scripts/train_sgns.py --data-dir data/output/ --out_dir src/sgns/
```
