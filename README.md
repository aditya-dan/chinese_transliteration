# chinese_transliteration

LING482 project

The objective of this project is to train an RNN-based encoder-decoder model and an SGNS-based encoder-only model to perform a transliteration from Pinyin to Hanzi. The main research question is: can the same accuracy be achieved using an SGNS-based model for a lower computational cost? As of now, we used Gensim for SGNS (that trains on a Hanzi corpus), and plan to use Torch for the RNN. We also fine tune a BERT model trained on a Pinyin corpus to give us embeddings for Pinyin words, and map the embeddings it produces with those from the SGNS.

=====IMPORTANT=====

The results we have as of now are mappings between Hanzi SGNS embeddings and BERT Pinyin embeddings. We have trained SGNS on a Hanzi corpus and BERT on the equivalent Pinyin corpus. The script [mapping_hanzi_and_pinyin_embeddings.py](mlm%2Fmapping_hanzi_and_pinyin_embeddings.py) loads these two models and trains a linear regression model taking pairs of BERT and SGNS embeddings. The script [test_regression.py](src%2Fmlm%2Ftest_regression.py) tests cosine similarities between embeddings. It also checks if the true Hanzi embeddings are within the top 5 nearest neighbours of the predicted embedding, and if there are other Hanzi for a given Pinyin that have closer embeddings than the true Hanzi.

## File Structure

```
pinyin-sgns/
│
├── data/
│   ├── pinyin/         # pinyin corpora
│   └── hanzi/          # hanzi corpora
│
├── src/
    ├── mlm/            # SGNS and BERT embeddings are mapped and tested here
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

## Instructions

### 1. Clone the repository

```
git clone https://github.com/aditya-dan/chinese_transliteration.git
```

### 2. Create and activate the conda environment

```
conda env create --name pinyin-sgns -f environment.yml
conda activate pinyin-sgns
```

### BERT-Style MLM (first approach)

#### BERT-Style MLM download link

https://drive.google.com/file/d/1cJ86dAbOqB83Egf2pB4K31oPVp79HstV/view?usp=drive_link

#### Hanzi SGNS model download link

https://drive.google.com/file/d/10JSiBrMlqK2p6Hf72s0OoTq0bWsdUFoL/view?usp=drive_link

#### Pinyin to Hanzi Linear Regression model download link

https://drive.google.com/file/d/1cJ86dAbOqB83Egf2pB4K31oPVp79HstV/view?usp=drive_link

Navigate to src/mlm, save the models hanzi_sgns_model, pinyin_bert and pinyin_to_hanzi_regression.joblib in the directory, and run test_regression.py.

```
cd src/mlm
python test_regression.py
```

### BERT-WWM Model instructions (Second Approach, Main Approach in Paper)

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

# train regression model
python scripts/train_reg.py \
    --data_dir ./data/output/ \
    --bert_model ./src/bert/final_model/ \
    --sgns_model ./src/sgns/hanzi_sgns.model \
    --out_path ./src/regression/pinyin_to_hanzi_regression.joblib
    --max_sentences 25000
```

### 5. Test the models

Commands can be found in configs/test_regression.md.
