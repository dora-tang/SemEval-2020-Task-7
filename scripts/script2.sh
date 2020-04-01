# ENVIRONEMNT
conda env create -f environment.yml
source activate humor
python -m spacy download en
# exit enviroment:
# source deactivate

cd src
# CBOW
python main2.py -epochs 10 -lr 0.001 -bsz 16 -model cbow -exp cbow

# RoBERTa
python main2.py -epochs 10 -bsz 16 \
-model transformer \
-transformer roberta-base \
-exp roberta-base \
-cache $HUGGINGFACE_TRANSFORMERS_CACHE \
-track

# BERT
python main2.py -epochs 10 -bsz 16 \
-model transformer \
-transformer bert-base-uncased \
-exp bert-base-uncased \
-cache $HUGGINGFACE_TRANSFORMERS_CACHE \
-track
