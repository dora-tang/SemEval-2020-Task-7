# ENVIRONEMNT
conda env create -f environment.yml
source activate humor
python -m spacy download en
# exit enviroment:
# source deactivate

cd src
# CACHE=cache/transformers
CACHE=/share/data/speech/Data/sjin/data/cache/transformers
#CACHE=/Users/jinshuning/Documents/Github/data/cache/transformers

#
# # TASK1
# # CBOW
# python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_predict -save_model -track \
# -model cbow -exp cbow
# ### python main.py -exp cbow -do_predict
# ### python main.py -exp cbow -do_eval
# # CBOW extra
# python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_predict -save_model -track \
# -model cbow -train_extra -exp cbow_extra
# # CBOW finetune
# python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_predict -save_model -track \
# -model cbow -finetune -exp cbow_fintune
# # CBOW extra finetune
# python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_predict -save_model -track \
# -model cbow -train_extra -finetune -exp cbow_extra_finetune
#
# # roberta
# python main.py -epochs 10 -bsz 32 -do_train -do_predict -cache $CACHE -save_model -track \
# -model transformer -transformer roberta-base -exp roberta-base
# ### python main.py -exp roberta-base -do_predict  -cache $CACHE
# ### python main.py -exp roberta-base -do_eval  -cache $CACHE
# # roberta finetune
# python main.py -epochs 10 -bsz 32 -do_train -do_predict -cache $CACHE -save_model -track \
# -model transformer -transformer roberta-base -exp roberta-base_finetune -finetune
# # roberta extra
# python main.py -epochs 10 -bsz 32 -do_train -do_predict -cache $CACHE -save_model -track \
# -model transformer -transformer roberta-base -train_extra -exp roberta-base_extra
# # roberta extra finetune
# python main.py -epochs 10 -bsz 32 -do_train -do_predict -cache $CACHE -save_model -track \
# -model transformer -transformer roberta-base -train_extra -exp roberta-base_extra_finetune -finetune
#
# # bert
# python main.py -epochs 10 -bsz 32 -do_train -do_predict -cache $CACHE -save_model -track \
# -model transformer -transformer bert-base-uncased -exp bert-base-uncased
# # bert finetune
# python main.py -epochs 10 -bsz 32 -do_train -do_predict -cache $CACHE -save_model -track \
# -model transformer -transformer bert-base-uncased -exp bert-base-uncased_finetune -finetune
# # bert extra
# python main.py -epochs 10 -bsz 32 -do_train -do_predict -cache $CACHE -save_model -track \
# -model transformer -transformer bert-base-uncased -train_extra -exp bert-base-uncased_extra
# # bert extra finetune
# python main.py -epochs 10 -bsz 32 -do_train -do_predict -cache $CACHE -save_model -track \
# -model transformer -transformer bert-base-uncased -train_extra -exp bert-base-uncased_extra_finetune -finetune


# # TASK2
#
# # CBOW
# python main2.py -epochs 10 -lr 0.001 -bsz 16 -model cbow -exp cbow
# python main2.py -epochs 10 -lr 0.001 -bsz 16 -model cbow -exp cbow_diff -diff
#
# python main2.py -epochs 10 -lr 0.001 -bsz 16 -model cbow -exp cbow_more -train_more
# python main2.py -epochs 10 -lr 0.001 -bsz 16 -model cbow -exp cbow_more_diff -train_more -diff
#
# # RoBERTa
# python main2.py -epochs 10 -bsz 16 \
# -model transformer \
# -transformer roberta-base \
# -exp roberta-base \
# -cache $CACHE \
# -track
# #
# # python main2.py -epochs 10 -bsz 16 \
# # -model transformer \
# # -transformer roberta-base \
# # -exp roberta-base_diff \
# # -cache $CACHE \
# # -track \
# # -diff
# #
# # python main2.py -epochs 10 -bsz 16 \
# # -model transformer \
# # -transformer roberta-base \
# # -exp roberta-base_ranking \
# # -cache $CACHE \
# # -track \
# # -ranking
# #
# # python main2.py -epochs 10 -bsz 16 \
# # -model transformer \
# # -transformer roberta-base \
# # -exp roberta-base_ranking0.4 \
# # -cache $CACHE \
# # -track \
# # -ranking 0.4
# #
# #
# #
# # python main2.py -epochs 10 -bsz 16 \
# # -model transformer \
# # -transformer roberta-base \
# # -cache $CACHE \
# # -track \
# # -exp roberta-base_ranking0.5_margin0.2 \
# # -ranking 0.5 \
# # -margin 0.2
#
# #
# # python main2.py -epochs 10 -bsz 16 \
# # -model transformer \
# # -transformer roberta-base \
# # -exp roberta-base_ranking2 \
# # -cache $CACHE \
# # -track \
# # -ranking 2
#
# python main2.py -epochs 10 -bsz 16 \
# -model transformer \
# -transformer roberta-base \
# -exp roberta-base_more \
# -train_more \
# -cache $CACHE \
# -track
#
# python main2.py -epochs 10 -bsz 16 \
# -model transformer \
# -transformer roberta-base \
# -exp roberta-base_more_diff \
# -train_more \
# -cache $CACHE \
# -track \
# -diff
#
#
# # BERT
# python main2.py -epochs 10 -bsz 16 \
# -model transformer \
# -transformer bert-base-uncased \
# -exp bert-base-uncased \
# -cache $CACHE \
# -track


python extract.py exp*/*/*log > results.txt
