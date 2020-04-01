# # ENVIRONEMNT
# conda env create -f environment.yml
# python -m spacy download en
#
# # put cache directory in bash_profile (mac) bashrc (linux)
# #HUGGINGFACE_TRANSFORMERS_CACHE=cache/transformers
# HUGGINGFACE_TRANSFORMERS_CACHE=/share/data/speech/Data/sjin/data/cache/transformers
# #HUGGINGFACE_TRANSFORMERS_CACHE=/Users/jinshuning/Documents/Github/data/cache/transformers
# case "$(uname -s)" in
#   Darwin)
#     echo export HUGGINGFACE_TRANSFORMERS_CACHE="$HUGGINGFACE_TRANSFORMERS_CACHE" >> ~/.bash_profile
#     source ~/.bash_profile
#   ;;
#   Linux)
#     echo export HUGGINGFACE_TRANSFORMERS_CACHE="$HUGGINGFACE_TRANSFORMERS_CACHE" >> ~/.bashrc
#     source ~/.bashrc
#   ;;
#   *)
#     echo 'Automatic path setup is only configured for MacOS and Linux.'
#   ;;
# esac
source activate humor
# exit enviroment:
# source deactivate
cd src


# TASK1
# CBOW
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track \
-model cbow -exp cbow_context -seed 1 -feature edit-context

python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track \
-model cbow -exp cbow_original -seed 1 -feature edit-original
### python main.py -exp cbow -do_eval
### python main.py -exp cbow -do_predict
# CBOW extra
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track \
-model cbow -train_extra -exp cbow_extra -seed 1
# CBOW finetune
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track \
-model cbow -finetune -exp cbow_fintune -seed 1
# CBOW extra finetune
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track \
-model cbow -train_extra -finetune -exp cbow_extra_finetune -seed 1

# roberta
python main.py -epochs 10 -bsz 32 -do_train -do_eval -save_model -track \
-model transformer -transformer roberta-base_context -exp roberta-base -seed 1 -feature edit_context
python main.py -epochs 10 -bsz 32 -do_train -do_eval -save_model -track \
-model transformer -transformer roberta-base_original -exp roberta-base -seed 1 -feature edit_original
### python main.py -exp roberta-base -do_eval
### python main.py -exp roberta-base -do_predict
# roberta finetune
python main.py -epochs 10 -bsz 32 -do_train -do_eval -save_model -track \
-model transformer -transformer roberta-base -exp roberta-base_finetune -finetune -seed 1
# roberta extra
python main.py -epochs 10 -bsz 32 -do_train -do_eval -save_model -track \
-model transformer -transformer roberta-base -train_extra -exp roberta-base_extra -seed 1
# roberta extra finetune
python main.py -epochs 10 -bsz 32 -do_train -do_eval -save_model -track \
-model transformer -transformer roberta-base -train_extra -exp roberta-base_extra_finetune -finetune -seed 1

# bert
python main.py -epochs 10 -bsz 32 -do_train -do_eval -save_model -track \
-model transformer -transformer bert-base-uncased -exp bert-base-uncased -seed 1
# bert finetune
python main.py -epochs 10 -bsz 32 -do_train -do_eval -save_model -track \
-model transformer -transformer bert-base-uncased -exp bert-base-uncased_finetune -finetune -seed 1
# bert extra
python main.py -epochs 10 -bsz 32 -do_train -do_eval -save_model -track \
-model transformer -transformer bert-base-uncased -train_extra -exp bert-base-uncased_extra -seed 1
# bert extra finetune
python main.py -epochs 10 -bsz 32 -do_train -do_eval -save_model -track \
-model transformer -transformer bert-base-uncased -train_extra -exp bert-base-uncased_extra_finetune -finetune -seed 1

#
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
# \
# -track
# #
# # python main2.py -epochs 10 -bsz 16 \
# # -model transformer \
# # -transformer roberta-base \
# # -exp roberta-base_diff \
# # \
# # -track \
# # -diff
# #
# # python main2.py -epochs 10 -bsz 16 \
# # -model transformer \
# # -transformer roberta-base \
# # -exp roberta-base_ranking \
# # \
# # -track \
# # -ranking
# #
# # python main2.py -epochs 10 -bsz 16 \
# # -model transformer \
# # -transformer roberta-base \
# # -exp roberta-base_ranking0.4 \
# # \
# # -track \
# # -ranking 0.4
# #
# #
# #
# # python main2.py -epochs 10 -bsz 16 \
# # -model transformer \
# # -transformer roberta-base \
# # \
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
# # \
# # -track \
# # -ranking 2
#
# python main2.py -epochs 10 -bsz 16 \
# -model transformer \
# -transformer roberta-base \
# -exp roberta-base_more \
# -train_more \
# \
# -track
#
# python main2.py -epochs 10 -bsz 16 \
# -model transformer \
# -transformer roberta-base \
# -exp roberta-base_more_diff \
# -train_more \
# \
# -track \
# -diff
#
#
# # BERT
# python main2.py -epochs 10 -bsz 16 \
# -model transformer \
# -transformer bert-base-uncased \
# -exp bert-base-uncased \
# \
# -track

#
# python extract.py exp*/*/*log > results.txt
