# cbow
python main.py -task task1 -bsz 32 \
-do_train -do_eval \
-save_model -track -tensorboard \
-model cbow -feature edit-context -finetune 0 -train_extra 0 -seed 1 \
-exp cbow_edit-context_fintune0_extra0_seed1 \
-epochs 10 -lr 0.001 -schedule reduce_on_plateau -optimizer adam

python main.py -task task2 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -do_predict -save_model -track -model cbow -feature edit-context -finetune 0 -train_extra 0 -seed 1 -exp cbow_edit-context_fintune0_extra0_seed1 -tensorboard -schedule linear_schedule_with_warmup -optimizer huggingface_adamw

python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-context -finetune 0 -train_extra 1 -seed 1 -exp cbow_edit-context_fintune0_extra1_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-context -finetune 1 -train_extra 0 -seed 1 -exp cbow_edit-context_fintune1_extra0_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-context -finetune 1 -train_extra 1 -seed 1 -exp cbow_edit-context_fintune1_extra1_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-original -finetune 0 -train_extra 0 -seed 1 -exp cbow_edit-original_fintune0_extra0_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-original -finetune 0 -train_extra 1 -seed 1 -exp cbow_edit-original_fintune0_extra1_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-original -finetune 1 -train_extra 0 -seed 1 -exp cbow_edit-original_fintune1_extra0_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-original -finetune 1 -train_extra 1 -seed 1 -exp cbow_edit-original_fintune1_extra1_seed1
# roberta

python main.py -task task1 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-context -finetune 1 -train_extra 0 -seed 1 \
-lr 2e-5 -schedule reduce -epochs 10 \
-exp roberta-base_edit-context_fintune1_2e-5_reduce \
-tensorboard

python main.py -task task1 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-context -finetune 1 -train_extra 0 -seed 1 \
-lr 2e-5 -schedule linear -epochs 3 \
-exp roberta-base_edit-context_fintune1_2e-5_linear

python main.py -task task1 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-context -finetune 1 -train_extra 0 -seed 1 \
-lr 2e-5 -schedule constant -epochs 10 \
-exp roberta-base_edit-context_fintune1_2e-5_constant


python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-context -finetune 0 -train_extra 0 -seed 1 -exp roberta-base_edit-context_fintune0_extra0_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-context -finetune 0 -train_extra 1 -seed 1 -exp roberta-base_edit-context_fintune0_extra1_seed1

python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-context -finetune 1 -train_extra 1 -seed 1 -exp roberta-base_edit-context_fintune1_extra1_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-original -finetune 0 -train_extra 0 -seed 1 -exp roberta-base_edit-original_fintune0_extra0_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-original -finetune 0 -train_extra 1 -seed 1 -exp roberta-base_edit-original_fintune0_extra1_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-original -finetune 1 -train_extra 0 -seed 1 -exp roberta-base_edit-original_fintune1_extra0_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-original -finetune 1 -train_extra 1 -seed 1 -exp roberta-base_edit-original_fintune1_extra1_seed1
# bert
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-context -finetune 0 -train_extra 0 -seed 1 -exp bert-base-uncased_edit-context_fintune0_extra0_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-context -finetune 0 -train_extra 1 -seed 1 -exp bert-base-uncased_edit-context_fintune0_extra1_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-context -finetune 1 -train_extra 0 -seed 1 -exp bert-base-uncased_edit-context_fintune1_extra0_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-context -finetune 1 -train_extra 1 -seed 1 -exp bert-base-uncased_edit-context_fintune1_extra1_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-original -finetune 0 -train_extra 0 -seed 1 -exp bert-base-uncased_edit-original_fintune0_extra0_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-original -finetune 0 -train_extra 1 -seed 1 -exp bert-base-uncased_edit-original_fintune0_extra1_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-original -finetune 1 -train_extra 0 -seed 1 -exp bert-base-uncased_edit-original_fintune1_extra0_seed1
python main.py -task task1 -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-original -finetune 1 -train_extra 1 -seed 1 -exp bert-base-uncased_edit-original_fintune1_extra1_seed1




# cbow, constant lr, adam
# no-fintuning
