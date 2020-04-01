# cbow
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-context -finetune 0 -train_extra 0 -seed 1 -exp cbow_edit-context_fintune0_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-context -finetune 0 -train_extra 1 -seed 1 -exp cbow_edit-context_fintune0_extra1_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-context -finetune 1 -train_extra 0 -seed 1 -exp cbow_edit-context_fintune1_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-context -finetune 1 -train_extra 1 -seed 1 -exp cbow_edit-context_fintune1_extra1_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-original -finetune 0 -train_extra 0 -seed 1 -exp cbow_edit-original_fintune0_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-original -finetune 0 -train_extra 1 -seed 1 -exp cbow_edit-original_fintune0_extra1_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-original -finetune 1 -train_extra 0 -seed 1 -exp cbow_edit-original_fintune1_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model cbow -feature edit-original -finetune 1 -train_extra 1 -seed 1 -exp cbow_edit-original_fintune1_extra1_seed1
# roberta
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-context -finetune 0 -train_extra 0 -seed 1 -exp roberta-base_edit-context_fintune0_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-context -finetune 0 -train_extra 1 -seed 1 -exp roberta-base_edit-context_fintune0_extra1_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-context -finetune 1 -train_extra 0 -seed 1 -exp roberta-base_edit-context_fintune1_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-context -finetune 1 -train_extra 1 -seed 1 -exp roberta-base_edit-context_fintune1_extra1_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-original -finetune 0 -train_extra 0 -seed 1 -exp roberta-base_edit-original_fintune0_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-original -finetune 0 -train_extra 1 -seed 1 -exp roberta-base_edit-original_fintune0_extra1_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-original -finetune 1 -train_extra 0 -seed 1 -exp roberta-base_edit-original_fintune1_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer roberta-base -feature edit-original -finetune 1 -train_extra 1 -seed 1 -exp roberta-base_edit-original_fintune1_extra1_seed1
# bert
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-context -finetune 0 -train_extra 0 -seed 1 -exp bert-base-uncased_edit-context_fintune0_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-context -finetune 0 -train_extra 1 -seed 1 -exp bert-base-uncased_edit-context_fintune0_extra1_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-context -finetune 1 -train_extra 0 -seed 1 -exp bert-base-uncased_edit-context_fintune1_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-context -finetune 1 -train_extra 1 -seed 1 -exp bert-base-uncased_edit-context_fintune1_extra1_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-original -finetune 0 -train_extra 0 -seed 1 -exp bert-base-uncased_edit-original_fintune0_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-original -finetune 0 -train_extra 1 -seed 1 -exp bert-base-uncased_edit-original_fintune0_extra1_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-original -finetune 1 -train_extra 0 -seed 1 -exp bert-base-uncased_edit-original_fintune1_extra0_seed1
python main.py -epochs 10 -lr 0.001 -bsz 32 -do_train -do_eval -save_model -track -model transformer -transformer bert-base-uncased -feature edit-original -finetune 1 -train_extra 1 -seed 1 -exp bert-base-uncased_edit-original_fintune1_extra1_seed1