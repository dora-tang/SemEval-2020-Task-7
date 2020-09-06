#roberta-context-task1: 0.525981
python main.py -task task1 -bsz 32 -do_train -do_eval -save_model -track -tensorboard -model transformer -transformer roberta-base -finetune 1 -lr 2e-05 -schedule linear_schedule_with_warmup -epoch 10 -feature edit-context -train_extra 1 -seed 1 -exp roberta-base_finetune_1_lr_2e-05_schedule_linear_schedule_with_warmup_epoch_10_feature_edit-context_extra_1_seed_1

# roberta-original-task1: 0.52207
python main.py -task task1 -bsz 32 -do_train -do_eval -save_model -track -tensorboard -model transformer -transformer roberta-base -finetune 1 -lr 2e-05 -schedule linear_schedule_with_warmup -epoch 3 -feature edit-original -train_extra 1 -seed 1 -exp roberta-base_finetune_1_lr_2e-05_schedule_linear_schedule_with_warmup_epoch_3_feature_edit-original_extra_1_seed_1

# roberta-edit-task1: 0.526453
python main.py -task task1 -bsz 32 -do_train -do_eval -save_model -track -tensorboard -model transformer -transformer roberta-base -finetune 1 -lr 2e-05 -schedule linear_schedule_with_warmup -epoch 10 -feature edit -train_extra 0 -seed 1 -exp roberta-base_finetune_1_lr_2e-05_schedule_linear_schedule_with_warmup_epoch_10_feature_edit_extra_0_seed_1

# roberta-context-task2: 0.649163
python main.py -task task2 -bsz 16 -do_train -do_eval -save_model -track -tensorboard -model transformer -transformer roberta-base -finetune 1 -lr 2e-05 -schedule linear_schedule_with_warmup -epoch 3 -feature edit-context -train_extra 0 -seed 1 -exp roberta-base_finetune_1_lr_2e-05_schedule_linear_schedule_with_warmup_epoch_3_feature_edit-context_extra_0_seed_1

# roberta-original-task2: 0.649543
python main.py -task task2 -bsz 16 -do_train -do_eval -save_model -track -tensorboard -model transformer -transformer roberta-base -finetune 1 -lr 2e-05 -schedule linear_schedule_with_warmup -epoch 3 -feature edit-original -train_extra 0 -seed 1 -exp roberta-base_finetune_1_lr_2e-05_schedule_linear_schedule_with_warmup_epoch_3_feature_edit-original_extra_0_seed_1

# roberta-edit-task2: 0.651065
python main.py -task task2 -bsz 16 -do_train -do_eval -save_model -track -tensorboard -model transformer -transformer roberta-base -finetune 1 -lr 2e-05 -schedule none -epoch 10 -feature edit -train_extra 1 -seed 1 -exp roberta-base_finetune_1_lr_2e-05_schedule_none_epoch_10_feature_edit_extra_1_seed_1
