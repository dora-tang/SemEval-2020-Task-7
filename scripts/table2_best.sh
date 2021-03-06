# cbow-task1: 0.542423
python main.py -task task1 -bsz 32 -do_train -do_eval -save_model -track -tensorboard -model cbow -finetune 0 -lr 0.0003 -schedule none -epoch 10 -feature edit-context -train_extra 0 -seed 1 -exp cbow_finetune_0_lr_0.0003_schedule_none_epoch_10_feature_edit-context_extra_0_seed_1

# bert-task1: 0.530364
python main.py -task task1 -bsz 32 -do_train -do_eval -save_model -track -tensorboard -model transformer -transformer bert-base-uncased -finetune 0 -lr 0.0003 -schedule none -epoch 10 -feature edit-context -train_extra 1 -seed 1 -exp bert-base-uncased_finetune_0_lr_0.0003_schedule_none_epoch_10_feature_edit-context_extra_1_seed_1

# roberta-task1: 0.52207
python main.py -task task1 -bsz 32 -do_train -do_eval -save_model -track -tensorboard -model transformer -transformer roberta-base -finetune 1 -lr 2e-05 -schedule linear_schedule_with_warmup -epoch 3 -feature edit-original -train_extra 1 -seed 1 -exp roberta-base_finetune_1_lr_2e-05_schedule_linear_schedule_with_warmup_epoch_3_feature_edit-original_extra_1_seed_1

# cbow-task2: 0.605784
python main.py -task task2 -bsz 16 -do_train -do_eval -save_model -track -tensorboard -model cbow -finetune 1 -lr 0.0003 -schedule none -epoch 10 -feature edit-context -train_extra 1 -seed 1 -exp cbow_finetune_1_lr_0.0003_schedule_none_epoch_10_feature_edit-context_extra_1_seed_1

# bert-task2: 0.635464
python main.py -task task2 -bsz 16 -do_train -do_eval -save_model -track -tensorboard -model transformer -transformer bert-base-uncased -finetune 1 -lr 5e-05 -schedule none -epoch 10 -feature edit-context -train_extra 0 -seed 1 -exp bert-base-uncased_finetune_1_lr_5e-05_schedule_none_epoch_10_feature_edit-context_extra_0_seed_1

# roberta-task2: 0.649543
python main.py -task task2 -bsz 16 -do_train -do_eval -save_model -track -tensorboard -model transformer -transformer roberta-base -finetune 1 -lr 2e-05 -schedule linear_schedule_with_warmup -epoch 3 -feature edit-original -train_extra 0 -seed 1 -exp roberta-base_finetune_1_lr_2e-05_schedule_linear_schedule_with_warmup_epoch_3_feature_edit-original_extra_0_seed_1
