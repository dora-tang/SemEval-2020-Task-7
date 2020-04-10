CODE_DIR=../code
DATA_DIR=../data

PRED_DIR=../baseline_output
mkdir -p $PRED_DIR

# run baseline 1
echo "Task 1 Baseline"

python $CODE_DIR/baseline_task_1.py $DATA_DIR/task-1/train.csv $DATA_DIR/task-1/dev.csv
GOLD=$DATA_DIR/task-1/dev.csv
PRED=$PRED_DIR/task-1-output.csv
python $CODE_DIR/score_task_1.py $GOLD $PRED
head $PRED -n 5
# RMSE = 0.578

echo

# run baseline 2
echo "Task 2 Baseline"
mkdir -p ../output
python $CODE_DIR/baseline_task_2.py $DATA_DIR/task-2/train.csv $DATA_DIR/task-2/dev.csv
GOLD=$DATA_DIR/task-2/dev.csv
PRED=$PRED_DIR/task-2-output.csv
python $CODE_DIR/score_task_2.py $GOLD $PRED
head $PRED -n 5
# Accuracy = 0.514
