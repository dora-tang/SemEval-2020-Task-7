
CODE_DIR=./code
DATA_DIR=./data

PRED_DIR=./baseline_output
mkdir -p $PRED_DIR
CODE_DIR2=./src



# run baseline 1
echo "=== Task 1 Baseline ==="

echo
echo "[ Dev Result ]"

PRED=$PRED_DIR/task-1-dev-output.csv
GOLD=$DATA_DIR/task-1/dev.csv

python $CODE_DIR/baseline_task_1.py $DATA_DIR/task-1/train.csv $GOLD $PRED
head -10 $PRED
python $CODE_DIR/score_task_1.py $GOLD $PRED
python $CODE_DIR2/evaluate_output.py task1 $GOLD $PRED
# RMSE = 0.578

echo
echo "[ Test Result ]"

PRED=$PRED_DIR/task-1-test-output.csv
GOLD=$DATA_DIR/task-1/test_with_label.csv

python $CODE_DIR/baseline_task_1.py $DATA_DIR/task-1/train.csv $GOLD $PRED
head -10 $PRED
python $CODE_DIR/score_task_1.py $GOLD $PRED
python $CODE_DIR2/evaluate_output.py task1 $GOLD $PRED
# 0.574710


# run baseline 2
echo
echo
echo "=== Task 2 Baseline ==="

echo
echo "[ Dev Result ]"

PRED=$PRED_DIR/task-2-dev-output.csv
GOLD=$DATA_DIR/task-2/dev.csv

python $CODE_DIR/baseline_task_2.py $DATA_DIR/task-2/train.csv $GOLD $PRED
head -10 $PRED
python $CODE_DIR/score_task_2.py $GOLD $PRED
python $CODE_DIR2/evaluate_output.py task2 $GOLD $PRED
# Accuracy = 0.514

echo
echo "[ Test Result ]"

PRED=$PRED_DIR/task-2-test-output.csv
GOLD=$DATA_DIR/task-2/test_with_label.csv

python $CODE_DIR/baseline_task_2.py $DATA_DIR/task-2/train.csv $GOLD $PRED
head -10 $PRED
python $CODE_DIR/score_task_2.py $GOLD $PRED
python $CODE_DIR2/evaluate_output.py task2 $GOLD $PRED
# 0.490487
