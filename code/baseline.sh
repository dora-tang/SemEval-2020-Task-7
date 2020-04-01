
echo "Task 1 Baseline"
python baseline_task_1.py ../data/task-1/train.csv ../data/task-1/dev.csv

GOLD=../data/task-1/dev.csv
PRED=../output/task-1-output.csv
python score_task_1.py $GOLD $PRED
# RMSE = 0.578

echo

echo "Task 2 Baseline"
python baseline_task_2.py ../data/task-2/train.csv ../data/task-2/dev.csv

GOLD=../data/task-2/dev.csv
PRED=../output/task-2-output.csv
python score_task_2.py $GOLD $PRED
# Accuracy = 0.514
