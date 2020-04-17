# -*- coding: utf-8 -*-
"""
@author: Nabil Hossain
         nhossain@cs.rochester.edu
         Dept. Computer Science
         University of Rochester, NY
"""

'''
A naive baseline system for task 2

This baseline always predicts the most frequent label in the training set.
'''

import pandas as pd
import numpy as np
import sys
import os
def baseline_task_2(train_loc, test_loc, out_loc):
    train = pd.read_csv(train_loc)
    test = pd.read_csv(test_loc)

    counts = train['label'].value_counts(sort=False)
    #counts = train['label'].value_counts()
    pred = np.argmax(counts.values)
    print(f'Count: {dict(counts)}')
    print(f'Most frequent label in training set: {pred}')
    test['pred'] = pred

    output = test[['id', 'pred']]
    output.to_csv(out_loc, index=False)

    print('Output file created:\n\t- '+os.path.abspath(out_loc))


if __name__ == '__main__':

    # expect sys.argv[1] = ../data/task-2/train.csv
    # expect sys.argv[2] = ../data/task-2/dev.csv
    # expect sys.argv[3] = '../baseline_output/task-2-output.csv'
    if len(sys.argv) <= 2:
        out_loc = '../baseline_output/task-2-output.csv'
    else:
        out_loc = sys.argv[3]
    baseline_task_2(sys.argv[1], sys.argv[2], out_loc)
