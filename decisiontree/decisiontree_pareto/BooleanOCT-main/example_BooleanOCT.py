# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:02:10 2023

@author: JC TU
"""



import time
from os import path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
import os
import csv
from datetime import datetime

import tree as miptree
from sklearn import tree


depths = [2, 3, 4]
seeds = [1, 2, 3, 4]

dataset = 'balance_scale_onehot'
# ================ Output file ======================================================================
current_datetime = datetime.now()
result_dir = os.getcwd() + '/decisiontree_pareto/BooleanOCT-main/results'
result_csv = f'{result_dir}/'+ f'{dataset}_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+'.csv'
with open(result_csv, mode='a', newline='') as all_result:
    writer = csv.writer(all_result)
    writer.writerow(['approach', 'dataset', 'sample', 'nrow', 'depth', 'restricted_class', 'beta_p', 'alpha', 'max_num_features', 
                     'time_limit', 'status', 'obj_value', 'gamma_value', 'solving_time', 'gap',
                     'train_acc', 'eval_acc','test_acc', 'train_prec', 'eval_prec','test_prec'])

# ================ Read data ========================================================================
data_path = os.getcwd() + '/decisiontree_pareto/BooleanOCT-main/data/'
df = pd.read_csv(data_path + dataset +'.csv')
x = df.drop('target',axis=1)
y = df['target']
x = np.array(x.values)
y = np.array(y.values)

# ================= Train process ====================================================================
approach_name = "U-BooleanOCT"
time_limit = 3600
restricted_label = 0
validation = 1
for depth in depths:
    for seed in seeds: 
        for alpha in [0.001, 0.01]:
            for N in [3, 5]:
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.25, stratify=y, random_state=42 + seed
                )

                if validation == 1:
                    x_train, x_val, y_train, y_val = train_test_split(
                        x_train, y_train, test_size=1/3, stratify=y_train, random_state=42 + seed 
                    )


                brtree = miptree.booleanoptimalDecisionTreeClassifier(max_depth=depth, min_samples_split=1, alpha=alpha, N=N, warmstart=True, timelimit=time_limit, output=True)

                brtree.fit(x_train, y_train)
                
                y_train_pred = brtree.predict(x_train)
                y_test_pred = brtree.predict(x_test)
                if validation == 1:
                    y_val_pred = brtree.predict(x_val)


                train_acc = accuracy_score(y_train, brtree.predict(x_train))        
                test_acc = accuracy_score(y_test, brtree.predict(x_test))
                if validation == 1:
                    val_acc = accuracy_score(y_val, brtree.predict(x_val))

                train_prec = precision_score(y_train, brtree.predict(x_train), labels=[restricted_label], average= None)[0]
                test_prec = precision_score(y_test, brtree.predict(x_test), labels=[restricted_label], average= None)[0]
                if validation == 1:
                    val_prec = precision_score(y_val, brtree.predict(x_val), labels=[restricted_label], average= None)[0]

                train_len = x_train.shape[0]
                with open(result_csv, mode='a') as results:
                    results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                    results_writer.writerow(
                        [approach_name, dataset, seed, train_len, depth, restricted_label, None, alpha, N, 
                            time_limit, brtree.status, brtree.objval, None, brtree.runtime, brtree.optgap, 
                            train_acc, val_acc, test_acc, train_prec, val_prec, test_prec])
                #print('train_acc:', train_acc, 'val_acc:', val_acc, 'test_acc:',test_acc)
