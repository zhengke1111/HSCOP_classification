import gurobipy as gp
import pandas as pd
from decisiontree import MIP_tree
from decisiontree import utils
from datetime import datetime
from collections import Counter
import csv
import os
import numpy as np
import copy
from pprint import pprint
import math
import random
import time


model = gp.Model('DecisionTree')

model.setParam('MIPFocus', 1)
model.setParam('IntegralityFocus', 1)
model.setParam('NumericFocus',3)
feasibility_tol = 1e-09
model.setParam('FeasibilityTol', feasibility_tol)
model.Params.LazyConstraints = 1



def decisiontree_constraint(dataset='blsc', data_splits = None, beta_p=None, D=2):
    
    current_datetime = datetime.now()
    # result_dir = f'result_tree/depth{D}_noreg'
    result_dir = f'result_tree/depth{D}'
    # result_csv = f'{result_dir}/'+ f'{dataset}_result_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+'.csv'
    # with open(result_csv, mode='a', newline='') as all_result:
    #     writer = csv.writer(all_result)
    #     writer.writerow(['depth','run', 'type', 'tau_0', 'varrho', 'key_beta_p', 'beta_p', 'objective_value', 'optimality_gap', 'time','actual_time', 'same_pieces', 'gamma', 
    #                     'z_frac','z_counts', 'train_frac', 'train_constraint_gap', 'train_counts', 'test_frac', 'test_constraint_gap', 'test_counts',
    #                     'test_train_gap', 'vio_asm_equal_0', 'vio_asm_interval','vio_feasibilitytol', 'vio_asm_equal_0_rate', 'vio_asm_interval_rate','vio_feasibilitytol_rate', 
    #                     'z_integrality_vio','train_acc','test_acc','train_prec','test_prec'])
    
    result_subdir = f'{result_dir}/'+ f'{dataset}_'+ current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(result_subdir)

    details_csv = f'{result_subdir}/'+ f'{dataset}_result_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+'.csv'
    with open(details_csv, mode='a', newline='') as details:
        writer = csv.writer(details)
        writer.writerow(['run', 'type', 'tau_0', 'varrho', 'beta_p','shrinkage_iter', 'piece', 'pip_iter', 'integer_rate', 'integers', 'objective_value', 'optimality_gap',
                        'final_improvement_time','time','train','test', 'test_constraint_gap',
                        'vio_asm_equal_0', 'vio_asm_interval','vio_feasibilitytol',
                        'vio_asm_equal_0_rate', 'vio_asm_interval_rate','vio_feasibilitytol_rate', 
                        'z_integrality_vio'])
    
    
    for run in range(1,5):
        result_sub2dir = result_subdir + f'/splits_{run}'
        os.makedirs(result_sub2dir)
        if D == 2:
            if dataset in ['nwth','wine']:
                base_rate= 40 
            else:
                base_rate = 20 
        elif D == 3:
            if dataset in ['nwth','wine']:
                base_rate = 20 
            else:
                base_rate = 10
        else:
            if dataset in ['ceva','ctmc','fish']:
                base_rate = 5
            else: 
                base_rate = 10

        ## ====== Train and test ======
        X_train = data_splits[run]['X_train']
        y_train = data_splits[run]['y_train']
        X_test = data_splits[run]['X_test']
        y_test = data_splits[run]['y_test']
        N = X_train.shape[0]
        p = X_train.shape[1]

        if p>5:
            ## ======= Same tau_0 as unconstrained ================
            # regularizer = 'hard_l0'
            # df=pd.read_csv(f"result_tree/{dataset}_unconstrained.csv")
            # df["dataset"] = df["dataset"].astype(str)
            # df["depth"]   = df["depth"].astype(int)
            # df["run"]     = df["run"].astype(int)
            # tau_map = df.set_index(["dataset", "depth", "run"])["tau_0"].to_dict()
            # best_tau_0 = tau_map[(dataset, D, run)]
            tau_lb = max(2,math.ceil(p/2)-3)
            tau_ub = min(p,math.ceil(p/2)+3)
            regularizer = 'hard_l0'
            timestamp = time.time()
            random.seed(timestamp)
            best_tau_0 =  random.choice(range(tau_lb, tau_ub + 1))
            print(best_tau_0)
            random.seed()
        else:
            regularizer = 'none'
            best_tau_0 = None
     
        data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'class_restricted': list(beta_p.keys())}
        start = utils.train_decisiontree_model(X_train, y_train, max_depth = D)
        
    
        for method in range(7,8):
            start_copy = copy.deepcopy(start)
            settings = {'method': method, 'epsilon':1e-4, 'epsilon_nu': 1e-1, 'beta_p':beta_p, 'D': D, 'enhanced_size': 4, 'rho': 1e4, 'feasibilitytol': feasibility_tol,
                        'regularizer':regularizer, 'tau_0':best_tau_0, 'varrho': 0, 'tune': False}
            # 'method': 1. 'full_mip', 2. 'base_fixed', 3. 'base_shrinkage', 4. 'simplified_arbitrary4_fixed', 5. 'simplified_arbitrary1_fixed',  6. 'simplified_arbitrary4_shrinkage', 7. 'simplified_arbitrary1_shrinkage' 
            stop_rule = {'timelimit': 3600, 'base_rate': base_rate, 'pip_max_rate': 60, 'unchanged_iters':3, 'max_iteration': 10, 'max_outer_iter': 4}
            file_path = {'dataset': dataset, 'result_csv': result_csv, 'details_csv': details_csv, 'result_dir': result_dir, 'result_sub2dir': result_sub2dir, 'run': run}
            MIP_tree.mip_tree(model, data, start_copy, settings, stop_rule, file_path)




dataset_list = ['wine', 'nwth', 'htds', 'dmtl', 'blsc', 'ctmc', 'ceva', 'fish']


current_datetime = datetime.now()
result_csv = f'result_tree/'+ f'result_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+'.csv'
with open(result_csv, mode='a', newline='') as all_result:
    writer = csv.writer(all_result)
    writer.writerow(['dataset', 'depth', 'run', 'type', 'tau_0', 'varrho', 'key_beta_p', 'beta_p', 'objective_value', 'optimality_gap', 'time','actual_time', 'same_pieces', 'gamma', 
                    'z_frac','z_counts', 'train_frac', 'train_constraint_gap', 'train_counts', 'test_frac', 'test_constraint_gap', 'test_counts',
                    'test_train_gap', 'vio_asm_equal_0', 'vio_asm_interval','vio_feasibilitytol', 'vio_asm_equal_0_rate', 'vio_asm_interval_rate','vio_feasibilitytol_rate', 
                    'z_integrality_vio','train_acc','test_acc','train_prec','test_prec'])
    
for D in range(2,5):
    for dataset in dataset_list:
        X, y = utils.sample_data(dataset=dataset)
        y = y.values + 1
        counter_result = Counter(y)
        max_count = max(counter_result.values())
        most_common_classes = [cls for cls, count in counter_result.items() if count == max_count]
        key_beta = min(most_common_classes).item()
        
        data_splits = {}
        initial_train = {}
        for run in range(1,5):
            data_splits[run] = utils.split_data(X, y, random_state=42+run)
            X_train = data_splits[run]['X_train']
            y_train = data_splits[run]['y_train']
            initial_solution = utils.train_decisiontree_model(X_train, y_train, max_depth = D)
            initial_train_results = utils.evaluate_tree(X_train, y_train, initial_solution['a'], initial_solution['b'], initial_solution['c'], D)
            
            initial_train[run] = initial_train_results['frac']
        
        prec_values = [initial_train[run][f'prec{key_beta}'] for run in initial_train]

        if -1 in prec_values:
            threshold = (np.ceil(np.mean(y == 1)*100)/100).item()
        elif all(p == 1 for p in prec_values):
            threshold = 1
        elif 1 in prec_values:
            threshold = (np.ceil(max(p for p in prec_values if p != 1)*100)/100).item()
        else:
            threshold = (np.ceil(max(prec_values)*100)/100).item()

        # beta_p = {key_beta: threshold}
        # decisiontree_constraint(dataset=dataset, data_splits=data_splits, beta_p=beta_p, D=D)
        threshold_dict = {2: [0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72], 
                          3: [0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82], 
                          4: [0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82]}
        
        for threshold in threshold_dict[D]:
            beta_p = {key_beta: threshold}
            decisiontree_constraint(dataset=dataset, data_splits=data_splits, beta_p=beta_p, D=D)

