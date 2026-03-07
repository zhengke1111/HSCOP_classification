import MIP_tree
import utils
from datetime import datetime
from collections import Counter
import gurobipy as gp
import csv
import os
import numpy as np
import copy
import math
import random
import time

# =========== Gurobi Settings ==========
model = gp.Model('DecisionTree')
model.setParam('MIPFocus', 1)
model.setParam('IntegralityFocus', 1)
model.setParam('Threads', 32)
model.setParam('NumericFocus',3)
feasibility_tol = 1e-09
model.setParam('FeasibilityTol', feasibility_tol)
model.Params.LazyConstraints = 1


# =========== Settings to Solve Decision Tree Problem with Precision Constraint ==========
def decisiontree_constraint(dataset='blsc', data_splits = None, beta_p=None, D=2):
    """
    Settings and output for decision tree classification problem with precision constraint

    Args:
        dataset (str, optional): Abbreviation of dataset name. Defaults to 'blsc'.
        data_splits (dict, optional): Dictionary to store the split data. Defaults to None.
        beta_p (dict, optional): {key_beta: threshold}: Rescticted Class: Precision (lower) threshold to be meet. Defaults to None.
        D (int, optional): Depth. Defaults to 2.
    """
    # Name the result document based on the start time of the experiment
    current_datetime = datetime.now()
    result_dir = f'decisiontree/results/depth{D}'

    # For each (depth, dataset, beta_p), a "result_csv" is created
    result_csv = f'{result_dir}/'+ f'{dataset}_result_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+'.csv'
    with open(result_csv, mode='a', newline='') as all_result:
        writer = csv.writer(all_result)
        writer.writerow(['dataset', 'depth', 'run', 'type', 'tau_0', 'key_beta_p', 'beta_p', 'objective_value', 'optimality_gap', 'time','actual_time', 'same_pieces', 'gamma', 
                        'z_frac','z_counts', 'train_frac', 'train_constraint_gap', 'train_counts', 'test_frac', 'test_constraint_gap', 'test_counts',
                        'test_train_gap', 'vio_asm_equal_0', 'vio_asm_interval','vio_feasibilitytol', 'vio_asm_equal_0_rate', 'vio_asm_interval_rate','vio_feasibilitytol_rate', 
                        'z_integrality_vio','train_acc','test_acc','train_prec','test_prec'])
    
    # "sub" means subdirectory
    result_subdir = f'{result_dir}/'+ f'{dataset}_'+ current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(result_subdir)

    # "details_csv" records the detailed results of each PIP iteration 
    details_csv = f'{result_subdir}/'+ f'{dataset}_result_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+'.csv'
    with open(details_csv, mode='a', newline='') as details:
        writer = csv.writer(details)
        # The following terms are explained in README.md
        writer.writerow(['run', 'type', 'tau_0', 'beta_p','shrinkage_iter', 'piece', 'pip_iter', 'integer_rate', 'integers', 'objective_value', 'optimality_gap',
                        'final_improvement_time','time','train','test', 'test_constraint_gap',
                        'vio_asm_equal_0', 'vio_asm_interval','vio_feasibilitytol',
                        'vio_asm_equal_0_rate', 'vio_asm_interval_rate','vio_feasibilitytol_rate', 
                        'z_integrality_vio'])
    
    
    for run in range(1,5):
        # "sub2" means subsubdirectory, for each run(split) there is a subsubdirectory
        result_sub2dir = result_subdir + f'/splits_{run}'
        os.makedirs(result_sub2dir)

        # Set integer ratios based on different depths and dataset sizes
        # The deeper the depth, the larger the dataset, and the lower the integer ratio
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

        # Extract training and test sets
        X_train = data_splits[run]['X_train']
        y_train = data_splits[run]['y_train']
        X_test = data_splits[run]['X_test']
        y_test = data_splits[run]['y_test']
        N = X_train.shape[0]
        p = X_train.shape[1]

        if p>5:                                                     # When the dimension of features p>5
            regularizer = 'hard_l0'                                 # Set \tau_0 as the max number of features
            tau_lb = max(2,math.ceil(p/2)-3)                        # Upper bound: \lceil p/2 \rceil +3
            tau_ub = min(p,math.ceil(p/2)+3)                        # Lower bound: \lceil p/2 \rceil -3
            timestamp = time.time()
            random.seed(timestamp)                                  # Random seeds depend on the realtime timestamp
            best_tau_0 =  random.choice(range(tau_lb, tau_ub + 1))  # Randomly select \tau_0 \in [\lceil p/2 \rceil -3, \lceil p/2 \rceil +3]
            random.seed()

            # ========== When we run pareto comparison, we keep the number of features of constrained PIP the same as unconstrained PIP ==========
            # regularizer = 'hard_l0'
            # df=pd.read_csv(f"decisiontree/results/{dataset}_unconstrained.csv")
            # df["dataset"] = df["dataset"].astype(str)
            # df["depth"]   = df["depth"].astype(int)
            # df["run"]     = df["run"].astype(int)
            # tau_map = df.set_index(["dataset", "depth", "run"])["tau_0"].to_dict()
            # best_tau_0 = tau_map[(dataset, D, run)]
        else:                                                       # When the dimension of features p\le 5
            regularizer = 'none'
            best_tau_0 = None                                       # Do not set \tau_0
     
        data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'class_restricted': list(beta_p.keys())}
        start = utils.train_decisiontree_model(X_train, y_train, max_depth = D) # Initial solution generated by DecisionTreeClassifier

        # There are totally 8 methods: 
        # 1: 'full_mip'                             Full MIP
        # 2: 'base_fixed'                           \varepsilon-fixed
        # 3: 'base_shrinkage'                       \varepsilon-shrinkage
        # 4: 'simplified_arbitrary4_fixed'          \varepsilon-fixed-arbitrary4
        # 5: 'simplified_arbitrary1_fixed'          \varepsilon-fixed-arbitrary1
        # 6: 'simplified_arbitrary4_shrinkage'      \varepsilon-shrinkage-arbitrary4
        # 7: 'simplified_arbitrary1_shrinkage'      \varepsilon-shrinkage-arbitrary1
        # 8: 'unconstrained'                        Unconstrained PIP
        for method in range(1,8):
            start_copy = copy.deepcopy(start)
            settings = {'method': method, 'epsilon':1e-4, 'epsilon_nu': 1e-1, 'beta_p':beta_p, 'D': D, 'enhanced_size': 4, 'rho': 1e4, 'feasibilitytol': feasibility_tol,
                        'regularizer':regularizer, 'tau_0':best_tau_0, 'tune': False}
            stop_rule = {'timelimit': 3600, 'base_rate': base_rate, 'pip_max_rate': 60, 'unchanged_iters':3, 'max_iteration': 10, 'max_outer_iter': 4}
            file_path = {'dataset': dataset, 'result_csv': result_csv, 'details_csv': details_csv, 'result_dir': result_dir, 'result_sub2dir': result_sub2dir, 'run': run}
            MIP_tree.mip_tree(model, data, start_copy, settings, stop_rule, file_path)


# Abbreviation of dataset name
dataset_list = ['wine', 'nwth', 'htds', 'dmtl', 'blsc', 'ctmc', 'ceva', 'fish']

# ========== If run grid thresholds for drawing pareto curve, integrate the results under different beta_p into one result_csv ==========
# current_datetime = datetime.now()
# result_csv = f'result_tree/'+ f'result_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+'.csv'
# with open(result_csv, mode='a', newline='') as all_result:
#     writer = csv.writer(all_result)
#     writer.writerow(['dataset', 'depth', 'run', 'type', 'tau_0', 'key_beta_p', 'beta_p', 'objective_value', 'optimality_gap', 'time','actual_time', 'same_pieces', 'gamma', 
#                     'z_frac','z_counts', 'train_frac', 'train_constraint_gap', 'train_counts', 'test_frac', 'test_constraint_gap', 'test_counts',
#                     'test_train_gap', 'vio_asm_equal_0', 'vio_asm_interval','vio_feasibilitytol', 'vio_asm_equal_0_rate', 'vio_asm_interval_rate','vio_feasibilitytol_rate', 
#                     'z_integrality_vio','train_acc','test_acc','train_prec','test_prec'])

def decisiontree_run():
    """
    Run constrained decision tree experiments over multiple datasets and tree depths.

    This function iterates over a collection of datasets and a range of maximum tree depths. For each dataset and each depth, it performs multiple random
    train/test splits, trains an initial decision tree model on each training split, and evaluates the resulting classifier on the training data.

    Based on the training precision of the majority class across runs, the function constructs a class-specific precision threshold `beta_p`. This
    threshold is then passed to `decisiontree_constraint(...)` to solve or evaluate the corresponding constrained decision tree problem.

    Procedure:
        1. For each tree depth `D` in `{2, 3, 4}`:
        2. For each dataset in `dataset_list`:
           - load the dataset,
           - shift labels by 1,
           - identify the majority class (breaking ties by choosing the smaller
             class label),
           - generate four random train/test splits,
           - train an initial decision tree on each training split,
           - evaluate the training precision for the selected majority class,
           - determine a threshold `beta_p` from these precision values,
           - call `decisiontree_constraint(...)` with the constructed threshold.

    Threshold construction:
        - If at least one recorded precision is `-1`(the corresponding denominator is 0), the threshold is set to the ceiling of the empirical proportion of class `1`, 
          rounded up to two decimal places.
        - If all precision values are `1`, the threshold is set to `1`.
        - If some precision values are `1` but not all, the threshold is set to the ceiling of the largest precision strictly less than `1`, rounded
          up to two decimal places.
        - Otherwise, the threshold is set to the ceiling of the maximum observed precision, rounded up to two decimal places.

    Notes:
        - The labels are transformed via `y = y.values + 1`, so the function assumes the original labels are encoded in a form compatible with
          this shift.
        - The majority class is determined from the full dataset, not from individual training splits.
        - The threshold `beta_p` is defined only for the selected majority class `key_beta`.
    """
    for D in range(2, 5):
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
                data_splits[run] = utils.split_data(X, y, random_state = 42 + run)
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

            beta_p = {key_beta: threshold}
            decisiontree_constraint(dataset=dataset, data_splits=data_splits, beta_p=beta_p, D=D)
                
            # ========== Grid thresholds for pareto comparison ==========
            # threshold_dict = {2: [0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72], 
            #                   3: [0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82], 
            #                   4: [0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82]}
            
            # for threshold in threshold_dict[D]:
            #     beta_p = {key_beta: threshold}
            #     decisiontree_constraint(dataset=dataset, data_splits=data_splits, beta_p=beta_p, D=D)


decisiontree_run()