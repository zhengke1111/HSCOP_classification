from decisiontree import PIP_unconstrained_single_iter_tree
from decisiontree import utils
from decisiontree import callback_data_tree
import csv
import time
import numpy as np
from collections import Counter


def pip_unconstrained_iterations(model, data, start, settings, stop_rule, file_path):
    """
    PIP iterations in PIP method to solve the decision tree classification problem without precision constraint

    Args:
        model (dict): Gurobi parameter settings, including {Name, 'MIPFocus', 'IntegralityFocus', 'Threads', 'NumericFocus', 'FeasibilityTol'}
        data (dict): Data splits, {X_train, y_train, X_test, y_test} split by some random seeds we set
        start (dict): Initial solution from the last iteration
        settings (dict): Settings of PIP
        stop_rule (dict): Timelimit, Base_rate, Feasible_rate, Pip_max_rate, Unchanged_iters, Max_iteration, Max_outer_iter
        file_path (dict): File path to store the output
    """
    # Below are similar to the function pip_iterations in PIP_iterations_tree.py
    X_train, y_train, X_test, y_test, class_restricted = data['X_train'], data['y_train'], data['X_test'], data['y_test'], data['class_restricted']
    method, D = settings['method'], settings['D']
    base_rate, pip_max_rate, unchanged_iters, max_iteration = stop_rule['base_rate'],  stop_rule['pip_max_rate'], stop_rule['unchanged_iters'], stop_rule['max_iteration']
    details_csv = file_path['details_csv']
    record_time = []
    actual_time = []
    iter_unchanged = 0
    integer_rate = base_rate
    method_list = ['full_mip', 'base_fixed', 'base_shrinkage', 'simplified_arbitrary4_fixed', 'simplified_arbitrary1_fixed', 'simplified_arbitrary4_shrinkage', 'simplified_arbitrary1_shrinkage', 'unconstrained']
    multi_piece_list = []
    J = list(set(y_train))

    for iteration in range(max_iteration):

        file_path['pip_iter'] = iteration
        objective_value_old = start['objective_value']

        start['z_plus_0'] = utils.calculate_z_plus_0(X_train, start['a'], start['b'], D)
        start['L'] = utils.calculate_L(X_train, y_train, start['c'], D, start['z_plus_0'])
        settings['delta_1'], settings['delta_2'] = utils.calculate_delta(X_train=X_train, a=start['a'], b=start['b'], D=D, selected_piece=None, epsilon=None, base_rate=integer_rate)
        
        objective_function_term, solution, counts_result = PIP_unconstrained_single_iter_tree.pip_unconstrained_single_iter_tree(model, data, start, settings, file_path)
        record_time.append(objective_function_term['runtime'])
        actual_time.append(objective_function_term['runtime'])

        train_result, test_result, train_constraint_gap, test_constraint_gap, test_train_gap = utils.train_test_results(X_train, y_train, X_test, y_test, solution, D, J, None, class_restricted)
        with open(details_csv, mode='a', newline='') as details:
            writer = csv.writer(details)
            writer.writerow([file_path['run'], method_list[method-1], settings['tau_0'], None, None, None, iteration, integer_rate/100, counts_result['num_integer_vars'], objective_function_term['objective_value'], objective_function_term['optimality_gap'],
                                None, objective_function_term['runtime'],
                            train_result['frac'], test_result['frac'],  test_constraint_gap,
                            counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'],
                            counts_result['violations_assumption_1_rate'], counts_result['violations_assumption_2_rate'],counts_result['violations_feasibilitytol_rate'], 
                            counts_result['z_integrality_vio']])
                
        start = solution
        objective_value = solution['objective_value']

        if objective_value - objective_value_old <= 1e-5:
            iter_unchanged += 1
            integer_rate = min(pip_max_rate, integer_rate + 10)
            
        else:
            iter_unchanged = 0
            integer_rate = max(base_rate, integer_rate - 10)

        if iter_unchanged >= unchanged_iters:
            max_iteration = iteration + 1
            break

    record_time = sum(record_time)
    actual_time = sum(actual_time)
    counts_result['multi_piece_list'] = multi_piece_list

    return objective_function_term, solution, counts_result, record_time, actual_time