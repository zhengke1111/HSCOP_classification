'''
Author: zhengke 1604367740@qq.com
Date: 2024-11-22 06:31:51
LastEditors: zhengke 1604367740@qq.com
LastEditTime: 2024-12-05 10:02:56
FilePath: /AHC_max_accuracy/decisiontree/full_mip_tree.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import pandas as pd
import csv
import os
from decisiontree import utils
from decisiontree import PIP_iterations_tree
from decisiontree import full_mip_tree
from decisiontree import callback_data_tree
from decisiontree import PIP_unconstrained_iterations_tree
import time
# from decisiontree import full_MIP_tree_callback
# from decisiontree import call_back_data




def mip_tree(model, data, start, settings, stop_rule, file_path):
    """
    :param lbd:
    :param model:
    :param obj_cons_num: 
    :param X_train: 
    :param y_train: 
    :param w_start: 
    :param b_start: 
    :param z_plus_start: 
    :param z_minus_start: 
    :param epsilon: 
    :param gamma_0: 
    :param M: 
    :param rho: 
    :param beta_p: 
    :param dirname: 
    :return: 
    """
    X_train, y_train, X_test, y_test, class_restricted = data['X_train'], data['y_train'], data['X_test'], data['y_test'], data['class_restricted']
    a_start, b_start, c_start = start['a'], start['b'], start['c']
    method, epsilon, beta_p, D, enhanced_size = settings['method'], settings['epsilon'], settings['beta_p'], settings['D'], settings['enhanced_size']
    base_rate = stop_rule['base_rate']
    result_sub2dir, result_csv, run = file_path['result_sub2dir'], file_path['result_csv'], file_path['run']
    file_path['shrinkage_iter'] = None
    file_path['piece_index'] = None
    multi_piece_list = None
    random.seed(42)
    J = list(set(y_train))
    
    model = model.copy()
    method_list = ['full_mip', 'base_fixed', 'base_shrinkage', 'simplified_arbitrary4_fixed', 'simplified_arbitrary1_fixed', 'simplified_arbitrary4_shrinkage', 'simplified_arbitrary1_shrinkage', 'unconstrained']

    if method == 1:  # 'full_mip'
        result_sub3dir = result_sub2dir + '/full_mip'
        os.makedirs(result_sub3dir, exist_ok=True)
        file_path['result_sub3dir'] = result_sub3dir
        objective_function_term, solution, counts_result, execution_time  = full_mip_tree.full_mip_tree(model, data, start, settings, stop_rule, file_path)
        train_result, test_result, train_constraint_gap, test_constraint_gap, test_train_gap = utils.train_test_results(X_train, y_train, X_test, y_test, solution, D, J, beta_p, class_restricted)
        with open(file_path['details_csv'], mode='a', newline='') as details:
            writer = csv.writer(details)
            writer.writerow([file_path['run'], method_list[method-1], None, None, beta_p, None, None, None, 1, counts_result['num_integer_vars'], objective_function_term['objective_value'], objective_function_term['optimality_gap'], 
                             objective_function_term['final_improvement_time'], objective_function_term['actual_time'], 
                             train_result['frac'], test_result['frac'], test_constraint_gap,
                            counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'], 
                            counts_result['violations_assumption_1_rate'], counts_result['violations_assumption_2_rate'],counts_result['violations_feasibilitytol_rate'],
                            counts_result['z_integrality_vio']])
        record_time = execution_time
        actual_time = objective_function_term['actual_time']
        
    
    if method == 2:  # 'base_fixed'
        result_sub3dir = result_sub2dir + '/base_fixed'
        os.makedirs(result_sub3dir, exist_ok=True)
        file_path['result_sub3dir'] = result_sub3dir
        callback_data_tree.timelimit = 100
        selected_piece = None
        settings['selected_piece'] = None
        start['objective_value'] = -np.inf
        objective_function_term, solution, counts_result, record_time, actual_time = PIP_iterations_tree.pip_iterations(model, data, start, settings, stop_rule, file_path)
        

    if method == 3:  # 'base_shrinkage'
        result_sub3dir = result_sub2dir + '/base_shrinkage'
        os.makedirs(result_sub3dir, exist_ok=True)
        file_path['result_sub3dir'] = result_sub3dir
        callback_data_tree.timelimit = 100
        selected_piece = None
        settings['selected_piece'] = None
        settings['epsilon'] = settings['epsilon_nu']
        epsilon = settings['epsilon']
        max_outer_iter = stop_rule['max_outer_iter']
        iter_start = start
        iter_start['objective_value'] = -np.inf
        record_time_iteration = []
        actual_time_iteration = []
        
        for iteration in range(max_outer_iter):
            file_path['shrinkage_iter'] = iteration
            objective_function_term, solution, counts_result, record_time, actual_time = PIP_iterations_tree.pip_iterations(model, data, iter_start, settings, stop_rule, file_path)
            record_time_iteration.append(record_time)
            actual_time_iteration.append(actual_time)
            iter_start['objective_value'] = solution['objective_value']
            iter_start['a'], iter_start['b'], iter_start['c'] = solution['a'], solution['b'], solution['c']
            settings['epsilon'] = 0.1 * settings['epsilon']
            epsilon = settings['epsilon']
        
        record_time = sum(record_time_iteration)
        actual_time = sum(actual_time_iteration)


    if method == 4:  # 'simplified_arbitrary4_fixed'
        callback_data_tree.timelimit = 30
        result_sub3dir = result_sub2dir + '/simplified_arbitrary4_fixed'
        os.makedirs(result_sub3dir, exist_ok=True)
        file_path['result_sub3dir'] = result_sub3dir
        start['objective_value'] = -np.inf
        objective_function_term, solution, counts_result, record_time, actual_time = PIP_iterations_tree.pip_iterations(model, data, start, settings, stop_rule, file_path)
        multi_piece_list = counts_result['multi_piece_list']


    if method == 5:  # 'simplified_arbitrary1_fixed'
        callback_data_tree.timelimit = 30
        if settings['tune'] == False:
            result_sub3dir = result_sub2dir + '/simplified_arbitrary1_fixed'
        else:
            if settings['regularizer'] == 'hard_l0':
                tau_0 = settings['tau_0']
                result_sub3dir = result_sub2dir + f'/tune/tau0_{tau_0}'
            if settings['regularizer'] == 'soft_l0':
                varrho_index = settings['varrho_index']
                result_sub3dir = result_sub2dir + f'/tune/varrho_{varrho_index}'
        os.makedirs(result_sub3dir, exist_ok=True)
        file_path['result_sub3dir'] = result_sub3dir
        start['objective_value'] = -np.inf
        objective_function_term, solution, counts_result, record_time, actual_time = PIP_iterations_tree.pip_iterations(model, data, start, settings, stop_rule, file_path)
        

    if method == 6:  #  'simplified_arbitrary4_shrinkage'
        callback_data_tree.timelimit = 30
        result_sub3dir = result_sub2dir + '/simplified_arbitrary4_shrinkage'
        os.makedirs(result_sub3dir, exist_ok=True)
        file_path['result_sub3dir'] = result_sub3dir
        max_outer_iter = stop_rule['max_outer_iter']
        iter_start = start
        iter_start['objective_value'] = -np.inf
        
        settings['epsilon'] = settings['epsilon_nu']
        epsilon = settings['epsilon']
        record_time_iteration = []
        actual_time_iteration = []
        max_objective = -np.inf
        multi_piece_list = {}

        for iteration in range(max_outer_iter):
            file_path['shrinkage_iter'] = iteration
            record_time_piece = []
            actual_time_piece = []
            time_start_generate_M = time.time()
            M_set_index, multi_piece = utils.generate_M(X_train, iter_start['a'], iter_start['b'], D, epsilon, base_rate)
            multi_piece_list[f'iter_{iteration}'] = multi_piece
            candidate_M_set_index = utils.generate_combinations(M_set_index)
            time_end_generate_M = time.time()
            time_generate_M = time_end_generate_M - time_start_generate_M
            piece_index = 0
            for selected_piece in candidate_M_set_index:
                piece_index += 1
                file_path['piece_index'] = piece_index
                settings['selected_piece'] = selected_piece
                objective_function_term, solution, counts_result, record_time, actual_time = PIP_iterations_tree.pip_iterations(model, data, iter_start, settings, stop_rule, file_path)
                record_time_piece.append(record_time)
                actual_time_piece.append(actual_time)
                if solution['objective_value'] >= max_objective:
                    max_objective = solution['objective_value']
                    best_objective_function_term, best_solution, best_counts_result = objective_function_term, solution, counts_result
            objective_function_term, solution, counts_result = best_objective_function_term, best_solution, best_counts_result
            record_time_iteration.append(sum(record_time_piece))
            actual_time_iteration.append(sum(actual_time_piece)+time_generate_M)
            iter_start['objective_value'] = solution['objective_value']
            iter_start['a'], iter_start['b'], iter_start['c'] = solution['a'], solution['b'], solution['c']
            settings['epsilon'] = 0.1 * settings['epsilon']
            epsilon = settings['epsilon']
    
        record_time = sum(record_time_iteration)
        actual_time = sum(actual_time_iteration)


    if method == 7:  # 'simplified_arbitrary1_shrinkage'
        callback_data_tree.timelimit = 30
        result_sub3dir = result_sub2dir + '/simplified_arbitrary1_shrinkage'
        os.makedirs(result_sub3dir, exist_ok=True)
        file_path['result_sub3dir'] = result_sub3dir
        max_outer_iter = stop_rule['max_outer_iter']
        iter_start = start
        iter_start['objective_value'] = -np.inf
        
        settings['epsilon'] = settings['epsilon_nu']
        epsilon = settings['epsilon']
        record_time_iteration = []
        actual_time_iteration = []
        max_objective = -np.inf

        for iteration in range(max_outer_iter):
            file_path['shrinkage_iter'] = iteration
            time_start_generate_M = time.time()
            selected_piece = utils.generate_random_combination(X_train, iter_start['a'], iter_start['b'], D, epsilon)
            time_end_generate_M = time.time()
            time_generate_M = time_end_generate_M - time_start_generate_M
            settings['selected_piece'] = selected_piece
            objective_function_term, solution, counts_result, record_time, actual_time = PIP_iterations_tree.pip_iterations(model, data, iter_start, settings, stop_rule, file_path)
            record_time_iteration.append(record_time)
            actual_time_iteration.append(actual_time+time_generate_M)
            iter_start['objective_value'] = solution['objective_value']
            iter_start['a'], iter_start['b'], iter_start['c'] = solution['a'], solution['b'], solution['c']
            settings['epsilon'] = 0.1 * settings['epsilon']
            epsilon = settings['epsilon']
        
        record_time = sum(record_time_iteration)
        actual_time = sum(actual_time_iteration)
    

    if method == 8: # unconstrained
        result_sub3dir = result_sub2dir + '/unconstrained'
        os.makedirs(result_sub3dir, exist_ok=True)
        file_path['result_sub3dir'] = result_sub3dir
        callback_data_tree.timelimit = 30
        start['objective_value'] = -np.inf
        epsilon, beta_p, settings['epsilon'], settings['beta_p'] = None, None, None, None
        objective_function_term, solution, counts_result, record_time, actual_time = PIP_unconstrained_iterations_tree.pip_unconstrained_iterations(model, data, start, settings, stop_rule, file_path)

    train_result, test_result, train_constraint_gap, test_constraint_gap, test_train_gap = utils.train_test_results(X_train, y_train, X_test, y_test, solution, D, J, beta_p, class_restricted)

    if settings['tune'] == False:
        with open(result_csv, mode='a', newline='') as all_result:
            writer = csv.writer(all_result)
            # The last 4 columns is applicable for the case that there is only one element in class_restricted
            writer.writerow([file_path['dataset'], D, run, method_list[method-1], settings['tau_0'], settings['varrho'], next(iter(beta_p)) if beta_p is not None else class_restricted, next(iter(beta_p.values())) if beta_p is not None else None, objective_function_term['objective_value'], objective_function_term['optimality_gap'], 
                            record_time, actual_time, multi_piece_list, next(iter(objective_function_term['gamma'].values())) if objective_function_term['gamma'] is not None else None, 
                            objective_function_term['z_frac'], objective_function_term['z_counts'], train_result['frac'],train_constraint_gap, train_result['counts'], 
                            test_result['frac'], test_constraint_gap, test_result['counts'],
                            test_train_gap,
                            counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'], 
                            counts_result['violations_assumption_1_rate'], counts_result['violations_assumption_2_rate'],counts_result['violations_feasibilitytol_rate'],
                            counts_result['z_integrality_vio'],
                            train_result['frac']['acc'], test_result['frac']['acc'], train_result['frac'][f'prec{class_restricted[0]}'], test_result['frac'][f'prec{class_restricted[0]}']])
    else:
        with open(result_csv, mode='a', newline='') as all_result:
            writer = csv.writer(all_result)
            writer.writerow([file_path['dataset'], D, run, 'tune', settings['tau_0'], settings['varrho'], next(iter(beta_p)) if beta_p is not None else class_restricted, next(iter(beta_p.values())) if beta_p is not None else None, objective_function_term['objective_value'], objective_function_term['optimality_gap'], 
                            record_time, actual_time, multi_piece_list, next(iter(objective_function_term['gamma'].values())) if objective_function_term['gamma'] is not None else None, 
                            objective_function_term['z_frac'], objective_function_term['z_counts'], train_result['frac'],train_constraint_gap, train_result['counts'], 
                            test_result['frac'], test_constraint_gap, test_result['counts'],
                            test_train_gap,
                            counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'], 
                            counts_result['violations_assumption_1_rate'], counts_result['violations_assumption_2_rate'],counts_result['violations_feasibilitytol_rate'],
                            counts_result['z_integrality_vio'],
                            train_result['frac']['acc'], test_result['frac']['acc'], train_result['frac'][f'prec{class_restricted[0]}'], test_result['frac'][f'prec{class_restricted[0]}']])
        return train_result['frac'], test_result['frac']


                    
    