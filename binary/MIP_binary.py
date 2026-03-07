import utils
import PIP_binary_iterations
import PIP_binary_unconstrained_iterations
import Full_MIP_binary
import numpy as np
import random
import csv
import os


def mip_binary(model, data, start, settings, stop_rule, file_path):
    """
    Choose a kind of MIP/PIP method to solve the binary classification problem with/without precision constraint

    Args:
        model (dict): Gurobi parameter settings, including {Name, 'MIPFocus', 'IntegralityFocus', 'Threads', 'FeasibilityTol'}
        data (dict): Data splits, {X_train, y_train, X_test, y_test} split by some random seeds we set
        start (dict): Initial solution from the last iteration
        settings (dict): Settings of PIP
        stop_rule (dict): Timelimit, Base_rate, Feasible_rate, Pip_max_rate, Unchanged_iters, Max_iteration, Max_outer_iter
        file_path (dict): File path to store the output
    """
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    method, beta_p = settings['method'], settings['beta_p']
    result_sub2dir, result_csv, run = file_path['result_sub2dir'], file_path['result_csv'], file_path['run']
    random.seed(42)
    model = model.copy()
    method_list = ['full_mip', 'full_mip_t300', 'full_mip_t600', 'fixed', 'shrinkage', 'unconstrained']
    # There are totally 6 methods: 
        # 1: 'full_mip'         Full MIP
        # 2: 'full_mip_t300'    Early MIP (T300)
        # 3: 'full_mip_t600',   Early MIP (T600)
        # 4: 'fixed',           \varepsilon-fixed PIP
        # 5: 'shrinkage',       \varepsilon-shrinkage PIP
        # 6: 'unconstrained'    PIP without precision constraint (for pareto comparison)
    
    if method == 1:  # 'full_mip'
        result_sub3dir = result_sub2dir + '/full_mip'
        settings['callback_type'] = 0
        os.makedirs(result_sub3dir)
        file_path['result_sub3dir'] = result_sub3dir
        objective_function_term, solution, counts_result = Full_MIP_binary.full_mip_binary(model, data, start, settings, stop_rule, file_path)
        with open(file_path['details_csv'], mode='a', newline='') as details:
            writer = csv.writer(details)
            writer.writerow([file_path['run'], method_list[method-1], beta_p, None, None, counts_result['num_integer_vars'],1, objective_function_term['objective_value'], objective_function_term['bestbd'], objective_function_term['optimality_gap'], 
                             solution['w'], solution['b'],
                            counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'], counts_result['z_integrality_vio']])
        record_time = objective_function_term['time']

    
    if method == 2:  # 'full_mip_t300'
        result_sub3dir = result_sub2dir + '/full_mip_t300'
        settings['callback_type'] = 1
        os.makedirs(result_sub3dir)
        file_path['result_sub3dir'] = result_sub3dir
        objective_function_term, solution, counts_result = Full_MIP_binary.full_mip_binary(model, data, start, settings, stop_rule, file_path)
        with open(file_path['details_csv'], mode='a', newline='') as details:
            writer = csv.writer(details)
            writer.writerow([file_path['run'], method_list[method-1], beta_p, None, None, counts_result['num_integer_vars'],1, objective_function_term['objective_value'], objective_function_term['bestbd'], objective_function_term['optimality_gap'], 
                            solution['w'], solution['b'],
                            counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'], counts_result['z_integrality_vio']])
        record_time = objective_function_term['time']


    if method == 3:  # 'full_mip_t600'
        result_sub3dir = result_sub2dir + '/full_mip_t600'
        settings['callback_type'] = 2
        os.makedirs(result_sub3dir)
        file_path['result_sub3dir'] = result_sub3dir
        objective_function_term, solution, counts_result = Full_MIP_binary.full_mip_binary(model, data, start, settings, stop_rule, file_path)
        with open(file_path['details_csv'], mode='a', newline='') as details:
            writer = csv.writer(details)
            writer.writerow([file_path['run'], method_list[method-1], beta_p, None, None, counts_result['num_integer_vars'], 1,objective_function_term['objective_value'], objective_function_term['bestbd'], objective_function_term['optimality_gap'], 
                            solution['w'], solution['b'],
                            counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'], counts_result['z_integrality_vio']])
        record_time = objective_function_term['time']
        
        
    if method == 4:  # 'fixed'
        result_sub3dir = result_sub2dir + '/base_fixed'
        os.makedirs(result_sub3dir)
        file_path['result_sub3dir'] = result_sub3dir
        start['objective_value'] = -np.inf
        objective_function_term, solution, counts_result, execution_time = PIP_binary_iterations.pip_binary_iterations(model, data, start, settings, stop_rule, file_path)
        record_time = sum(execution_time) 

    if method == 5:  # 'shrinkage'
        result_sub3dir = result_sub2dir + '/base_shrinkage'
        os.makedirs(result_sub3dir)
        file_path['result_sub3dir'] = result_sub3dir
        settings['epsilon'] = settings['epsilon_nu']
        max_outer_iter = stop_rule['max_outer_iter']
        iter_start = start
        iter_start['objective_value'] = -np.inf
        execution_time_iteration = []
        
        for iter in range(max_outer_iter):
            file_path['shrinkage_iter'] = iter
            objective_function_term, solution, counts_result, execution_time = PIP_binary_iterations.pip_binary_iterations(model, data, iter_start, settings, stop_rule, file_path)
            execution_time_iteration.append(sum(execution_time))
            iter_start['objective_value'] = solution['objective_value']
            iter_start['w'], iter_start['b'] = solution['w'], solution['b']
            settings['epsilon'] = 0.1 * settings['epsilon']

        objective_function_term, solution, counts_result = objective_function_term, solution, counts_result 
        record_time = sum(execution_time_iteration)

    if method == 6:  # 'unconstrained'
        result_sub3dir = result_sub2dir + '/unconstrained'
        os.makedirs(result_sub3dir)
        file_path['result_sub3dir'] = result_sub3dir
        
        iter_start = start
        iter_start['objective_value'] = -np.inf
        execution_time_iteration = []
        
        objective_function_term, solution, counts_result, execution_time = PIP_binary_unconstrained_iterations.pip_binary_unconstrained_iterations(model, data, start, settings, stop_rule, file_path)
        objective_function_term['gamma_in_obj'] = None
        record_time = sum(execution_time)

    train_result = utils.evaluate_binary(X_train, y_train, solution['w'], solution['b'])
    test_result = utils.evaluate_binary(X_test, y_test, solution['w'], solution['b'])

    with open(result_csv, mode='a', newline='') as all_result:
        writer = csv.writer(all_result)
        writer.writerow([run, method_list[method-1], beta_p, objective_function_term['objective_value'], objective_function_term['bestbd'], objective_function_term['optimality_gap'], 
                         record_time, solution['w'], solution['b'], objective_function_term['gamma_in_obj'], 
                         train_result['acc_margin'],train_result['accuracy'], train_result['precision'], train_result['recall'], 
                         test_result['acc_margin'], test_result['accuracy'], test_result['precision'], test_result['recall'], 
                         counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'], counts_result['z_integrality_vio']])


                    

    