import PIP_tree_solve_partial_problem as PIP_tree_solve_partial_problem
import utils
import csv
import time
import numpy as np


def pip_tree_control_termination(model, data, start, settings, stop_rule, file_path):
    """
    Determine whether to continue or stop, and, if continuing, decide whether to enlarge or shrink the in-between sets ${\cal J}$. 
    For tree-based classification problem with precision constraint. 

    Args:
        model (dict): Gurobi parameter settings, including {Name, 'MIPFocus', 'IntegralityFocus', 'Threads', 'NumericFocus', 'FeasibilityTol'}
        data (dict): Data splits, {X_train, y_train, X_test, y_test} split by some random seeds we set
        start (dict): Initial solution from the last iteration
        settings (dict): Settings of PIP
        stop_rule (dict): Timelimit, Base_rate, Feasible_rate, Pip_max_rate, Unchanged_iters, Max_iteration, Max_outer_iter
        file_path (dict): File path to store the output
    """
    X_train, y_train, X_test, y_test, class_restricted = data['X_train'], data['y_train'], data['X_test'], data['y_test'], data['class_restricted']
    method, epsilon, beta_p, D, enhanced_size = settings['method'], settings['epsilon'], settings['beta_p'], settings['D'], settings['enhanced_size']
    base_rate, pip_max_rate, expansion_rate, unchanged_iters, max_iteration = stop_rule['base_rate'],  stop_rule['pip_max_rate'], stop_rule['expansion_rate'], stop_rule['unchanged_iters'], stop_rule['max_iteration']
    details_csv = file_path['details_csv']
    record_time = []
    iter_unchanged = 0
    integer_rate = base_rate
    method_list = method_list = ['Full MIP', 'F-PIP', 'ISA_PIP', 'FD4-PIP', 'FD-PIP', 'IDSA4-PIP', 'IDSA-PIP', 'U-PIP']
    multi_piece_list = []
    J = list(set(y_train))

    # PIP iterations
    for iteration in range(max_iteration):
        time_start_generate_M = time.time()

        if method == 4:  # simplified_arbitrary4_fixed
            M_set_index, multi_piece = utils.generate_M(X_train, start['a'], start['b'], D, epsilon, integer_rate, enhanced_size)
            candidate_M_set_index = utils.generate_combinations(M_set_index)  # Generate {\cal L}^4_{st}(a,b)
            multi_piece_list.append(multi_piece)
        if method == 5:
            settings['selected_piece'] = utils.generate_random_combination(X_train, start['a'], start['b'], D, epsilon)  # Generate {\cal L}_{st}(a,b)
            
        time_end_generate_M = time.time()
        # When enhanced_size is very high, it may cost a long time to generate a pieces set
        time_generate_M = time_end_generate_M - time_start_generate_M

        file_path['pip_iter'] = iteration
        objective_value_old = start['objective_value']
        max_objective = -np.inf

        if method == 4:
            piece_index = 0
            # Traverse each selected piece
            for selected_piece in candidate_M_set_index:
                settings['selected_piece'] = selected_piece
                piece_index += 1
                file_path['piece_index'] = piece_index
                start['gamma'], start['z_plus_0'], start['z_plus'], start['z_minus'] = utils.calculate_gamma(X_train, y_train, start['a'], start['b'], start['c'], D, settings['selected_piece'], beta_p, epsilon, class_restricted)
                start['eta'], start['zeta'], start['L'] = utils.calculate_eta_zeta_L(X_train, y_train, start['c'], D, start['z_plus_0'], start['z_plus'], start['z_minus'])
                settings['delta_1'], settings['delta_2'] = utils.calculate_delta(X_train=X_train, a=start['a'], b=start['b'], D=D, selected_piece=settings['selected_piece'], epsilon=epsilon, base_rate=integer_rate)
                
                objective_function_term, solution, counts_result = PIP_tree_solve_partial_problem.pip_tree_solve_partial_problem(model, data, start, settings, file_path)
                train_result, test_result, train_constraint_gap, test_constraint_gap, test_train_gap = utils.train_test_results(X_train, y_train, X_test, y_test, solution, D, J, beta_p, class_restricted)

                with open(details_csv, mode='a', newline='') as details:
                    writer = csv.writer(details)
                    writer.writerow([file_path['split'], method_list[method-1], settings['tau_0'], beta_p, file_path['shrinkage_iter'], file_path['piece_index'], iteration, integer_rate/100, counts_result['num_integer_vars'], objective_function_term['objective_value'], objective_function_term['optimality_gap'], 
                                      None, objective_function_term['runtime'],
                                    train_result['frac'], test_result['frac'], test_constraint_gap,
                                    counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'],
                                    counts_result['violations_assumption_1_rate'], counts_result['violations_assumption_2_rate'],counts_result['violations_feasibilitytol_rate'], 
                                    counts_result['z_integrality_vio']])
                
                if solution['objective_value'] >= max_objective:
                    max_objective = solution['objective_value']
                    best_objective_function_term, best_solution, best_counts_result = objective_function_term, solution, counts_result
            objective_function_term, solution, counts_result = best_objective_function_term, best_solution, best_counts_result
            record_time.append(objective_function_term['runtime'])

        else:  # In each PIP iteration, only one piece is selected 
            start['gamma'], start['z_plus_0'], start['z_plus'], start['z_minus'] = utils.calculate_gamma(X_train, y_train, start['a'], start['b'], start['c'], D, settings['selected_piece'], beta_p, epsilon, class_restricted)
            start['eta'], start['zeta'], start['L'] = utils.calculate_eta_zeta_L(X_train, y_train, start['c'], D, start['z_plus_0'], start['z_plus'], start['z_minus'])
            settings['delta_1'], settings['delta_2'] = utils.calculate_delta(X_train=X_train, a=start['a'], b=start['b'], D=D, selected_piece=settings['selected_piece'], epsilon=epsilon, base_rate=integer_rate)
            
            objective_function_term, solution, counts_result = PIP_tree_solve_partial_problem.pip_tree_solve_partial_problem(model, data, start, settings, file_path)
            record_time.append(objective_function_term['runtime'])

            train_result, test_result, train_constraint_gap, test_constraint_gap, test_train_gap = utils.train_test_results(X_train, y_train, X_test, y_test, solution, D, J, beta_p, class_restricted)
            with open(details_csv, mode='a', newline='') as details:
                writer = csv.writer(details)
                writer.writerow([file_path['split'], method_list[method-1], settings['tau_0'], beta_p, file_path['shrinkage_iter'], file_path['piece_index'], iteration, integer_rate/100, counts_result['num_integer_vars'], objective_function_term['objective_value'], objective_function_term['optimality_gap'],
                                 None, objective_function_term['runtime'],
                                train_result['frac'], test_result['frac'],  test_constraint_gap,
                                counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'],
                                counts_result['violations_assumption_1_rate'], counts_result['violations_assumption_2_rate'],counts_result['violations_feasibilitytol_rate'], 
                                counts_result['z_integrality_vio']])
                
        start = solution
        objective_value = solution['objective_value']
    
        if objective_value - objective_value_old <= 1e-5:           # Objective value remains unchanged
            iter_unchanged += 1
            integer_rate = min(pip_max_rate, integer_rate + expansion_rate)     # Enlarge the In-between index set
            
        else:                                                       # Objective value increases
            iter_unchanged = 0
            integer_rate = max(base_rate, integer_rate - expansion_rate)        # Shrink the In-between index set

        if iter_unchanged >= unchanged_iters:                       # Reach \tilde{\mu}_{max}
            max_iteration = iteration + 1
            break

    record_time = sum(record_time)
    counts_result['multi_piece_list'] = multi_piece_list

    return objective_function_term, solution, counts_result, record_time



def pip_unconstrained_tree_control_termination(model, data, start, settings, stop_rule, file_path):
    """
    Determine whether to continue or stop, and, if continuing, decide whether to enlarge or shrink the in-between sets ${\cal J}$. 
    For tree-based classification problem without precision constraint. 

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
    base_rate, pip_max_rate, expansion_rate, unchanged_iters, max_iteration = stop_rule['base_rate'],  stop_rule['pip_max_rate'], stop_rule['expansion_rate'], stop_rule['unchanged_iters'], stop_rule['max_iteration']
    details_csv = file_path['details_csv']
    record_time = []
    iter_unchanged = 0
    integer_rate = base_rate
    method_list = method_list = ['Full MIP', 'F-PIP', 'ISA_PIP', 'FD4-PIP', 'FD-PIP', 'IDSA4-PIP', 'IDSA-PIP', 'U-PIP']
    multi_piece_list = []
    J = list(set(y_train))

    for iteration in range(max_iteration):

        file_path['pip_iter'] = iteration
        objective_value_old = start['objective_value']

        start['z_plus_0'] = utils.calculate_z_plus_0(X_train, start['a'], start['b'], D)
        start['L'] = utils.calculate_L(X_train, y_train, start['c'], D, start['z_plus_0'])
        settings['delta_1'], settings['delta_2'] = utils.calculate_delta(X_train=X_train, a=start['a'], b=start['b'], D=D, selected_piece=None, epsilon=None, base_rate=integer_rate)
        
        objective_function_term, solution, counts_result = PIP_tree_solve_partial_problem.pip_unconstrained_tree_solve_partial_problem(model, data, start, settings, file_path)
        record_time.append(objective_function_term['runtime'])

        train_result, test_result, train_constraint_gap, test_constraint_gap, test_train_gap = utils.train_test_results(X_train, y_train, X_test, y_test, solution, D, J, None, class_restricted)
        with open(details_csv, mode='a', newline='') as details:
            writer = csv.writer(details)
            writer.writerow([file_path['split'], method_list[method-1], settings['tau_0'], None, None, None, iteration, integer_rate/100, counts_result['num_integer_vars'], objective_function_term['objective_value'], objective_function_term['optimality_gap'],
                                None, objective_function_term['runtime'],
                            train_result['frac'], test_result['frac'],  test_constraint_gap,
                            counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'],
                            counts_result['violations_assumption_1_rate'], counts_result['violations_assumption_2_rate'],counts_result['violations_feasibilitytol_rate'], 
                            counts_result['z_integrality_vio']])
                
        start = solution
        objective_value = solution['objective_value']

        if objective_value - objective_value_old <= 1e-5:
            iter_unchanged += 1
            integer_rate = min(pip_max_rate, integer_rate + expansion_rate)
            
        else:
            iter_unchanged = 0
            integer_rate = max(base_rate, integer_rate - expansion_rate)

        if iter_unchanged >= unchanged_iters:
            max_iteration = iteration + 1
            break

    record_time = sum(record_time)
    counts_result['multi_piece_list'] = multi_piece_list

    return objective_function_term, solution, counts_result, record_time