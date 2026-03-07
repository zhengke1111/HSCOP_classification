import utils
import PIP_iterations_tree
import Full_MIP_tree
import callback_data_tree
import PIP_unconstrained_iterations_tree
import numpy as np
import random
import csv
import os
import time


def mip_tree(model, data, start, settings, stop_rule, file_path):
    """
    Run a tree-based MIP/PIP procedure under a specified solution method and record the resulting train/test performance.

    This function serves as the main driver for solving a decision-tree-based optimization model under one of several supported methods, including the
    full MIP formulation, various fixed or shrinkage PIP schemes, simplified piecewise methods, and the unconstrained variant.

    Depending on the selected method in `settings['method']`, the function:
    - prepares the corresponding result directory,
    - configures callback time limits,
    - optionally performs multiple outer shrinkage iterations,
    - optionally generates candidate pieces or random pieces,
    - calls the appropriate solver routine,
    - evaluates the obtained solution on both training and test sets,
    - writes detailed experiment results to CSV files.

    Supported methods:
        1. `full_mip`
            Solve the full mixed-integer programming formulation.
        2. `base_fixed`
            Run the epsilon-fixed PIP method.
        3. `base_shrinkage`
            Run the epsilon-shrinkage PIP method, where epsilon is reduced geometrically across outer iterations.
        4. `simplified_arbitrary4_fixed`
            Run the simplified epsilon-fixed method with 4 arbitrarily selected generated candidate pieces.
        5. `simplified_arbitrary1_fixed`
            Run the simplified epsilon-fixed method with 1 arbitrarily selected piece.
        6. `simplified_arbitrary4_shrinkage`
            Run the simplified epsilon-shrinkage method with 4 arbitrarily selected generated candidate pieces at each outer iteration.
        7. `simplified_arbitrary1_shrinkage`
            Run the simplified epsilon-shrinkage method with 1 arbitrarily selected piece at each outer iteration.
        8. `unconstrained`
            Run the unconstrained (without precision constraint) PIP formulation.

    Args:
        model:
            A base optimization model. The function creates an internal copy via `model.copy()` before solving.

        data:
            A dictionary containing the dataset and related information. It is expected to contain at least:
            - `X_train`: training feature matrix,
            - `y_train`: training labels,
            - `X_test`: test feature matrix,
            - `y_test`: test labels,
            - `class_restricted`: restricted class set used in the precision constraints.

        start:
            A dictionary specifying the warm-start or initial solution information. Depending on the method, it may contain:
            - `objective_value`, `a`, `b`, `c`.

        settings:
            A dictionary of method-specific settings. It is expected to contain entries such as:
            - `method`: integer code specifying the solution method,
            - `epsilon`: current epsilon value,
            - `epsilon_nu`: initial epsilon value for shrinkage schemes,
            - `beta_p`: class-specific precision threshold,
            - `D`: tree depth,
            - `tune`: whether the run is for tuning,
            - `tau_0`: tuning/regularization parameter,
            - `regularizer`,
            - `selected_piece` (set internally for some methods).

        stop_rule:
            A dictionary containing stopping-rule parameters, including:
            - `base_rate`,
            - `max_outer_iter`.

        file_path:
            A dictionary of paths and run metadata. It is expected to include
            items such as:
            - `result_sub2dir`,
            - `result_csv`,
            - `details_csv`,
            - `dataset`,
            - `run`.

            The function also updates this dictionary internally with entries
            such as:
            - `result_sub3dir`,
            - `shrinkage_iter`,
            - `piece_index`.

    Returns:
        If `settings['tune']` is `False`, the function does not explicitly return a value.

        If `settings['tune']` is `True`, the function returns:
            tuple:
                - `train_result['frac']`: training performance summary,
                - `test_result['frac']`: test performance summary.

    Side Effects:
        - Creates result subdirectories if they do not already exist.
        - Modifies parts of `settings`, `start`, and `file_path` in place.
        - Updates callback time limits through `callback_data_tree.timelimit`.
        - Writes experiment summaries to CSV files.
        - Calls method-specific solver routines such as
          `full_mip_tree.full_mip_tree`,
          `PIP_iterations_tree.pip_iterations`, and
          `PIP_unconstrained_iterations_tree.pip_unconstrained_iterations`.

    Notes:
        - For shrinkage-based methods, epsilon is repeatedly updated by multiplying it by `0.1` after each outer iteration.
        - For piecewise simplified methods, candidate pieces are generated from the current iterate and the best objective value is retained.
        - The final solution is always evaluated on both the training and test sets using `utils.train_test_results(...)`.
        - In the unconstrained case, both `epsilon` and `beta_p` are set to `None`.
    """
    X_train, y_train, X_test, y_test, class_restricted = data['X_train'], data['y_train'], data['X_test'], data['y_test'], data['class_restricted']
    method, epsilon, beta_p, D, enhanced_size = settings['method'], settings['epsilon'], settings['beta_p'], settings['D'], settings['enhanced_size']
    base_rate = stop_rule['base_rate']
    result_sub2dir, result_csv, run = file_path['result_sub2dir'], file_path['result_csv'], file_path['run']
    file_path['shrinkage_iter'], file_path['piece_index'], multi_piece_list = None, None, None
    random.seed(42)
    J = list(set(y_train))
    model = model.copy()
    method_list = ['full_mip', 'base_fixed', 'base_shrinkage', 'simplified_arbitrary4_fixed', 'simplified_arbitrary1_fixed', 'simplified_arbitrary4_shrinkage', 'simplified_arbitrary1_shrinkage', 'unconstrained']
    # There are totally 8 methods: 
    # 1: 'full_mip'                             Full MIP
    # 2: 'base_fixed'                           \varepsilon-fixed
    # 3: 'base_shrinkage'                       \varepsilon-shrinkage
    # 4: 'simplified_arbitrary4_fixed'          \varepsilon-fixed-arbitrary4
    # 5: 'simplified_arbitrary1_fixed'          \varepsilon-fixed-arbitrary1
    # 6: 'simplified_arbitrary4_shrinkage'      \varepsilon-shrinkage-arbitrary4
    # 7: 'simplified_arbitrary1_shrinkage'      \varepsilon-shrinkage-arbitrary1
    # 8: 'unconstrained'                        Unconstrained PIP

    if method == 1:  # 'full_mip'
        result_sub3dir = result_sub2dir + '/full_mip'
        os.makedirs(result_sub3dir, exist_ok=True)
        file_path['result_sub3dir'] = result_sub3dir
        objective_function_term, solution, counts_result, execution_time  = Full_MIP_tree.full_mip_tree(model, data, start, settings, stop_rule, file_path)
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
            M_set_index, multi_piece = utils.generate_M(X_train, iter_start['a'], iter_start['b'], D, epsilon, base_rate, enhanced_size)
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
            writer.writerow([file_path['dataset'], D, run, method_list[method-1], settings['tau_0'], next(iter(beta_p)) if beta_p is not None else class_restricted, next(iter(beta_p.values())) if beta_p is not None else None, objective_function_term['objective_value'], objective_function_term['optimality_gap'], 
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
            writer.writerow([file_path['dataset'], D, run, 'tune', settings['tau_0'], next(iter(beta_p)) if beta_p is not None else class_restricted, next(iter(beta_p.values())) if beta_p is not None else None, objective_function_term['objective_value'], objective_function_term['optimality_gap'], 
                            record_time, actual_time, multi_piece_list, next(iter(objective_function_term['gamma'].values())) if objective_function_term['gamma'] is not None else None, 
                            objective_function_term['z_frac'], objective_function_term['z_counts'], train_result['frac'],train_constraint_gap, train_result['counts'], 
                            test_result['frac'], test_constraint_gap, test_result['counts'],
                            test_train_gap,
                            counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'], 
                            counts_result['violations_assumption_1_rate'], counts_result['violations_assumption_2_rate'],counts_result['violations_feasibilitytol_rate'],
                            counts_result['z_integrality_vio'],
                            train_result['frac']['acc'], test_result['frac']['acc'], train_result['frac'][f'prec{class_restricted[0]}'], test_result['frac'][f'prec{class_restricted[0]}']])
        return train_result['frac'], test_result['frac']
