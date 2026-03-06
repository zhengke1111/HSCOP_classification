from binary import PIP_binary
from binary import utils
import csv


def pip_binary_iterations(model, data, start, settings, stop_rule, file_path):
    """
    PIP iterations in PIP method to solve the binary classification problem with precision constraint

    Args:
        model (dict): Gurobi parameter settings, including {Name, 'MIPFocus', 'IntegralityFocus', 'Threads', 'FeasibilityTol'}
        data (dict): Data splits, {X_train, y_train, X_test, y_test} split by some random seeds we set
        start (dict): Initial solution from the last iteration
        settings (dict): Settings of PIP
        stop_rule (dict): Timelimit, Base_rate, Feasible_rate, Pip_max_rate, Unchanged_iters, Max_iteration, Max_outer_iter
        file_path (dict): File path to store the output

    Returns:
        Tuple(dict, dict, dict, list): objective_function_term, solution, counts_result, execution_time
    """

    X_train, y_train = data['X_train'], data['y_train']
    objective_value = start['objective_value']
    method, epsilon, beta_p = settings['method'], settings['epsilon'], settings['beta_p']
    # r'_{init}, r_{init}, r_{max}, \tilde{\mu}_{max}
    base_rate, feasible_rate, pip_max_rate, unchanged_iters, max_iteration = stop_rule['base_rate'],  stop_rule['feasible_rate'], stop_rule['pip_max_rate'], stop_rule['unchanged_iters'], stop_rule['max_iteration']
    method_list = ['full_mip', 'full_mip_t300', 'full_mip_t600', 'fixed', 'shrinkage']
    if method == 4:
        file_path['shrinkage_iter'] = None

    shrinkage_list, execution_time = [], []
    iter_unchanged, feasible_iter, infeasible_iter = 0, 0, 0
    
    for iteration in range(max_iteration):
        file_path['pip_iter'] = iteration
        if objective_value < 0:                 # Still infeasible due to the large penalty coefficient \rho
            infeasible_iter += 1
            if infeasible_iter == 1:            # The first iteration is infeasible
                integer_rate = base_rate        # r'_{init}
        if objective_value >= 0:                # Feasible
            feasible_iter += 1
            if feasible_iter == 1:              # The first iteration to be feasible
                integer_rate = feasible_rate    # r_{init}

        objective_value_old = objective_value
        
        # Initial value of gamma and z
        start['gamma_0'], start['z_plus_0'], start['z_plus'], start['z_minus'] = utils.calculate_gamma(X_train, y_train, start['w'], start['b'], beta_p, epsilon)
        # In-between set threshold of \phi
        settings['delta_1'], settings['delta_2'] = utils.calculate_delta(X_train=X_train, y_train=y_train, weight=start['w'], bias=start['b'], epsilon=epsilon, base_rate=integer_rate)
        
        objective_function_term, solution, counts_result = PIP_binary.pip_binary(model, data, start, settings, file_path)

        with open(file_path['details_csv'], mode='a', newline='') as details:
            writer = csv.writer(details)
            writer.writerow([file_path['run'], method_list[method-1], beta_p, file_path['shrinkage_iter'], iteration, counts_result['num_integer_vars'],integer_rate, objective_function_term['objective_value'], objective_function_term['bestbd'], objective_function_term['optimality_gap'], 
                            solution['w'], solution['b'], counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'], counts_result['z_integrality_vio']])
            
        start = solution
        objective_value = solution['objective_value']

        if objective_value - objective_value_old <= 1e-5:               # The objective value remains unchanged
            if objective_value > 0:                                     # Feasible
                iter_unchanged += 1                                     # The counter of consecutive unchanged iterations
            shrinkage_list.append(0)                                    # Record as Enlargement (shrinkage = 0)
            integer_rate = min(pip_max_rate, integer_rate + 10)         # Enlarge the In-between index set
            
        else:
            iter_unchanged = 0                                          # Reset the counter to zero
            shrinkage_list.append(1)                                    # Record as Shrinkage (shrinkage = 1)
            if objective_value > 0:                                     # Feasible
                integer_rate = max(feasible_rate, integer_rate - 10)    # Shrink the In-between index set
            else:                                                       # Infeasible
                integer_rate = integer_rate                             # Do not change the integer ratio
        
        execution_time.append(objective_function_term['runtime'])
        
        if iter_unchanged >= unchanged_iters:                           # Reach \tilde{\mu}_{max}
            max_iteration = iteration + 1
            break
    
    return objective_function_term, solution, counts_result, execution_time