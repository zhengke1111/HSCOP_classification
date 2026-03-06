from binary import PIP_binary_unconstrained
from binary import utils
import csv


def pip_binary_unconstrained_iterations(model, data, start, settings, stop_rule, file_path):
    """
    PIP iterations in PIP method to solve the binary classification problem without precision constraint

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
    # Below are similar to the function pip_binary_iterations in PIP_binary_iterations.py
    X_train, y_train = data['X_train'], data['y_train']
    objective_value= start['objective_value']
    method = settings['method']
    feasible_rate, pip_max_rate, unchanged_iters, max_iteration = stop_rule['feasible_rate'], stop_rule['pip_max_rate'], stop_rule['unchanged_iters'], stop_rule['max_iteration']
    method_list = ['full_mip', 'full_mip_t300', 'full_mip_t600','fixed', 'shrinkage','unconstrained']

    shrinkage_list, execution_time = [], []
    iter_unchanged = 0
    integer_rate = feasible_rate

    for iteration in range(max_iteration):
        file_path['pip_iter'] = iteration

        objective_value_old = objective_value
        
        start['z_plus_0'] = utils.calculate_z(X_train, y_train, start['w'], start['b'])
        settings['delta_1'], settings['delta_2'] = utils.calculate_delta_unconstrained(X_train=X_train, y_train=y_train, weight=start['w'], bias=start['b'], base_rate=integer_rate)
        
        objective_function_term, solution, counts_result = PIP_binary_unconstrained.pip_binary_unconstrained(model, data, start, settings, file_path)

        with open(file_path['details_csv'], mode='a', newline='') as details:
            writer = csv.writer(details)
            writer.writerow([file_path['run'], method_list[method-1], None, None, iteration, counts_result['num_integer_vars'],integer_rate, objective_function_term['objective_value'], objective_function_term['bestbd'], objective_function_term['optimality_gap'], 
                            solution['w'], solution['b'], counts_result['violations_assumption_1'], counts_result['violations_assumption_2'],counts_result['violations_feasibilitytol'], counts_result['z_integrality_vio']])
            
        start = solution
        objective_value = solution['objective_value']
        vio_feasibilitytol_indicator = counts_result['vio_feasibilitytol_indicator']

        if objective_value - objective_value_old <= 1e-5:
            iter_unchanged += 1
            shrinkage_list.append(0)
            integer_rate = min(pip_max_rate, integer_rate + 10)
            
        else:
            iter_unchanged = 0
            shrinkage_list.append(1)
            integer_rate = max(feasible_rate, integer_rate - 10)
            
        execution_time.append(objective_function_term['runtime'])
        
        if iter_unchanged >= unchanged_iters:
            max_iteration = iteration + 1
            break
    
    return objective_function_term, solution, counts_result, execution_time