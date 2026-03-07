import utils
import MIP_binary_callback
import callback_data_binary
from gurobipy import GRB
import gurobipy as gp
import numpy as np
import pandas as pd
import os


def full_mip_binary(model, data, start, settings, stop_rule, file_path):
    """
    A Full MILP to solve the binary classification problem with precision constraint

    Args:
        model (dict): Gurobi parameter settings, including {Name, 'MIPFocus', 'IntegralityFocus', 'Threads', 'FeasibilityTol'}
        data (dict): Data splits, {X_train, y_train, X_test, y_test} split by some random seeds we set
        start (dict): Initial solution from the last iteration
        settings (dict): Settings of PIP
        stop_rule: Timelimit
        file_path (dict): File path to store the output

    Returns:
        Tuple(dict, dict, dict): objective_function_term, solution, counts_result 
    """
    # Below are similar to the function pip_binary in PIP_binary.py
    model = model.copy()
    X_train, y_train = data['X_train'], data['y_train']
    w_start, b_start = start['w'], start['b']
    epsilon, rho, beta_p, feasibility_tol, callback_type = settings['epsilon'], settings['rho'], settings['beta_p'], settings['feasibilitytol'], settings['callback_type']
    gamma_0, z_plus_0_start, z_plus_start, z_minus_start = utils.calculate_gamma(X_train, y_train, w_start, b_start, beta_p, epsilon)
    timelimit = stop_rule['timelimit']
    result_sub3dir = file_path['result_sub3dir']

    dim = X_train.shape[1]
    N = X_train.shape[0]

    positive_index = np.where(y_train == 1)[0].tolist()
    negative_index = np.where(y_train == -1)[0].tolist()

    z_plus_0 = {}
    z_plus = {}
    z_minus = {}

    solution_z_plus_0 = {}
    solution_z_plus = {}
    solution_z_minus = {}

    tau = 100
    b_ub = tau*max(abs(X_train[s][p]) for p in range(dim) for s in range(N)) + 1
    big_M = 2*(tau*max(abs(X_train[s][p]) for p in range(dim) for s in range(N)) + 1)

    w = model.addVars(dim, lb=-tau, ub=tau, vtype=GRB.CONTINUOUS, name="w")
    b = model.addVar(lb=-b_ub, ub=b_ub, vtype=GRB.CONTINUOUS, name="b")
    abs_w = model.addVars(dim, lb=0, ub=tau, vtype=GRB.CONTINUOUS, name="abs_w")
    abs_b = model.addVar(lb=0, ub=b_ub, vtype=GRB.CONTINUOUS, name="abs_b")
    gamma = model.addVar(lb=0, ub=gamma_0, vtype=GRB.CONTINUOUS, name="gamma")

    for p in range(dim):
        w[p].setAttr(gp.GRB.Attr.Start, w_start[p])
        model.addConstr(w[p] <= abs_w[p])
        model.addConstr(-w[p] <= abs_w[p])
    b.setAttr(gp.GRB.Attr.Start, b_start)
    model.addConstr(b <= abs_b)
    model.addConstr(-b <= abs_b)

    for s in positive_index:
        # z_{0s}^+
        z_plus_0[s] = model.addVar(vtype=GRB.BINARY, name="z_plus_0_" + str(s))
        z_plus_0[s].setAttr(gp.GRB.Attr.Start, z_plus_0_start[s])
        model.addConstr(gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) + b -1 -feasibility_tol>= -big_M * (1 - z_plus_0[s]))
        # z_{1s}^+
        z_plus[s] = model.addVar(vtype=GRB.BINARY, name="z_plus_" + str(s))
        z_plus[s].setAttr(gp.GRB.Attr.Start, z_plus_start[s])
        model.addConstr(gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) + b -feasibility_tol>= -big_M * (1 - z_plus[s]))
    
    for s in negative_index:
        # z_{0s}^+
        z_plus_0[s] = model.addVar(vtype=GRB.BINARY, name="z_plus_0_" + str(s))
        z_plus_0[s].setAttr(gp.GRB.Attr.Start, z_plus_0_start[s])
        model.addConstr(-gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) - b - 1 -feasibility_tol>= -big_M * (1 - z_plus_0[s]))
        # z_{1s}^-
        z_minus[s] = model.addVar(vtype=GRB.BINARY, name="z_minus_" + str(s))
        z_minus[s].setAttr(gp.GRB.Attr.Start, z_minus_start[s])
        model.addConstr(-gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) - b -feasibility_tol>= -big_M * (
                            1 - z_minus[s]) + epsilon)
        
    model.addConstr((1-beta_p)*gp.quicksum(z_plus[s] for s in positive_index) - beta_p*gp.quicksum((1 - z_minus[s]) for s in negative_index) + gamma >= 0)
    model.addConstr(gp.quicksum(abs_w[p] for p in range(dim)) <= tau) 
    
    obj = (1/N)*(gp.quicksum(z_plus_0[s] for s in positive_index) + gp.quicksum(z_plus_0[s] for s in negative_index)) - rho * gamma 
    model.addConstr(obj <= 1, "manual_upper_bound")
    model.setObjective(obj, GRB.MAXIMIZE)
    model.update()
    num_integer_vars = sum(1 for v in model.getVars() if v.vType == gp.GRB.BINARY)

    model.setParam("Timelimit", timelimit)
    
    os.makedirs(os.path.join(result_sub3dir, 'LogFile'), exist_ok=True)
    model.setParam('LogFile', os.path.join(result_sub3dir, 'LogFile', f'log_file_full_mip_{callback_type}.txt'))

    callback_data_binary.time_for_finding_feasible_solution = 0
    callback_data_binary.log_data=[]
    
    if callback_type == 0:                                                  # Default callback mechanism, for Full MIP (Final)
        model.optimize(MIP_binary_callback.full_mip_callback)
    elif callback_type == 1:                                                # Callback mechanism for Early MIP (T300)
        model.optimize(MIP_binary_callback.full_mip_callback_t300)
    else:                                                                   # Callback mechanism for Early MIP (T600)
        model.optimize(MIP_binary_callback.full_mip_callback_t600)
    log_data_list = callback_data_binary.log_data
    log_df = pd.DataFrame(log_data_list)
    last_row = log_df.iloc[-1]
    
    if len(log_df)>0:
        last_row = log_df.iloc[-1]
        if model.objVal<=0:
            final_improvement_time = model.Runtime                          # If infeasible after termination, record the final improvement time as the actual run time
            final_improvement_gap = model.MIPGap
        else:
            final_improvement_time = last_row['final_improvement_time']     # If feasible after termination, record the final improvement time as the last row of final improvement time
            final_improvement_gap = last_row['final_improvement_gap']
        runtime = model.Runtime
    else:
        final_improvement_time = model.Runtime
        final_improvement_gap = model.MIPGap

    log_df_file_path = os.path.join(result_sub3dir + '/LogFile', f'log_file_full_mip{callback_type}.csv')
    log_df.to_csv(log_df_file_path, index=False)

    objective_value = model.objVal
    optimality_gap = model.MIPGap
    bestbd = model.ObjBound

    solution_w = [w[p].X for p in range(dim)]
    solution_b = b.X

    # The following "violations_assumption_1, violations_assumption_2, violations_feasibility_tol", See PIP_binary.py
    violations_assumption_1 = 0
    violations_assumption_2 = 0
    violations_feasibilitytol = {'phi^+_0':0,'phi^+':0,'phi^-':0}

    z_integrality_vio = 0
    for s in positive_index:
        solution_z_plus_0[s] = z_plus_0[s].X
        if z_plus_0[s].X > 0 and z_plus_0[s].X < 1:
                z_integrality_vio += 1
        solution_z_plus[s] = z_plus[s].X
        if z_plus[s].X > 0 and z_plus[s].X < 1:
                z_integrality_vio += 1
        if (np.dot(solution_w, X_train[s]) + solution_b -1 < 0) & (np.dot(solution_w, X_train[s]) + solution_b -1>= -feasibility_tol):
                violations_feasibilitytol['phi^+_0'] += 1
        if (np.dot(solution_w, X_train[s]) + solution_b < 0) & (np.dot(solution_w, X_train[s]) + solution_b >= -feasibility_tol):
            violations_feasibilitytol['phi^+'] += 1
    for s in negative_index:
        solution_z_plus_0[s] = z_plus_0[s].X
        if z_plus_0[s].X > 0 and z_plus_0[s].X < 1:
                z_integrality_vio += 1
        solution_z_minus[s] = z_minus[s].X
        if z_minus[s].X > 0 and z_minus[s].X < 1:
                z_integrality_vio += 1
        if (-np.dot(solution_w, X_train[s]) - solution_b -1 < 0) & (-np.dot(solution_w, X_train[s]) - solution_b -1>= -feasibility_tol):
                violations_feasibilitytol['phi^+_0'] += 1
        if (-(np.dot(solution_w, X_train[s]) + solution_b) - epsilon < 0) & (-(np.dot(solution_w, X_train[s]) + solution_b)-epsilon >= -feasibility_tol):
            violations_feasibilitytol['phi^-'] += 1
        if np.dot(solution_w, X_train[s]) + solution_b == 0:
            violations_assumption_1 += 1
        if (np.dot(solution_w, X_train[s]) + solution_b < 0) & (np.dot(solution_w, X_train[s]) + solution_b >= -epsilon):
            violations_assumption_2 += 1

    
    os.makedirs(os.path.join(result_sub3dir, 'Solution'), exist_ok=True)
    with (open(result_sub3dir + '/Solution/' + f'solution_full_mip{callback_type}.txt',
            'a') as f):
        for s in positive_index:
            print("y_" + str(s) +'=', y_train[s], file=f)
            print("z^+_0_" + str(s) + '=', solution_z_plus_0[s], file=f)
            print('\phi^+_0_' + str(s) + '=', np.dot(solution_w, X_train[s]) + solution_b - 1, file=f)
            print("z^+_" + str(s) + '=', solution_z_plus[s], file=f)
            print('\phi^+_' + str(s) + '=', np.dot(solution_w, X_train[s]) + solution_b, file=f)

        for s in negative_index:
            print("y_" + str(s) +'=', y_train[s], file=f)
            print("z^+_0_" + str(s) + '=', solution_z_plus_0[s], file=f)
            print('\phi^+_0_' + str(s) + '=', -np.dot(solution_w, X_train[s]) - solution_b - 1, file=f)
            print("z^-_" + str(s) + '=', solution_z_minus[s], file=f)
            print('-\phi^-_' + str(s) + '-epsilon =', -(np.dot(solution_w, X_train[s]) + solution_b) - epsilon, file=f)

    runtime = model.Runtime

    objective_function_term = {'objective_value': objective_value, 'bestbd':bestbd, 'optimality_gap': optimality_gap, 'gamma_in_obj': gamma.X, 'time': final_improvement_time if callback_type == 0 else runtime, 'gap': final_improvement_gap}
    solution = {'objective_value': objective_value, 'w': solution_w, 'b': solution_b, 'z_plus_0': solution_z_plus_0, 'z_plus': solution_z_plus, 'z_minus': solution_z_minus, 'gamma_0': gamma.X}
    violations_feasibilitytol = violations_feasibilitytol['phi^+_0'] + violations_feasibilitytol['phi^+'] + violations_feasibilitytol['phi^-']
    counts_result = {
        'num_integer_vars': num_integer_vars,
        'violations_assumption_1': violations_assumption_1, 
        'violations_assumption_2': violations_assumption_2, 
        'violations_feasibilitytol': violations_feasibilitytol,
        'number_of_integers': num_integer_vars,
        'z_integrality_vio': z_integrality_vio
    }

    return objective_function_term, solution, counts_result