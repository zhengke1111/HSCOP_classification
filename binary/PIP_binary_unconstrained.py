
import MIP_binary_callback
import callback_data_binary
from gurobipy import GRB
import gurobipy as gp
import numpy as np
import pandas as pd
import os


def pip_binary_unconstrained(model, data, start, settings, file_path):
    """
    A single PIP iteration in PIP method to solve the binary classification problem without precision constraint

    Args:
        model (dict): Gurobi parameter settings, including {Name, 'MIPFocus', 'IntegralityFocus', 'Threads', 'FeasibilityTol'}
        data (dict): Data splits, {X_train, y_train, X_test, y_test} split by some random seeds we set
        start (dict): Initial solution from the last iteration
        settings (dict): Settings of PIP
        file_path (dict): File path to store the output

    Returns:
        Tuple(dict, dict, dict): objective_function_term, solution, counts_result 
    """
    # Below are similar to the function pip_binary in PIP_binary.py
    model = model.copy()
    X_train, y_train = data['X_train'], data['y_train']
    w_start, b_start, z_plus_0_start = start['w'], start['b'], start['z_plus_0']
    method, delta_1, delta_2, feasibility_tol = settings['method'], settings['delta_1'], settings['delta_2'], settings['feasibilitytol']
    result_sub3dir, pip_iter = file_path['result_sub3dir'], file_path['pip_iter']
    

    dim = X_train.shape[1]
    N = X_train.shape[0]

    positive_index = np.where(y_train == 1)[0].tolist()
    negative_index = np.where(y_train == -1)[0].tolist()

    z_plus_0 = {}

    solution_z_plus_0 = {}

    J_0_plus_0, J_1_plus_0, J_2_plus_0 = [], [], []

    tau = 10
    b_ub = tau*max(abs(X_train[s][p]) for p in range(dim) for s in range(N)) + 1
    big_M = 2*(tau*max(abs(X_train[s][p]) for p in range(dim) for s in range(N)) + 1)

    w = model.addVars(dim, lb=-tau, ub=tau, vtype=GRB.CONTINUOUS, name="w")
    b = model.addVar(lb=-b_ub, ub=b_ub, vtype=GRB.CONTINUOUS, name="b")
    abs_w = model.addVars(dim, lb=0, ub=tau, vtype=GRB.CONTINUOUS, name="abs_w")
    abs_b = model.addVar(lb=0, ub=b_ub, vtype=GRB.CONTINUOUS, name="abs_b")

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
        if np.dot(w_start, X_train[s]) + b_start -1 >= delta_1[0]:
            J_1_plus_0.append(s)
            model.remove(z_plus_0[s])
            model.addConstr(gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) + b - 1 -feasibility_tol>= 0)
        elif np.dot(w_start, X_train[s]) + b_start-1 <= -delta_2[0]: 
            J_2_plus_0.append(s)
            model.remove(z_plus_0[s])
        else:
            J_0_plus_0.append(s)
            z_plus_0[s].setAttr(gp.GRB.Attr.Start, z_plus_0_start[s])
            model.addConstr(gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) + b -1 -feasibility_tol>= -big_M * (1 - z_plus_0[s]))

    for s in negative_index:
        # z_{0s}^+
        z_plus_0[s] = model.addVar(vtype=GRB.BINARY, name="z_plus_0_" + str(s))
        if -np.dot(w_start, X_train[s]) - b_start -1 >= delta_1[0]:
            J_1_plus_0.append(s)
            model.remove(z_plus_0[s])
            model.addConstr(-gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) - b - 1 -feasibility_tol>= 0)
        elif -np.dot(w_start, X_train[s]) - b_start -1 <= -delta_2[0]: 
            J_2_plus_0.append(s)
            model.remove(z_plus_0[s])
        else:
            J_0_plus_0.append(s)
            z_plus_0[s].setAttr(gp.GRB.Attr.Start, z_plus_0_start[s])
            model.addConstr(-gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) - b - 1 -feasibility_tol>= -big_M * (1 - z_plus_0[s]))

        
    model.addConstr(gp.quicksum(abs_w[p] for p in range(dim)) <= tau)

    # Set Objective function
    J_0_plus_0_positive = list(set(J_0_plus_0) & set(positive_index))
    J_0_plus_0_negative = list(set(J_0_plus_0) & set(negative_index))
    J_1_plus_0_positive = list(set(J_1_plus_0) & set(positive_index))
    J_1_plus_0_negative = list(set(J_1_plus_0) & set(negative_index))
    
    obj = (1/N)*(gp.quicksum(z_plus_0[s] for s in J_0_plus_0_positive)+ gp.quicksum(z_plus_0[s] for s in J_0_plus_0_negative) + sum(1 for _ in J_1_plus_0_positive)+sum(1 for _ in J_1_plus_0_negative)) 
    model.addConstr(obj <= 1, "manual_upper_bound")
    model.setObjective(obj, GRB.MAXIMIZE)
    model.update()
    num_integer_vars = sum(1 for v in model.getVars() if v.vType == gp.GRB.BINARY)

    model.setParam("Timelimit", 300)
    if method == 6:
        postfix = f'pip_iter_{pip_iter}'
    
    os.makedirs(result_sub3dir + '/Logfile', exist_ok=True)
    model.setParam('LogFile', os.path.join(result_sub3dir + '/Logfile', f'log_{postfix}.txt'))

    callback_data_binary.log_data=[]
    callback_data_binary.time_for_finding_feasible_solution = 0
    model.optimize(MIP_binary_callback.mip_binary_callback)
    log_data_list = callback_data_binary.log_data
    log_df = pd.DataFrame(log_data_list)
    time_for_finding_feasible_solution = callback_data_binary.time_for_finding_feasible_solution
    try:
        last_row = log_df.iloc[-1]
    except IndexError:
        last_row = {'BestBd': 0}


    objective_value = model.objVal
    bestbd = model.ObjBound
    try:
        optimality_gap = model.MIPGap
    except AttributeError:
        optimality_gap = -1
    runtime = model.Runtime
    solution_w = [w[p].X for p in range(dim)]
    solution_b = b.X
    
    z_integrality_vio = 0
    
    for s in J_0_plus_0:
        solution_z_plus_0[s] = z_plus_0[s].X
        if z_plus_0[s].X > 0 and z_plus_0[s].X < 1:
            z_integrality_vio += 1
    for s in J_1_plus_0:
        solution_z_plus_0[s] = 1
    for s in J_2_plus_0:
        solution_z_plus_0[s] = 0


    os.makedirs(result_sub3dir + '/Solution', exist_ok=True)
    with open(result_sub3dir + '/Solution/' + f'solution_{postfix}.txt', 'a') as f:
        # The following "violations_assumption_1, violations_assumption_2, violations_feasibility_tol", See PIP_binary.py
        violations_assumption_1 = 0
        violations_assumption_2 = 0
        violations_feasibilitytol = {'phi^+_0':0}

        for s in positive_index:
            if (np.dot(solution_w, X_train[s]) + solution_b - 1 < 0) & (np.dot(solution_w, X_train[s]) + solution_b - 1>= -feasibility_tol):
                violations_feasibilitytol['phi^+_0'] += 1
    
        for s in negative_index:
            if (-np.dot(solution_w, X_train[s]) - solution_b - 1 < 0) & (-np.dot(solution_w, X_train[s]) - solution_b - 1>= -feasibility_tol):
                violations_feasibilitytol['phi^+_0'] += 1
            

    objective_function_term = {
        'objective_value': objective_value,'bestbd': bestbd,'optimality_gap': optimality_gap,
        'best_bd':last_row['BestBd'], 'runtime': runtime}
    solution = {'objective_value': objective_value, 'w': solution_w, 'b': solution_b, 'z_plus_0': solution_z_plus_0}
    violations_feasibilitytol = violations_feasibilitytol['phi^+_0']
    
    vio_feasibilitytol_indicator = False
    log_path = result_sub3dir + '/Logfile/'+f'log_{postfix}.txt'
    with open(log_path, 'r') as log_file:
        log_content = log_file.read()
    if "max constraint violation" in log_content:
        vio_feasibilitytol_indicator = True

    counts_result = {
        'num_integer_vars': num_integer_vars,
        'violations_assumption_1': violations_assumption_1, 'violations_assumption_2': violations_assumption_2, 
        'violations_feasibilitytol': violations_feasibilitytol,
        'number_of_integers': num_integer_vars,
        'z_integrality_vio': z_integrality_vio, 
        'vio_feasibilitytol_indicator': vio_feasibilitytol_indicator
    }

    return objective_function_term, solution, counts_result
