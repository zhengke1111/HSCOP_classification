import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import pandas as pd
import csv
import os
from collections import Counter
from decisiontree import callback_data_tree
from decisiontree import MIP_tree_callback
from decisiontree import utils


def pip_unconstrained_single_iter_tree(model, data, start, settings, file_path):
    """
    A single PIP iteration in PIP method to solve the decision tree classification problem without precision constraint

    Args:
        model (dict): Gurobi parameter settings, including {Name, 'MIPFocus', 'IntegralityFocus', 'Threads', 'NumericFocus', 'FeasibilityTol'}
        data (dict): Data splits, {X_train, y_train, X_test, y_test} split by some random seeds we set
        start (dict): Initial solution from the last iteration
        settings (dict): Settings of PIP
        file_path (dict): File path to store the output

    Returns:
        Tuple(dict, dict, dict): objective_function_term, solution, counts_result 
    """
    #   Below are similar to the function pip_single_iter_tree in PIP_single_iter_tree.py
    X_train, y_train, class_restricted = data['X_train'], data['y_train'], data['class_restricted']
    a_start, b_start, c_start, z_plus_0_start = start['a'], start['b'], start['c'], start['z_plus_0']
    L_start = None
    if all(key in start for key in ['L']):
        L_start = start['L']
    
    method, delta_1, delta_2, D, feasibility_tol = settings['method'], settings['delta_1'], settings['delta_2'], settings['D'], settings['feasibilitytol']
    regularizer = settings['regularizer']
    result_sub3dir, pip_iter = file_path['result_sub3dir'], file_path['pip_iter']
    if method in [3,6,7]:
        shrinkage_iter = file_path['shrinkage_iter']
    if method in [4,6]:
        piece_index = file_path['piece_index']
    random.seed(42)

    p = X_train.shape[1]                        # dimension
    total_class_num = len(Counter(y_train))
    J = range(1, total_class_num+1)             # classes
    I = class_restricted                        # class restricted
    class_index = {cls: np.where(y_train == cls)[0] for cls in J}  # use to select samples of certain class

    model = model.copy()
    N = X_train.shape[0]
    a = {}
    a_abs ={}
    b = {}
    L = {}

    I_plus_0, I_plus_1, I_plus_2 = {}, {}, {}  # Index sets for z_plus_0  

    tau_1 = 100
    b_ub = tau_1*abs(max(abs(X_train[s][i]) for i in range(p) for s in range(N))) + 1

    M_z_plus_0 = 2*(tau_1*abs(max(abs(X_train[s][i]) for i in range(p) for s in range(N))) + 1)
    
    for k in range(2**D-1):
        a[k] = model.addVars(p, lb=-tau_1, ub=tau_1, vtype=GRB.CONTINUOUS, name="a_" + str(k))
        a_abs[k] = model.addVars(p, lb=0, ub=tau_1, vtype=GRB.CONTINUOUS, name="a_abs_" + str(k))
        model.addConstrs((a[k][i]<=a_abs[k][i]) for i in range(p))
        model.addConstrs((-a[k][i]<=a_abs[k][i]) for i in range(p))
        b[k] = model.addVar(lb=-b_ub, ub = b_ub, vtype=GRB.CONTINUOUS, name="b_" + str(k))

        if a_start is not None:
            for i in range(p):
                a[k][i].setAttr(gp.GRB.Attr.Start, a_start[k][i])
        if b_start is not None:
            b[k].setAttr(gp.GRB.Attr.Start, b_start[k])
        model.addConstr(gp.quicksum(a_abs[k][i] for i in range(p))<=tau_1)
    
    if regularizer == 'hard_l0':
        tau_0 = settings['tau_0']
        u={}
        for k in range(2**D-1):
            u[k] = model.addVars(p, vtype=GRB.BINARY, name='u_'+str(k))
            model.addConstrs((a[k][i]>=-tau_1*(1-u[k][i])) for i in range(p))
            model.addConstrs((-a[k][i]>=-tau_1*(1-u[k][i])) for i in range(p))
            model.addConstr(gp.quicksum((1-u[k][i]) for i in range(p))<=tau_0)

    key_z_plus_0 = [(s, t) for s in range(N) for t in range(2**D)]
    z_plus_0 = model.addVars(key_z_plus_0, vtype=GRB.BINARY, name='z_plus_0')
    

    keys_c = [(j, t) for j in J for t in range(2**D)]
    c = model.addVars(keys_c, vtype=GRB.BINARY, name="c")
    if c_start is not None:
        for (j,t) in keys_c:
            c[j,t].setAttr(gp.GRB.Attr.Start, c_start[j,t])

    A_L, A_R = utils.ancestors(D)

    for t in range(2**D):
        I_plus_0[t], I_plus_1[t], I_plus_2[t] = [], [], []  # Index sets for z_plus_0  
        
        L[t] = model.addVar(lb=0, ub=N, vtype=GRB.CONTINUOUS, name="L_" + str(t))
        if L_start is not None:
            if t in L_start:
                L[t].setAttr(gp.GRB.Attr.Start, L_start[t])

        for s in range(N):
            phi_0_plus = min([sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k] - 1 for k in A_R[t]]+ [-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - 1 for k in A_L[t]])
            if phi_0_plus > delta_1[0]:
                model.remove(z_plus_0[s, t])
                I_plus_1[t].append(s)
                model.addConstrs((gp.quicksum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] - 1 -feasibility_tol>= 0) for k in A_R[t])
                model.addConstrs((gp.quicksum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k] - 1 -feasibility_tol>= 0) for k in A_L[t])
            elif phi_0_plus >= -delta_2[0]:
                if z_plus_0_start[s, t] == 0 or z_plus_0_start[s, t] == 1:
                    z_plus_0[s, t].setAttr(gp.GRB.Attr.Start, z_plus_0_start[s, t])
                I_plus_0[t].append(s)
                model.addConstrs((gp.quicksum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] - 1 -feasibility_tol>= -M_z_plus_0 * (1-z_plus_0[s,t])) for k in A_R[t])
                model.addConstrs((gp.quicksum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k] - 1 -feasibility_tol>= -M_z_plus_0 * (1-z_plus_0[s,t])) for k in A_L[t])
            else:
                model.remove(z_plus_0[s, t])
                I_plus_2[t].append(s)
            
            
        model.addConstrs((L[t] <= gp.quicksum(z_plus_0[s, t] for s in list(set(class_index[j]) & set(I_plus_0[t]))) + sum(1 for _ in list(set(class_index[j]) & set(I_plus_1[t]))) + N * (1-c[j,t])) for j in J)
        model.addConstr(1 == gp.quicksum(c[j, t] for j in J)) 

    obj = 1/N*gp.quicksum(L[t] for t in range(2**D)) 
    model.addConstr(obj <= 1, "manual_upper_bound")
    model.setObjective(obj,GRB.MAXIMIZE)
    model.update()
    num_integer_vars = sum(1 for v in model.getVars() if v.vType == gp.GRB.BINARY)

    if method in [2,3]:
        model.setParam("Timelimit", 600)
    else:
        model.setParam("Timelimit", 300)

    os.makedirs(result_sub3dir + '/Logfile', exist_ok=True)
    if method == 2 or method == 8:
        postfix = f'pip_iter_{pip_iter}'
    if method == 3 or method == 7:
        postfix = f'shrinkage_iter_{shrinkage_iter}_pip_iter_{pip_iter}'
    if method == 4:
        postfix = f'pip_iter_{pip_iter}_piece_{piece_index}'
    if method == 5:
        postfix = f'pip_iter_{pip_iter}'
    if method == 6:
        postfix = f'shrinkage_iter_{shrinkage_iter}_piece_{piece_index}_pip_iter_{pip_iter}'
    model.setParam('LogFile', os.path.join(result_sub3dir + '/Logfile', f'log_{postfix}.txt'))

    callback_data_tree.log_data=[]
    model._vars = model.getVars()
    model.optimize(MIP_tree_callback.mip_tree_callback)
    objective_value = model.objVal
    runtime = model.Runtime

    try:
        optimality_gap = model.MIPGap
    except AttributeError:
        optimality_gap = -1

    output_abc = []  
    for k in range(2**D-1):
        a_vector = [a[k][i].X for i in range(p)]  
        a_vector_str = ", ".join(f"{value:.4f}" for value in a_vector)  
        output_abc.append(f"a_{k}: [{a_vector_str}]")  
        output_abc.append(f"b_{k}: {b[k].X:.4f}")  
    for j in J:
        c_vector = [c[j, t].X for t in range(2**D)]  
        c_vector_str = ", ".join(f"{value}" for value in c_vector)  
        output_abc.append(f"c_{j}: [{c_vector_str}]")  
    os.makedirs(result_sub3dir + '/Solution', exist_ok=True)
    with open(result_sub3dir + '/Solution/' + f'solution_{postfix}.txt', 'a') as f:
        f.write("\n".join(output_abc))  

    solution_a = {}
    solution_b = {}
    solution_c = {}
    solution_L = {}
    for k in a.keys():
        solution_a[k] = {i: a[k][i].X for i in range(p)}
    for k in b.keys():
        solution_b[k] = b[k].X
    for key in keys_c:
        solution_c[key] = c[key].X
    for t in range(2**D):
        solution_L[t] = L[t].X

    z_integrality_vio = 0
    solution_z_plus_0 = {}

    for (s, t) in key_z_plus_0:
        if s in I_plus_0[t]:
            solution_z_plus_0[s, t] = z_plus_0[s, t].X
            if 1> z_plus_0[s, t].X >0:
                z_integrality_vio += 1
        elif s in I_plus_1[t]:
            solution_z_plus_0[s, t] = 1
        else:
            solution_z_plus_0[s, t] = 0
    
    z_frac = {}
    z_counts = {}
    z_frac['acc_margin'] = round(1/N*sum(solution_c[y_train[s],t]*solution_z_plus_0[s, t] for s in range(N) for t in range(2**D)),3)
    z_counts['acc_margin'] = {'nm':sum(solution_c[y_train[s],t]*solution_z_plus_0[s, t] for s in range(N) for t in range(2**D)),'dm':N}
    
    objective_function_term = {'objective_value': objective_value, 'optimality_gap': optimality_gap, 
                               'gamma': None, 'runtime': runtime, 'z_frac': z_frac, 'z_counts':z_counts}
    
    solution = {'objective_value': objective_value, 
                'a': solution_a,
                'b': solution_b,
                'c': solution_c,
                'L': solution_L,
                'z_plus_0': solution_z_plus_0}
    
    violations_feasibilitytol = 0

    for s in range(N):
        for t in range(2**D):
            lhs_feas = min([sum(solution_a[k][i]*X_train[s][i] for i in range(p)) - solution_b[k] - 1 for k in A_R[t]] + [-sum(solution_a[k][i]*X_train[s][i] for i in range(p)) + solution_b[k] - 1 for k in A_L[t]])
            if 0 > lhs_feas >= -feasibility_tol:
                violations_feasibilitytol += 1

    vio_feasibilitytol_indicator = False
    log_path = result_sub3dir + '/Logfile/'+f'log_{postfix}.txt'
    with open(log_path, 'r') as log_file:
        log_content = log_file.read()
    if "max constraint violation" in log_content:
        vio_feasibilitytol_indicator = True
    counts_result = {'num_integer_vars': num_integer_vars, 'violations_assumption_1': None, 'violations_assumption_1_rate': None,
                     'violations_assumption_2': None, 'violations_assumption_2_rate': None, 
                     'violations_feasibilitytol': violations_feasibilitytol, 'violations_feasibilitytol_rate': violations_feasibilitytol/(N*(2**D)),
                     'z_integrality_vio': z_integrality_vio, 'vio_feasibilitytol_indicator': vio_feasibilitytol_indicator}
    
    return objective_function_term, solution, counts_result
    