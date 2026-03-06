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


def pip_single_iter_tree(model, data, start, settings, file_path):
    """
    :param lbd:
    :param model:
    :param obj_cons_num: 
    :param X_train: 
    :param y_train: 
    :param w_start: 
    :param b_start
    :param z_plus_start: 
    :param z_minus_start: 
    :param epsilon: 
    :param gamma: 
    :param M: 
    :param rho: 
    :param beta_p: 
    :param dirname: 
    :return: 
    """
    X_train, y_train, class_restricted = data['X_train'], data['y_train'], data['class_restricted']
    a_start, b_start, c_start, z_plus_0_start, z_plus_start, z_minus_start, gamma_start = start['a'], start['b'], start['c'], start['z_plus_0'], start['z_plus'], start['z_minus'], start['gamma']
    eta_start, zeta_start, L_start = None, None, None
    if all(key in start for key in ['eta', 'zeta', 'L']):
        eta_start, zeta_start, L_start = start['eta'], start['zeta'], start['L']
    # else:
    #     #### c_start = start['c']
    #     c_start, eta_start, zeta_start, L_start = start['c'], None, None, None
    method, selected_piece, epsilon, delta_1, delta_2, rho, beta_p, D, feasibility_tol = settings['method'], settings['selected_piece'], settings['epsilon'], settings['delta_1'], settings['delta_2'], settings['rho'], settings['beta_p'], settings['D'], settings['feasibilitytol']
    regularizer = settings['regularizer']
    result_sub3dir, pip_iter = file_path['result_sub3dir'], file_path['pip_iter']
    if method in [3,6,7]:
        shrinkage_iter = file_path['shrinkage_iter']
    if method in [4,6]:
        piece_index = file_path['piece_index']
    # temporary variables for debug
    random.seed(42)
    # samples
    p = X_train.shape[1]  # dimension
    total_class_num = len(Counter(y_train))
    J = range(1, total_class_num+1)  # classes
    I = class_restricted  # class restricted
    class_index = {cls: np.where(y_train == cls)[0] for cls in J}  # use to select samples of certain class

    ###############################
    model = model.copy()
    N = X_train.shape[0]
    a = {}
    a_abs ={}
    b = {}
    L = {}

    I_plus_0, I_plus_1, I_plus_2 = {}, {}, {}  # Index sets for z_plus_0  
    J_plus_0, J_plus_1, J_plus_2 = {}, {}, {}  # Index sets for z_plus
    J_minus_0, J_minus_1, J_minus_2 = {}, {}, {}  # Index sets for z_minus

    tau_1 = 100
    b_ub = tau_1*abs(max(abs(X_train[s][i]) for i in range(p) for s in range(N))) + 1

    M_z_plus_0 = 2*(tau_1*abs(max(abs(X_train[s][i]) for i in range(p) for s in range(N))) + 1)
    M_z = 2*(tau_1*abs(max(abs(X_train[s][i]) for i in range(p) for s in range(N))) + 1)
    M_y = 4*(tau_1*abs(max(abs(X_train[s][i]) for i in range(p) for s in range(N))) + 1)
    M_eta = {cls: sum(1 for _ in class_index[cls]) for cls in J}
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

    if regularizer == 'soft_l0':
        # TODO
        varrho = settings['varrho']
        tau_0 = settings['tau_0']
        u={}
        v={}
        for k in range(2**D-1):
            u[k] = model.addVars(p, vtype=GRB.BINARY, name='u_'+str(k))
            v[k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='v_'+str(k))
            model.addConstrs((a[k][i]>=-tau_1*(1-u[k][i])) for i in range(p))
            model.addConstrs((-a[k][i]>=-tau_1*(1-u[k][i])) for i in range(p))
            model.addConstr(gp.quicksum((1-u[k][i]) for i in range(p))<=tau_0+v[k])

    key_z_plus_0 = [(s, t) for s in range(N) for t in range(2**D)]
    z_plus_0 = model.addVars(key_z_plus_0, vtype=GRB.BINARY, name='z_plus_0')
    
    keys_z_plus = [(s, t) for s in range(N) for t in range(2**D)]
    z_plus = model.addVars(keys_z_plus, vtype=GRB.BINARY, name="z_plus")

    keys_z_minus = [(s, t) for s in range(N) for t in range(2**D)]
    z_minus = model.addVars(keys_z_minus, vtype=GRB.BINARY, name="z_minus")

    gamma = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="gamma")
    if gamma_start is not None:
        for j in I:
            gamma[j].setAttr(gp.GRB.Attr.UB, gamma_start[j])

    keys_c = [(j, t) for j in J for t in range(2**D)]
    c = model.addVars(keys_c, vtype=GRB.BINARY, name="c")
    if c_start is not None:
        for (j,t) in keys_c:
            c[j,t].setAttr(gp.GRB.Attr.Start, c_start[j,t])
    
    keys_eta = [(j, t) for j in I for t in range(2**D)]
    eta= model.addVars(keys_eta, lb=0, ub=N, vtype=GRB.CONTINUOUS, name="eta")
    if eta_start is not None:
        for (j,t) in keys_eta:
            eta[j,t].setAttr(gp.GRB.Attr.Start, eta_start[j,t])

    keys_zeta = [(j, t) for j in I for t in range(2**D)]
    zeta= model.addVars(keys_zeta, lb=0, ub=N, vtype=GRB.CONTINUOUS, name="zeta")
    if zeta_start is not None:
        for (j,t) in keys_zeta:
            zeta[j,t].setAttr(gp.GRB.Attr.Start, zeta_start[j,t])

    A_L, A_R = utils.ancestors(D)

    for t in range(2**D):
        I_plus_0[t], I_plus_1[t], I_plus_2[t] = [], [], []  # Index sets for z_plus_0  
        J_plus_0[t], J_plus_1[t], J_plus_2[t] = [], [], []   # Index sets for z_plus
        J_minus_0[t], J_minus_1[t], J_minus_2[t] = [], [], []   # Index sets for z_minus
        
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
            
            phi_plus = min([sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k] for k in A_R[t]]+ [-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon for k in A_L[t]])
            if phi_plus > delta_1[1]:
                model.remove(z_plus[s, t])
                J_plus_1[t].append(s)
                model.addConstrs((gp.quicksum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] -feasibility_tol>= 0) for k in A_R[t])
                model.addConstrs((gp.quicksum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k] - epsilon -feasibility_tol>= 0) for k in A_L[t])
            elif phi_plus >= -delta_2[1]:
                if z_plus_start[s, t] == 0 or z_plus_start[s, t] == 1:
                    z_plus[s, t].setAttr(gp.GRB.Attr.Start, z_plus_start[s, t])
                J_plus_0[t].append(s)
                model.addConstrs((gp.quicksum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] -feasibility_tol>= -M_z * (1-z_plus[s,t])) for k in A_R[t])
                model.addConstrs((gp.quicksum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k] - epsilon -feasibility_tol>= -M_z * (1-z_plus[s,t])) for k in A_L[t])
            else:
                model.remove(z_plus[s, t])
                J_plus_2[t].append(s)

        model.addConstrs((L[t] <= gp.quicksum(z_plus_0[s, t] for s in list(set(class_index[j]) & set(I_plus_0[t]))) + sum(1 for _ in list(set(class_index[j]) & set(I_plus_1[t]))) + N * (1-c[j,t])) for j in J)
        model.addConstr(1 == gp.quicksum(c[j, t] for j in J)) 

    if selected_piece is not None:
        for t in range(2**D):
            for s in range(N):    
                k = selected_piece[s][t]
                if k in A_L[t]:
                    phi_minus = sum(-a_start[k][i] * X_train[s][i] for i in range(p)) + b_start[k]
                    if -phi_minus > delta_1[1]:
                        model.remove(z_minus[s, t])
                        J_minus_1[t].append(s)
                        model.addConstr(-(gp.quicksum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k]) -feasibility_tol>= 0)
                    elif -phi_minus >= -delta_2[1]:
                        if z_minus_start[s, t] == 0 or z_minus_start[s, t] == 1:
                            z_minus[s, t].setAttr(gp.GRB.Attr.Start, z_minus_start[s, t])
                        J_minus_0[t].append(s)
                        model.addConstr(-(gp.quicksum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k]) -feasibility_tol>= -M_z * (1-z_minus[s,t]))
                    else:
                        model.remove(z_minus[s, t])
                        J_minus_2[t].append(s)
                        
                elif k in A_R[t]:
                    phi_minus = sum(a_start[k][i] * X_train[s][i] for i in range(p)) - b_start[k] + epsilon
                    if -phi_minus > delta_1[1]:
                        model.remove(z_minus[s, t])
                        J_minus_1[t].append(s)
                        model.addConstr(-(gp.quicksum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] + epsilon) -feasibility_tol>= 0)
                    elif -phi_minus >= -delta_2[1]:
                        if z_minus_start[s, t] == 0 or z_minus_start[s, t] == 1:
                            z_minus[s, t].setAttr(gp.GRB.Attr.Start, z_minus_start[s, t])
                        J_minus_0[t].append(s)
                        model.addConstr(-(gp.quicksum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] + epsilon) -feasibility_tol>= -M_z * (1-z_minus[s,t]))
                    else:
                        model.remove(z_minus[s, t])
                        J_minus_2[t].append(s)
    
    else:
        key_phi_max = [(s, t) for s in range(N) for t in range(2**D)]
        phi_max = model.addVars(key_phi_max, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="phi_max")
        key_phi_minus_stk = [(s, t, k) for s in range(N) for t in range(2**D) for k in (A_L[t] + A_R[t])]
        phi_minus_stk = model.addVars(key_phi_minus_stk, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="phi_minus_stk")
        phi_list = {}
        for s in range(N):
            phi_list[s]={}
            for t in range(2**D):  
                phi_list[s][t] = []
                underline_phi = max([sum(a_start[k][i] * X_train[s][i] for i in range(p)) - b_start[k] for k in A_L[t]] + [sum(-a_start[k][i] * X_train[s][i] for i in range(p)) + b_start[k] - epsilon for k in A_R[t]])
                if underline_phi < -delta_2[1]:
                    model.remove(z_minus[s, t])
                    model.remove(phi_max[s, t])
                    for k in (A_L[t] + A_R[t]):
                        model.remove(phi_minus_stk[s, t, k])
                    J_minus_2[t].append(s)
                else:
                    model.addConstrs((phi_minus_stk[s, t, k] == gp.quicksum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k] - epsilon) for k in A_R[t])
                    model.addConstrs((phi_minus_stk[s, t, k] == gp.quicksum(a[k][i] * X_train[s][i] for i in range(p)) - b[k]) for k in A_L[t])
                    for k in (A_L[t] + A_R[t]):
                        phi_list[s][t].append(phi_minus_stk[s, t, k])
                    model.addGenConstrMax(phi_max[s, t], phi_list[s][t], constant=None)
                    if underline_phi <= delta_1[1]:
                        if z_minus_start[s, t] == 0 or z_minus_start[s, t] == 1:
                            z_minus[s, t].setAttr(gp.GRB.Attr.Start, z_minus_start[s, t])
                        J_minus_0[t].append(s)
                        model.addConstr(phi_max[s, t] -feasibility_tol>= -M_z*(1-z_minus[s, t]))
                    else:
                        model.remove(z_minus[s, t])
                        J_minus_1[t].append(s)
                        model.addConstr(phi_max[s, t] -feasibility_tol>= 0) 
        
    model.addConstrs((eta[j,t]<= M_eta[j]*c[j,t]) for j in I for t in range(2**D))
    model.addConstrs((eta[j,t]<=gp.quicksum(z_plus[s,t] for s in list(set(class_index[j]) & set(J_plus_0[t]))) + sum(1 for _ in list(set(class_index[j]) & set(J_plus_1[t]))) + M_eta[j]*(1-c[j,t])) for j in I for t in range(2**D))

    model.addConstrs((zeta[j,t]>=-N*c[j,t]) for j in I for t in range(2**D)) 
    model.addConstrs((zeta[j,t]>=gp.quicksum((1-z_minus[s,t]) for s in J_minus_0[t]) + sum(1 for _ in J_minus_2[t]) - N*(1-c[j,t])) for j in I for t in range(2**D)) 
        
    model.addConstrs((gp.quicksum(eta[j,t] for t in range(2**D)) - beta_p[j]* gp.quicksum(zeta[j,t] for t in range(2**D)) + gamma[j] >= 0) for j in I)

    model.addConstrs((gp.quicksum(eta[j,t] for t in range(2**D)) + 0.01 * gamma[j]>=1) for j in I)

    obj = 1/N*gp.quicksum(L[t] for t in range(2**D)) - rho*gp.quicksum(gamma[j] for j in I)
    if regularizer == 'soft_l0': 
        obj = 1/N*gp.quicksum(L[t] for t in range(2**D)) - rho*gp.quicksum(gamma[j] for j in I) - varrho/N*gp.quicksum(v[k] for k in range(2**D-1))
    model.addConstr(obj <= 1, "manual_upper_bound")
    model.setObjective(obj,GRB.MAXIMIZE)
    model.update()
    num_integer_vars = sum(1 for v in model.getVars() if v.vType == gp.GRB.BINARY)

    if method in [2,3]:
        model.setParam("Timelimit", 600)
    else:
        model.setParam("Timelimit", 300)

    os.makedirs(result_sub3dir + '/Logfile', exist_ok=True)
    if method == 2:
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
    # os.makedirs(result_sub3dir + '/Model', exist_ok=True)
    # model.write(result_sub3dir + '/Model' + f'/model_{postfix}.lp')

    callback_data_tree.log_data=[]
    callback_data_tree.time_for_finding_feasible_solution = 0
    model._vars = model.getVars()
    model.optimize(MIP_tree_callback.mip_tree_callback)
    log_data_list = callback_data_tree.log_data
    log_df = pd.DataFrame(log_data_list)
    time_for_finding_feasible_solution = callback_data_tree.time_for_finding_feasible_solution
    objective_value = model.objVal
    runtime = model.Runtime

    try:
        last_row = log_df.iloc[-1]
    except IndexError:
        last_row = {'BestBd': model.objVal}

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
    solution_eta = {}
    solution_zeta = {}
    solution_L = {}
    for k in a.keys():
        solution_a[k] = {i: a[k][i].X for i in range(p)}
    for k in b.keys():
        solution_b[k] = b[k].X
    for key in keys_c:
        solution_c[key] = c[key].X
    for key in keys_eta:
        solution_eta[key] = eta[key].X
    for key in keys_eta:
        solution_zeta[key] = zeta[key].X
    for t in range(2**D):
        solution_L[t] = L[t].X

    z_integrality_vio = 0
    solution_z_plus_0, solution_z_plus, solution_z_minus = {}, {}, {}

    for (s, t) in key_z_plus_0:
        if s in I_plus_0[t]:
            solution_z_plus_0[s, t] = z_plus_0[s, t].X
            if 1> z_plus_0[s, t].X >0:
                z_integrality_vio += 1
        elif s in I_plus_1[t]:
            solution_z_plus_0[s, t] = 1
        else:
            solution_z_plus_0[s, t] = 0
    
    for (s, t) in keys_z_plus:
        if s in J_plus_0[t]:
            solution_z_plus[s, t] = z_plus[s, t].X
            if 1> z_plus[s, t].X >0:
                z_integrality_vio += 1
        elif s in J_plus_1[t]:
            solution_z_plus[s, t] = 1
        else:
            solution_z_plus[s, t] = 0
    
    for (s, t) in keys_z_minus:
        if s in J_minus_0[t]:
            solution_z_minus[s, t] = z_minus[s, t].X
            if 1> z_minus[s, t].X >0:
                z_integrality_vio += 1
        elif s in J_minus_1[t]:
            solution_z_minus[s, t] = 1
        else:
            solution_z_minus[s, t] = 0

    solution_gamma = {}
    for j in I:
        solution_gamma[j] = gamma[j].X
    
    z_frac = {}
    z_counts = {}
    z_frac['acc_margin'] = round(1/N*sum(solution_c[y_train[s],t]*solution_z_plus_0[s, t] for s in range(N) for t in range(2**D)),3)
    z_counts['acc_margin'] = {'nm':sum(solution_c[y_train[s],t]*solution_z_plus_0[s, t] for s in range(N) for t in range(2**D)),'dm':N}
    for j in J:  
        z_frac[f'prec{j}'] = round(sum(solution_c[j,t]*solution_z_plus[s, t] for s in class_index[j] for t in range(2**D))/sum(solution_c[j,t]*(1-solution_z_minus[s, t]) for s in range(N) for t in range(2**D)),3) if sum(solution_c[j,t]*(1-solution_z_minus[s, t]) for s in range(N) for t in range(2**D))>0 else -1
        z_counts[j] = {'nm': sum(solution_c[j,t]*solution_z_plus[s, t] for s in class_index[j] for t in range(2**D)),'dm':sum(solution_c[j,t]*(1-solution_z_minus[s, t]) for s in range(N) for t in range(2**D))}

    objective_function_term = {'objective_value': objective_value, 'optimality_gap': optimality_gap, 
                               'gamma': solution_gamma, 'runtime': runtime, 'z_frac': z_frac, 'z_counts':z_counts}
    
    solution = {'objective_value': objective_value, 
                'a': solution_a,
                'b': solution_b,
                'c': solution_c,
                'eta': solution_eta,
                'zeta': solution_zeta,
                'L': solution_L,
                'z_plus_0': solution_z_plus_0,
                'z_plus': solution_z_plus,
                'z_minus': solution_z_minus,
                'gamma': {key: gamma[key].X for key in gamma.keys()}}
    
    violations_assumption_1 = 0
    violations_assumption_2 = 0
    violations_feasibilitytol = 0

    for s in range(N):
        for t in range(2**D):
            lhs_feas = min([sum(solution_a[k][i]*X_train[s][i] for i in range(p)) - solution_b[k] - 1 for k in A_R[t]] + [-sum(solution_a[k][i]*X_train[s][i] for i in range(p)) + solution_b[k] - 1 for k in A_L[t]])
            if 0 > lhs_feas >= -feasibility_tol:
                violations_feasibilitytol += 1

    for s in range(N):
        for t in range(2**D):
            lhs_feas = min([sum(solution_a[k][i] * X_train[s][i] for i in range(p)) - solution_b[k] for k in A_R[t]] + [sum(-solution_a[k][i] * X_train[s][i] for i in range(p)) + solution_b[k] - epsilon for k in A_L[t]])
            if 0 > lhs_feas >= -feasibility_tol:
                violations_feasibilitytol += 1
            
            lhs_assum = min([sum(solution_a[k][i] * X_train[s][i] for i in range(p)) - solution_b[k] for k in A_R[t]] + [sum(-solution_a[k][i] * X_train[s][i] for i in range(p)) + solution_b[k] - epsilon for k in A_L[t]])
            if lhs_assum == -epsilon:
                violations_assumption_1 += 1
            elif -epsilon< lhs_assum <= 0:  
                violations_assumption_2 += 1 
                    
                        
    for s in range(N):
        for t in range(2**D):
            if selected_piece is None:
                if s in J_minus_0[t]:
                    lhs_feas = phi_max[s, t].X
                    if 0 > lhs_feas >= -feasibility_tol:
                        violations_feasibilitytol += 1

                    lhs_assum = min([sum(solution_a[k][i] * X_train[s][i] for i in range(p)) - solution_b[k] for k in A_R[t]] + [sum(-solution_a[k][i] * X_train[s][i] for i in range(p)) + solution_b[k] - epsilon for k in A_L[t]])
                    if lhs_assum == 0:
                        violations_assumption_1 += 1
                    elif -epsilon<= lhs_assum < 0:
                        violations_assumption_2 += 1 
                        
            else:
                if s in J_minus_0[t]:
                    k = selected_piece[s][t]
                    if k in A_L[t]:
                        lhs_feas = - (sum(-solution_a[k][i] * X_train[s][i] for i in range(p)) + solution_b[k])
                    elif k in A_R[t]:
                        lhs_feas = - (sum(solution_a[k][i] * X_train[s][i] for i in range(p)) - solution_b[k] + epsilon)
                        lhs_assum = sum(solution_a[k][i] * X_train[s][i] for i in range(p)) - solution_b[k]
                        if lhs_assum == 0:
                            violations_assumption_1 += 1
                        elif -epsilon <= lhs_assum <0:
                            violations_assumption_2 += 1 
                    if 0 > lhs_feas >= -feasibility_tol:
                        violations_feasibilitytol += 1

    vio_feasibilitytol_indicator = False
    log_path = result_sub3dir + '/Logfile/'+f'log_{postfix}.txt'
    with open(log_path, 'r') as log_file:
        log_content = log_file.read()
    if "max constraint violation" in log_content:
        vio_feasibilitytol_indicator = True
    counts_result = {'num_integer_vars': num_integer_vars, 'violations_assumption_1': violations_assumption_1, 'violations_assumption_1_rate': violations_assumption_1/(2*N*(2**D)),
                     'violations_assumption_2': violations_assumption_2, 'violations_assumption_2_rate': violations_assumption_2/(2*N*(2**D)), 
                     'violations_feasibilitytol': violations_feasibilitytol, 'violations_feasibilitytol_rate': violations_feasibilitytol/(3*N*(2**D)),
                     'z_integrality_vio': z_integrality_vio, 'vio_feasibilitytol_indicator': vio_feasibilitytol_indicator}
    
    return objective_function_term, solution, counts_result
    