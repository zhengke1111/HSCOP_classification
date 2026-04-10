import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import time
import pandas as pd
import csv
import os
from collections import Counter
import utils
import callback
import callback_data_tree

def full_mip_tree(model, data, start, settings, stop_rule, file_path):

    # temporary variables for debug
    X_train, y_train, class_restricted = data['X_train'], data['y_train'], data['class_restricted']
    a_start, b_start, c_start = start['a'], start['b'], start['c']
    epsilon, rho, beta_p, D, feasibility_tol = settings['epsilon'], settings['rho'], settings['beta_p'], settings['D'], settings['feasibilitytol']
    time_start_calculate = time.time()
    gamma_start, z_plus_0_start, z_plus_start, z_minus_start = utils.calculate_gamma(X_train, y_train, a_start, b_start, c_start, D, None, beta_p, epsilon, class_restricted)
    eta_start, zeta_start, L_start = None, None, None
    # eta_start, zeta_start, L_start = utils.calculate_eta_zeta_L(X_train, y_train, start['c'], D, z_plus_0_start, z_plus_start, z_minus_start)
    time_end_calculate = time.time()
    time_calculate = time_end_calculate - time_start_calculate
    timelimit = stop_rule['timelimit']
    result_sub3dir = file_path['result_sub3dir']
    total_class_num = len(Counter(y_train))
    regularizer = settings['regularizer']
     # samples
    p = X_train.shape[1]  # dimension
    J = range(1, total_class_num+1)  # classes
    I = class_restricted  # class restricted
    class_index = {cls: np.where(y_train == cls)[0] for cls in J}  # use to select samples of certain class

    model = model.copy()
    N = X_train.shape[0]
    a = {}
    a_abs = {}
    b = {}
    A_L = {}
    A_R = {}
    L = {}
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
        b[k] = model.addVar(lb=-b_ub, ub=b_ub, vtype=GRB.CONTINUOUS, name="b_" + str(k))
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
    if z_plus_0_start is not None:
        for s in range(N):
            for t in range(2**D):
                z_plus_0[s, t].setAttr(gp.GRB.Attr.Start, z_plus_0_start[s, t])
    
    keys_z_plus = [(s, t) for s in range(N) for t in range(2**D)]
    z_plus = model.addVars(keys_z_plus, vtype=GRB.BINARY, name="z_plus")
    if z_plus_start is not None:
        for s in range(N):
            for t in range(2**D):
                z_plus[s, t].setAttr(gp.GRB.Attr.Start, z_plus_start[s, t])

    keys_z_minus = [(s, t) for s in range(N) for t in range(2**D)]
    z_minus = model.addVars(keys_z_minus, vtype=GRB.BINARY, name="z_minus")
    if z_minus_start is not None:
        for s in range(N):
            for t in range(2**D):
                z_minus[s, t].setAttr(gp.GRB.Attr.Start, z_minus_start[s, t])

    key_phi_max = [(s, t) for s in range(N) for t in range(2**D)]
    phi_max = model.addVars(key_phi_max, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="phi_max")

    gamma = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="gamma")
    if gamma_start is not None:
        for j in I:
            gamma[j].setAttr(gp.GRB.Attr.UB, gamma_start[j])

    keys_c = [(j, t) for j in J for t in range(2**D)]
    c = model.addVars(keys_c, vtype=GRB.BINARY, name="c")
    
    keys_eta = [(j, t) for j in J for t in range(2**D)]
    eta= model.addVars(keys_eta, lb=0, ub=N, vtype=GRB.CONTINUOUS, name="eta")
    if eta_start is not None:
        for (j,t) in keys_eta:
            eta[j,t].setAttr(gp.GRB.Attr.Start, eta_start[j,t])

    keys_zeta = [(j, t) for j in J for t in range(2**D)]
    zeta= model.addVars(keys_zeta, lb=0, ub=N, vtype=GRB.CONTINUOUS, name="zeta")
    if zeta_start is not None:
        for (j,t) in keys_zeta:
            zeta[j,t].setAttr(gp.GRB.Attr.Start, zeta_start[j,t])

    for t in range(2**D):
        L[t] = model.addVar(lb=0, ub=N, vtype=GRB.CONTINUOUS, name="L_" + str(t))
        if L_start is not None:
            if t in L_start:
                L[t].setAttr(gp.GRB.Attr.Start, L_start[t])
                
        A_L[t] = []
        A_R[t] = []
        real_t = t + 2 ** D - 1
        current_node = real_t
        while current_node != 0:
            parent_node = (current_node - 1) // 2
            if current_node == 2 * parent_node + 1:
                A_L[t].append(parent_node)
            else:
                A_R[t].append(parent_node)
            current_node = parent_node

        model.addConstrs((gp.quicksum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] - 1 -feasibility_tol>= -M_z_plus_0 * (1-z_plus_0[s,t])) for k in A_R[t] for s in range(N))
        model.addConstrs((gp.quicksum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k] - 1 -feasibility_tol>= -M_z_plus_0 * (1-z_plus_0[s,t])) for k in A_L[t] for s in range(N))

        model.addConstrs((L[t] <= gp.quicksum(z_plus_0[s, t] for s in class_index[j])+N*(1-c[j,t])) for j in J)
        model.addConstr(1 == gp.quicksum(c[j, t] for j in J))
        
        # constraints related to precision are as below
        model.addConstrs((gp.quicksum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] -feasibility_tol>= -M_z * (1-z_plus[s,t])) for k in A_R[t] for s in range(N))
        model.addConstrs((gp.quicksum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k] - epsilon -feasibility_tol>= -M_z * (1-z_plus[s,t])) for k in A_L[t] for s in range(N))
        
    model.addConstrs((eta[j,t]<= M_eta[j]*c[j,t]) for j in I for t in range(2**D))
    model.addConstrs((eta[j,t]<=gp.quicksum(z_plus[s,t] for s in class_index[j]) + M_eta[j]*(1-c[j,t])) for j in I for t in range(2**D))

    model.addConstrs((zeta[j,t]>=-N*c[j,t]) for j in I for t in range(2**D)) 
    model.addConstrs((zeta[j,t]>=gp.quicksum((1-z_minus[s,t]) for s in range(N)) - N*(1-c[j,t])) for j in I for t in range(2**D)) 
    # Problem: if the precision cannot be satisfied, let c[j,t] (j \notin I)=1, then the nomiator and denomiator = 0, gamma = 0

    model.addConstrs((gp.quicksum(eta[j,t] for t in range(2**D))-beta_p[j]* gp.quicksum(zeta[j,t] for t in range(2**D)) + gamma[j] >= 0) for j in I)

    model.addConstrs((gp.quicksum(eta[j,t] for t in range(2**D)) + 0.01 * gamma[j] >=1) for j in I)

    key_phi_minus_stk = [(s, t, k) for s in range(N) for t in range(2**D) for k in (A_L[t] + A_R[t])]
    phi_minus_stk = model.addVars(key_phi_minus_stk, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="phi_minus_stk")

    phi_list = {}
    for s in range(N):
        phi_list[s]={}
        for t in range(2**D):
            phi_list[s][t] = []
            model.addConstrs((phi_minus_stk[s, t, k] == gp.quicksum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k] - epsilon) for k in A_R[t])
            model.addConstrs((phi_minus_stk[s, t, k] == gp.quicksum(a[k][i] * X_train[s][i] for i in range(p)) - b[k]) for k in A_L[t])
            for k in (A_L[t] + A_R[t]):
                phi_list[s][t].append(phi_minus_stk[s, t, k])
            model.addGenConstrMax(phi_max[s, t], phi_list[s][t], constant=None)
            
    model.addConstrs((phi_max[s, t] -feasibility_tol>= -M_z*(1-z_minus[s, t])) for s in range(N) for t in range(2**D))

    obj = 1/N*gp.quicksum(L[t] for t in range(2**D)) - rho*gp.quicksum(gamma[j] for j in I)
    
    model.addConstr(obj <= 1, "manual_upper_bound")
    model.setObjective(obj,GRB.MAXIMIZE)
    model.setParam("Timelimit", timelimit)
    model.update()
    num_integer_vars = sum(1 for v in model.getVars() if v.vType == gp.GRB.BINARY)

    os.makedirs(result_sub3dir + '/Logfile')
    model.setParam('LogFile', os.path.join(result_sub3dir + '/Logfile', f'log_file.txt'))

    callback_data_tree.log_data=[]
    model._vars = model.getVars()
    model.optimize(callback.full_mip_callback)
    
    log_data_list = callback_data_tree.log_data
    log_df = pd.DataFrame(log_data_list)
    
    objective_value = model.objVal
    try:
        optimality_gap = model.MIPGap
    except AttributeError:
        optimality_gap = -1

    if len(log_df)>0:
        last_row = log_df.iloc[-1]
        if model.objVal<0:
            final_improvement_time = model.Runtime
        elif optimality_gap == 0: 
            final_improvement_time = model.Runtime
        else:
            final_improvement_time = last_row['final_improvement_time']
        runtime = model.Runtime
    else:
        final_improvement_time = model.Runtime
        runtime = model.Runtime


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
    os.makedirs(result_sub3dir + '/Solution')
    with open(result_sub3dir + '/Solution/' + 'solution.txt', 'a') as f:
        f.write("\n".join(output_abc))  

    solution_a = {}
    solution_b = {}
    solution_c = {}
    for k in a.keys():
        solution_a[k] = {i: a[k][i].X for i in range(p)}
    for k in b.keys():
        solution_b[k] = b[k].X
    for key in keys_c:
        solution_c[key] = c[key].X

    z_integrality_vio = 0
    solution_z_plus_0 = {}
    solution_z_plus = {}
    solution_z_minus = {}
    for key in key_z_plus_0:
        solution_z_plus_0[key] = z_plus_0[key].X
        if 1 > solution_z_plus_0[key] > 0:
            z_integrality_vio += 1
    for key in keys_z_plus:
        solution_z_plus[key] = z_plus[key].X
        if 1 > solution_z_plus[key] > 0:
            z_integrality_vio += 1
    for key in keys_z_minus:
        solution_z_minus[key] = z_minus[key].X
        if 1 > solution_z_minus[key] > 0:
            z_integrality_vio += 1

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

    objective_function_term = {'objective_value': objective_value, 'optimality_gap': optimality_gap, 'gamma': solution_gamma, 'final_improvement_time': final_improvement_time, 'actual_time': runtime, 'z_frac': z_frac, 'z_counts':z_counts}

    solution = {'objective_value': objective_value, 'a': solution_a, 'b': solution_b, 'c': solution_c, 'z_plus_0': solution_z_plus_0, 'z_plus': solution_z_plus, 'z_minus': solution_z_minus, 'gamma': solution_gamma}

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
            lhs_assum = min([sum(solution_a[k][i] * X_train[s][i] for i in range(p)) - solution_b[k] for k in A_R[t]] + [sum(-solution_a[k][i] * X_train[s][i] for i in range(p)) + solution_b[k] - epsilon for k in A_L[t]])
            if lhs_assum == 0:
                violations_assumption_1 += 1
            elif -epsilon<= lhs_assum < 0:
                violations_assumption_2 += 1 

            lhs_feas = phi_max[s, t].X
            if 0 > lhs_feas >= -feasibility_tol:
                violations_feasibilitytol += 1

    counts_result = {'num_integer_vars': num_integer_vars, 'violations_assumption_1': violations_assumption_1, 'violations_assumption_1_rate': violations_assumption_1/(2*N*(2**D)),
                     'violations_assumption_2': violations_assumption_2, 'violations_assumption_2_rate': violations_assumption_2/(2*N*(2**D)), 
                     'violations_feasibilitytol': violations_feasibilitytol, 'violations_feasibilitytol_rate': violations_feasibilitytol/(3*N*(2**D)), 
                     'z_integrality_vio': z_integrality_vio}
    
    execution_time = final_improvement_time
    return objective_function_term, solution, counts_result, execution_time
