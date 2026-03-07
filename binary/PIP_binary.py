from gurobipy import GRB
import MIP_binary_callback, callback_data_binary
import os, numpy as np, pandas as pd, gurobipy as gp


def pip_binary(model, data, start, settings, file_path):
    """
    A single PIP iteration in PIP method to solve the binary classification problem with precision constraint

    Args:
        model (dict): Gurobi parameter settings, including {Name, 'MIPFocus', 'IntegralityFocus', 'Threads', 'FeasibilityTol'}
        data (dict): Data splits, {X_train, y_train, X_test, y_test} split by some random seeds we set
        start (dict): Initial solution from the last iteration
        settings (dict): Settings of PIP
        file_path (dict): File path to store the output

    Returns:
        Tuple(dict, dict, dict): objective_function_term, solution, counts_result 
    """

    model = model.copy()  # Copy the Gurobi parameter settings
    X_train, y_train = data['X_train'], data['y_train']  # Data used to train MIP model
    
    # The initial solutions including continuous w,b,gamma and discrete z 
    w_start, b_start, z_plus_0_start, z_plus_start, z_minus_start, gamma_0 = start['w'], start['b'], start['z_plus_0'], start['z_plus'], start['z_minus'], start['gamma_0']
    
    # Settings of PIP: 
    ## method: full_mip / full_mip_t300 / full_mip_t600 / fixed / shrinkage / unconstrained
    ## epsilon: epsilon-approximation in not l.s.c. Heaviside functions
    ## delta_1, delta_2: thresholds of in-between index sets
    ## rho: infeasibility penalty parameter
    ## beta_p: precision threshold
    ## feasibility_tol: Gurobi parameter
    method, epsilon, delta_1, delta_2, rho, beta_p, feasibility_tol = settings['method'], settings['epsilon'], settings['delta_1'], settings['delta_2'], settings['rho'], settings['beta_p'], settings['feasibilitytol']
    result_sub3dir, pip_iter = file_path['result_sub3dir'], file_path['pip_iter']  # The output of a single PIP iteration is stored in the third level subdirectory, using pip_iter to distinguish
    if method == 5:
        shrinkage_iter = file_path['shrinkage_iter']  # In epsilon-shrinkage, there is the shrinkage iteration number

    N, dim = X_train.shape[0], X_train.shape[1]  # Sample size N, features dimension dim
    positive_index, negative_index = np.where(y_train == 1)[0].tolist(), np.where(y_train == -1)[0].tolist()  # Indices of samples with true labels as positive / negative
    z_plus_0, z_plus, z_minus = {}, {}, {}  # Dictionaries to store the variables
    solution_z_plus_0, solution_z_plus, solution_z_minus = {}, {}, {}  # Dictionaries to store the values of the variables

    # J_0 in-between set, J_1: integers fixed as 1, J_2: integers free, seen as 0
    # z_{plus_0: in the objective function, plus: in constraint with positive coefficients, minus: in constraint with negative coefficients}
    J_0_plus_0, J_1_plus_0, J_2_plus_0, J_0_plus, J_1_plus, J_2_plus, J_0_minus, J_1_minus, J_2_minus = [], [], [], [], [], [], [], [], []

    tau = 10  # \ell-1 norm constraint
    b_ub = tau*max(abs(X_train[s][p]) for p in range(dim) for s in range(N)) + 1  # Upperbound of the absolute value of bias b 
    big_M = 2*(tau*max(abs(X_train[s][p]) for p in range(dim) for s in range(N)) + 1)  # Big-M

    # Define variables w,b and their absolute upper bounds, and gamma
    w, b = model.addVars(dim, lb=-tau, ub=tau, vtype=GRB.CONTINUOUS, name="w"), model.addVar(lb=-b_ub, ub=b_ub, vtype=GRB.CONTINUOUS, name="b")
    abs_w, abs_b = model.addVars(dim, lb=0, ub=tau, vtype=GRB.CONTINUOUS, name="abs_w"), model.addVar(lb=0, ub=b_ub, vtype=GRB.CONTINUOUS, name="abs_b")
    gamma = model.addVar(lb=0, ub=gamma_0, vtype=GRB.CONTINUOUS, name="gamma")
    
    # \ell-1 norm constraint of w,b
    for p in range(dim):
        w[p].setAttr(gp.GRB.Attr.Start, w_start[p])
        model.addConstr(w[p] <= abs_w[p])
        model.addConstr(-w[p] <= abs_w[p])
    b.setAttr(gp.GRB.Attr.Start, b_start)
    model.addConstr(b <= abs_b)
    model.addConstr(-b <= abs_b)
    model.addConstr(gp.quicksum(abs_w[p] for p in range(dim)) <= tau)

    # Constraints of Heaviside binary variables
    for s in positive_index:

        # z_{0s}^+, in objective function
        z_plus_0[s] = model.addVar(vtype=GRB.BINARY, name="z_plus_0_" + str(s))
        if np.dot(w_start, X_train[s]) + b_start -1 >= delta_1[0]:  # Binary variables fixed as 1
            J_1_plus_0.append(s)
            model.remove(z_plus_0[s])
            model.addConstr(gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) + b - 1 - feasibility_tol >= 0)  # - feasibility_tol: avoid gurobi treat -1e-9 as 0 in left-hand-side
        elif np.dot(w_start, X_train[s]) + b_start-1 <= -delta_2[0]:  # Binary variables free, seen as 0 in objective function and constraints
            J_2_plus_0.append(s)
            model.remove(z_plus_0[s])
        else:  # Binary variables in in-between sets
            J_0_plus_0.append(s)
            z_plus_0[s].setAttr(gp.GRB.Attr.Start, z_plus_0_start[s])
            model.addConstr(gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) + b - 1 - feasibility_tol >= -big_M * (1 - z_plus_0[s]))

        # z_{1s}^+, in constraints, with nonnegative coefficients
        z_plus[s] = model.addVar(vtype=GRB.BINARY, name="z_plus_" + str(s))
        if np.dot(w_start, X_train[s]) + b_start >= delta_1[1]:  # Binary variables fixed as 1
            J_1_plus.append(s)
            model.remove(z_plus[s])
            model.addConstr(gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) + b - feasibility_tol >= 0)
        elif np.dot(w_start, X_train[s]) + b_start <= -delta_2[1]:  # Binary variables free, seen as 0 in objective function and constraints
            J_2_plus.append(s)
            model.remove(z_plus[s])
        else:  # Binary variables in in-between sets
            J_0_plus.append(s)
            z_plus[s].setAttr(gp.GRB.Attr.Start, z_plus_start[s])
            model.addConstr(gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) + b - feasibility_tol >= -big_M * (1 - z_plus[s]))

    for s in negative_index:
        # z_{0s}^+, in objective function
        z_plus_0[s] = model.addVar(vtype=GRB.BINARY, name="z_plus_0_" + str(s))
        if -np.dot(w_start, X_train[s]) - b_start -1 >= delta_1[0]:  # Binary variables fixed as 1
            J_1_plus_0.append(s)
            model.remove(z_plus_0[s])
            model.addConstr(-gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) - b - 1 - feasibility_tol >= 0)
        elif -np.dot(w_start, X_train[s]) - b_start -1 <= -delta_2[0]:  # Binary variables free, seen as 0 in objective function and constraints
            J_2_plus_0.append(s)
            model.remove(z_plus_0[s])
        else:  # Binary variables in in-between sets
            J_0_plus_0.append(s)
            z_plus_0[s].setAttr(gp.GRB.Attr.Start, z_plus_0_start[s])
            model.addConstr(-gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) - b - 1 - feasibility_tol >= -big_M * (1 - z_plus_0[s]))

        # z_{1s}^-, in constraints, with nonpositive coefficients
        z_minus[s] = model.addVar(vtype=GRB.BINARY, name="z_minus_" + str(s))
        if -(np.dot(w_start, X_train[s]) + b_start) - epsilon >= delta_1[1]:  # Binary variables fixed as 1
            J_1_minus.append(s)
            model.remove(z_minus[s])
            model.addConstr(-gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) - b - feasibility_tol >= epsilon)
        elif -(np.dot(w_start, X_train[s]) + b_start) - epsilon <= -delta_2[1]:  # Binary variables free, seen as 0 in objective function and constraints
            J_2_minus.append(s)
            model.remove(z_minus[s])
        else:  # Binary variables in in-between sets
            J_0_minus.append(s)
            z_minus[s].setAttr(gp.GRB.Attr.Start, z_minus_start[s])
            model.addConstr(-gp.quicksum(w[p] * X_train[s][p] for p in range(dim)) - b - feasibility_tol >= -big_M * (1 - z_minus[s]) + epsilon)

    # Precision constraint
    model.addConstr((1-beta_p)*(gp.quicksum(z_plus[s] for s in J_0_plus) + sum(1 for _ in J_1_plus)) - beta_p*(gp.quicksum((1 - z_minus[s]) for s in J_0_minus) + sum(1 for _ in J_2_minus)) + gamma >= 0)
    
    # Set Objective function
    # obj = (1/N)*(gp.quicksum(z_plus_0[s] for s in J_0_plus_0) + sum(1 for _ in J_1_plus_0)) - rho * gamma 
    J_0_plus_0_positive, J_0_plus_0_negative, J_1_plus_0_positive, J_1_plus_0_negative = list(set(J_0_plus_0) & set(positive_index)), list(set(J_0_plus_0) & set(negative_index)), list(set(J_1_plus_0) & set(positive_index)), list(set(J_1_plus_0) & set(negative_index))
    obj = (1/N)*(gp.quicksum(z_plus_0[s] for s in J_0_plus_0_positive) + gp.quicksum(z_plus_0[s] for s in J_0_plus_0_negative) + sum(1 for _ in J_1_plus_0_positive) + sum(1 for _ in J_1_plus_0_negative)) - rho * gamma 

    # Manually set the upper bound = 1 since the accuracy out of margin is always <= 1
    model.addConstr(obj <= 1, "manual_upper_bound")
    model.setObjective(obj, GRB.MAXIMIZE)
    model.update()
    num_integer_vars = sum(1 for v in model.getVars() if v.vType == gp.GRB.BINARY)  # Number of integer variables

    # Timelimit of a single PIP iteration
    model.setParam("Timelimit", 300)
    if method == 4:
        postfix = f'pip_iter_{pip_iter}'  # epsilon-fixed problem, only pip_iter
    if method == 5:
        postfix = f'shrinkage_iter_{shrinkage_iter}_pip_iter_{pip_iter}'  # epsilon-shrinkage problem, shrinkage_iter and pip_iter
    
    os.makedirs(result_sub3dir + '/Logfile', exist_ok=True)  # Set the path of logfiles
    model.setParam('LogFile', os.path.join(result_sub3dir + '/Logfile', f'log_{postfix}.txt'))

    # Callback mechanism, store a obj-time log
    callback_data_binary.log_data = []
    callback_data_binary.time_for_finding_feasible_solution = 0

    # Run the MIP model
    model.optimize(MIP_binary_callback.mip_binary_callback)

    # Store callback records
    log_data_list = callback_data_binary.log_data
    log_df = pd.DataFrame(log_data_list)
    # In earlier versions, time records were used to search for feasible solutions using Full MIP
    time_for_finding_feasible_solution = callback_data_binary.time_for_finding_feasible_solution
    try:
        last_row = log_df.iloc[-1]  # The last line is the current result
    except IndexError:
        last_row = {'BestBd': 0}  # If for some reason, such as infeasible, the record is empty, this line is to prevent code errors

    objective_value = model.objVal  # The objective value
    bestbd = model.ObjBound  # The BestBd 
    try:
        optimality_gap = model.MIPGap  # Call the optimality gap 
    except AttributeError:
        optimality_gap = -1  # If there is no optimality gap output 
    runtime = model.Runtime  # Record the runtime of model output by gurobi
    solution_w = [w[p].X for p in range(dim)]  # value of w 
    solution_b = b.X  # value of b
    
    z_integrality_vio = 0  # Numbers of the binary variables that are not integer, such as 0.99999...
    
    # Record the value of binary variables z
    for s in J_0_plus_0:
        solution_z_plus_0[s] = z_plus_0[s].X  # Record the value of the real (unfixed) binary variables in the model
        if z_plus_0[s].X > 0 and z_plus_0[s].X < 1:
            z_integrality_vio += 1  # If the value is in (0,1), record the violation
    for s in J_1_plus_0:
        solution_z_plus_0[s] = 1  # Record the value of the binary variables fixed as 1
    for s in J_2_plus_0:
        solution_z_plus_0[s] = 0  # Record the value of the binary variables fixed as 0

    for s in J_0_plus:
        solution_z_plus[s] = z_plus[s].X
        if z_plus[s].X > 0 and z_plus[s].X < 1:
                z_integrality_vio += 1
    for s in J_1_plus:
        solution_z_plus[s] = 1
    for s in J_2_plus:
        solution_z_plus[s] = 0

    for s in J_0_minus:
        solution_z_minus[s] = z_minus[s].X
        if z_minus[s].X > 0 and z_minus[s].X < 1:
                z_integrality_vio += 1
    for s in J_1_minus:
        solution_z_minus[s] = 1
    for s in J_2_minus:
        solution_z_minus[s] = 0

    os.makedirs(result_sub3dir + '/Solution', exist_ok=True)
    with open(result_sub3dir + '/Solution/' + f'solution_{postfix}.txt', 'a') as f:
        # Assumption 1: $\phi_{ij}$ is nonnegative near $\bar{x}$ for all $j\in {\cal J}^-_{i,0}(\bar x)$ and all $i=0,1,\cdots, I$ [this is the local sign invariance property of $\bar x$
        # and is vacuously valid if ${\cal J}^-_{i,0}(\bar x)$ is empty for all $i=0,1,\cdots,I$] 
        # In binary classification case, ${\cal J}^-_{i,0}(\bar{w},\bar{b}) = \{s|\bar{w}^\top X^s + \bar{b} = 0\}$, $\exists (w',b')=(\bar{w},\bar{b}-\Delta b) \in {\cal N}_{(\bar{w},\bar{b})}, \Delta b > 0$, 
        # such that ${w'}^{\top} X^s + b' < 0$. 
        # Thus, the number of data points not satisfying Assumption 1: $|\{s|\bar{w}^\top X^s + \bar{b} = 0\}|$
        violations_assumption_1 = 0  
        # Assumption 2: for some $\varepsilon>0$ (which must exist), a neighborhood ${\cal N}$ of $\bar x$ exists such that the equality below
        # $\psi^-_{ij}\mathbf{1}_{(-\varepsilon,\infty)}(\phi_{ij}(x)) = \psi^-_{ij}\mathbf{1}_{[0,\infty)}(\phi_{ij}(x)) $
        # holds for all $x\in {\cal N}$ and all $(i,j)$ with $j\not\in {\cal J}^-_{i,0}(\bar x)$ and $i=0,1,\cdots,I$.
        # In binary classification case, as long as $\phi_{is}(\bar{W},\bar{b})\not\in [-\varepsilon,0]$, there exists a neighborhood ${\cal N}$ of $(\bar{w},\bar{b})$ s.t. 
        # $\psi^-_{is}\mathbf{1}_{(-\varepsilon,\infty)}(\phi_{is}(w,b))=\psi^-_{is}\mathbf{1}_{[0,\infty)}(\phi_{is}(w,b))$ holds for all $(w,b)\in {\cal N}$ and all $(i,s)$ with 
        # $s\not\in {\cal J}^-_{i,0}(\bar{w},\bar{b})$ and $i=0,1,\cdots,I$.
        # Thus, the number of data points not satisfying Assumption 2: $|\{s|\bar{w}^\top X^s + \bar{b} \in [-\varepsilon,0)\}|$ (not [-\varepsilon,0] since $s\not\in {\cal J}^-_{i,0}(\bar{w},\bar{b})$)
        violations_assumption_2 = 0
        violations_feasibilitytol = {'phi^+_0':0,'phi^+':0,'phi^-':0}

        for s in positive_index:
            if (np.dot(solution_w, X_train[s]) + solution_b - 1 < 0) & (np.dot(solution_w, X_train[s]) + solution_b - 1>= -feasibility_tol):
                violations_feasibilitytol['phi^+_0'] += 1  # Record the violation of the feasibility tolerance, Numerical Issues
            if (np.dot(solution_w, X_train[s]) + solution_b < 0) & (np.dot(solution_w, X_train[s]) + solution_b >= -feasibility_tol):
                violations_feasibilitytol['phi^+'] += 1  # Record the violation of the feasibility tolerance, Numerical Issues
    
        for s in negative_index:
            if (-np.dot(solution_w, X_train[s]) - solution_b - 1 < 0) & (-np.dot(solution_w, X_train[s]) - solution_b - 1>= -feasibility_tol):
                violations_feasibilitytol['phi^+_0'] += 1
            if (-(np.dot(solution_w, X_train[s]) + solution_b) - epsilon < 0) & (-(np.dot(solution_w, X_train[s]) + solution_b)-epsilon >= -feasibility_tol):
                violations_feasibilitytol['phi^-'] += 1
            if np.dot(solution_w, X_train[s]) + solution_b == 0:
                violations_assumption_1 += 1
            if (np.dot(solution_w, X_train[s]) + solution_b < 0) & (np.dot(solution_w, X_train[s]) + solution_b >= -epsilon):
                violations_assumption_2 += 1

    objective_function_term = {
        'objective_value': objective_value, 
        'bestbd': bestbd,
        'optimality_gap': optimality_gap, 
        'gamma_in_obj': gamma.X, 
        'best_bd':last_row['BestBd'], 
        'runtime': runtime}
    
    solution = {'objective_value': objective_value, 'w': solution_w, 'b': solution_b, 'z_plus_0': solution_z_plus_0, 'z_plus': solution_z_plus, 'z_minus': solution_z_minus, 'gamma_0': gamma.X}
    violations_feasibilitytol = violations_feasibilitytol['phi^+_0'] + violations_feasibilitytol['phi^+'] + violations_feasibilitytol['phi^-']
    
    vio_feasibilitytol_indicator = False
    log_path = result_sub3dir + '/Logfile/'+f'log_{postfix}.txt'
    with open(log_path, 'r') as log_file:
        log_content = log_file.read()
    if "max constraint violation" in log_content:
        # If "max constraint violation" in log file, the max constraint violation exceed the feasibility tolerance of Gurobi Setting
        vio_feasibilitytol_indicator = True

    counts_result = {
        'num_integer_vars': num_integer_vars,
        'violations_assumption_1': violations_assumption_1, 'violations_assumption_2': violations_assumption_2, 
        'violations_feasibilitytol': violations_feasibilitytol,
        'number_of_integers': num_integer_vars,
        'z_integrality_vio': z_integrality_vio, 
        'vio_feasibilitytol_indicator': vio_feasibilitytol_indicator
    }  # counts_result: counts of integer variables in the model and counts of points that violate some conditions

    return objective_function_term, solution, counts_result
