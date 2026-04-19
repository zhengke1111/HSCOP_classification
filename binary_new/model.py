from gurobipy import GRB
from parameter import *
from callback import *
from utils import *
import os, numpy as np, pandas as pd, gurobipy as gp


class Model:

    def __init__(self, X, y, epsilon, beta, model_type, delta_plus, delta_minus, model_params, 
                 model_dir, model_name, save_log=False, console_log=False):
        
        self.X = X
        self.y = y
        self.epsilon = epsilon
        self.beta = beta

        self.model_type = model_type
        self.delta_plus = delta_plus    # None if full model
        self.delta_minus = delta_minus  # None if full model

        self.sample_size = self.X.shape[0]
        self.N = range(self.X.shape[0])                                         # range(N)
        self.p = range(self.X.shape[1])  
        self.positive_index = np.where(y == 1)[0].tolist()
        self.negative_index = np.where(y == -1)[0].tolist() 

        self.M = 2*(TAU_1*max(abs(X[s][i]) for i in self.p for s in self.N) + 1)  # Big-M
        self.b_ub = TAU_1*max(abs(X[s][i]) for i in self.p for s in self.N) + 1 

        self.model = gp.Model()
        for para in model_params.keys():
            self.model.setParam(para, model_params[para])
        self.model.update()

        self.dir = model_dir
        self.model_name = model_name
        self.save_log = save_log
        self.console_log = console_log
        self.var = {}
        self.var_val = {}
        self.num_int = 0
        self.model_state = 0  

        self.z_plus_0_active, self.z_plus_0_fixed_as_1, self.z_plus_0_fixed_as_0 = [], [], []      # Index sets for z_plus_0  
        self.z_plus_active, self.z_plus_fixed_as_1, self.z_plus_fixed_as_0 = [], [], []                  # Index sets for z_plus
        self.z_minus_active, self.z_minus_fixed_as_1, self.z_minus_fixed_as_0 = [], [], []            # Index sets for z_minus

    def model_optimize(self, callback):
        """
        Execute Gurobi optimization with error handling.
        """
        try:
            self.model.optimize(callback)
            if self.model.SolCount > 0:
                self.model_state = 1  # Feasible solution found
            else:
                self.model_state = -1  # No feasible solution
                if self.save_log:
                    with open(self.model.Params.LogFile, 'a') as f:
                        print('======================================================================================\n'
                              'Cannot find feasible solution\n'
                              '======================================================================================\n',
                              file=f)
        except gp.GurobiError as e:
            self.model_state = -1
            if self.save_log:
                with open(self.model.Params.LogFile, 'a') as f:
                    print('======================================================================================\n'
                          f"Error code {e.errno}: {e}\n"
                          '======================================================================================\n',
                          file=f)
        except AttributeError:
            self.model_state = -1
            if self.save_log:
                with open(self.model.Params.LogFile, 'a') as f:
                    print('======================================================================================\n'
                          'Model not built yet.\n'
                          '======================================================================================\n',
                          file=f)
                    
    def add_basic_var(self, w_start, b_start):

        self.var['w'] = self.model.addVars(self.p, lb=-TAU_1, ub=TAU_1, vtype=GRB.CONTINUOUS, name="w")
        self.var['w_abs'] = self.model.addVars(self.p, lb=0, ub=TAU_1, vtype=GRB.CONTINUOUS, name="w_abs")
        self.model.addConstrs((self.var['w'][i] <= self.var['w_abs'][i] for i in self.p), name = "w_l1_pos")
        self.model.addConstrs((-self.var['w'][i] <= self.var['w_abs'][i] for i in self.p), name = "w_l1_neg")
        self.model.addConstr(gp.quicksum(self.var['w_abs'][i] for i in self.p) <= TAU_1, name = "w_sum_l1")
        
        if w_start is not None:
            for i in self.p:
                self.var['w'][i].setAttr(gp.GRB.Attr.Start, w_start[i])

        self.var['b'] = self.model.addVar(lb=-self.b_ub, ub=self.b_ub, vtype=GRB.CONTINUOUS, name="b")

        if b_start is not None:
            self.var['b'].setAttr(gp.GRB.Attr.Start, b_start)
        
        gamma_start, z_plus_0_start, z_plus_start, z_minus_start = calculate_gamma(self.X, self.y, w_start, b_start, self.beta, self.epsilon)

        self.var['gamma'] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="gamma")
        self.var['gamma'].setAttr(gp.GRB.Attr.UB, gamma_start)

        self.var['z_plus_0'] = self.model.addVars(self.N, vtype=GRB.BINARY, name="z_plus_0")
        for s in self.N:
            self.var['z_plus_0'][s].setAttr(gp.GRB.Attr.Start, z_plus_0_start[s])

        self.var['z_plus'] = self.model.addVars(self.positive_index, vtype=GRB.BINARY, name="z_plus")
        for s in self.positive_index:
            self.var['z_plus'][s].setAttr(gp.GRB.Attr.Start, z_plus_start[s])

        self.var['z_minus'] = self.model.addVars(self.negative_index, vtype=GRB.BINARY, name="z_minus")
        for s in self.negative_index:
            self.var['z_minus'][s].setAttr(gp.GRB.Attr.Start, z_minus_start[s])

    def add_full_constr_z_plus_0(self):
        self.model.addConstrs((gp.quicksum(self.var['w'][i] * self.X[s][i] for i in self.p) + self.var['b'] - 1 - FEASIBILITY_TOL >= - self.M * (1 - self.var['z_plus_0'][s])) for s in self.positive_index)
        self.model.addConstrs((gp.quicksum(-self.var['w'][i] * self.X[s][i] for i in self.p) - self.var['b'] - 1 - FEASIBILITY_TOL >= - self.M * (1 - self.var['z_plus_0'][s])) for s in self.negative_index)

    def add_partial_constr_z_plus_0(self, w_start, b_start):
        for s in self.positive_index:
            phi_0_positive = sum(w_start[i]* self.X[s][i] for i in self.p) + b_start - 1
            if phi_0_positive >= self.delta_plus[0]:
                self.z_plus_0_fixed_as_1.append(s)
                self.model.remove(self.var['z_plus_0'][s])
                self.model.addConstr(gp.quicksum(self.var['w'][i] * self.X[s][i] for i in self.p) + self.var['b'] - 1 - FEASIBILITY_TOL >= 0)  # - feasibility_tol: avoid gurobi treat -1e-9 as 0 in left-hand-side
            elif phi_0_positive <= -self.delta_minus[0]:  # Binary variables free, seen as 0 in objective function and constraints
                self.z_plus_0_fixed_as_0.append(s)
                self.model.remove(self.var['z_plus_0'][s])
            else:  # Binary variables in in-between sets
                self.z_plus_0_active.append(s)
                self.model.addConstr(gp.quicksum(self.var['w'][i] * self.X[s][i] for i in self.p) + self.var['b'] - 1 - FEASIBILITY_TOL >= - self.M * (1 - self.var['z_plus_0'][s]))

        for s in self.negative_index:
            phi_0_negative = -sum(w_start[i]* self.X[s][i] for i in self.p) - b_start - 1
            if phi_0_negative >= self.delta_plus[0]:  # Binary variables fixed as 1
                self.z_plus_0_fixed_as_1.append(s)
                self.model.remove(self.var['z_plus_0'][s])
                self.model.addConstr(-gp.quicksum(self.var['w'][i] * self.X[s][i] for i in self.p) - self.var['b'] - 1 - FEASIBILITY_TOL >= 0)
            elif phi_0_negative <= - self.delta_minus[0]:  # Binary variables free, seen as 0 in objective function and constraints
                self.z_plus_0_fixed_as_0.append(s)
                self.model.remove(self.var['z_plus_0'][s])
            else:  # Binary variables in in-between sets
                self.z_plus_0_active.append(s)
                self.model.addConstr(-gp.quicksum(self.var['w'][i] * self.X[s][i] for i in self.p) - self.var['b'] - 1 - FEASIBILITY_TOL >= - self.M * (1 - self.var['z_plus_0'][s]))

    def add_full_constr_z_plus(self):
        self.model.addConstrs((gp.quicksum(self.var['w'][i] * self.X[s][i] for i in self.p) + self.var['b'] - FEASIBILITY_TOL >= - self.M * (1 - self.var['z_plus'][s])) for s in self.positive_index)

    def add_partial_constr_z_plus(self, w_start, b_start):
        for s in self.positive_index:
            phi_1_positive = sum(w_start[i]*self.X[s][i] for i in self.p) + b_start 
            if phi_1_positive >= self.delta_plus[1]:
                self.z_plus_fixed_as_1.append(s)
                self.model.remove(self.var['z_plus'][s])
                self.model.addConstr(gp.quicksum(self.var['w'][i] * self.X[s][i] for i in self.p) + self.var['b'] - FEASIBILITY_TOL >= 0)  # - feasibility_tol: avoid gurobi treat -1e-9 as 0 in left-hand-side
            elif phi_1_positive <= -self.delta_minus[1]:  # Binary variables free, seen as 0 in objective function and constraints
                self.z_plus_fixed_as_0.append(s)
                self.model.remove(self.var['z_plus'][s])
            else:  # Binary variables in in-between sets
                self.z_plus_active.append(s)
                self.model.addConstr(gp.quicksum(self.var['w'][i] * self.X[s][i] for i in self.p) + self.var['b'] - FEASIBILITY_TOL >= - self.M * (1 - self.var['z_plus'][s]))

    def add_full_constr_z_minus(self):
        self.model.addConstrs((gp.quicksum(-self.var['w'][i] * self.X[s][i] for i in self.p) - self.var['b'] - self.epsilon - FEASIBILITY_TOL >= - self.M * (1 - self.var['z_minus'][s])) for s in self.negative_index)

    def add_partial_constr_z_minus(self, w_start, b_start):
        for s in self.negative_index:
            phi_1_negative = - sum(w_start[i]* self.X[s][i] for i in self.p) - b_start - self.epsilon
            if phi_1_negative >= self.delta_plus[1]:
                self.z_minus_fixed_as_1.append(s)
                self.model.remove(self.var['z_minus'][s])
                self.model.addConstr(gp.quicksum(-self.var['w'][i] * self.X[s][i] for i in self.p) - self.var['b'] - self.epsilon - FEASIBILITY_TOL >= 0)  # - feasibility_tol: avoid gurobi treat -1e-9 as 0 in left-hand-side
            elif phi_1_negative <= -self.delta_minus[1]:
                self.z_minus_fixed_as_0.append(s)
                self.model.remove(self.var['z_minus'][s])
            else:
                self.z_minus_active.append(s)
                self.model.addConstr(gp.quicksum(-self.var['w'][i] * self.X[s][i] for i in self.p) - self.var['b'] - self.epsilon - FEASIBILITY_TOL >= - self.M * (1 - self.var['z_minus'][s]))

    def add_full_precision_constr(self):
        self.model.addConstr((1-self.beta)*gp.quicksum(self.var['z_plus'][s] for s in self.positive_index) - self.beta*gp.quicksum((1 - self.var['z_minus'][s]) for s in self.negative_index) + self.var['gamma'] >= 0)
    
    def add_partial_precision_constr(self):
        self.model.addConstr((1-self.beta)*(gp.quicksum(self.var['z_plus'][s] for s in self.z_plus_active) + sum(1 for _ in self.z_plus_fixed_as_1)) - self.beta*(gp.quicksum((1 - self.var['z_minus'][s]) for s in self.z_minus_active) + sum(1 for _ in self.z_minus_fixed_as_0)) + self.var['gamma'] >= 0)

    def add_full_acc_margin(self):
        obj = (1/self.sample_size)*(gp.quicksum(self.var['z_plus_0'][s] for s in self.positive_index) + gp.quicksum(self.var['z_plus_0'][s] for s in self.negative_index)) - RHO * self.var['gamma'] 
        self.model.addConstr(obj <= 1, "manual_upper_bound")
        self.model.setObjective(obj, GRB.MAXIMIZE) 

    def add_partial_acc_margin(self):
        obj = (1/self.sample_size)*(gp.quicksum(self.var['z_plus_0'][s] for s in list(set(self.z_plus_0_active) & set(self.positive_index))) + gp.quicksum(self.var['z_plus_0'][s] for s in list(set(self.z_plus_0_active) & set(self.negative_index))) + sum(1 for _ in list(set(self.z_plus_0_fixed_as_1) & set(self.positive_index))) + sum(1 for _ in list(set(self.z_plus_0_fixed_as_1) & set(self.negative_index)))) - RHO * self.var['gamma'] 
        self.model.addConstr(obj <= 1, "manual_upper_bound")
        self.model.setObjective(obj, GRB.MAXIMIZE) 

    def formulate_model(self, w_start, b_start):
        if self.model_type == 'full':
            self.add_basic_var(w_start, b_start)
            self.model.update()
            self.add_full_constr_z_plus_0()
            self.add_full_constr_z_plus()
            self.add_full_constr_z_minus()
            self.add_full_precision_constr()
            self.add_full_acc_margin()
            self.model.update()

        elif self.model_type == 'partial':
            self.add_basic_var(w_start, b_start)
            self.model.update()
            self.add_partial_constr_z_plus_0(w_start, b_start)
            self.add_partial_constr_z_plus(w_start, b_start)
            self.add_partial_constr_z_minus(w_start, b_start)
            self.add_partial_precision_constr()
            self.add_partial_acc_margin()
            self.model.update()

    def solve_model(self):
        if self.model_type == 'full':
            time_limit = FULL_MODEL_TIME_LIMIT
            self.model.__dict__['last_time'] = 0
            self.model.__dict__['last_obj'] = -np.inf
            # final_improvement_time of 'Full MIP': output as 'time' column of 'Full MIP'
            self.model.__dict__['final_improvement_time'] = 0
            self.model.__dict__['optimality_gap'] = -1
            self.model.__dict__['time_for_feasible'] = 0
            self.model.__dict__['time_limit'] = time_limit
            callback = full_model_callback
        elif self.model_type == 'partial':
            time_limit = PARTIAL_MODEL_TIME_LIMIT
            self.model.__dict__['unchanged_tolerance'] = UNCHANGED_TOLERANCE
            self.model.__dict__['last_time'] = 0
            self.model.__dict__['last_obj'] = -np.inf
            self.model.__dict__['final_improvement_time'] = 0
            self.model.__dict__['time_for_feasible'] = 0
            self.model.__dict__['time_limit'] = time_limit
            callback = partial_model_callback

        self.model.setParam("Timelimit", time_limit)

        if self.save_log:
            os.makedirs(self.dir, exist_ok=True)
            log_file = os.path.join(self.dir, f"LogFile_{self.model_name}.txt")
            self.model.setParam('LogFile', log_file)
        if not self.console_log:
            self.model.setParam('LogToConsole', 0)
        self.model.update()
        self.model._vars = self.model.getVars()
        self.model_optimize(callback)

        if self.model_state > 0:
            if self.model_type == 'unconstrained_partial':
                self.var_val['w'] = {k: self.var['w'][k].X for k in self.var['w'].keys()}
                self.var_val['b'] = self.var['b'].X

            else:
                self.var_val['w'] = {k: self.var['w'][k].X for k in self.var['w'].keys()}
                self.var_val['b'] = self.var['b'].X
                self.var_val['gamma'] = self.var['gamma'].X

    def write_integrated_results(self, dataset_results_csv, dataset, split, method, beta, X_test, y_test):
        train_results = evaluate_binary(self.X, self.y, self.var_val['w'], self.var_val['b'])
        test_results = evaluate_binary(X_test, y_test, self.var_val['w'], self.var_val['b'])

        write_single_integrated_result(results_csv=dataset_results_csv,
                                       dataset=dataset,
                                       split=split,
                                       method=method,
                                       beta=beta,
                                       objective_value=self.model.ObjVal,
                                       optimality_gap=self.model.MIPGap,
                                       time=self.model.__dict__['final_improvement_time'],
                                       actual_time=self.model.Runtime,
                                       gamma=self.var_val['gamma'],
                                       train_acc_margin=train_results['acc_margin'],
                                       test_acc_margin=test_results['acc_margin'],
                                       train_acc=train_results['accuracy'],
                                       test_acc=test_results['accuracy'],
                                       train_prec=train_results['precision'],
                                       test_prec=test_results['precision'],
                                       train_recall=train_results['recall'],
                                       test_recall=test_results['recall']
                                       )


    