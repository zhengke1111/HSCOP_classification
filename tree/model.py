import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
from collections import Counter
from callback import *
from utils import *
from parameter import *


class Model:
    """
    Gurobi MIP model for tree-based multi-class classification. With or without precision constraints (with: 'full', 'partial'; without: 'unconstrained_partial').
    """
    def __init__(self, X, y, depth, tau_0, class_restrict, epsilon, beta, model_type, ell, delta_plus, delta_minus, model_params,
                 model_dir, model_name, save_log=False, console_log=False):
        """
        Initialize MIP model

        Args:
            X: Feature matrix
            y: Label vector
            depth: Depth of the decision tree
            tau_0: \ell_0 norm constraint of the branching coefficient at each branching node k, a_k
            class_restrict: Class(es) to apply precision constraints
            epsilon: Approximation parameter
            beta: Precision constraint parameters per class
            model_type: 'full', 'partial', 'unconstrained_partial'
            ell: PA piece set (None for un-PA-decomposed methods or unconstrained model)
            delta_plus/delta_minus: Thresholds for fixing variables in partial model
            model_params: Gurobi solver parameters
            model_dir: Directory for logs
            model_name: Name for log file
            save_log (bool, optional): True if save the Gurobi log. Defaults to False.
            console_log (bool, optional): True if output the Gurobi solving log. Defaults to False.
        """
        
        self.X = X
        self.y = y
        self.depth = depth
        self.tau_0 = tau_0
        self.class_restrict = class_restrict
        self.beta = beta
        self.epsilon = epsilon

        self.model_type = model_type    # full, partial or unconstrained_partial
        self.ell = ell                  # None if un-PA-decomposed model
        self.delta_plus = delta_plus    # None if full model
        self.delta_minus = delta_minus  # None if full model

        self.sample_size = self.X.shape[0]                                      # Sample size N
        self.N = range(self.X.shape[0])                                         # range(N)
        self.p = range(self.X.shape[1])                                         # Dimension of features
        self.J = range(1, len(Counter(y)) + 1)                                  # [J]
        self.class_index = {cls: np.where(y == cls)[0] for cls in self.J}       # Select samples of each class

        self.branch_node = range(2**depth-1)
        self.leaf_node = range(2**depth)  
        self.A_L, self.A_R = ancestors(depth)                                                                         
        self.b_ub = TAU_1*abs(max(abs(self.X[s][i]) for i in self.p for s in self.N)) + 1   

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
        self.model_state = 0            # -1: infeasible, 0: default, 1: feasible

        self.M_z_plus_0 = 2*(TAU_1*abs(max(abs(X[s][i]) for i in self.p for s in self.N)) + 1)          # Big-M: M_{\xi} 
        # Note: z_plus_0 is \xi in the tree-based classification model
        self.M_z = 2*(TAU_1*abs(max(abs(X[s][i]) for i in self.p for s in self.N)) + 1)                 # Big-M: M_{z}
        self.M_eta = {cls: sum(1 for _ in self.class_index[cls]) for cls in self.J}                     # Big-M: M_{\eta, j} for j in J
        self.M_zeta = self.sample_size                                                                  # Big-M: M_{\zeta} is set as sample size N

        # active: active binary variables; fixed_as_1/0: binary variables that are fixed as 1/0 in the current partial problem
        self.z_plus_0_active, self.z_plus_0_fixed_as_1, self.z_plus_0_fixed_as_0 = {}, {}, {}           # Index sets for z_plus_0  
        self.z_plus_active, self.z_plus_fixed_as_1, self.z_plus_fixed_as_0 = {}, {}, {}                 # Index sets for z_plus
        self.z_minus_active, self.z_minus_fixed_as_1, self.z_minus_fixed_as_0 = {}, {}, {}              # Index sets for z_minus

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
                    
    def add_basic_var(self, a_start, b_start, c_start):
        """Add common variables (for both full and partial models): a, b, c, z_plus_0, z_plus, z_minus, gamma, eta, zeta, L"""
        # Branching coefficient (weight) a
        key_a = [(k, i) for k in self.branch_node for i in self.p]
        self.var['a'] = self.model.addVars(key_a, lb=-100, ub=100, vtype=GRB.CONTINUOUS, name="a")
        self.var['a_abs'] = self.model.addVars(key_a, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="a_abs")
        self.model.addConstrs((self.var['a'][k, i]<=self.var['a_abs'][k, i] for k in self.branch_node for i in self.p), name = 'a_l1_pos')
        self.model.addConstrs((-self.var['a'][k, i]<=self.var['a_abs'][k, i] for k in self.branch_node for i in self.p), name = 'a_l1_neg')
        self.model.addConstrs((gp.quicksum(self.var['a_abs'][k, i] for i in self.p) <= TAU_1 for k in self.branch_node), name = 'a_sum_l1')
        if a_start is not None:
            for k in self.branch_node:
                for i in self.p:
                    self.var['a'][k, i].setAttr(gp.GRB.Attr.Start, a_start[k, i])
        
        # \ell_0 norm constraint for a
        if self.tau_0 is not None:
            self.var['u'] = self.model.addVars(key_a, vtype = GRB.BINARY, name = 'u')
            self.model.addConstrs((self.var['a'][k, i] >= -TAU_1*(1-self.var['u'][k, i]) for k in self.branch_node for i in self.p), name = 'a_l0_pos')
            self.model.addConstrs((-self.var['a'][k, i] >= -TAU_1*(1-self.var['u'][k, i]) for k in self.branch_node for i in self.p), name = 'a_l0_neg')
            self.model.addConstrs((gp.quicksum((1-self.var['u'][k, i]) for i in self.p) <= self.tau_0 for k in self.branch_node), name = 'a_sum_l0')
        
        # Branching coefficient (bias) b
        self.var['b'] = self.model.addVars(self.branch_node, lb=-self.b_ub, ub=self.b_ub, vtype=GRB.CONTINUOUS, name="b")
        if b_start is not None:
            for k in self.branch_node:
                self.var['b'][k].setAttr(gp.GRB.Attr.Start, b_start[k])

        # Precision constraint feasibility ensurance \gamma; warm starts for \xi, z^+, z^-
        gamma_start, z_plus_0_start, z_plus_start, z_minus_start = calculate_gamma(self.X, self.y, a_start, b_start, c_start, self.depth, self.ell, self.beta, self.epsilon, self.class_restrict)
        # Numerator of precision \eta; denominator of precision \zeta; denominator of accuracy (out of margin 1): L
        eta_start, zeta_start, L_start = calculate_eta_zeta_L(self.X, self.y, c_start, self.depth, z_plus_0_start, z_plus_start, z_minus_start)

        key_z_plus_0 = [(s, t) for s in self.N for t in self.leaf_node]              # \xi_{st}
        self.var['z_plus_0'] = self.model.addVars(key_z_plus_0, vtype=GRB.BINARY, name='z_plus_0')
        for t in self.leaf_node:
            for s in self.N:
                if z_plus_0_start[s, t] == 0 or z_plus_0_start[s, t] == 1:
                        self.var['z_plus_0'][s, t].setAttr(gp.GRB.Attr.Start, z_plus_0_start[s, t])
        
        key_z_plus = [(s, t) for s in self.N for t in self.leaf_node]                # z^+_{st}
        self.var['z_plus'] = self.model.addVars(key_z_plus, vtype=GRB.BINARY, name="z_plus")
        for t in self.leaf_node:
            for s in self.N:
                if z_plus_start[s, t] == 0 or z_plus_start[s, t] == 1:
                        self.var['z_plus'][s, t].setAttr(gp.GRB.Attr.Start, z_plus_start[s, t])

        key_z_minus = [(s, t) for s in self.N for t in self.leaf_node]               # z^-_{st}
        self.var['z_minus'] = self.model.addVars(key_z_minus, vtype=GRB.BINARY, name="z_minus")
        for t in self.leaf_node:
            for s in self.N:
                if z_minus_start[s, t] == 0 or z_minus_start[s, t] == 1:
                        self.var['z_minus'][s, t].setAttr(gp.GRB.Attr.Start, z_minus_start[s, t])
        
        self.var['gamma'] = self.model.addVars(self.class_restrict, lb=0, vtype=GRB.CONTINUOUS, name="gamma")
        if gamma_start is not None:
            for j in self.class_restrict:
                self.var['gamma'][j].setAttr(gp.GRB.Attr.UB, gamma_start[j])

        # ================= Binary variable for class assignment, c =================
        key_c = [(j, t) for j in self.J for t in self.leaf_node]                                       # c_{jt}
        self.var['c'] = self.model.addVars(key_c, vtype=GRB.BINARY, name="c")
        if c_start is not None:
            for (j, t) in key_c:
                self.var['c'][j, t].setAttr(gp.GRB.Attr.Start, c_start[j, t])
        
        # ================= Auxiliary variable of the numerator of precision, \eta =================
        key_eta = [(j, t) for j in self.class_restrict for t in self.leaf_node]                         # \eta_{jt}
        self.var['eta'] = self.model.addVars(key_eta, lb=0, ub=self.sample_size, vtype=GRB.CONTINUOUS, name="eta")  
        if eta_start is not None:
            for (j, t) in key_eta:
                self.var['eta'][j, t].setAttr(gp.GRB.Attr.Start, eta_start[j, t])

        # ================= Auxiliary variable of the denominator of precision, \zeta =================
        key_zeta = [(j, t) for j in self.class_restrict for t in self.leaf_node]                        # \zeta_{jt}
        self.var['zeta'] = self.model.addVars(key_zeta, lb = 0, ub=self.sample_size, vtype=GRB.CONTINUOUS, name="zeta")
        if zeta_start is not None:
            for (j, t) in key_zeta:
                self.var['zeta'][j, t].setAttr(gp.GRB.Attr.Start, zeta_start[j, t])

        # ================= Auxiliary variable of the denominator of accuracy, L =================
        self.var['L'] = self.model.addVars(self.leaf_node, lb=0, ub=self.sample_size, vtype=GRB.CONTINUOUS, name="L")    # L_t
        if L_start is not None:
            for t in self.leaf_node:
                if t in L_start:
                    self.var['L'][t].setAttr(gp.GRB.Attr.Start, L_start[t])

    def add_unconstrained_basic_var(self, a_start, b_start, c_start):
        """Add variables (for unconstrained_partial): a, b, c, z_plus_0, L"""
        # Branching coefficient (weight) a
        key_a = [(k, i) for k in self.branch_node for i in self.p]
        self.var['a'] = self.model.addVars(key_a, lb=-100, ub=100, vtype=GRB.CONTINUOUS, name="a")
        self.var['a_abs'] = self.model.addVars(key_a, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="a_abs")
        self.model.addConstrs((self.var['a'][k, i]<=self.var['a_abs'][k, i] for k in self.branch_node for i in self.p), name = 'a_l1_pos')
        self.model.addConstrs((-self.var['a'][k, i]<=self.var['a_abs'][k, i] for k in self.branch_node for i in self.p), name = 'a_l1_neg')
        self.model.addConstrs((gp.quicksum(self.var['a_abs'][k, i] for i in self.p) <= TAU_1 for k in self.branch_node), name = 'a_sum_l1')
        if a_start is not None:
            for k in self.branch_node:
                for i in self.p:
                    self.var['a'][k, i].setAttr(gp.GRB.Attr.Start, a_start[k, i])

        # \ell_0 norm for a
        if self.tau_0 is not None:
            self.var['u'] = self.model.addVars(key_a, vtype = GRB.BINARY, name = 'u')
            self.model.addConstrs((self.var['a'][k, i] >= -TAU_1*(1-self.var['u'][k, i]) for k in self.branch_node for i in self.p), name = 'a_l0_pos')
            self.model.addConstrs((-self.var['a'][k, i] >= -TAU_1*(1-self.var['u'][k, i]) for k in self.branch_node for i in self.p), name = 'a_l0_neg')
            self.model.addConstrs((gp.quicksum((1-self.var['u'][k, i]) for i in self.p) <= self.tau_0 for k in self.branch_node), name = 'a_sum_l0')
        
        # Branching coefficient (bias) b 
        self.var['b'] = self.model.addVars(self.branch_node, lb=-self.b_ub, ub=self.b_ub, vtype=GRB.CONTINUOUS, name="b")
        if b_start is not None:
            for k in self.branch_node:
                self.var['b'][k].setAttr(gp.GRB.Attr.Start, b_start[k])

        # Warm starts for \xi,
        z_plus_0_start = calculate_z_plus_0(self.X, a_start, b_start, self.depth)
        L_start = calculate_L(self.X, self.y, c_start, self.depth, z_plus_0_start)

        key_z_plus_0 = [(s, t) for s in self.N for t in self.leaf_node]              
        self.var['z_plus_0'] = self.model.addVars(key_z_plus_0, vtype=GRB.BINARY, name='z_plus_0')
        for t in self.leaf_node:
            for s in self.N:
                if z_plus_0_start[s, t] == 0 or z_plus_0_start[s, t] == 1:
                        self.var['z_plus_0'][s, t].setAttr(gp.GRB.Attr.Start, z_plus_0_start[s, t])

        # ================= Binary variable for class assignment =================
        key_c = [(j, t) for j in self.J for t in self.leaf_node]                           # c_{jt}
        self.var['c'] = self.model.addVars(key_c, vtype=GRB.BINARY, name="c")
        if c_start is not None:
            for (j, t) in key_c:
                self.var['c'][j, t].setAttr(gp.GRB.Attr.Start, c_start[j, t])

        # ================= Auxiliary variable of the denominator of accuracy =================
        self.var['L'] = self.model.addVars(self.leaf_node, lb=0, ub=self.sample_size, vtype=GRB.CONTINUOUS, name="L")    # L_t
        if L_start is not None:
            for t in self.leaf_node:
                if t in L_start:
                    self.var['L'][t].setAttr(gp.GRB.Attr.Start, L_start[t])

    def add_full_constr_z_0st_plus(self):
        r"""For full model: add objective-related z_plus_0 (\xi) constraints for all (s,t)"""
        for t in self.leaf_node:
            for s in self.N:
                self.model.addConstrs((gp.quicksum(self.var['a'][k, i] * self.X[s][i] for i in self.p) - self.var['b'][k] - 1 - FEASIBILITYTOL >= - self.M_z_plus_0 * (1 - self.var['z_plus_0'][s, t])) for k in self.A_R[t])
                self.model.addConstrs((gp.quicksum(-self.var['a'][k, i] * self.X[s][i] for i in self.p) + self.var['b'][k] - 1 - FEASIBILITYTOL >= - self.M_z_plus_0 * (1 - self.var['z_plus_0'][s, t])) for k in self.A_L[t])

    def add_partial_constr_z_0st_plus(self, a_start, b_start):
        r"""For partial model: add objective-related z_plus_0 (\xi) constraints"""
        for t in self.leaf_node:
            self.z_plus_0_fixed_as_1[t], self.z_plus_0_active[t], self.z_plus_0_fixed_as_0[t] = [], [], []
            for s in self.N:
                # ================= Relaionship between \phi_{0;st} and \xi_{st} =================
                phi_0_plus = min([sum(a_start[k, i]*self.X[s][i] for i in self.p) - b_start[k] - 1 for k in self.A_R[t]]+ [-sum(a_start[k, i]*self.X[s][i] for i in self.p) + b_start[k] - 1 for k in self.A_L[t]])
                # =========== {\cal J}_{0,>} ==========
                if phi_0_plus > self.delta_plus[0]:
                    self.model.remove(self.var['z_plus_0'][s, t])
                    self.z_plus_0_fixed_as_1[t].append(s)
                    self.model.addConstrs((gp.quicksum(self.var['a'][k, i] * self.X[s][i] for i in self.p) - self.var['b'][k] - 1 - FEASIBILITYTOL >= 0) for k in self.A_R[t])
                    self.model.addConstrs((gp.quicksum(-self.var['a'][k, i] * self.X[s][i] for i in self.p) + self.var['b'][k] - 1 - FEASIBILITYTOL >= 0) for k in self.A_L[t])
                # =========== {\cal J}_{0,0} ==========
                elif phi_0_plus >= -self.delta_minus[0]:
                    self.z_plus_0_active[t].append(s)
                    self.model.addConstrs((gp.quicksum(self.var['a'][k, i] * self.X[s][i] for i in self.p) - self.var['b'][k] - 1 - FEASIBILITYTOL >= - self.M_z_plus_0 * (1 - self.var['z_plus_0'][s, t])) for k in self.A_R[t])
                    self.model.addConstrs((gp.quicksum(-self.var['a'][k, i] * self.X[s][i] for i in self.p) + self.var['b'][k] - 1 - FEASIBILITYTOL >= - self.M_z_plus_0 * (1 - self.var['z_plus_0'][s, t])) for k in self.A_L[t])
                # =========== {\cal J}_{0,<} ==========
                else:
                    self.model.remove(self.var['z_plus_0'][s, t])
                    self.z_plus_0_fixed_as_0[t].append(s)

    def add_full_constr_z_st_plus(self):
        """For full model: add precision-related z_plus constraints for all (s,t)"""
        for t in self.leaf_node:
            for s in self.N:
                self.model.addConstrs((gp.quicksum(self.var['a'][k, i] * self.X[s][i] for i in self.p) - self.var['b'][k] - FEASIBILITYTOL >= - self.M_z * (1- self.var['z_plus'][s, t])) for k in self.A_R[t])
                self.model.addConstrs((gp.quicksum(-self.var['a'][k, i] * self.X[s][i] for i in self.p) + self.var['b'][k] - self.epsilon - FEASIBILITYTOL >= - self.M_z * (1- self.var['z_plus'][s, t])) for k in self.A_L[t])

    def add_partial_constr_z_st_plus(self, a_start, b_start):
        """For partial model: add precision-related z_plus constraints"""
        for t in self.leaf_node:
            self.z_plus_fixed_as_1[t], self.z_plus_active[t], self.z_plus_fixed_as_0[t] = [], [], []
            for s in self.N:
                # ================= Relaionship between \phi_{1;st} and \z^+_{st} =================
                phi_plus = min([sum(a_start[k, i]*self.X[s][i] for i in self.p) - b_start[k] for k in self.A_R[t]]+ [-sum(a_start[k, i]*self.X[s][i] for i in self.p) + b_start[k] - self.epsilon for k in self.A_L[t]])
                # =========== {\cal J}_{1,>}(a,b) ==========
                if phi_plus > self.delta_plus[1]:
                    self.model.remove(self.var['z_plus'][s, t])
                    self.z_plus_fixed_as_1[t].append(s)
                    self.model.addConstrs((gp.quicksum(self.var['a'][k, i] * self.X[s][i] for i in self.p) - self.var['b'][k] - FEASIBILITYTOL >= 0) for k in self.A_R[t])
                    self.model.addConstrs((gp.quicksum(-self.var['a'][k, i] * self.X[s][i] for i in self.p) + self.var['b'][k] - self.epsilon - FEASIBILITYTOL>= 0) for k in self.A_L[t])
                # =========== {\cal J}_{1,0}(a,b) ==========
                elif phi_plus >= - self.delta_minus[1]:
                    self.z_plus_active[t].append(s)
                    self.model.addConstrs((gp.quicksum(self.var['a'][k, i] * self.X[s][i] for i in self.p) - self.var['b'][k] - FEASIBILITYTOL >= - self.M_z * (1- self.var['z_plus'][s, t])) for k in self.A_R[t])
                    self.model.addConstrs((gp.quicksum(-self.var['a'][k, i] * self.X[s][i] for i in self.p) + self.var['b'][k] - self.epsilon - FEASIBILITYTOL >= - self.M_z * (1- self.var['z_plus'][s, t])) for k in self.A_L[t])
                # =========== {\cal J}_{1,<}(a,b) ==========
                else:
                    self.model.remove(self.var['z_plus'][s, t])
                    self.z_plus_fixed_as_0[t].append(s)

    def add_full_constr_z_st_minus(self):
        """For full model: add precision-related z_minus constraints for all (s,t)"""
        key_phi_max = [(s, t) for s in self.N for t in self.leaf_node]
        phi_max = self.model.addVars(key_phi_max, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="phi_max")
        key_phi_minus_stk = [(s, t, k) for s in self.N for t in self.leaf_node for k in (self.A_L[t] + self.A_R[t])]
        phi_minus_stk = self.model.addVars(key_phi_minus_stk, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="phi_minus_stk")
        phi_list = {}
        
        for s in self.N:
            phi_list[s]={}
            for t in self.leaf_node:  
                phi_list[s][t] = []
                self.model.addConstrs((phi_minus_stk[s, t, k] == gp.quicksum(-self.var['a'][k, i] * self.X[s][i] for i in self.p) + self.var['b'][k] - self.epsilon) for k in self.A_R[t])
                self.model.addConstrs((phi_minus_stk[s, t, k] == gp.quicksum(self.var['a'][k, i] * self.X[s][i] for i in self.p) - self.var['b'][k]) for k in self.A_L[t])
                for k in (self.A_L[t] + self.A_R[t]):
                    phi_list[s][t].append(phi_minus_stk[s, t, k])
                self.model.addGenConstrMax(phi_max[s, t], phi_list[s][t], constant=None)
                self.model.addConstr(phi_max[s, t] - FEASIBILITYTOL >= - self.M_z*(1-self.var['z_minus'][s, t]))
                    
    def add_partial_constr_z_st_minus(self, a_start, b_start):
        """For partial model: add precision-related z_minus constraints for all (s,t)"""
        # self.ell is None: No P(iecewise)A(ffine) Decomposition 
        if self.ell is None:
            key_phi_max = [(s, t) for s in self.N for t in self.leaf_node]
            phi_max = self.model.addVars(key_phi_max, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="phi_max")
            key_phi_minus_stk = [(s, t, k) for s in self.N for t in self.leaf_node for k in (self.A_L[t] + self.A_R[t])]
            phi_minus_stk = self.model.addVars(key_phi_minus_stk, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="phi_minus_stk")
            phi_list = {}
            for t in self.leaf_node:
                self.z_minus_fixed_as_1[t], self.z_minus_active[t], self.z_minus_fixed_as_0[t] = [], [], []

            for s in self.N:
                phi_list[s]={}
                for t in self.leaf_node:  
                    phi_list[s][t] = []
                    underline_phi = max([sum(a_start[k, i] * self.X[s][i] for i in self.p) - b_start[k] for k in self.A_L[t]] + [sum(-a_start[k, i] * self.X[s][i] for i in self.p) + b_start[k] - self.epsilon for k in self.A_R[t]])
                    # ================= Relaionship between \underline{\phi}^{PA}_{st} and \z^-_{st} =================
                    if underline_phi < - self.delta_minus[1]:
                        self.model.remove(self.var['z_minus'][s, t])
                        self.model.remove(phi_max[s, t])
                        for k in (self.A_L[t] + self.A_R[t]):
                            self.model.remove(phi_minus_stk[s, t, k])
                        self.z_minus_fixed_as_0[t].append(s)
                    else:
                        self.model.addConstrs((phi_minus_stk[s, t, k] == gp.quicksum(-self.var['a'][k, i] * self.X[s][i] for i in self.p) + self.var['b'][k] - self.epsilon) for k in self.A_R[t])
                        self.model.addConstrs((phi_minus_stk[s, t, k] == gp.quicksum(self.var['a'][k, i] * self.X[s][i] for i in self.p) - self.var['b'][k]) for k in self.A_L[t])
                        for k in (self.A_L[t] + self.A_R[t]):
                            phi_list[s][t].append(phi_minus_stk[s, t, k])
                        self.model.addGenConstrMax(phi_max[s, t], phi_list[s][t], constant=None)
                        if underline_phi <= self.delta_plus[1]:
                            self.z_minus_active[t].append(s)
                            self.model.addConstr(phi_max[s, t] - FEASIBILITYTOL >= - self.M_z*(1-self.var['z_minus'][s, t]))
                        else:
                            self.model.remove(self.var['z_minus'][s, t])
                            self.z_minus_fixed_as_1[t].append(s)
                            self.model.addConstr(phi_max[s, t] - FEASIBILITYTOL >= 0) 
        # self.ell is not None: With P(iecewise)A(ffine) Decomposition, piece \ell
        else:
            for t in self.leaf_node:
                self.z_minus_fixed_as_1[t], self.z_minus_active[t], self.z_minus_fixed_as_0[t] = [], [], []
                for s in self.N:    
                    k = self.ell[s][t]
                    # ================= Relaionship between \phi^{\ell_{st};-}_{st} and \z^-_{st} =================
                    if k in self.A_L[t]:
                        phi_minus = sum(-a_start[k, i] * self.X[s][i] for i in self.p) + b_start[k]
                        if -phi_minus > self.delta_plus[1]:
                            self.model.remove(self.var['z_minus'][s, t])
                            self.z_minus_fixed_as_1[t].append(s)
                            self.model.addConstr(-(gp.quicksum(-self.var['a'][k, i] * self.X[s][i] for i in self.p) + self.var['b'][k]) - FEASIBILITYTOL >= 0)
                        elif -phi_minus >= -self.delta_minus[1]:                            
                            self.z_minus_active[t].append(s)
                            self.model.addConstr(-(gp.quicksum(-self.var['a'][k, i] * self.X[s][i] for i in self.p) + self.var['b'][k]) - FEASIBILITYTOL >= -self.M_z * (1- self.var['z_minus'][s,t]))
                        else:
                            self.model.remove(self.var['z_minus'][s, t])
                            self.z_minus_fixed_as_0[t].append(s)
                    elif k in self.A_R[t]:
                        phi_minus = sum(a_start[k, i] * self.X[s][i] for i in self.p) - b_start[k] + self.epsilon
                        if -phi_minus > self.delta_plus[1]:
                            self.model.remove(self.var['z_minus'][s, t])
                            self.z_minus_fixed_as_1[t].append(s)
                            self.model.addConstr(-(gp.quicksum(self.var['a'][k, i] * self.X[s][i] for i in self.p) - self.var['b'][k] + self.epsilon) - FEASIBILITYTOL >= 0)
                        elif -phi_minus >= -self.delta_minus[1]:
                            
                            self.z_minus_active[t].append(s)
                            self.model.addConstr(-(gp.quicksum(self.var['a'][k, i] * self.X[s][i] for i in self.p) - self.var['b'][k] + self.epsilon) - FEASIBILITYTOL >= - self.M_z * (1-self.var['z_minus'][s,t]))
                        else:
                            self.model.remove(self.var['z_minus'][s, t])
                            self.z_minus_fixed_as_0[t].append(s)

    def add_full_constr_L_and_c(self):
        for t in self.leaf_node:
            # ================= Constraint: L_t \le \sum_{s=1}^N \mathbf{1}\{j_s=j\}\xi_{st} + M_L*(1-c_{jt}) =================
            self.model.addConstrs((self.var['L'][t] <= gp.quicksum(self.var['z_plus_0'][s, t] for s in self.class_index[j]) + self.sample_size * (1-self.var['c'][j,t])) for j in self.J)
            # ================= Constraint: \sum_{j\in [J]} c_{jt} = 1 =================
            self.model.addConstr(1 == gp.quicksum(self.var['c'][j, t] for j in self.J)) 


    def add_partial_constr_L_and_c(self):
        for t in self.leaf_node:
            # ================= Constraint: L_t \le \sum_{s=1}^N \mathbf{1}\{j_s=j\}\xi_{st} + M_L*(1-c_{jt}) =================
            self.model.addConstrs((self.var['L'][t] <= gp.quicksum(self.var['z_plus_0'][s, t] for s in list(set(self.class_index[j]) & set(self.z_plus_0_active[t]))) + sum(1 for _ in list(set(self.class_index[j]) & set(self.z_plus_0_fixed_as_1[t]))) + self.sample_size * (1-self.var['c'][j,t])) for j in self.J)
            # ================= Constraint: \sum_{j\in [J]} c_{jt} = 1 =================
            self.model.addConstr(1 == gp.quicksum(self.var['c'][j, t] for j in self.J)) 

    def add_full_precision_constr(self):
        # ======================= Constraint: \eta_{jt} \le M_{\eta,j} * c_{jt} =======================
        self.model.addConstrs((self.var['eta'][j,t]<= self.M_eta[j]* self.var['c'][j,t]) for j in self.class_restrict for t in self.leaf_node)
        # ======================= Constraint: \eta_{jt} \le \sum_{s=1}^N \mathbf{1}\{j_s = j\} z^+_{st} + M_{\eta,j} * (1 - c_{jt}) =======================
        self.model.addConstrs((self.var['eta'][j,t]<=gp.quicksum(self.var['z_plus'][s,t] for s in self.class_index[j]) + self.M_eta[j]*(1-self.var['c'][j,t])) for j in self.class_restrict for t in self.leaf_node)
        # ======================= Constraint: \zeta_{jt} \ge -M_{\zeta} * c_{jt} =======================
        self.model.addConstrs((self.var['zeta'][j,t]>=-self.M_zeta*self.var['c'][j,t]) for j in self.class_restrict for t in self.leaf_node) 
        # ======================= Constraint: \zeta_{jt} \ge \sum_{s=1}^N (1-z^-_{st}) - M_{\zeta} * (1 - c_{jt}) =======================
        self.model.addConstrs((self.var['zeta'][j,t]>=gp.quicksum((1-self.var['z_minus'][s,t]) for s in self.N) - self.M_zeta*(1-self.var['c'][j,t])) for j in self.class_restrict for t in self.leaf_node) 
        # ======================= Constraint: \sum_{t\in {\cal T}_\ell} \eta_{jt} - \beta_{pj} * \sum_{t\in {\cal T}_\ell} \zeta_{jt} + \gamma_j \ge 0 =======================
        self.model.addConstrs((gp.quicksum(self.var['eta'][j,t] for t in self.leaf_node) - self.beta[j]* gp.quicksum(self.var['zeta'][j,t] for t in self.leaf_node) + self.var['gamma'][j] >= 0) for j in self.class_restrict)
        # ======================= Constraint: \sum_{t\in {\cal T}_\ell} \eta_{jt} + \tilde{\lambda} \gamma_j \ge 1 =======================
        self.model.addConstrs((gp.quicksum(self.var['eta'][j,t] for t in self.leaf_node) + 0.01 * self.var['gamma'][j]>=1) for j in self.class_restrict)

    def add_partial_precision_constr(self):
        # ======================= Constraint: \eta_{jt} \le M_{\eta,j} * c_{jt} =======================
        self.model.addConstrs((self.var['eta'][j,t]<= self.M_eta[j]* self.var['c'][j,t]) for j in self.class_restrict for t in self.leaf_node)
        # ======================= Constraint: \eta_{jt} \le \sum_{s=1}^N \mathbf{1}\{j_s = j\} z^+_{st} + M_{\eta,j} * (1 - c_{jt}) =======================
        self.model.addConstrs((self.var['eta'][j,t]<=gp.quicksum(self.var['z_plus'][s,t] for s in list(set(self.class_index[j]) & set(self.z_plus_active[t]))) + sum(1 for _ in list(set(self.class_index[j]) & set(self.z_plus_fixed_as_1[t]))) + self.M_eta[j]*(1-self.var['c'][j,t])) for j in self.class_restrict for t in self.leaf_node)
        # ======================= Constraint: \zeta_{jt} \ge -M_{\zeta} * c_{jt} =======================
        self.model.addConstrs((self.var['zeta'][j,t]>=-self.M_zeta*self.var['c'][j,t]) for j in self.class_restrict for t in self.leaf_node) 
        # ======================= Constraint: \zeta_{jt} \ge \sum_{s=1}^N (1-z^-_{st}) - M_{\zeta} * (1 - c_{jt}) =======================
        self.model.addConstrs((self.var['zeta'][j,t]>=gp.quicksum((1-self.var['z_minus'][s,t]) for s in self.z_minus_active[t]) + sum(1 for _ in self.z_minus_fixed_as_0[t]) - self.M_zeta*(1-self.var['c'][j,t])) for j in self.class_restrict for t in self.leaf_node) 
        # ======================= Constraint: \sum_{t\in {\cal T}_\ell} \eta_{jt} - \beta_{pj} * \sum_{t\in {\cal T}_\ell} \zeta_{jt} + \gamma_j \ge 0 =======================
        self.model.addConstrs((gp.quicksum(self.var['eta'][j,t] for t in self.leaf_node) - self.beta[j]* gp.quicksum(self.var['zeta'][j,t] for t in self.leaf_node) + self.var['gamma'][j] >= 0) for j in self.class_restrict)
        # ======================= Constraint: \sum_{t\in {\cal T}_\ell} \eta_{jt} + \tilde{\lambda} \gamma_j \ge 1 =======================
        self.model.addConstrs((gp.quicksum(self.var['eta'][j,t] for t in self.leaf_node) + 0.01 * self.var['gamma'][j]>=1) for j in self.class_restrict)

    def add_acc_margin_obj(self):
        """Set objective: maximize accuracy outside margin - RHO * sum(gamma)."""
        obj = 1/self.sample_size * gp.quicksum(self.var['L'][t] for t in self.leaf_node) - RHO*gp.quicksum(self.var['gamma'][j] for j in self.class_restrict)
        self.model.addConstr(obj <= 1, "manual_upper_bound")
        self.model.setObjective(obj,GRB.MAXIMIZE)

    def add_unconstrained_acc_margin_obj(self):
        """Set objective (for unconstrained problem, no \gamma): maximize accuracy outside margin"""
        obj = 1/self.sample_size * gp.quicksum(self.var['L'][t] for t in self.leaf_node)
        self.model.addConstr(obj <= 1, "manual_upper_bound")
        self.model.setObjective(obj,GRB.MAXIMIZE)

    def formulate_model(self, a_start, b_start, c_start):
        """Build and add all variables, constraints, and objective to the model."""
        if self.model_type == 'full':
            self.add_basic_var(a_start, b_start, c_start)
            self.model.update()
            self.add_full_constr_z_0st_plus()
            self.add_full_constr_z_st_plus()
            self.add_full_constr_z_st_minus()
            self.add_full_constr_L_and_c()
            self.add_full_precision_constr()
            self.add_acc_margin_obj()
            self.model.update()

        elif self.model_type == 'partial':
            self.add_basic_var(a_start, b_start, c_start)
            self.model.update()
            self.add_partial_constr_z_0st_plus(a_start, b_start)
            self.add_partial_constr_z_st_plus(a_start, b_start)
            self.add_partial_constr_z_st_minus(a_start, b_start)
            self.add_partial_constr_L_and_c()
            self.add_partial_precision_constr()
            self.add_acc_margin_obj()
            self.model.update()

        elif self.model_type == 'unconstrained_partial':
            self.add_unconstrained_basic_var(a_start, b_start, c_start)
            self.model.update()
            self.add_partial_constr_z_0st_plus(a_start, b_start)
            self.add_partial_constr_L_and_c()
            self.add_unconstrained_acc_margin_obj()
            self.model.update()


    
    def solve_model(self):
        """Set solver params, configure callback, run optimization, extract variable values."""
        if self.model_type == 'full':
            self.model.Params.LazyConstraints = 1
            time_limit = FULL_MODEL_TIME_LIMIT
            self.model.__dict__['last_time'] = 0
            self.model.__dict__['last_obj'] = -np.inf
            # final_improvement_time of 'Full MIP': output as 'time' column of 'Full MIP'
            self.model.__dict__['final_improvement_time'] = 0
            self.model.__dict__['optimality_gap'] = -1
            self.model.__dict__['time_for_feasible'] = 0
            self.model.__dict__['time_limit'] = time_limit
            callback = full_model_callback
        elif self.model_type == 'partial' or 'unconstrained_partial':
            self.model.Params.LazyConstraints = 1
            if self.ell is None and self.model_type == 'partial' :
                time_limit = UNPA_PARTIAL_MODEL_TIME_LIMIT
                self.model.__dict__['unchanged_tolerance'] = UNPA_UNCHANGED_TOLERANCE
            else:
                time_limit = PARTIAL_MODEL_TIME_LIMIT
                self.model.__dict__['unchanged_tolerance'] = UNCHANGED_TOLERANCE
            self.model.__dict__['last_time'] = 0
            self.model.__dict__['last_obj'] = -np.inf
            # final_improvement_time of partial problems: record the last improvement in callback, 
            # terminate if the difference of two consecutive final_improvement time >= UNCHANGED_TOLERANCE
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
                for key in ['a', 'b', 'c']:
                    self.var_val[key] = {keys: self.var[key][keys].getAttr(GRB.Attr.X) for keys in self.var[key].keys()}
            else:
                for key in ['a', 'b', 'c', 'gamma']:
                    self.var_val[key] = {keys: self.var[key][keys].getAttr(GRB.Attr.X) for keys in self.var[key].keys()}

    def write_integrated_results(self, dataset_results_csv, dataset, split, method, tau_0, beta, X_test, y_test):
        """Compute train/test metrics and write results to CSV."""
        train_results = evaluate_tree(self.X, self.y, self.var_val['a'], self.var_val['b'], self.var_val['c'], self.depth)
        test_results = evaluate_tree(X_test, y_test, self.var_val['a'], self.var_val['b'], self.var_val['c'], self.depth)
        write_single_integrated_result(results_csv=dataset_results_csv,
                                       dataset=dataset,
                                       depth=self.depth,
                                       split=split,
                                       method=method,
                                       tau_0=tau_0,
                                       beta=beta,
                                       objective_value=self.model.ObjVal,
                                       optimality_gap=self.model.MIPGap,
                                       time=self.model.__dict__['final_improvement_time'],
                                       actual_time=self.model.Runtime,
                                       gamma=next(iter(self.var_val['gamma'].values())),
                                       train_acc_margin=train_results['frac']['acc_margin'], 
                                       test_acc_margin=test_results['frac']['acc_margin'], 
                                       train_acc=train_results['frac']['acc'], 
                                       test_acc=test_results['frac']['acc'], 
                                       train_prec=train_results['frac'][f'prec{self.class_restrict[0]}'], 
                                       test_prec=test_results['frac'][f'prec{self.class_restrict[0]}']
                                       )