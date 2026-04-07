import gurobipy as gp
import os.path
from utils import *
from callback import *
from parameter import *


class Model:
    """Gurobi MIP model for multi-class classification with precision constraints (full or partial)."""
    def __init__(self, X, y, class_restrict, epsilon, beta, model_type, ell, delta_plus, delta_minus, model_params,
                 model_dir, model_name, save_log=False, console_log=False):
        """
        Initialize MIP model.

        Args:
            X: Feature matrix
            y: Label vector
            class_restrict: Classes to apply precision constraints
            epsilon: Approximation parameter
            beta: Precision constraint parameters per class
            model_type: "full" or "partial"
            ell: PA piece set (None for un-PA-decomposed)
            delta_plus/delta_minus: Thresholds for fixing variables in partial model
            model_params: Gurobi solver parameters
            model_dir: Directory for logs
            model_name: Name for log file
        """
        self.X = X
        self.y = y
        self.class_restrict = class_restrict
        self.beta = beta
        self.epsilon = epsilon

        self.model_type = model_type  # full or partial
        self.ell = ell  # None if un-PA-decomposed model
        self.delta_plus = delta_plus  # None if full model
        self.delta_minus = delta_minus  # None if full model

        self.M = generate_bigM(self.X)
        self.N = range(self.X.shape[0])
        self.p = range(self.X.shape[1])
        self.I = range(len(np.unique(self.y)))
        self.class_index = {cls: np.where(y == cls)[0] for cls in self.I}  # use to select samples of certain class
        self.class_index_rest = {cls: np.where(y != cls)[0] for cls in self.I}

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
        self.model_state = 0  # -1: infeasible, 0: default, 1: feasible

    def model_optimize(self, callback):
        """Execute Gurobi optimization with error handling."""
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

    def add_basic_var(self, W_start, b_start):
        """Add common variables: W, b, z_plus_0, z_plus, z_minus, gamma (with warm start)."""
        self.var['b'] = self.model.addVars(self.I, lb=-10, ub=10, vtype=GRB.CONTINUOUS, name="b")
        # variables that used in both full and partial model
        self.var['W'] = self.model.addVars(self.I, self.p, lb=-10, ub=10, vtype=GRB.CONTINUOUS, name='W')
        if W_start is not None:
            for i in self.I:
                for d in self.p:
                    self.var['W'][i, d].setAttr(gp.GRB.Attr.Start, W_start[i, d])
        if b_start is not None:
            for i in self.I:
                self.var['b'][i].setAttr(gp.GRB.Attr.Start, b_start[i])

        self.var['W_l1'] = self.model.addVars(self.I, lb=0, ub=10, vtype=GRB.CONTINUOUS, name='W_l1')
        # Constraint for l1-norm of W
        self.model.addConstrs((self.var['W_l1'][i] == gp.norm(self.var['W'].select(i, '*'), 1)
                               for i in self.I), name="constr_W_l1_norm")

        self.var['z_plus_0'] = self.model.addVars(self.N, vtype=GRB.BINARY, name="z_plus_0")
        keys_z_plus = [(i, j) for i in self.class_restrict for j in self.class_index[i]]
        self.var['z_plus'] = self.model.addVars(keys_z_plus, vtype=GRB.BINARY, name="z_plus")

        keys_z_minus = [(i, j) for i in self.class_restrict for j in self.class_index_rest[i]]
        self.var['z_minus'] = self.model.addVars(keys_z_minus, vtype=GRB.BINARY, name="z_minus")

        self.var['gamma'] = self.model.addVars(self.class_restrict, lb=0, vtype=GRB.CONTINUOUS, name="gamma")
        for i in self.class_restrict:
            gamma_ub = max([self.beta[i] * len(self.class_index_rest[i]) + 1, 0.1 * len(self.class_index[i])])
            self.var['gamma'][i].setAttr(gp.GRB.Attr.UB, gamma_ub)

        if (W_start is not None) and (b_start is not None):
            z_plus_0_start, z_plus_start, z_minus_start \
                = generate_z_start(self.X, self.y, W_start, b_start, self.epsilon, self.class_restrict, self.ell)
            gamma_start \
                = generate_gamma_start(self.X, self.y, W_start, b_start, self.beta, self.epsilon, self.class_restrict,
                                       self.ell)
            for j in self.N:
                self.var['z_plus_0'][j].setAttr(gp.GRB.Attr.Start, z_plus_0_start[j])
            for i in self.class_restrict:
                for j in self.class_index[i]:
                    self.var['z_plus'][i, j].setAttr(gp.GRB.Attr.Start, z_plus_start[i, j])
                for j in self.class_index_rest[i]:
                    self.var['z_minus'][i, j].setAttr(gp.GRB.Attr.Start, z_minus_start[i, j])
                self.var['gamma'][i].setAttr(gp.GRB.Attr.Start, gamma_start[i])

    def add_unPA_var(self):
        """Add auxiliary variables (h_ijk, phi) for un-PA-decomposed models."""
        # variable that only used in un-PA-decomposed model. PA-decomposed model does not need extra variable
        keys_h_ijk = [(i, j, k) for i in self.class_restrict for j in self.class_index_rest[i]
                      for k in self.I if k != i]
        self.var['h_ijk'] = self.model.addVars(keys_h_ijk, lb=-self.M, vtype=GRB.CONTINUOUS, name="h_ijk")
        keys_phi = [(i, j) for i in self.class_restrict for j in self.class_index_rest[i]]
        self.var['phi'] = self.model.addVars(keys_phi, lb=-self.M, vtype=GRB.CONTINUOUS, name='phi')

    def gp_expression_h_mn(self, m, n, W, b, j):
        """Return (W[m]·X[j] + b[m]) - (W[n]·X[j] + b[n]) as Gurobi LinExpr."""
        expression = ((gp.quicksum(W[m, d] * self.X.iloc[j, d] for d in self.p) + b[m])
                      - (gp.quicksum(W[n, d] * self.X.iloc[j, d] for d in self.p) + b[n]))
        return expression

    def add_constr_z_0s_plus(self, j):
        """Add objective-related z_plus_0 constraints for sample j."""
        self.model.addConstrs(
            (self.gp_expression_h_mn(self.y[j], m, self.var['W'], self.var['b'], j) - 1 - FEASIBILITYTOL
             >= -self.M * (1 - self.var['z_plus_0'][j])
             for m in (m for m in self.I if m != self.y[j])), name=f"constr_Heaviside_IP_obj[{j}]")

    def add_constr_z_ms_plus(self, i, j):
        """Add z_plus[i,j] constraints for classes > i and < i."""
        self.model.addConstrs((self.gp_expression_h_mn(i, m, self.var['W'], self.var['b'], j) - FEASIBILITYTOL
                               >= -self.M * (1 - self.var['z_plus'][i, j])
                               for m in (m for m in self.I if m > i)), name=f"constr_Heaviside_IP_plus[{i},{j}]")
        self.model.addConstrs(
            (self.gp_expression_h_mn(i, n, self.var['W'], self.var['b'], j) - self.epsilon - FEASIBILITYTOL
             >= -self.M * (1 - self.var['z_plus'][i, j])
             for n in (n for n in self.I if n < i)), name=f"constr_Heaviside_IP_plus[{i},{j}]")

    def add_constr_z_ms_minus(self, i, j):
        """
        Add constraints for z_minus[i,j]。
        Handles both un-PA-decomposed and PA-decomposed model logic.
        """
        if self.ell is None:
            # un-PA-decomposed model
            self.model.addConstr(self.var['phi'][i, j] - self.epsilon - FEASIBILITYTOL
                                 >= -self.M * (1 - self.var['z_minus'][i, j]),
                                 name=f"constr_Heaviside_IP_minus[{i},{j}]")
            self.model.addConstrs((self.var['h_ijk'][i, j, m]
                                   == -self.gp_expression_h_mn(i, m, self.var['W'], self.var['b'], j)
                                   for m in (m for m in self.I if m > i)), name=f"constr_value_of_h[{i},{j}]")
            self.model.addConstrs((self.var['h_ijk'][i, j, n]
                                   == -self.gp_expression_h_mn(i, n, self.var['W'], self.var['b'], j) + self.epsilon
                                   for n in (n for n in self.I if n < i)), name=f"constr_value_of_h[{i},{j}]")
            self.model.addGenConstrMax(self.var['phi'][i, j],
                                       [self.var['h_ijk'][i, j, m] for m in (m for m in self.I if m != i)],
                                       constant=None, name=f"constr_max[{i},{j}]")
        else:
            if self.ell[i, j] < i:
                self.model.addConstr(
                    self.gp_expression_h_mn(self.ell[i, j], i, self.var['W'], self.var['b'], j) - FEASIBILITYTOL
                    >= -self.M * (1 - self.var['z_minus'][i, j]), name=f"constr_Heaviside_IP_minus[{i},{j}]")
            else:
                self.model.addConstr(
                    self.gp_expression_h_mn(self.ell[i, j], i, self.var['W'], self.var['b'], j)
                    - self.epsilon - FEASIBILITYTOL
                    >= -self.M * (1 - self.var['z_minus'][i, j]), name=f"constr_Heaviside_IP_minus[{i},{j}]")

    def add_full_constr_z_plus(self):
        """Add all z_plus constraints for full model."""
        for s in self.N:
            self.add_constr_z_0s_plus(s)
        for m in self.class_restrict:
            for s in self.class_index[m]:
                # Constraint for z_{ms}
                self.add_constr_z_ms_plus(m, s)

    def add_partial_constr_z_plus(self, phi):
        """Add z_plus constraints for partial model: fix binary vars where phi is outside delta thresholds."""
        for s in self.N:
            if phi['obj', s] <= -self.delta_minus['obj']:
                self.model.addConstr(self.var['z_plus_0'][s] == 0, name=f"constr_fix_z_plus_0[{s}]")
                self.var['z_plus_0'][s].setAttr(gp.GRB.Attr.Start, 0)
                self.num_int -= 1
            else:
                self.add_constr_z_0s_plus(s)
                if phi['obj', s] >= self.delta_plus['obj']:
                    self.num_int -= 1
                    self.model.addConstr(self.var['z_plus_0'][s] == 1, name=f"constr_fix_z_plus_0[{s}]")
                    self.var['z_plus_0'][s].setAttr(gp.GRB.Attr.Start, 1)
        for m in self.class_restrict:
            for s in self.class_index[m]:
                if phi[m, s] <= -self.delta_minus[m]:
                    self.num_int -= 1
                    self.model.addConstr(self.var['z_plus'][m, s] == 0, name=f"constr_fix_z_plus[{m},{s}]")
                    self.var['z_plus'][m, s].setAttr(gp.GRB.Attr.Start, 0)
                else:
                    self.add_constr_z_ms_plus(m, s)
                    if phi[m, s] >= self.delta_plus[m]:
                        self.num_int -= 1
                        self.model.addConstr(self.var['z_plus'][m, s] == 1, name=f"constr_fix_z_plus[{s}]")
                        self.var['z_plus'][m, s].setAttr(gp.GRB.Attr.Start, 1)

    def add_full_constr_z_minus(self):
        """Add z_minus constraints for full model."""
        for m in self.class_restrict:
            for s in self.class_index_rest[m]:
                self.add_constr_z_ms_minus(m, s)

    def add_partial_constr_z_minus(self, phi):
        """Add z_minus constraints for partial model: fix binary vars where phi is outside delta thresholds."""
        for m in self.class_restrict:
            for s in self.class_index_rest[m]:
                if -phi[m, s] - self.epsilon <= -self.delta_minus[m]:
                    self.num_int -= 1
                    self.model.addConstr(self.var['z_minus'][m, s] == 0, name=f"constr_fix_z_minus[{m},{s}]")
                    self.var['z_minus'][m, s].setAttr(gp.GRB.Attr.Start, 0)
                else:
                    self.add_constr_z_ms_minus(m, s)
                    if -phi[m, s] - self.epsilon >= self.delta_plus[m]:
                        self.num_int -= 1
                        self.model.addConstr(self.var['z_minus'][m, s] == 1, name=f"constr_fix_z_minus[{m},{s}]")
                        self.var['z_minus'][m, s].setAttr(gp.GRB.Attr.Start, 1)

    def add_precision_constr(self):
        """Add precision and recall (>=10%) constraints."""
        # Constraint for precision
        self.model.addConstrs(((1 - self.beta[i]) * gp.quicksum(self.var['z_plus'].select(i, '*'))
                               + self.beta[i] * gp.quicksum(self.var['z_minus'].select(i, '*'))
                               >= self.beta[i] * len(self.class_index_rest[i]) - self.var['gamma'][i]
                               for i in self.class_restrict), name="constr_precision")
        # recall >= 0.1
        self.model.addConstrs((gp.quicksum(self.var['z_plus'].select(i, '*')) + self.var['gamma'][i]
                               >= 0.1 * len(self.class_index[i])
                               for i in self.class_restrict), name="constr_recall")

    def add_acc_margin_obj(self):
        """Set objective: maximize accuracy outside margin - RHO * sum(gamma)."""
        obj = (gp.quicksum(self.var['z_plus_0'].select('*')) / self.X.shape[0]
               - RHO * gp.quicksum(self.var['gamma'].select('*')))
        self.model.setObjective(obj, GRB.MAXIMIZE)

    def formulate_model(self, W_start, b_start):
        """Build and add all variables, constraints, and objective to the model."""
        self.add_basic_var(W_start, b_start)
        if self.ell is None:
            self.add_unPA_var()
        self.model.update()
        self.num_int = sum(1 for v in self.model.getVars() if v.vType == gp.GRB.BINARY)
        if self.model_type == 'full':
            self.add_full_constr_z_plus()
            self.add_full_constr_z_minus()
        elif self.model_type == 'partial':
            phi = inner_function(self.X, self.y, self.class_restrict, self.epsilon, W_start, b_start, self.ell)
            self.add_partial_constr_z_plus(phi)
            self.add_partial_constr_z_minus(phi)
        self.add_precision_constr()

        self.add_acc_margin_obj()
        self.model.update()

    def solve_model(self):
        """Set solver params, configure callback, run optimization, extract variable values."""
        if self.model_type == 'full':
            time_limit = FULL_MODEL_TIME_LIMIT
            self.model.__dict__['last_time'] = 0
            self.model.__dict__['last_obj'] = -np.inf
            self.model.__dict__['final_improvement_time'] = 0
            self.model.__dict__['optimality_gap'] = -1
            self.model.__dict__['time_for_feasible'] = 0
            callback = full_model_callback
        elif self.model_type == 'partial':
            self.model.Params.LazyConstraints = 1
            if self.ell is None:
                time_limit = UNPA_PARTIAL_MODEL_TIME_LIMIT
                self.model.__dict__['unchanged_tolerance'] = UNPA_UNCHANGED_TOLERANCE
            else:
                time_limit = PARTIAL_MODEL_TIME_LIMIT
                self.model.__dict__['unchanged_tolerance'] = UNCHANGED_TOLERANCE
            self.model.__dict__['last_time'] = 0
            self.model.__dict__['last_obj'] = -np.inf
            self.model.__dict__['final_improvement_time'] = 0
            self.model.__dict__['time_for_feasible'] = 0
            callback = partial_model_callback

        self.model.setParam("Timelimit", time_limit)
        if self.save_log:
            os.makedirs(self.dir, exist_ok=True)
            log_file = os.path.join(self.dir, f"LogFile_{self.model_name}.txt")
            self.model.setParam('LogFile', log_file)
        if not self.console_log:
            self.model.setParam('LogToConsole', 0)
        self.model.update()
        self.model_optimize(callback)
        if self.model_state > 0:
            for key in self.var:
                if key not in ['h_ijk', 'phi']:
                    self.var_val[key] = {keys: self.var[key][keys].getAttr(GRB.Attr.X) for keys in self.var[key].keys()}

    def write_integrated_results(self, integrated_csv, X_test, y_test, precision_threshold=None, fold=None):
        """Compute train/test metrics and write results to CSV."""
        train_results = classification_metric(self.X, self.y, self.var_val['W'], self.var_val['b'])
        test_results = classification_metric(X_test, y_test, self.var_val['W'], self.var_val['b'])
        constr_precision = precision_in_constraint(self.y, self.var_val['z_plus'], self.var_val['z_minus'],
                                                   self.class_restrict)
        write_single_integrated_result(integrated_csv=integrated_csv, title='Full MIP', execution_time=None,
                                       model_time=self.model.Runtime,
                                       final_improvement_time=self.model.__dict__['final_improvement_time'],
                                       obj_val=self.model.objVal, train_acc_margined=train_results['acc_margined'],
                                       train_acc=train_results['accuracy'],
                                       train_precision=train_results['precision'], train_recall=train_results['recall'],
                                       test_acc_margined=test_results['acc_margined'],
                                       test_acc=test_results['accuracy'], test_precision=test_results['precision'],
                                       test_recall=test_results['recall'], precision_in_constr=constr_precision,
                                       opt_gap=self.model.__dict__['optimality_gap'], num_kl=None,
                                       W=self.var_val['W'], b=self.var_val['b'],
                                       precision_threshold=precision_threshold, fold=fold)
