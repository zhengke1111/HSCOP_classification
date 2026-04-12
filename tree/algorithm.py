from model import *
import utils
import csv
import time
import numpy as np
from typing import Dict, Optional, Union

class PIP:
    def __init__(self, X_train, y_train, dataset, depth, tau_0, class_restrict, epsilon, beta, model_params, ell, algorithm_params, alg_dir,
                 save_log=False, console_log=False):
        self.X_train = X_train
        self.y_train = y_train
        self.dataset = dataset
        self.depth = depth
        self.tau_0 = tau_0
        self.class_restrict = class_restrict
        self.epsilon = epsilon
        self.beta = beta
        self.ell = ell
        self.model_params = model_params

        # Extract iteration control parameters
        self.unchanged_iter = algorithm_params['iteration']['unchanged_iter']
        self.max_iter = algorithm_params['iteration']['max_iter']

        # Extract ratio update parameters
        self.max_ratio = algorithm_params['ratio']['max_ratio']
        self.base_ratio = algorithm_params['ratio']['base_ratio'][self.depth][dataset]
        self.min_ratio = self.base_ratio
        self.change_ratio = algorithm_params['ratio']['change_ratio']

        # Initialize runtime tracking and model storage
        self.execution_time_list = []
        self.model_dict: Dict[int, Optional[Union[Model, Dict[int, Model]]]] = {
            key: None for key in range(self.max_iter)
        }
        self.algorithm_state = 0  # 0: initial, 1: fixed-piece success, 2: arbitrary-4 success, 3: arbitrary-1 success
        self.alg_dir = alg_dir
        self.save_log = save_log
        self.console_log = console_log
        if self.save_log:
            os.makedirs(self.alg_dir, exist_ok=True)  # Create directory if not exists

        # Initialize final output container
        self.output = {
            'obj_val': -np.inf,
            'a': None,  
            'b': None,  
            'c': None
        }
        
    def ratio_update_rule(self, ratio, obj_val, obj_val_old, iter_unchanged):
        if -1e-5 <= obj_val - obj_val_old <= 1e-5:
            new_ratio = min(ratio + self.change_ratio, self.max_ratio) 
            iter_unchanged_new = iter_unchanged + 1
        else:
            new_ratio = max(ratio - self.change_ratio, self.min_ratio)
            iter_unchanged_new = 0
        return new_ratio, iter_unchanged_new

    def formulate_and_solve_partial_model(self, model_dir, delta_1, delta_2, ell, iter_model_name, a_start, b_start, c_start):
        """Build and solve a partial MIP model with given delta thresholds and piece set."""
        # Initialize partial model with problem parameters
        partial_model = Model(
            X=self.X_train,
            y=self.y_train,
            depth=self.depth,
            tau_0=self.tau_0,
            class_restrict=self.class_restrict,
            epsilon=self.epsilon,
            beta=self.beta,
            model_type='partial',
            ell=ell,
            delta_plus=delta_1,
            delta_minus=delta_2,
            model_params=self.model_params,
            model_dir=model_dir,
            model_name=iter_model_name,
            save_log=self.save_log,
            console_log=self.console_log
        )

        # Formulate the model with warm start values and solve
        partial_model.formulate_model(a_start, b_start, c_start)
        partial_model.solve_model()
        return partial_model
    
    def formulate_and_solve_unconstrained_partial_model(self, model_dir, delta_1, delta_2, iter_model_name, a_start, b_start, c_start):
        """Build and solve a partial MIP model with given delta thresholds and piece set."""
        # Initialize partial model with problem parameters
        partial_model = Model(
            X=self.X_train,
            y=self.y_train,
            depth=self.depth,
            tau_0=self.tau_0,
            class_restrict=self.class_restrict,
            epsilon=self.epsilon,
            beta=self.beta,
            model_type='unconstrained_partial',
            ell=None,
            delta_plus=delta_1,
            delta_minus=delta_2,
            model_params=self.model_params,
            model_dir=model_dir,
            model_name=iter_model_name,
            save_log=self.save_log,
            console_log=self.console_log
        )

        # Formulate the model with warm start values and solve
        partial_model.formulate_model(a_start, b_start, c_start)
        partial_model.solve_model()
        return partial_model
    
    def iteration_process_fixed_piece(self, alg_name, iteration, start, ratio, a_start, b_start, c_start):
        delta_1, delta_2 = calculate_delta(self.X_train, a_start, b_start, self.depth, self.ell, self.epsilon, ratio)

        iter_model_name = alg_name + f"_iter_{iteration}"

        partial_model = self.formulate_and_solve_partial_model(self.alg_dir, delta_1, delta_2, self.ell, iter_model_name, a_start, b_start, c_start)

        self.execution_time_list.append(time.time() - start)
        return partial_model

    def iteration_process_enhanced_arbitrary_4(self, alg_name, iteration, start, ratio, a_start, b_start, c_start):
        if 'ell_comb_list' not in self.__dict__:
            self.__dict__['ell_comb_list'] = []  # Track piece combination lengths
            self.__dict__['a_list'] = []  # Track weights per iteration
            self.__dict__['b_list'] = []  # Track biases per iteration
            self.__dict__['c_list'] = []  # Track biases per iteration
            self.__dict__['obj_list'] = []  # Track objective values per iteration

        ell_set_index, multi_piece = utils.generate_M(self.X_train, a_start, b_start, self.depth, self.epsilon, ratio, ENHANCED_SIZE)
        ell_comb = utils.generate_combinations(ell_set_index)  # Generate {\cal L}^4_{st}(a,b)
        self.__dict__['ell_comb_list'].append(multi_piece)

        sub_prob_counter = 0
        model_dict = {}
        execution_time_iteration = {}
        for ell in ell_comb:
            sub_start = time.time()
            # Define unique model name for subproblem
            iter_model_name = alg_name + f"_iter_{iteration}" + f"_sub_prob_{sub_prob_counter}"
            # Create subdirectory for current iteration
            model_dir = os.path.join(self.alg_dir, f"iter_{iteration}")
            if self.save_log:
                os.makedirs(model_dir, exist_ok=True)

            # Formulate and solve partial model for current piece combination
            delta_1, delta_2 = calculate_delta(self.X_train, a_start, b_start, self.depth, ell, self.epsilon, ratio)
            partial_model = self.formulate_and_solve_partial_model(model_dir, delta_1, delta_2, ell, iter_model_name, a_start, b_start, c_start)

            # Store solved model and execution time
            model_dict.update({sub_prob_counter: partial_model})
            execution_time_iteration.update({sub_prob_counter: time.time() - sub_start})
            sub_prob_counter += 1

        # Record total iteration execution time
        execution_time_iteration.update({'total': time.time() - start})
        self.execution_time_list.append(execution_time_iteration)
        return model_dict

    def iteration_process_enhanced_arbitrary_1(self, alg_name, iteration, start, ratio, a_start, b_start, c_start):
        if 'ell_comb_list' not in self.__dict__:
            self.__dict__['ell_comb_list'] = []  # Track piece combination lengths
        
        ell = utils.generate_random_combination(self.X_train, a_start, b_start, self.depth, self.epsilon) 
        self.__dict__['ell_comb_list'].append(ell) #
        delta_1, delta_2 = calculate_delta(self.X_train, a_start, b_start, self.depth, ell, self.epsilon, ratio)

        # Define unique model name for current iteration
        iter_model_name = alg_name + f"_iter_{iteration}"
        model_dir = self.alg_dir

        # Formulate and solve partial model
        partial_model = self.formulate_and_solve_partial_model(model_dir, delta_1, delta_2, ell, iter_model_name, a_start, b_start, c_start)

        # Record iteration execution time
        self.execution_time_list.append(time.time() - start)
        return partial_model
    
    def iteration_process_unconstrained(self, alg_name, iteration, start, ratio, a_start, b_start, c_start):
        delta_1, delta_2 = calculate_delta(self.X_train, a_start, b_start, self.depth, None, None, ratio)

        # Define unique model name for current iteration
        iter_model_name = alg_name + f"_iter_{iteration}"
        model_dir = self.alg_dir

        # Formulate and solve partial model
        partial_model = self.formulate_and_solve_unconstrained_partial_model(model_dir, delta_1, delta_2, iter_model_name, a_start, b_start, c_start)

        # Record iteration execution time
        self.execution_time_list.append(time.time() - start)
        return partial_model
    
    def main_computation(self, iteration_process, alg_name, a, b, c):
        gamma_start, z_plus_0_start, z_plus_start, z_minus_start = calculate_gamma(self.X_train, self.y_train, a, b, c, self.depth, self.ell, self.beta, self.epsilon, self.class_restrict)
        eta_start, zeta_start, L_start = calculate_eta_zeta_L(self.X_train, self.y_train, c, self.depth, z_plus_0_start, z_plus_start, z_minus_start)
        obj_val = (sum(L_t for L_t in L_start.values()) / self.X_train.shape[0] - RHO * (sum(gamma for gamma in gamma_start.values())))

        a_start = a
        b_start = b
        c_start = c
        iter_unchanged = 0  # Counter for consecutive unchanged iterations
        ratio = self.base_ratio  # Initial adaptive ratio

        for iteration in range(self.max_iter):
            start = time.time()
            obj_val_old = obj_val  # Store previous objective value

            solution = iteration_process(alg_name, iteration, start, ratio, a_start, b_start, c_start)

            # Handle multi-subproblem solution (arbitrary-4)
            if isinstance(solution, dict):
                obj_benchmark = obj_val_old
                a_benchmark = a_start
                b_benchmark = b_start
                c_benchmark = c_start
                
                self.model_dict[iteration] = {}
                # Evaluate all subproblems and select best solution
                for sub_prob, model in solution.items():
                    if model.model_state == 1:  # Check if model solved successfully
                        obj_val_sub_prob = model.model.objVal
                        # Update benchmark if current subproblem is better
                        if obj_val_sub_prob > obj_benchmark:
                            obj_benchmark = obj_val_sub_prob
                            a_benchmark = model.var_val['a']
                            b_benchmark = model.var_val['b']
                            c_benchmark = model.var_val['c']
                        self.model_dict[iteration].update({sub_prob: model})
                    else:
                        # Mark algorithm as failed if any subproblem fails
                        self.algorithm_state = - iteration - 1
                        break

                # Update parameters if all subproblems succeeded
                if self.algorithm_state >= 0:
                    self.__dict__['obj_list'].append(obj_benchmark)
                    self.__dict__['a_list'].append(a_benchmark)
                    self.__dict__['b_list'].append(b_benchmark)
                    self.__dict__['c_list'].append(b_benchmark)
                    
                    # Update warm start parameters for next iteration
                    a_start = a_benchmark
                    b_start = b_benchmark
                    c_start = c_benchmark
                    obj_val = obj_benchmark

                    # Update adaptive ratio and unchanged iteration counter
                    ratio, iter_unchanged = self.ratio_update_rule(ratio, obj_val, obj_val_old, iter_unchanged)
                else:
                    break  # Exit loop if algorithm failed
            
            # Handle single-model solution (fixed-piece/arbitrary-1)
            else:
                if solution.model_state == 1:  # Check if model solved successfully
                    # Update objective value and warm start parameters
                    obj_val = solution.model.objVal
                    a_start = solution.var_val['a']
                    b_start = solution.var_val['b']
                    c_start = solution.var_val['c']

                    # Update adaptive ratio and unchanged iteration counter
                    ratio, iter_unchanged = self.ratio_update_rule(ratio, obj_val, obj_val_old, iter_unchanged)
                    self.model_dict[iteration] = solution
                else:
                    # Mark algorithm as failed if model solving failed
                    self.algorithm_state = -iteration - 1
                    break

            # Early stop if objective value unchanged for max allowed iterations
            if iter_unchanged >= self.unchanged_iter:
                self.max_iter = iteration + 1
                break

        if self.algorithm_state >= 0:
            # Set algorithm state based on strategy type
            if 'ell_comb_list' in self.__dict__:
                if isinstance(self.model_dict[0], dict):
                    self.algorithm_state = 2  # arbitrary-4 success
                else:
                    self.algorithm_state = 3  # arbitrary-1 success
            else:
                self.algorithm_state = 1  # fixed-piece success

            # Extract final results based on algorithm type
            if self.algorithm_state == 1 or self.algorithm_state == 3:
                # Fixed-piece or arbitrary-1 (single model per iteration)
                final_model = self.model_dict[self.max_iter - 1]
                self.output['obj_val'] = final_model.model.objVal
                self.output['a'] = final_model.var_val['a']
                self.output['b'] = final_model.var_val['b']
                self.output['c'] = final_model.var_val['c']
                
            else:
                # Arbitrary-4 (multiple models per iteration)
                self.output['obj_val'] = self.__dict__['obj_list'][-1]
                self.output['a'] = self.__dict__['a_list'][-1]
                self.output['b'] = self.__dict__['b_list'][-1]
                self.output['c'] = self.__dict__['c_list'][-1]

    def main_computation_unconstrained(self, iteration_process, alg_name, a, b, c):
        z_plus_0_start = calculate_z_plus_0(self.X_train, a, b, self.depth)
        L_start = calculate_L(self.X_train, self.y_train, c, self.depth, z_plus_0_start)
        obj_val = (sum(L_t for L_t in L_start.values()) / self.X_train.shape[0])

        a_start = a
        b_start = b
        c_start = c
        iter_unchanged = 0  # Counter for consecutive unchanged iterations
        ratio = self.base_ratio  # Initial adaptive ratio

        for iteration in range(self.max_iter):
            start = time.time()
            obj_val_old = obj_val  # Store previous objective value

            solution = iteration_process(alg_name, iteration, start, ratio, a_start, b_start, c_start)

            if solution.model_state == 1:  # Check if model solved successfully
                # Update objective value and warm start parameters
                obj_val = solution.model.objVal
                a_start = solution.var_val['a']
                b_start = solution.var_val['b']
                c_start = solution.var_val['c']

                # Update adaptive ratio and unchanged iteration counter
                ratio, iter_unchanged = self.ratio_update_rule(ratio, obj_val, obj_val_old, iter_unchanged)
                self.model_dict[iteration] = solution
            else:
                # Mark algorithm as failed if model solving failed
                self.algorithm_state = -iteration - 1
                break

            # Early stop if objective value unchanged for max allowed iterations
            if iter_unchanged >= self.unchanged_iter:
                self.max_iter = iteration + 1
                break

        if self.algorithm_state >= 0:
            self.algorithm_state = 1  # fixed-piece success
            final_model = self.model_dict[self.max_iter - 1]
            self.output['obj_val'] = final_model.model.objVal
            self.output['a'] = final_model.var_val['a']
            self.output['b'] = final_model.var_val['b']
            self.output['c'] = final_model.var_val['c']
                
            
                
    def write_integrated_results(self, integrated_csv, alg_name, X_test, y_test, precision_threshold=None, fold=None):
        pass # TODO



class IterativeShrinkage:
    def __init__(self, X_train, y_train, dataset, depth, tau_0, class_restrict, beta, model_params, pip_params, alg_dir,
                 save_log=False, console_log=False):

        self.X_train = X_train
        self.y_train = y_train
        self.dataset = dataset
        self.depth = depth
        self.tau_0 = tau_0
        self.class_restrict = class_restrict
        self.beta = beta 
        self.model_params = model_params

        # Shrinkage control parameters
        self.max_outer_iter = SHRINKAGE_MAX_OUT_ITER
        self.pip_params = pip_params
        self.alg_dir = alg_dir
        self.save_log = save_log
        self.console_log = console_log

        # Initialize runtime tracking and PIP instance storage
        self.execution_time_list = []
        self.pip_alg_dict: Dict[int, Optional[Union[PIP, Dict[int, PIP]]]] = {
            key: None for key in range(self.max_outer_iter)
        }
        self.algorithm_state = 0  # 0: initial, positive: success, negative: failed

        # Initialize final output container
        self.output: Dict[str, Optional[Union[dict, float]]] = {
            'obj_val': -np.inf,
            'a': None,
            'b': None,
            'c': None
        }

    def iteration_process_enhanced_arbitrary_4(self, alg_name, epsilon, iteration, start, a_start, b_start, c_start):

        if 'ell_comb_list' not in self.__dict__:
            self.__dict__['ell_comb_list'] = []
            self.__dict__['a_list'] = []
            self.__dict__['b_list'] = []
            self.__dict__['c_list'] = []
            self.__dict__['obj_list'] = []

        ell_set_index, multi_piece = utils.generate_M(self.X_train, a_start, b_start, self.depth, epsilon[iteration], self.pip_params['ratio']['base_ratio'], ENHANCED_SIZE)
        ell_comb = utils.generate_combinations(ell_set_index)  # Generate {\cal L}^4_{st}(a,b)
        self.__dict__['ell_comb_list'].append(multi_piece)

        sub_prob_counter = 0
        pip_dict = {}
        execution_time_iteration = {}
        for ell in ell_comb:
            sub_start = time.time()
            # Create subdirectory for current outer iteration/subproblem
            pip_alg_dir = os.path.join(
                self.alg_dir, f"outer_iter_{iteration}", f"sub_prob_{sub_prob_counter}"
            )

            # Initialize PIP instance for current piece combination and epsilon
            pip = PIP(
                self.X_train, self.y_train, self.dataset, self.depth, self.tau_0, self.class_restrict, epsilon[iteration], self.beta,
                self.model_params, ell, self.pip_params, pip_alg_dir,
                self.save_log, self.console_log
            )

            # Run PIP main computation with fixed-piece strategy
            pip.main_computation(pip.iteration_process_fixed_piece, alg_name, a_start, b_start, c_start)

            # Store PIP instance and execution time
            pip_dict.update({sub_prob_counter: pip})
            execution_time_iteration.update({sub_prob_counter: time.time() - sub_start})
            sub_prob_counter += 1

        # Record total outer iteration execution time
        execution_time_iteration.update({'total': time.time() - start})
        self.execution_time_list.append(execution_time_iteration)
        return pip_dict
    
    def iteration_process_enhanced_arbitrary_1(self, alg_name, epsilon, iteration, start, a_start, b_start, c_start):
        if 'ell_comb_list' not in self.__dict__:
            self.__dict__['ell_comb_list'] = []
        
        ell = utils.generate_random_combination(self.X_train, a_start, b_start, self.depth, epsilon[iteration]) 
        self.__dict__['ell_comb_list'].append(ell) #

        # Create directory for current outer iteration's PIP outputs
        pip_alg_dir = os.path.join(self.alg_dir, f"outer_iter_{iteration}")

        # Initialize PIP instance with random piece combination and current epsilon
        pip = PIP(
            self.X_train, self.y_train, self.dataset, self.depth, self.tau_0, self.class_restrict, epsilon[iteration], self.beta,
            self.model_params, ell, self.pip_params, pip_alg_dir,
            self.save_log, self.console_log
        )

        # Create unique PIP algorithm name with outer iteration identifier
        pip_alg_name = alg_name + f"_outer_iter_{iteration}"

        # Run PIP main computation with fixed-piece iteration strategy
        pip.main_computation(pip.iteration_process_fixed_piece, pip_alg_name, a_start, b_start, c_start)

        # Record total execution time for current outer iteration
        self.execution_time_list.append(time.time() - start)
        return pip

    def main_computation(self, iteration_process, alg_name, a, b, c):
        """Main outer loop: generate shrinking epsilon sequence, iterate, update warm start."""
        # Generate epsilon sequence (progressively shrinks each iteration)
        epsilon = generate_epsilon(self.max_outer_iter)

        # Initialize warm-start parameters for outer iterations
        a_start = a
        b_start = b
        c_start = c

        # Execute outer shrinkage iterations
        for iteration in range(self.max_outer_iter):
            # Record start time of current outer iteration
            start = time.time()

            # Execute specified outer iteration strategy
            solution = iteration_process(alg_name, epsilon, iteration, start, a_start, b_start, c_start)

            # Case 1: Solution is dictionary (multi-subproblem arbitrary-4 strategy)
            if isinstance(solution, dict):
                # Initialize benchmark values for optimal solution selection
                obj_benchmark = -np.inf
                a_benchmark = a_start
                b_benchmark = b_start
                c_benchmark = c_start
                
                self.pip_alg_dict[iteration] = {}

                # Evaluate all subproblems and select optimal solution
                for sub_prob, pip in solution.items():
                    # Check if subproblem PIP executed successfully
                    if pip.algorithm_state >= 0:
                        obj_val_sub_prob = pip.output['obj_val']
                        # Update benchmark if current subproblem has better objective value
                        if obj_val_sub_prob > obj_benchmark:
                            obj_benchmark = obj_val_sub_prob
                            a_benchmark = pip.output['a']
                            b_benchmark = pip.output['b']
                            c_benchmark = pip.output['c']
                            
                        # Store successful subproblem PIP instance
                        self.pip_alg_dict[iteration].update({sub_prob: pip})
                    # Terminate iteration if any subproblem fails
                    else:
                        self.algorithm_state = -iteration - 1
                        break

                # Update parameters if all subproblems succeeded
                if self.algorithm_state >= 0:
                    # Initialize result lists if not exists
                    for attr in ['obj_list', 'a_list', 'b_list', 'c_list']:
                        if attr not in self.__dict__:
                            self.__dict__[attr] = []

                    # Append optimal results from current iteration
                    self.__dict__['obj_list'].append(obj_benchmark)
                    self.__dict__['a_list'].append(a_benchmark)
                    self.__dict__['b_list'].append(b_benchmark)
                    self.__dict__['c_list'].append(c_benchmark)

                    # Update warm-start parameters for next outer iteration
                    a_start = a_benchmark
                    b_start = b_benchmark
                    c_start = c_benchmark
                # Terminate outer iteration loop if subproblem failed
                else:
                    break

            # Case 2: Solution is single PIP instance (arbitrary-1 or inner update strategy)
            else:
                # Check if PIP instance executed successfully
                if solution.algorithm_state >= 0:
                    # Store successful PIP instance
                    self.pip_alg_dict[iteration] = solution
                    # Update warm-start parameters with current PIP results
                    a_start = solution.output['a']
                    b_start = solution.output['b']
                    c_start = solution.output['c']
                # Terminate iteration if PIP instance failed
                else:
                    self.algorithm_state = -iteration - 1
                    break

        # Set final output if algorithm completed successfully
        if self.algorithm_state >= 0:
            # Determine algorithm state code based on execution strategy
            if 'ell_comb_list' in self.__dict__:
                if isinstance(self.pip_alg_dict[0], dict):
                    self.algorithm_state = 1  # enhanced outer update (arbitrary-4)
                else:
                    self.algorithm_state = 2  # enhanced outer update (arbitrary-1)
            else:
                self.algorithm_state = 3  # inner update (no predefined pieces)

            # Set final output based on algorithm state
            if self.algorithm_state == 2 or self.algorithm_state == 3:
                # For arbitrary-1/inner update: use last iteration's PIP results
                last_pip = self.pip_alg_dict[self.max_outer_iter - 1]
                self.output['obj_val'] = last_pip.output['obj_val']
                self.output['a'] = last_pip.output['a']
                self.output['b'] = last_pip.output['b']
                self.output['c'] = last_pip.output['c']
          
            else:
                # For arbitrary-4: use best results from last iteration
                self.output['obj_val'] = self.__dict__['obj_list'][-1]
                self.output['a'] = self.__dict__['a_list'][-1]
                self.output['b'] = self.__dict__['b_list'][-1]
                self.output['c'] = self.__dict__['c_list'][-1]
                
    def write_integrated_results(self, integrated_csv, alg_name, X_test, y_test, precision_threshold=None, fold=None):
        pass # TODO
