from model import *
import utils
import csv
import time
import numpy as np
from typing import Dict, Optional, Union

class PIP:
    def __init__(self, X_train, y_train, dataset, epsilon, beta, model_params, algorithm_params, alg_dir, save_log=False, console_log=False):

        self.X_train = X_train
        self.y_train = y_train
        self.dataset = dataset

        self.epsilon = epsilon
        self.beta = beta

        self.model_params = model_params

        # Extract iteration control parameters
        self.unchanged_iter = algorithm_params['iteration']['unchanged_iter']
        self.max_iter = algorithm_params['iteration']['max_iter']

        self.initial_ratio_prime = algorithm_params['ratio']['initial_ratio_prime']
        self.max_ratio = algorithm_params['ratio']['max_ratio']
        self.base_ratio = algorithm_params['ratio']['base_ratio'][dataset]
        self.min_ratio = self.base_ratio
        self.change_ratio = algorithm_params['ratio']['change_ratio']

        self.execution_time_list = []
        self.model_dict: Dict[int, Optional[Union[Model, Dict[int, Model]]]] = {
            key: None for key in range(self.max_iter)
        }
        self.algorithm_state = 0  
        self.alg_dir = alg_dir
        self.save_log = save_log
        self.console_log = console_log
        if self.save_log:
            os.makedirs(self.alg_dir, exist_ok=True)  # Create directory if not exists

        self.output = {
            'obj_val': -np.inf,
            'w': None,  
            'b': None,  
            'gamma': None
        }

    def ratio_update_rule(self, ratio, obj_val, obj_val_old, iter_unchanged):
        if  obj_val - obj_val_old <= 1e-5:
            new_ratio = min(ratio + self.change_ratio, self.max_ratio) 
            if obj_val > 0:
                iter_unchanged_new = iter_unchanged + 1
        else:
            iter_unchanged_new = 0
            if obj_val > 0:
                new_ratio = max(ratio - self.change_ratio, self.min_ratio)
            else:
                new_ratio = ratio

        if obj_val >= 0 and obj_val_old < 0:
            new_ratio = self.base_ratio
            iter_unchanged_new = 0
            
        return new_ratio, iter_unchanged_new
    
    def formulate_and_solve_partial_model(self, model_dir, delta_1, delta_2, iter_model_name, w_start, b_start):
        partial_model = Model(
            X=self.X_train,
            y=self.y_train,
            epsilon=self.epsilon,
            beta=self.beta,
            model_type='partial',
            delta_plus=delta_1,
            delta_minus=delta_2,
            model_params=self.model_params,
            model_dir=model_dir,
            model_name=iter_model_name,
            save_log=self.save_log,
            console_log=self.console_log
        )

        partial_model.formulate_model(w_start, b_start)
        partial_model.solve_model()
        return partial_model
    
    def iteration_process(self, alg_name, iteration, start, ratio, w_start, b_start):
        delta_1, delta_2 = calculate_delta(self.X_train, self.y_train, w_start, b_start, self.epsilon, ratio)

        iter_model_name = alg_name + f"_iter_{iteration}"

        partial_model = self.formulate_and_solve_partial_model(self.alg_dir, delta_1, delta_2, iter_model_name, w_start, b_start)

        # Record iteration execution time
        self.execution_time_list.append(time.time() - start)
        return partial_model
    
    def main_computation(self, iteration_process, alg_name, w, b):
        gamma_start, z_plus_0_start, z_plus_start, z_minus_start = calculate_gamma(self.X_train, self.y_train, w, b, self.beta, self.epsilon)
        obj_val = (
                sum(z_plus_0 for z_plus_0 in z_plus_0_start.values()) / self.X_train.shape[0]
                - RHO * (sum(gamma for gamma in gamma_start.values()))
        )

        w_start = w
        b_start = b
        iter_unchanged = 0
        if obj_val >= 0:
            ratio = self.base_ratio
        else:
            ratio = self.initial_ratio_prime
        
        for iteration in range(self.max_iter):
            start = time.time()
            obj_val_old = obj_val

            solution = iteration_process(alg_name, iteration, start, ratio, w_start, b_start)

            if solution.model_state == 1:  # Check if model solved successfully
                # Update objective value and warm start parameters
                obj_val = solution.model.objVal
                w_start = solution.var_val['w']
                b_start = solution.var_val['b']
                gamma_start = solution.var_val['gamma']

                # Update adaptive ratio and unchanged iteration counter
                ratio, iter_unchanged = self.ratio_update_rule(ratio, obj_val, obj_val_old, iter_unchanged)
                self.model_dict[iteration] = solution
            else:
                # Mark algorithm as failed if model solving failed
                self.algorithm_state = - iteration - 1
                break

            if iter_unchanged >= self.unchanged_iter:
                self.max_iter = iteration + 1
                break
        
        if self.algorithm_state >= 0:
            final_model = self.model_dict[self.max_iter - 1]
            self.output['obj_val'] = final_model.model.objVal
            self.output['w'] = final_model.var_val['w']
            self.output['b'] = final_model.var_val['b']
            self.output['gamma'] = final_model.var_val['gamma']

    def write_integrated_results(self):
        pass # TODO


class IterativeShrinkage:
    def __init__(self, X_train, y_train, dataset, beta, model_params, pip_params, alg_dir, save_log=False, console_log=False):

        self.X_train = X_train
        self.y_train = y_train
        self.dataset = dataset

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
        self.algorithm_state = 0  # 0: initial

        # Initialize final output container
        self.output: Dict[str, Optional[Union[dict, float]]] = {
            'obj_val': -np.inf,
            'w': None,
            'b': None,
            'gamma': None
        }

    def outer_iteration_process(self, alg_name, epsilon, iteration, start, w_start, b_start):
        # Create directory for current outer iteration's PIP outputs
        pip_alg_dir = os.path.join(self.alg_dir, f"outer_iter_{iteration}")

        pip = PIP(
            self.X_train, self.y_train, self.dataset, epsilon[iteration], self.beta, 
            self.model_params, self.pip_params, pip_alg_dir, self.save_log, self.console_log
        )

        # Create unique PIP algorithm name with outer iteration identifier
        pip_alg_name = alg_name + f"_outer_iter_{iteration}"

        pip.main_computation(pip.iteration_process, pip_alg_name, w_start, b_start)

        self.execution_time_list.append(time.time() - start)
        return pip
    
    def main_computation(self, iteration_process, alg_name, w, b):
        epsilon = generate_epsilon(self.max_outer_iter)

        w_start = w
        b_start = b

        # Execute outer shrinkage iterations
        for iteration in range(self.max_outer_iter):

            start = time.time()

            solution = iteration_process(alg_name, epsilon, iteration, start, w_start, b_start)

            if solution.algorithm_state >= 0:
                # Store successful PIP instance
                self.pip_alg_dict[iteration] = solution
                # Update warm-start parameters with current PIP results
                w_start = solution.output['w']
                b_start = solution.output['b']
            # Terminate iteration if PIP instance failed
            else:
                self.algorithm_state = - iteration - 1
                break
        
        if self.algorithm_state >= 0:
            last_pip = self.pip_alg_dict[self.max_outer_iter - 1]
            self.output['obj_val'] = last_pip.output['obj_val']
            self.output['w'] = last_pip.output['w']
            self.output['b'] = last_pip.output['b']
            self.output['gamma'] = last_pip.output['gamma']

    def write_integrated_results(self):
        pass # TODO