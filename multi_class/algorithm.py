import time
from typing import Dict, Optional
from model import *

class PIP:
    """
    PIP Algorithm Core Class
    Solves classification optimization problems with precision constraints, supporting two iterative strategies:
    - Fixed-piece: Uses pre-defined piece sets for iteration
    - Arbitrary-piece (enhanced): Dynamically selects piece combinations during iteration

    Attributes:
        X_train (np.ndarray): Training feature matrix with shape (n_samples, n_features)
        y_train (np.ndarray): Training label array with shape (n_samples,)
        class_restrict (list): List of classes to impose precision constraints on
        epsilon (float): Epsilon approximation parameter
        beta (float): Lower bound threshold for precision constraints
        ell (Optional[dict]): Fixed piece set (only for fixed-PA model; None for arbitrary strategies)
        model_params (dict): Gurobi solver parameters
        unchanged_iter (int): Max iterations with unchanged objective value (early stop trigger)
        max_iter (int): Maximum number of main iterations for PIP algorithm
        min_ratio (float): Minimum value for adaptive ratio update
        max_ratio (float): Maximum value for adaptive ratio update
        base_ratio (float): Initial adaptive ratio value
        change_ratio (float): Step size for ratio adjustment
        execution_time_list (list): Records execution time of each iteration
        model_dict (Dict[int, Optional[Union[Model, Dict[int, Model]]]]): Stores solved models per iteration
        algorithm_state (int): State flag (0=initial, 1=fixed-piece success, 2=arbitrary-4 success,
                               3=arbitrary-1 success, negative=failed iteration)
        alg_dir (str): Directory to save algorithm outputs (models, results)
        output (dict): Final algorithm outputs (objective value, weights, biases, z variables)
    """
    def __init__(self, X_train, y_train, class_restrict, epsilon, beta, model_params, ell, algorithm_params, alg_dir):

        """
        Initialize PIP algorithm instance

        Args:
            X_train (np.ndarray): Training feature matrix
            y_train (np.ndarray): Training label array
            class_restrict (list): Classes to impose precision constraints on
            epsilon (float): Epsilon approximation parameter
            beta (dict): Lower bound for precision constraints
            model_params (dict): Gurobi model parameters (e.g., TimeLimit, MIPGap)
            ell (Optional[dict]): Fixed piece set (None for arbitrary-piece strategies)
            algorithm_params (dict): PIP control parameters with structure:
                {
                    'iteration': {'unchanged_iter': int, 'max_iter': int},
                    'ratio': {'min_ratio': float, 'max_ratio': float, 'base_ratio': float, 'change_ratio': float}
                }
            alg_dir (str): Directory path to save algorithm outputs
        """
        self.X_train = X_train
        self.y_train = y_train
        self.class_restrict = class_restrict
        self.epsilon = epsilon
        self.beta = beta
        self.ell = ell
        self.model_params = model_params

        # Extract iteration control parameters
        self.unchanged_iter = algorithm_params['iteration']['unchanged_iter']
        self.max_iter = algorithm_params['iteration']['max_iter']

        # Extract ratio update parameters
        self.min_ratio = algorithm_params['ratio']['min_ratio']
        self.max_ratio = algorithm_params['ratio']['max_ratio']
        self.base_ratio = algorithm_params['ratio']['base_ratio']
        self.change_ratio = algorithm_params['ratio']['change_ratio']

        # Initialize runtime tracking and model storage
        self.execution_time_list = []
        self.model_dict: Dict[int, Optional[Union[Model, Dict[int, Model]]]] = {
            key: None for key in range(self.max_iter)
        }
        self.algorithm_state = 0  # 0: initial, 1: fixed-piece success, 2: arbitrary-4 success, 3: arbitrary-1 success
        self.alg_dir = alg_dir
        os.makedirs(self.alg_dir, exist_ok=True)  # Create directory if not exists

        # Initialize final output container
        self.output = {
            'obj_val': -np.inf,
            'W': None,  # Weight matrix
            'b': None,  # Bias vector
            'z_plus': None,  # z+ variables
            'z_minus': None  # z- variables
        }

    def ratio_update_rule(self, ratio, obj_val, obj_val_old, iter_unchanged):
        """
        Adaptive ratio update rule based on objective value change

        Adjusts the ratio parameter to balance exploration/exploitation:
        - Increase ratio if objective value is unchanged (stagnation)
        - Decrease ratio if objective value improves

        Args:
            ratio (float): Current ratio value
            obj_val (float): Current iteration objective value
            obj_val_old (float): Previous iteration objective value
            iter_unchanged (int): Number of consecutive iterations with unchanged objective

        Returns:
            tuple: (new_ratio, new_iter_unchanged)
                - new_ratio: Updated ratio value (clamped to [min_ratio, max_ratio])
                - new_iter_unchanged: Updated count of unchanged iterations
        """
        # Check if objective value is unchanged (within numerical tolerance)
        if -1e-5 <= obj_val - obj_val_old <= 1e-5:
            new_ratio = min(ratio + self.change_ratio + 5, self.max_ratio)
            iter_unchanged_new = iter_unchanged + 1
        else:
            new_ratio = max(ratio - self.change_ratio, self.min_ratio)
            iter_unchanged_new = 0
        return new_ratio, iter_unchanged_new

    def formulate_and_solve_partial_model(self, model_dir, delta_1, delta_2, ell, iter_model_name, W_start, b_start):
        """
        Formulate and solve the partial optimization model (core subproblem)

        Creates a Model instance for partial optimization, formulates the mathematical model
        with initial values, and solves it using Gurobi solver.

        Args:
            model_dir (str): Directory to save the model file
            delta_1 (dict): delta+ parameters for fixing variables
            delta_2 (dict): delta- parameters for fixing variables
            ell (dict): Piece set for the current subproblem
            iter_model_name (str): Unique name for the iteration model
            W_start (dict): Initial weight matrix for warm start
            b_start (dict): Initial bias vector for warm start

        Returns:
            Model: Solved partial model instance with results (objVal, var_val, etc.)
        """
        # Initialize partial model with problem parameters
        partial_model = Model(
            X=self.X_train,
            y=self.y_train,
            class_restrict=self.class_restrict,
            epsilon=self.epsilon,
            beta=self.beta,
            model_type='partial',
            ell=ell,
            delta_plus=delta_1,
            delta_minus=delta_2,
            model_params=self.model_params,
            model_dir=model_dir,
            model_name=iter_model_name
        )

        # Formulate the model with warm start values and solve
        partial_model.formulate_model(W_start, b_start)
        partial_model.solve_model()
        return partial_model

    def iteration_process_fixed_piece(self, alg_name, iteration, start, ratio, W_start, b_start):
        """
        Iteration process for fixed-piece PIP algorithm

        Calculates delta parameters, solves partial model with fixed piece set,
        and records execution time.

        Args:
            alg_name (str): Algorithm name (for model naming)
            iteration (int): Current iteration number
            start (float): Start time of the iteration (time.time())
            ratio (float): Current adaptive ratio value
            W_start (dict): Initial weight matrix for warm start
            b_start (dict): Initial bias vector for warm start

        Returns:
            Model: Solved partial model instance for fixed-piece iteration
        """
        # Calculate delta+ and delta- parameters using fixed piece set
        delta_1, delta_2 = delta_of_J(
            self.X_train, self.y_train, W_start, b_start,
            self.class_restrict, self.epsilon, ratio, self.min_ratio, self.ell
        )

        # Define unique model name for current iteration
        iter_model_name = alg_name + f"_iter_{iteration}"

        # Formulate and solve partial model
        partial_model = self.formulate_and_solve_partial_model(
            self.alg_dir, delta_1, delta_2, self.ell,
            iter_model_name, W_start, b_start
        )

        # Record iteration execution time
        self.execution_time_list.append(time.time() - start)
        return partial_model

    def iteration_process_enhanced_arbitrary_4(self, alg_name, iteration, start, ratio, W_start, b_start):
        """
        Iteration process for enhanced arbitrary-piece PIP algorithm (4 combinations)

        Dynamically generates piece sets, filters them based on inner function values,
        selects 4 random piece combinations, solves subproblems for each combination,
        and tracks execution time for each subproblem.

        Args:
            alg_name (str): Algorithm name (for model naming)
            iteration (int): Current iteration number
            start (float): Start time of the iteration (time.time())
            ratio (float): Current adaptive ratio value
            W_start (dict): Initial weight matrix for warm start
            b_start (dict): Initial bias vector for warm start

        Returns:
            Dict[int, Model]: Dictionary of solved partial models (subproblem index -> Model instance)
        """
        # Initialize storage for arbitrary-piece strategy if not exists
        if 'ell_comb_len_list' not in self.__dict__:
            self.__dict__['ell_comb_len_list'] = []  # Track piece combination lengths
            self.__dict__['W_list'] = []  # Track weights per iteration
            self.__dict__['b_list'] = []  # Track biases per iteration
            self.__dict__['obj_list'] = []  # Track objective values per iteration
            self.__dict__['z_plus_list'] = []  # Track z+ variables per iteration
            self.__dict__['z_minus_list'] = []  # Track z- variables per iteration

        # Calculate delta+ and delta- parameters (ell is None for arbitrary strategy)
        delta_1, delta_2 = delta_of_J(
            self.X_train, self.y_train, W_start, b_start,
            self.class_restrict, self.epsilon, ratio, self.min_ratio, self.ell
        )

        # Generate piece set (ELL) based on current model parameters
        ELL = piece_set(
            self.X_train, self.y_train, W_start, b_start,
            self.class_restrict, 0, self.epsilon
        )

        # Calculate inner function values for piece set filtering
        phi = inner_function(
            self.X_train, self.y_train, self.class_restrict,
            self.epsilon, W_start, b_start, None
        )

        # Filter piece set based on inner function and delta constraints
        ELL_filtered = {}
        for key, value in ELL.items():
            m = key[0]  # Class index
            s = key[1]  # Piece index
            if -phi[m, s] - self.epsilon <= -delta_2[m]:
                # Select single random piece if constraint is satisfied
                ELL_filtered[key] = [random.choice(value)]
            else:
                # Keep all pieces if constraint not satisfied
                ELL_filtered[key] = value

        # Select 4 random piece combinations from filtered set
        ell_comb = arbitrary_choose_piece_combination(ELL_filtered, num=4)

        # Record total piece combination length for tracking
        self.__dict__['ell_comb_len_list'].append(prod([len(piece) for piece in ELL.values()]))

        # Solve subproblems for each piece combination
        sub_prob_counter = 0
        model_dict = {}
        execution_time_iteration = {}
        for ell in ell_comb:
            sub_start = time.time()
            # Define unique model name for subproblem
            iter_model_name = alg_name + f"_iter_{iteration}" + f"_sub_prob_{sub_prob_counter}"
            # Create subdirectory for current iteration
            model_dir = os.path.join(self.alg_dir, f"iter_{iteration}")
            os.makedirs(model_dir, exist_ok=True)

            # Formulate and solve partial model for current piece combination
            partial_model = self.formulate_and_solve_partial_model(
                model_dir, delta_1, delta_2, ell, iter_model_name,
                W_start, b_start
            )

            # Store solved model and execution time
            model_dict.update({sub_prob_counter: partial_model})
            execution_time_iteration.update({sub_prob_counter: time.time() - sub_start})
            sub_prob_counter += 1

        # Record total iteration execution time
        execution_time_iteration.update({'total': time.time() - start})
        self.execution_time_list.append(execution_time_iteration)
        return model_dict

    def iteration_process_enhanced_arbitrary_1(self, alg_name, iteration, start, ratio, W_start, b_start):
        """
        Iteration process for enhanced arbitrary-piece PIP algorithm (1 combination)

        Dynamically generates piece sets, selects 1 random piece per class,
        solves the partial model, and records execution time.

        Args:
            alg_name (str): Algorithm name (for model naming)
            iteration (int): Current iteration number
            start (float): Start time of the iteration (time.time())
            ratio (float): Current adaptive ratio value
            W_start (dict): Initial weight matrix for warm start
            b_start (dict): Initial bias vector for warm start

        Returns:
            Model: Solved partial model instance for arbitrary-piece iteration
        """
        # Initialize piece combination length tracker if not exists
        if 'ell_comb_len_list' not in self.__dict__:
            self.__dict__['ell_comb_len_list'] = []

        # Generate piece set (ELL) based on current model parameters
        ELL = piece_set(
            self.X_train, self.y_train, W_start, b_start,
            self.class_restrict, 0, self.epsilon
        )

        # Randomly select 1 piece per class from the generated set
        ell = {}
        for key in list(ELL.keys()):
            ell[key] = random.choice(ELL[key])

        # Record total piece combination length
        self.__dict__['ell_comb_len_list'].append(prod([len(piece) for piece in ELL.values()]))

        # Calculate delta+ and delta- parameters using selected piece set
        delta_1, delta_2 = delta_of_J(
            self.X_train, self.y_train, W_start, b_start,
            self.class_restrict, self.epsilon, ratio, self.min_ratio, ell
        )

        # Define unique model name for current iteration
        iter_model_name = alg_name + f"_iter_{iteration}"
        model_dir = self.alg_dir

        # Formulate and solve partial model
        partial_model = self.formulate_and_solve_partial_model(
            model_dir, delta_1, delta_2, ell, iter_model_name,
            W_start, b_start
        )

        # Record iteration execution time
        self.execution_time_list.append(time.time() - start)
        return partial_model

    def main_computation(self, iteration_process, alg_name, W, b):
        """
        Main iterative computation loop for PIP algorithm

        Executes the core iteration process (fixed/arbitrary piece), updates adaptive ratio,
        tracks convergence (unchanged objective value), and stores final results.

        Args:
            iteration_process (callable): Iteration process function (fixed/arbitrary)
            alg_name (str): Algorithm name (for tracking)
            W (dict): Initial weight matrix
            b (dict): Initial bias vector
        """
        # Generate initial z and gamma variables for warm start
        z_plus_0_start, z_plus_start, z_minus_start = generate_z_start(
            self.X_train, self.y_train, W, b, self.epsilon, self.class_restrict, self.ell
        )
        gamma_start = generate_gamma_start(
            self.X_train, self.y_train, W, b, self.beta, self.epsilon,
            self.class_restrict, self.ell
        )

        # Calculate initial objective value
        obj_val = (
                sum(z_plus_0 for z_plus_0 in z_plus_0_start.values()) / self.X_train.shape[0]
                - RHO * (sum(gamma for gamma in gamma_start.values()))
        )

        # Initialize warm start parameters
        W_start = W
        b_start = b
        iter_unchanged = 0  # Counter for consecutive unchanged iterations
        ratio = self.base_ratio  # Initial adaptive ratio

        # Main iteration loop
        for iteration in range(self.max_iter):
            start = time.time()
            obj_val_old = obj_val  # Store previous objective value

            # Execute iteration process (fixed/arbitrary)
            solution = iteration_process(alg_name, iteration, start, ratio, W_start, b_start)

            # Handle multi-subproblem solution (arbitrary-4)
            if isinstance(solution, dict):
                obj_benchmark = obj_val_old
                W_benchmark = W_start
                b_benchmark = b_start
                z_plus_benchmark = z_plus_start
                z_minus_benchmark = z_minus_start

                self.model_dict[iteration] = {}
                # Evaluate all subproblems and select best solution
                for sub_prob, model in solution.items():
                    if model.model_state == 1:  # Check if model solved successfully
                        obj_val_sub_prob = model.model.objVal
                        # Update benchmark if current subproblem is better
                        if obj_val_sub_prob > obj_benchmark:
                            obj_benchmark = obj_val_sub_prob
                            W_benchmark = model.var_val['W']
                            b_benchmark = model.var_val['b']
                            z_plus_benchmark = model.var_val['z_plus']
                            z_minus_benchmark = model.var_val['z_minus']
                        self.model_dict[iteration].update({sub_prob: model})
                    else:
                        # Mark algorithm as failed if any subproblem fails
                        self.algorithm_state = -iteration - 1
                        break

                # Update parameters if all subproblems succeeded
                if self.algorithm_state >= 0:
                    self.__dict__['obj_list'].append(obj_benchmark)
                    self.__dict__['W_list'].append(W_benchmark)
                    self.__dict__['b_list'].append(b_benchmark)
                    self.__dict__['z_plus_list'].append(z_plus_benchmark)
                    self.__dict__['z_minus_list'].append(z_minus_benchmark)

                    # Update warm start parameters for next iteration
                    W_start = W_benchmark
                    b_start = b_benchmark
                    z_plus_start = z_plus_benchmark
                    z_minus_start = z_minus_benchmark
                    obj_val = obj_benchmark

                    # Update adaptive ratio and unchanged iteration counter
                    ratio, iter_unchanged = self.ratio_update_rule(
                        ratio, obj_val, obj_val_old, iter_unchanged
                    )
                else:
                    break  # Exit loop if algorithm failed

            # Handle single-model solution (fixed-piece/arbitrary-1)
            else:
                if solution.model_state == 1:  # Check if model solved successfully
                    # Update objective value and warm start parameters
                    obj_val = solution.model.objVal
                    W_start = solution.var_val['W']
                    b_start = solution.var_val['b']

                    # Update adaptive ratio and unchanged iteration counter
                    ratio, iter_unchanged = self.ratio_update_rule(
                        ratio, obj_val, obj_val_old, iter_unchanged
                    )
                    self.model_dict[iteration] = solution
                else:
                    # Mark algorithm as failed if model solving failed
                    self.algorithm_state = -iteration - 1
                    break

            # Early stop if objective value unchanged for max allowed iterations
            if iter_unchanged >= self.unchanged_iter:
                self.max_iter = iteration + 1
                break

        # Update final output if algorithm succeeded
        if self.algorithm_state >= 0:
            # Set algorithm state based on strategy type
            if 'ell_comb_len_list' in self.__dict__:
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
                self.output['W'] = final_model.var_val['W']
                self.output['b'] = final_model.var_val['b']
                self.output['z_plus'] = final_model.var_val['z_plus']
                self.output['z_minus'] = final_model.var_val['z_minus']
            else:
                # Arbitrary-4 (multiple models per iteration)
                self.output['obj_val'] = self.__dict__['obj_list'][-1]
                self.output['W'] = self.__dict__['W_list'][-1]
                self.output['b'] = self.__dict__['b_list'][-1]
                self.output['z_plus'] = self.__dict__['z_plus_list'][-1]
                self.output['z_minus'] = self.__dict__['z_minus_list'][-1]


    def write_integrated_results(self, integrated_csv, alg_name, X_test, y_test):
        """
        Write integrated final results

        Computes classification metrics (accuracy, precision, recall) for train/test sets,
        checks precision constraint satisfaction, and writes all integrated results to CSV.

        Args:
            integrated_csv (str): Path to integrated results CSV file
            alg_name (str): Algorithm name (for result labeling)
            X_test (np.ndarray): Test feature matrix
            y_test (np.ndarray): Test label array
        """
        if self.algorithm_state >= 0:
            # Calculate precision in constraint (satisfaction check)
            constr_precision = precision_in_constraint(
                self.y_train, self.output['z_plus'], self.output['z_minus'],
                self.class_restrict
            )

            # Calculate classification metrics for train/test sets
            train_results = classification_metric(
                self.X_train, self.y_train, self.output['W'], self.output['b']
            )
            test_results = classification_metric(
                X_test, y_test, self.output['W'], self.output['b']
            )

            # Calculate total execution time
            execution_time = 0
            for val in self.execution_time_list:
                if isinstance(val, dict):
                    execution_time += val['total']
                else:
                    execution_time += val

            # Get piece combination length (or fixed piece set)
            if 'ell_comb_len_list' in self.__dict__:
                num_kl = self.__dict__['ell_comb_len_list']
            else:
                num_kl = self.ell

            # Calculate total solver time across all models
            all_models = extract_inner_values(self.model_dict)
            model_time = np.sum([model.model.Runtime for model in all_models if model is not None])

            # Write integrated results to CSV
            write_single_integrated_result(
                integrated_csv=integrated_csv, title=alg_name, execution_time=execution_time,
                model_time=model_time, final_improvement_time=None,
                obj_val=self.output['obj_val'],
                train_acc_margined=train_results['acc_margined'],
                train_acc=train_results['accuracy'],
                train_precision=train_results['precision'],
                train_recall=train_results['recall'],
                test_acc_margined=test_results['acc_margined'],
                test_acc=test_results['accuracy'],
                test_precision=test_results['precision'], test_recall=test_results['recall'],
                precision_in_constr=constr_precision, opt_gap=None, num_kl=num_kl,
                W=self.output['W'], b=self.output['b']
            )
        else:
            # Write failure message if algorithm failed
            with open(integrated_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([alg_name, f'Failed in the {-self.algorithm_state + 1} iteration.'])


class IterativeShrinkage:
    """
    Iterative Shrinkage Algorithm Class
    Wraps the PIP algorithm to perform iterative shrinkage of the epsilon parameter,

    Attributes:
        X_train (np.ndarray): Training feature matrix
        y_train (np.ndarray): Training label array
        class_restrict (list): List of classes to impose precision constraints on
        beta (dict): Lower bound threshold for precision constraints
        model_params (dict): Gurobi solver parameters
        max_outer_iter (int): Maximum number of outer shrinkage iterations (from config.SHRINKAGE_MAX_OUT_ITER)
        pip_params (dict): PIP algorithm control parameters
        alg_dir (str): Directory to save shrinkage algorithm outputs
        execution_time_list (list): Records execution time of each outer iteration
        pip_alg_dict (Dict[int, Optional[Union[PIP, Dict[int, PIP]]]]): Stores PIP instances per outer iteration
        algorithm_state (int): State flag (0=initial, positive=success, negative=failed iteration)
        output (dict): Final algorithm outputs (objective value, weights, biases, z variables)
    """
    def __init__(self, X_train, y_train, class_restrict, beta, model_params, pip_params, alg_dir):
        """
        Initialize Iterative Shrinkage algorithm instance

        Args:
            X_train (np.ndarray): Training feature matrix
            y_train (np.ndarray): Training label array
            class_restrict (list): Classes to impose precision constraints on
            beta (float): Lower bound for precision constraints
            model_params (dict): Gurobi model parameters
            pip_params (dict): PIP algorithm control parameters (passed to PIP class)
            alg_dir (str): Directory path to save shrinkage algorithm outputs
        """
        self.X_train = X_train
        self.y_train = y_train
        self.class_restrict = class_restrict
        self.beta = beta
        self.model_params = model_params

        # Shrinkage control parameters
        self.max_outer_iter = SHRINKAGE_MAX_OUT_ITER
        self.pip_params = pip_params
        self.alg_dir = alg_dir

        # Initialize runtime tracking and PIP instance storage
        self.execution_time_list = []
        self.pip_alg_dict: Dict[int, Optional[Union[PIP, Dict[int, PIP]]]] = {
            key: None for key in range(self.max_outer_iter)
        }
        self.algorithm_state = 0  # 0: initial, positive: success, negative: failed

        # Initialize final output container
        self.output: Dict[str, Optional[Union[dict, float]]] = {
            'obj_val': -np.inf,
            'W': None,
            'b': None,
            'z_plus': None,
            'z_minus': None
        }

    def iteration_process_enhanced_arbitrary_4(self, alg_name, epsilon, iteration, start, W_start, b_start):
        """
        Outer iteration process for enhanced arbitrary-piece (4 combinations) shrinkage

        Creates PIP instance for current epsilon value, runs PIP main computation,
        and tracks piece combination lengths and solution variables.

        Args:
            alg_name (str): Algorithm name (for PIP naming)
            epsilon (list): Epsilon values for each outer iteration
            iteration (int): Current outer iteration number
            start (float): Start time of the outer iteration (time.time())
            W_start (dict): Initial weight matrix for PIP warm start
            b_start (dict): Initial bias vector for PIP warm start

        Returns:
            Dict[int, PIP]: Dictionary of PIP instances (subproblem index -> PIP instance)
        """
        # Initialize storage for shrinkage strategy if not exists
        if 'ell_comb_len_list' not in self.__dict__:
            self.__dict__['ell_comb_len_list'] = []
            self.__dict__['W_list'] = []
            self.__dict__['b_list'] = []
            self.__dict__['z_plus_list'] = []
            self.__dict__['z_minus_list'] = []
            self.__dict__['obj_list'] = []

        # Generate piece set based on current epsilon and model parameters
        ELL = piece_set(
            self.X_train, self.y_train, W_start, b_start,
            self.class_restrict, 0, epsilon[iteration]
        )

        # Generate arbitrary piece combinations
        ell_comb = arbitrary_choose_piece_combination(ELL)
        self.__dict__['ell_comb_len_list'].append(prod([len(piece) for piece in ELL.values()]))

        # Solve PIP for each piece combination
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
                self.X_train, self.y_train, self.class_restrict, epsilon[iteration], self.beta,
                self.model_params, ell, self.pip_params, pip_alg_dir
            )

            # Run PIP main computation with fixed-piece strategy
            pip.main_computation(pip.iteration_process_fixed_piece, alg_name, W_start, b_start)

            # Store PIP instance and execution time
            pip_dict.update({sub_prob_counter: pip})
            execution_time_iteration.update({sub_prob_counter: time.time() - sub_start})
            sub_prob_counter += 1

        # Record total outer iteration execution time
        execution_time_iteration.update({'total': time.time() - start})
        self.execution_time_list.append(execution_time_iteration)
        return pip_dict

    def iteration_process_enhanced_arbitrary_1(self, alg_name, epsilon, iteration, start, W_start, b_start):
        """
        Outer iteration process for enhanced arbitrary-piece (1 random combination) shrinkage

        Generates a single random piece combination, initializes a PIP instance with this combination,
        runs fixed-piece PIP computation, and tracks execution time and piece combination metrics.

        Args:
            alg_name (str): Base algorithm name (for PIP instance naming)
            epsilon (list): Epsilon values for each outer iteration (shrinks each iteration)
            iteration (int): Current outer iteration number (0-indexed)
            start (float): Start time of the outer iteration (time.time() timestamp)
            W_start (dict): Initial weight matrix for PIP warm start
            b_start (dict): Initial bias vector for PIP warm start

        Returns:
            PIP: Executed PIP instance with results for the random piece combination
        """
        # Initialize piece combination length tracker if not exists
        if 'ell_comb_len_list' not in self.__dict__:
            self.__dict__['ell_comb_len_list'] = []

        # Generate piece set based on current epsilon and warm start parameters
        ELL = piece_set(
            self.X_train, self.y_train, W_start, b_start,
            self.class_restrict, 0, epsilon[iteration]
        )

        # Randomly select one piece per class/segment to form ell dictionary
        ell = {}
        for key in list(ELL.keys()):
            ell[key] = random.choice(ELL[key])

        # Record total piece combination length (product of subset lengths)
        self.__dict__['ell_comb_len_list'].append(prod([len(val) for val in ELL.values()]))

        # Create directory for current outer iteration's PIP outputs
        pip_alg_dir = os.path.join(self.alg_dir, f"outer_iter_{iteration}")

        # Initialize PIP instance with random piece combination and current epsilon
        pip = PIP(
            self.X_train, self.y_train, self.class_restrict, epsilon[iteration], self.beta,
            self.model_params, ell, self.pip_params, pip_alg_dir
        )

        # Create unique PIP algorithm name with outer iteration identifier
        pip_alg_name = alg_name + f"_outer_iter_{iteration}"

        # Run PIP main computation with fixed-piece iteration strategy
        pip.main_computation(pip.iteration_process_fixed_piece, pip_alg_name, W_start, b_start)

        # Record total execution time for current outer iteration
        self.execution_time_list.append(time.time() - start)
        return pip

    def iteration_process_enhanced_inner_update(self, alg_name, epsilon, iteration, start, W_start, b_start):
        """
        Outer iteration process for enhanced inner-update shrinkage

        Initializes a PIP instance with NO predefined piece configuration (ell=None),
        runs PIP computation with the enhanced arbitrary-4 piece strategy, and tracks execution time.
        This strategy lets PIP dynamically determine piece combinations internally.

        Args:
            alg_name (str): Base algorithm name (for PIP instance naming)
            epsilon (list): Epsilon values for each outer iteration (shrinks each iteration)
            iteration (int): Current outer iteration number (0-indexed)
            start (float): Start time of the outer iteration (time.time() timestamp)
            W_start (dict): Initial weight matrix for PIP warm start
            b_start (dict): Initial bias vector for PIP warm start

        Returns:
            PIP: Executed PIP instance with results from inner arbitrary-4 piece strategy
        """
        # Create directory for current outer iteration's PIP outputs
        pip_alg_dir = os.path.join(self.alg_dir, f"outer_iteration_{iteration}")

        # Initialize PIP instance with NO predefined piece configuration (ell=None)
        pip = PIP(
            self.X_train, self.y_train, self.class_restrict, epsilon[iteration], self.beta,
            self.model_params, None, self.pip_params, pip_alg_dir
        )

        # Create unique PIP algorithm name with outer iteration identifier
        pip_alg_name = alg_name + f"_outer_iter_{iteration}"

        # Run PIP main computation with enhanced arbitrary-4 piece strategy
        pip.main_computation(pip.iteration_process_enhanced_arbitrary_4, pip_alg_name, W_start, b_start)

        # Record total execution time for current outer iteration
        self.execution_time_list.append(time.time() - start)
        return pip

    def main_computation(self, iteration_process, alg_name, W, b):
        """
        Main outer iteration driver for the Iterative Shrinkage algorithm

        Orchestrates the complete shrinkage process:
        1. Generates epsilon sequence (progressively shrinking values)
        2. Iterates through outer shrinkage iterations with specified strategy
        3. Processes PIP results (single/multi-subproblem)
        4. Selects optimal solutions and updates warm-start parameters
        5. Sets final output based on algorithm state and strategy type

        Args:
            iteration_process (callable): Outer iteration strategy to execute
                - self.iteration_process_enhanced_arbitrary_4
                - self.iteration_process_enhanced_arbitrary_1
                - self.iteration_process_enhanced_inner_update
            alg_name (str): Base algorithm name for result labeling
            W (dict): Initial weight matrix for outer iteration warm start
            b (dict): Initial bias vector for outer iteration warm start

        State Codes (self.algorithm_state):
            1: Enhanced outer update (multi-subproblem arbitrary-4)
            2: Enhanced outer update (single-subproblem arbitrary-1)
            3: Inner update (no predefined pieces)
            Negative values: Failed iteration (value = -failed_iteration - 1)
        """
        # Generate epsilon sequence (progressively shrinks each iteration)
        epsilon = generate_epsilon(self.max_outer_iter)

        # Initialize warm-start parameters for outer iterations
        W_start = W
        b_start = b

        # Execute outer shrinkage iterations
        for iteration in range(self.max_outer_iter):
            # Record start time of current outer iteration
            start = time.time()

            # Execute specified outer iteration strategy
            solution = iteration_process(alg_name, epsilon, iteration, start, W_start, b_start)

            # Case 1: Solution is dictionary (multi-subproblem arbitrary-4 strategy)
            if isinstance(solution, dict):
                # Initialize benchmark values for optimal solution selection
                obj_benchmark = -np.inf
                W_benchmark = W_start
                b_benchmark = b_start
                z_plus_benchmark = None
                z_minus_benchmark = None
                self.pip_alg_dict[iteration] = {}

                # Evaluate all subproblems and select optimal solution
                for sub_prob, pip in solution.items():
                    # Check if subproblem PIP executed successfully
                    if pip.algorithm_state >= 0:
                        obj_val_sub_prob = pip.output['obj_val']
                        # Update benchmark if current subproblem has better objective value
                        if obj_val_sub_prob > obj_benchmark:
                            obj_benchmark = obj_val_sub_prob
                            W_benchmark = pip.output['W']
                            b_benchmark = pip.output['b']
                            z_plus_benchmark = pip.output['z_plus']
                            z_minus_benchmark = pip.output['z_minus']
                        # Store successful subproblem PIP instance
                        self.pip_alg_dict[iteration].update({sub_prob: pip})
                    # Terminate iteration if any subproblem fails
                    else:
                        self.algorithm_state = -iteration - 1
                        break

                # Update parameters if all subproblems succeeded
                if self.algorithm_state >= 0:
                    # Initialize result lists if not exists
                    for attr in ['obj_list', 'W_list', 'b_list', 'z_plus_list', 'z_minus_list']:
                        if attr not in self.__dict__:
                            self.__dict__[attr] = []

                    # Append optimal results from current iteration
                    self.__dict__['obj_list'].append(obj_benchmark)
                    self.__dict__['W_list'].append(W_benchmark)
                    self.__dict__['b_list'].append(b_benchmark)
                    self.__dict__['z_plus_list'].append(z_plus_benchmark)
                    self.__dict__['z_minus_list'].append(z_minus_benchmark)

                    # Update warm-start parameters for next outer iteration
                    W_start = W_benchmark
                    b_start = b_benchmark
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
                    W_start = solution.output['W']
                    b_start = solution.output['b']
                # Terminate iteration if PIP instance failed
                else:
                    self.algorithm_state = -iteration - 1
                    break

        # Set final output if algorithm completed successfully
        if self.algorithm_state >= 0:
            # Determine algorithm state code based on execution strategy
            if 'ell_comb_len_list' in self.__dict__:
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
                self.output['W'] = last_pip.output['W']
                self.output['b'] = last_pip.output['b']
                self.output['z_plus'] = last_pip.output['z_plus']
                self.output['z_minus'] = last_pip.output['z_minus']
            else:
                # For arbitrary-4: use best results from last iteration
                self.output['obj_val'] = self.__dict__['obj_list'][-1]
                self.output['W'] = self.__dict__['W_list'][-1]
                self.output['b'] = self.__dict__['b_list'][-1]
                self.output['z_plus'] = self.__dict__['z_plus_list'][-1]
                self.output['z_minus'] = self.__dict__['z_minus_list'][-1]

    def write_integrated_results(self, integrated_csv, alg_name, X_test, y_test):
        """
        Write integrated shrinkage algorithm results to CSV file

        Aggregates and writes comprehensive results including:
        - Execution time metrics (total runtime, model solve time)
        - Classification metrics (train/test accuracy, precision, recall)
        - Precision constraint compliance
        - Learned model parameters (weights, biases)

        Args:
            integrated_csv (str): Path to output CSV file for integrated results
            alg_name (str): Algorithm name for result labeling in CSV
            X_test (np.ndarray): Test feature matrix (shape: n_test_samples × n_features)
            y_test (np.ndarray): Test label array (shape: n_test_samples × 1)

        Metrics Included:
            - Execution time: Total outer iteration runtime
            - Model time: Aggregated solver runtime from all PIP instances
            - Objective value: Final optimization objective value
            - Train/test metrics: acc_margined, accuracy, precision, recall
            - Precision constraint compliance: How well precision constraints are satisfied
            - num_kl: Piece combination length (only for arbitrary piece strategies)
            - Model parameters: W (weights), b (biases)
        """
        # Write complete results for successful execution
        if self.algorithm_state >= 0:
            # Calculate precision constraint compliance (training set)
            constr_precision = precision_in_constraint(
                self.y_train, self.output['z_plus'], self.output['z_minus'],
                self.class_restrict
            )

            # Calculate classification metrics for training and test sets
            train_results = classification_metric(
                self.X_train, self.y_train, self.output['W'], self.output['b']
            )
            test_results = classification_metric(
                X_test, y_test, self.output['W'], self.output['b']
            )

            # Calculate total execution time (supports dict/float time entries)
            execution_time = 0
            for val in self.execution_time_list:
                if isinstance(val, dict):
                    execution_time += val['total']
                else:
                    execution_time += val

            # Calculate total model solve time across all PIP instances
            all_pips = extract_inner_values(self.pip_alg_dict)
            model_time = 0
            for pip in all_pips:
                all_models = extract_inner_values(pip.model_dict)
                model_time += np.sum([
                    model.model.Runtime for model in all_models if model is not None
                ])

            # Get piece combination length list (only for arbitrary piece strategies)
            num_kl = self.__dict__.get('ell_comb_len_list', None)

            # Write integrated results to CSV
            write_single_integrated_result(
                integrated_csv=integrated_csv,
                title=alg_name,
                execution_time=execution_time,
                model_time=model_time,
                final_improvement_time=None,
                obj_val=self.output['obj_val'],
                train_acc_margined=train_results['acc_margined'],
                train_acc=train_results['accuracy'],
                train_precision=train_results['precision'],
                train_recall=train_results['recall'],
                test_acc_margined=test_results['acc_margined'],
                test_acc=test_results['accuracy'],
                test_precision=test_results['precision'],
                test_recall=test_results['recall'],
                precision_in_constr=constr_precision,
                opt_gap=None,
                num_kl=num_kl,
                W=self.output['W'],
                b=self.output['b']
            )
        # Write failure information for unsuccessful execution
        else:
            with open(integrated_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Calculate failed iteration number (algorithm_state = -iteration -1 → iteration = -state -1)
                failed_iteration = -self.algorithm_state - 1
                writer.writerow([alg_name, f'Failed in outer iteration {failed_iteration}'])
