import os
from algorithm import *
from parameter import *
import pprint


# =========== Settings to Solve Multi-class Classification Problem with Precision Constraint ==========
def solve_multi_class_classification_prob(param):
    """
    Main function to solve multi-class classification with precision constraints
    Executes specified algorithms and writes results to CSV

    Args:
        param (dict): Configuration dictionary containing:
            - data_set (str): Name of dataset to use (e.g., 'red_wine', 'fish')
            - sample_size (int): Number of samples to use from dataset (None = all samples)
            - n_splits (int): Number of cross-validation folds
            - folds (list/None): Specific folds to run (None = run all folds)
            - method (dict): Algorithms to execute (keys = method names, values = boolean)
            - start_sol (str/dict): Warm start initialization method ('logistic', 'SVM' or precomputed solutions)
            - model_param (dict): Hyperparameters for gurobi.
            - beta (dict): Precision threshold parameters for each class

    Returns:
        None: Results are written to CSV files in the generated results directory
    """

    # --------------------------
    # Inner Function: Solve MCC with specific algorithm
    # --------------------------
    def run_algorithm(method, X_train, y_train, beta, epsilon, model_params, result_dir, W_start, b_start):
        """
        Execute specific multi-class classification algorithm and return solution object.

        Args:
            method (str): Algorithm identifier ('MIP', 'F', 'S', 'F_Sim', 'F_Sim_A', 'S_En_Out', 'S_En_A_Out', 'S_En_In')
            X_train (np.ndarray): Training feature matrix
            y_train (np.ndarray): Training label vector
            beta (dict): Precision threshold parameters for each class
            epsilon (float): Approximating parameter in the model
            model_params (dict): Hyperparameters for gurobi.
            result_dir (str): Directory to save method-specific results
            W_start (np.ndarray): Initial weight matrix for warm start
            b_start (np.ndarray): Initial bias vector for warm start

        Returns:
            Union[Model, PIP, IterativeShrinkage, List[PIP], None]:
                Solution object(s) from the executed algorithm
        """
        # Get class restrictions from beta parameters
        class_restrict = list(beta.keys())

        # Mixed Integer Programming (MIP) approach
        if method == "MIP":
            mip = Model(X=X_train, y=y_train, class_restrict=class_restrict, epsilon=epsilon, beta=beta,
                        model_type='full', ell=None, delta_plus=None, delta_minus=None, model_params=model_params,
                        model_dir=result_dir, model_name='FullModel')
            mip.formulate_model(W_start, b_start)
            mip.solve_model()
            return mip

        # epsilon-fixed approach
        elif method == "F":
            fixed = PIP(
                X_train, y_train, class_restrict, epsilon, beta,
                model_params, None, ALG_PARAM, result_dir
            )
            alg_name = 'F'
            fixed.main_computation(fixed.iteration_process_fixed_piece, alg_name, W_start, b_start)
            return fixed

        # epsilon-shrinkage approach
        elif method == "S":
            max_out_iter = SHRINKAGE_MAX_OUT_ITER
            epsilon_ = generate_epsilon(max_out_iter)
            shrinkage = []

            for iteration in range(max_out_iter):
                pip_dir = os.path.join(result_dir, f'S_out_iter_{iteration}')
                pip = PIP(
                    X_train, y_train, class_restrict, epsilon_[iteration], beta,
                    model_params, None, ALG_PARAM, pip_dir
                )
                alg_name = f'S_out_iter_{iteration}'
                pip.main_computation(pip.iteration_process_fixed_piece, alg_name, W_start, b_start)
                shrinkage.append(pip)

                # Update warm start if algorithm succeeded
                if pip.algorithm_state >= 0:
                    W_start, b_start = pip.output['W'], pip.output['b']
                else:
                    break  # Terminate early on failure
            return shrinkage

        # Enhanced epsilon-fixed approach
        elif method == "F_Sim":
            fixed_simplified = PIP(
                X_train, y_train, class_restrict, epsilon, beta,
                model_params, None, ALG_PARAM, result_dir
            )
            alg_name = f'F_Sim'
            fixed_simplified.main_computation(
                fixed_simplified.iteration_process_enhanced_arbitrary_4,
                alg_name, W_start, b_start
            )
            return fixed_simplified

        # epsilon-fixed arbitrary-1 approach
        elif method == "F_Sim_A":
            fixed_simplified_arbitrary = PIP(
                X_train, y_train, class_restrict, epsilon, beta,
                model_params, None, ALG_PARAM, result_dir
            )
            alg_name = f'F_Sim_A'
            fixed_simplified_arbitrary.main_computation(
                fixed_simplified_arbitrary.iteration_process_enhanced_arbitrary_1,
                alg_name, W_start, b_start
            )
            return fixed_simplified_arbitrary

        # Enhanced epsilon-shrinkage approach
        elif method == "S_En_Out":
            shrinkage_enhanced_outer = IterativeShrinkage(
                X_train, y_train, class_restrict, beta, model_params,
                ALG_PARAM, result_dir
            )
            shrinkage_enhanced_outer.main_computation(
                shrinkage_enhanced_outer.iteration_process_enhanced_arbitrary_4,
                method, W_start, b_start
            )
            return shrinkage_enhanced_outer

        # epsilon-shrinkage arbitrary-1 approach
        elif method == "S_En_A_Out":
            shrinkage_enhanced_arb_outer = IterativeShrinkage(
                X_train, y_train, class_restrict, beta, model_params,
                ALG_PARAM, result_dir
            )
            alg_name = f'S_En_Out'
            shrinkage_enhanced_arb_outer.main_computation(
                shrinkage_enhanced_arb_outer.iteration_process_enhanced_arbitrary_1,
                alg_name, W_start, b_start
            )
            return shrinkage_enhanced_arb_outer

        # Enhanced epsilon-shrinkage with inner updating
        elif method == "S_En_In":
            shrinkage_enhanced_inner = IterativeShrinkage(
                X_train, y_train, class_restrict, beta, model_params,
                ALG_PARAM, result_dir
            )
            alg_name = f'S_En_In'
            shrinkage_enhanced_inner.main_computation(
                shrinkage_enhanced_inner.iteration_process_enhanced_inner_update,
                alg_name, W_start, b_start
            )
            return shrinkage_enhanced_inner

        # Return None for unsupported methods
        return None

    # --------------------------
    # Main Execution Pipeline
    # --------------------------
    # Configuration parameters
    epsilon = EPSILON  # Approximation parameter
    random.seed(42)  # Fix random seed for reproducibility
    time_str = time.strftime('%y%m%d-%H%M%S')  # Timestamp for result directory

    # Load cross-validation folds
    data_folds = split_folds(param['data_set'], param['sample_size'], param['n_splits'])
    sample_size = data_folds[1]['X_train'].shape[0] + data_folds[1]['X_test'].shape[0]

    # Create directory postfix from beta values (rounded to 2 decimal places)
    precision_str = [str(np.round(value, 2)) for value in param['beta'].values()]
    dir_post = '_'.join(precision_str)

    # Create main results directory
    results_dir = (
        f"results/our_results/multi_class_run/{param['data_set']}/"
        f"{time_str}_{param['data_set']}_{sample_size}_{epsilon:.0e}_{dir_post}"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Save parameters to text file for reproducibility
    param_txt = os.path.join(results_dir, 'parameters.txt')
    with open(param_txt, 'w') as file:
        file.write(f"Parameters for solving Multi-class classification on data set {param['data_set']}:\n")
        file.write(pprint.pformat(param))
        file.write("\nAlgorithm parameters:\n")
        file.write(pprint.pformat(ALG_PARAM))

    # Path for integrated results CSV
    integrated_result_csv = os.path.join(
        results_dir,
        f'integrated_result_{sample_size}_{epsilon:.0e}.csv'
    )
    n_splits = param['n_splits']

    # Determine which folds to execute
    if param['folds'] is None:
        folds = range(1, param['n_splits'] + 1)
    else:
        folds = param['folds']

    # Process each fold
    for i in folds:
        datafold = data_folds[i]

        # Generate warm start solution
        if param['start_sol'] == 'logistic':
            warm_start = generate_logistic_start(datafold['X_train'], datafold['y_train'])
        elif param['start_sol'] == 'SVM':
            warm_start = generate_svm_start(datafold['X_train'], datafold['y_train'])
        else:
            warm_start = param['start_sol'][i]

        # Calculate initial metrics from warm start
        train_result = classification_metric(
            datafold['X_train'], datafold['y_train'],
            warm_start['W'], warm_start['b']
        )
        test_result = classification_metric(
            datafold['X_test'], datafold['y_test'],
            warm_start['W'], warm_start['b']
        )

        # Write warm start results to CSV
        write_single_integrated_result(
            integrated_csv=integrated_result_csv,
            title='start_sol',
            execution_time=None,
            model_time=None,
            final_improvement_time=None,
            obj_val=None,
            train_acc_margined=train_result['acc_margined'],
            train_acc=train_result['accuracy'],
            train_precision=train_result['precision'],
            train_recall=train_result['recall'],
            test_acc_margined=test_result['acc_margined'],
            test_acc=test_result['accuracy'],
            test_precision=test_result['precision'],
            test_recall=test_result['recall'],
            precision_in_constr=None,
            opt_gap=None,
            num_kl=None,
            W=warm_start['W'],
            b=warm_start['b']
        )

        # Execute each selected algorithm
        for method, state in param['method'].items():
            if state is True:
                print(f'=============================Start {method} on fold {i}')
                train_size = datafold['X_train'].shape[0]
                result_dir = f'{results_dir}/{method}_{i}-{n_splits}_{train_size}'
                os.makedirs(results_dir, exist_ok=True)  # Ensure directory exists

                # Run the specified algorithm
                solution = run_algorithm(
                    method, datafold['X_train'], datafold['y_train'], param['beta'], epsilon, param['model_param'],
                    result_dir, warm_start['W'], warm_start['b']
                )

                # Write algorithm results to CSV
                write_results(method, solution, integrated_result_csv, datafold['X_test'], datafold['y_test'])


def run_multi_class_classification_experiment():
    """
    Organize multi-class classification experiment across datasets and precision thresholds.
    Iterates over specified datasets, precision threshold combinations, and executes selected algorithms.

    Key Configurations:
        method (dict): Dictionary of algorithms to execute - keys = algorithm names, values = boolean (True = run algorithm):
            - 'MIP': Full MIP approach
            - 'F': epsilon-fixed approach
            - 'S': epsilon-shrinkage approach
            - 'F_Sim_A': epsilon-fixed arbitrary-1 approach
            - 'S_En_A_Out': epsilon-shrinkage arbitrary-1 approach

        start_sol (dict): Warm start initialization strategy per dataset (matches solve_multi_class_classification_prob):
            - Type 1: 'logistic' = Logistic regression warm start
            - Type 2: 'SVM' = Support Vector Machine warm start
            - Type 3: Precomputed MIP start solutions by solving unconstrained problem

        precision_threshold (dict): Class-specific precision threshold configurations per dataset:
            - Keys: Dataset names ('red_wine', 'fish', 'robot_2', 'segmentation', 'vehicle', 'wave')
            - Values: List of dictionaries where each dict defines precision thresholds for specific classes
              (e.g., {0: 0.60, 3: 0.4} = class 0 requires 60% precision, class 3 requires 40% precision)

    Execution Flow:
        1. For each dataset in the target list
        2. For each precision threshold combination for the dataset
        3. Build parameter dictionary for solve_multi_class_classification_prob
        4. Execute classification pipeline with specified algorithms/settings
        5. Results are automatically saved to timestamped directories under Result/[dataset]/

    Returns:
        None: All results are written to CSV files and parameter logs in the Result directory
    """
    method = {'MIP': True, 'F': True, 'S': True, 'F_Sim_A': True, 'S_En_A_Out': True}
    start_sol = {
        'vehicle': VEHICLE_MIP_START,
        'red_wine': RED_WINE_MIP_START,
        'segmentation': SEGMENTATION_MIP_START,
        'fish': FISH_MIP_START,
        'wave': WAVE_MIP_START,
        'robot_2': ROBOT_2_MIP_START,
    }
    precision_threshold = {
        'vehicle': [{0: 0.60, 3: 0.4}, {0: 0.62, 3: 0.4}, {0: 0.55, 1: 0.99, 3: 0.4}, {0: 0.60, 1: 0.99, 3: 0.4}],
        'red_wine': [{1: 0.85}, {1: 0.90}, {1: 0.85, 2: 0.65}, {1: 0.90, 2: 0.65}],
        'segmentation': [{2: 0.8}, {2: 0.85}, {2: 0.8, 4: 0.75}, {2: 0.85, 4: 0.75}],
        'fish': [{0: 0.55, 3: 0.4}, {0: 0.6, 3: 0.4}, {0: 0.55, 1: 0.99, 3: 0.4}, {0: 0.6, 1: 0.99, 3: 0.4}],
        'wave': [{0: 0.89}, {0: 0.91}, {0: 0.89, 1: 0.8}, {0: 0.91, 1: 0.8}],
        'robot_2': [{2: 0.81}, {2: 0.86}, {2: 0.81, 1: 0.9}, {2: 0.86, 1: 0.9}],
    }
    for data_set in ['red_wine', 'fish', 'robot_2', 'segmentation', 'vehicle', 'wave']:
        for precision in precision_threshold[data_set]:
            param = {'data_set': data_set, 'sample_size': None, 'n_splits': 4, 'folds': None, 'method': method,
                     'start_sol': start_sol[data_set], 'model_param': MODEL_PARAM['Param_1'], 'beta': precision}
            solve_multi_class_classification_prob(param)


run_multi_class_classification_experiment()
