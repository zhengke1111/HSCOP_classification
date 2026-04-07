import os
from algorithm import *
from parameter import *
import pprint


# =========== Settings to Solve Multi-class Classification Problem with Precision Constraint ==========
def solve_multi_class_classification_prob(param, dataset_results_csv, dataset_dir):
    """
    Solve multi-class classification with precision constraints for a single precision setting.
    Results are appended to a shared dataset-level CSV, and LogFiles are stored in precision-specific subdirectories.

    Args:
        param (dict): Configuration dictionary containing:
            - data_set (str): Name of dataset (e.g., 'wine', 'fish', 'vehi')
            - sample_size (int): Number of samples to use from dataset (None = all samples)
            - n_splits (int): Number of cross-validation folds
            - folds (list/None): Specific folds to run (None = run all folds)
            - method (dict): Algorithms to execute (keys = method names, values = boolean)
            - start_sol (str/dict): Warm start initialization method ('logistic', 'SVM' or precomputed solutions)
            - model_param (dict): Hyperparameters for gurobi.
            - beta (dict): Precision threshold parameters for each class
            - save_log (bool): Whether to save Gurobi log files
            - console_log (bool): Whether to output Gurobi logs to console
        dataset_results_csv (str): Path to shared dataset-level results CSV
        dataset_dir (str): Path to dataset-level results directory (for creating precision subdirectories)

    Returns:
        None: Results are appended to dataset_results_csv
    """

    # --------------------------
    # Inner Function: Solve MCC with specific algorithm
    # --------------------------
    def run_algorithm(method, X_train, y_train, beta, epsilon, model_params, result_dir, W_start, b_start,
                      save_log=False, console_log=False):
        """
        Execute specific multi-class classification algorithm and return solution object.

        Args:
            method (str): Algorithm identifier ('Full MIP', 'PIP', 'ISA-PIP', 'IDSA-PIP')
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

        # Full MIP
        if method == "Full MIP":
            mip = Model(X=X_train, y=y_train, class_restrict=class_restrict, epsilon=epsilon, beta=beta,
                        model_type='full', ell=None, delta_plus=None, delta_minus=None, model_params=model_params,
                        model_dir=result_dir, model_name='FullModel', save_log=save_log, console_log=console_log)
            mip.formulate_model(W_start, b_start)
            mip.solve_model()
            return mip

        # PIP
        elif method == "PIP":
            pip = PIP(
                X_train, y_train, class_restrict, epsilon, beta,
                model_params, None, ALG_PARAM, result_dir,
                save_log, console_log
            )
            pip.main_computation(pip.iteration_process_fixed_piece, method, W_start, b_start)
            return pip

        # ISA-PIP
        elif method == "ISA-PIP":
            max_out_iter = SHRINKAGE_MAX_OUT_ITER
            epsilon_ = generate_epsilon(max_out_iter)
            shrinkage = []

            for iteration in range(max_out_iter):
                pip_dir = os.path.join(result_dir, f'ISA-PIP_out_iter_{iteration}')
                pip = PIP(
                    X_train, y_train, class_restrict, epsilon_[iteration], beta,
                    model_params, None, ALG_PARAM, pip_dir,
                    save_log, console_log
                )
                alg_name = f'ISA-PIP_out_iter_{iteration}'
                pip.main_computation(pip.iteration_process_fixed_piece, alg_name, W_start, b_start)
                shrinkage.append(pip)

                # Update warm start if algorithm succeeded
                if pip.algorithm_state >= 0:
                    W_start, b_start = pip.output['W'], pip.output['b']
                else:
                    break
            return shrinkage

        # D4-PIP: Use PA-decomposition in the basic PIP algorithm and select at most 4 arbitrary pieces in one iteration
        elif method == "D4-PIP":
            fixed_simplified = PIP(
                X_train, y_train, class_restrict, epsilon, beta,
                model_params, None, ALG_PARAM, result_dir,
                save_log, console_log
            )
            fixed_simplified.main_computation(
                fixed_simplified.iteration_process_enhanced_arbitrary_4,
                method, W_start, b_start
            )
            return fixed_simplified

        # D-PIP: Use PA-decomposition in the basic PIP algorithm and select 1 arbitrary piece in one iteration
        elif method == "D-PIP":
            fixed_simplified_arbitrary = PIP(
                X_train, y_train, class_restrict, epsilon, beta,
                model_params, None, ALG_PARAM, result_dir,
                save_log, console_log
            )
            fixed_simplified_arbitrary.main_computation(
                fixed_simplified_arbitrary.iteration_process_enhanced_arbitrary_1,
                method, W_start, b_start
            )
            return fixed_simplified_arbitrary

        # IDSA4-PIP: A variant of IDSA-PIP where at most 4 arbitrary pieces will be selected in one outer iteration
        elif method == "IDSA4-PIP":
            shrinkage_enhanced_outer = IterativeShrinkage(
                X_train, y_train, class_restrict, beta, model_params,
                ALG_PARAM, result_dir, save_log, console_log
            )
            shrinkage_enhanced_outer.main_computation(
                shrinkage_enhanced_outer.iteration_process_enhanced_arbitrary_4,
                method, W_start, b_start
            )
            return shrinkage_enhanced_outer

        # IDSA-PIP
        elif method == "IDSA-PIP":
            shrinkage_enhanced_arb_outer = IterativeShrinkage(
                X_train, y_train, class_restrict, beta, model_params,
                ALG_PARAM, result_dir, save_log, console_log
            )
            shrinkage_enhanced_arb_outer.main_computation(
                shrinkage_enhanced_arb_outer.iteration_process_enhanced_arbitrary_1,
                method, W_start, b_start
            )
            return shrinkage_enhanced_arb_outer

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

    # Create precision-specific subdirectory (only for LogFiles)
    precision_dir = os.path.join(
        dataset_dir,
        f"{param['data_set']}_{dir_post}_{time_str}"
    )
    # Note: directory will be created by Model/PIP only if save_log=True

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

        # Execute each selected algorithm
        for method, state in param['method'].items():
            if state is True:
                print(f'=============================Start {method} on fold {i}')
                train_size = datafold['X_train'].shape[0]
                result_dir = f'{precision_dir}/{method}_{i}-{n_splits}_{train_size}'
                if param['save_log']:
                    os.makedirs(result_dir, exist_ok=True)  # Ensure directory exists

                # Run the specified algorithm
                solution = run_algorithm(
                    method, datafold['X_train'], datafold['y_train'], param['beta'], epsilon, param['model_param'],
                    result_dir, warm_start['W'], warm_start['b'],
                    param['save_log'], param['console_log']
                )

                # Write algorithm results to shared dataset CSV with precision_threshold
                write_results(method, solution, dataset_results_csv, datafold['X_test'], datafold['y_test'],
                              precision_threshold=param['beta'], fold=i)


def run_multi_class_classification_experiment():
    """
    Organize multi-class classification experiment across datasets and precision thresholds.

    For each dataset:
        - Creates one dataset-level directory with parameters.txt and multi_class_{dataset}_results.csv
        - Iterates over precision threshold combinations
        - Each precision creates a subdirectory for LogFiles only

    Output structure per dataset:
        results/our_results/multi_class_run/{data_set}/
        ├── parameters.txt                          (all precision configurations)
        ├── multi_class_{data_set}_results.csv      (all results with Precision_threshold column)
        ├── {data_set}_{precision_1}_{timestamp}/   (LogFile subdirectories)
        ├── {data_set}_{precision_2}_{timestamp}/
        └── ...

    Key Configurations:
        method (dict): Dictionary of algorithms to execute - keys = algorithm names, values = boolean:
            - 'Full MIP': Full MIP approach
            - 'PIP': PIP approach
            - 'ISA-PIP': ISA-PIP approach
            - 'IDSA-PIP': IDSA-PIP approach

        start_sol (dict): Warm start initialization strategy per dataset:
            - Type 1: 'logistic' = Logistic regression warm start
            - Type 2: 'SVM' = Support Vector Machine warm start
            - Type 3: Precomputed MIP start solutions by solving unconstrained problem

        precision_threshold (dict): Class-specific precision threshold configurations per dataset:
            - Keys: Dataset names
            - Values: List of dictionaries defining precision thresholds for specific classes

    Returns:
        None: All results are written to CSV files and parameter logs
    """
    method = {'Full MIP': True, 'PIP': True, 'ISA-PIP': True, 'IDSA-PIP': True}
    start_sol = {
        'vehi': VEHICLE_MIP_START,
        'wine': RED_WINE_MIP_START,
        'segm': SEGMENTATION_MIP_START,
        'fish': FISH_MIP_START,
        'wave': WAVE_MIP_START,
        'robo': ROBOT_2_MIP_START,
    }
    precision_threshold = {
        'vehi': [{0: 0.62, 3: 0.8}, {0: 0.67, 3: 0.8}, {0: 0.62, 2: 0.8, 3: 0.8}, {0: 0.67, 2: 0.8, 3: 0.8}],
        'wine': [{1: 0.85}, {1: 0.85, 2: 0.65}, {1: 0.9}, {1: 0.9, 2: 0.65}],
        'segm': [{2: 0.8}, {2: 0.85}, {2: 0.8, 4: 0.75}, {2: 0.85, 4: 0.75}],
        'fish': [{0: 0.55, 3: 0.4}, {0: 0.6, 3: 0.4}, {0: 0.55, 1: 0.99, 3: 0.4}, {0: 0.6, 1: 0.99, 3: 0.4}],
        'wave': [{0: 0.89}, {0: 0.91}, {0: 0.89, 1: 0.8}, {0: 0.91, 1: 0.8}],
        'robo': [{2: 0.81}, {2: 0.86}, {2: 0.81, 1: 0.9}, {2: 0.86, 1: 0.9}],
    }

    for data_set in ['wine', 'fish', 'robo', 'segm', 'vehi', 'wave']:
        # Create dataset-level directory
        dataset_dir = f"results/our_results/multi_class_run/{data_set}"
        os.makedirs(dataset_dir, exist_ok=True)

        # Create shared results CSV at dataset level
        dataset_results_csv = os.path.join(dataset_dir, f'multi_class_{data_set}_results.csv')

        # Get sample size for parameters.txt
        data_folds = split_folds(data_set, None, 4)
        sample_size = data_folds[1]['X_train'].shape[0] + data_folds[1]['X_test'].shape[0]

        # Create parameters.txt at dataset level (records all precision thresholds)
        param_txt = os.path.join(dataset_dir, 'parameters.txt')
        param_copy = {
            'data_set': data_set,
            'sample_size': None,
            'n_splits': 4,
            'n_actual_samples': sample_size,
            'epsilon': EPSILON,
            'method': method,
            'model_param': MODEL_PARAM['Param_1'],
            'precision_thresholds': precision_threshold[data_set],
        }
        with open(param_txt, 'w') as file:
            file.write(f"Parameters for solving Multi-class classification on data set {data_set}:\n")
            file.write(pprint.pformat(param_copy))
            file.write("\nAlgorithm parameters:\n")
            file.write(pprint.pformat(ALG_PARAM))

        # Run each precision threshold
        for precision in precision_threshold[data_set]:
            param = {'data_set': data_set, 'sample_size': None, 'n_splits': 4, 'folds': None, 'method': method,
                     'start_sol': start_sol[data_set], 'model_param': MODEL_PARAM['Param_1'], 'beta': precision,
                     'save_log': False, 'console_log': False}
            solve_multi_class_classification_prob(param, dataset_results_csv, dataset_dir)


run_multi_class_classification_experiment()