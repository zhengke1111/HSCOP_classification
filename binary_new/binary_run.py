import os
from algorithm import *
from parameter import *
from utils import *
import pprint
import copy
import random

def solve_binary_classification_prob(param, dataset_results_csv, dataset_dir):

    def run_algorithm(method, X_train, y_train, dataset, beta, epsilon, model_params, result_dir, w_start, b_start, save_log=False, console_log=False):

        if method == "Full MIP":
            mip = Model(X=X_train, y=y_train, epsilon=epsilon, beta=beta, model_type='full', delta_plus=None, delta_minus=None, model_params=model_params, model_dir=result_dir,
                        model_name='FullModel', save_log=save_log, console_log=console_log)
            mip.formulate_model(w_start, b_start)
            mip.solve_model()
            return mip
        
        elif method == "Fixed PIP":
            pip = PIP(X_train=X_train, y_train=y_train, dataset=dataset, epsilon=epsilon, beta=beta, model_params=model_params, algorithm_params=ALG_PARAM, alg_dir=result_dir, save_log=save_log, console_log=console_log)
            pip.main_computation(pip.iteration_process, method, w_start, b_start)
            return pip
        
        elif method == "Shrinkage PIP":
            max_out_iter = SHRINKAGE_MAX_OUT_ITER
            epsilon_ = generate_epsilon(max_out_iter)
            shrinkage = []

            for iteration in range(max_out_iter):
                pip_dir = os.path.join(result_dir, f'Shrinkage-PIP_out_iter_{iteration}')
                pip = PIP(X_train=X_train, y_train=y_train, dataset=dataset, epsilon=epsilon_[iteration], beta=beta, model_params=model_params, algorithm_params=ALG_PARAM, alg_dir=pip_dir, save_log=save_log, console_log=console_log)
                alg_name = f'Shrinkage-PIP_out_iter_{iteration}'
                pip.main_computation(pip.iteration_process, alg_name, w_start, b_start)
                shrinkage.append(pip)

                if pip.algorithm_state >= 0:
                    w_start, b_start = pip.output['w'], pip.output['b']
                else:
                    break
            return shrinkage

        elif method == "U-PIP":
            unconstrained = PIP(X_train=X_train, y_train=y_train, dataset=dataset, epsilon=None, beta=None, model_params=model_params, algorithm_params=ALG_PARAM, alg_dir=result_dir, save_log=save_log, console_log=console_log)
            unconstrained.main_computation_unconstrained(unconstrained.iteration_process_unconstrained, method, w_start, b_start)
            return unconstrained
        


    # --------------------------
    # Main Execution Pipeline
    # --------------------------
    # Configuration parameters
    epsilon = EPSILON  # Approximation parameter

    for run in range(1,5):
        X_train = param['data_splits'][run]['X_train']
        y_train = param['data_splits'][run]['y_train']
        X_test = param['data_splits'][run]['X_test']
        y_test = param['data_splits'][run]['y_test']
        y_train = 2*y_train.values - 1                  # convert 0/1 to 1/-1
        y_test = 2*y_test.values - 1

        for method, state in param['method'].items():
            if state is True:
                start_sol_copy = copy.deepcopy(param['start_sol'][run])
                result_dir = os.path.join(dataset_dir, f"{param['dataset']}_run-{run}")
                result_dir_method = os.path.join(result_dir, f"{method}")
                if param['save_log']:
                    os.makedirs(result_dir, exist_ok=True)
                    os.makedirs(result_dir_method, exist_ok=True)

                if method != 'U-PIP':
                    for beta in THRESHOLD_GRID[param['dataset']]:
                        beta_str = str(np.round(beta, 2))
                        result_dir_method_beta = os.path.join(result_dir_method, f'beta-{beta_str}')
                        solution = run_algorithm(method, X_train, y_train, param['dataset'], beta, epsilon, param['model_param'], result_dir_method_beta, 
                                            start_sol_copy['w'], start_sol_copy['b'], param['save_log'], param['console_log'])

                        if method == 'Full MIP':
                            solution.write_integrated_results(dataset_results_csv=dataset_results_csv, dataset=param['dataset'], split=run, method=method, beta=beta, X_test=X_test, y_test=y_test)

                        if method == 'Fixed PIP':
                            solution.write_integrated_results(dataset_results_csv=dataset_results_csv, split=run, method=method, beta=beta, X_test=X_test, y_test=y_test)

                        if method == 'Shrinkage PIP':
                            final_pip = solution[-1]

                            train_results = evaluate_binary(X_train, y_train, final_pip.output['w'], final_pip.output['b'])
                            test_results = evaluate_binary(X_test, y_test, final_pip.output['w'], final_pip.output['b'])

                            total_model_time = 0
                            for pip in solution:
                                all_models = extract_inner_values(pip.model_dict)
                                total_model_time += np.sum([model.model.Runtime for model in all_models if model is not None])

                            write_single_integrated_result(results_csv=dataset_results_csv, dataset=param['dataset'], split=run, method=method, beta=beta, objective_value=final_pip.output['obj_val'],
                                                            optimality_gap=None, time=total_model_time, actual_time=None, gamma=final_pip.output['gamma'] if final_pip.output['gamma'] is not None else None,
                                                            train_acc_margin=train_results['acc_margin'], test_acc_margin=test_results['acc_margin'], train_acc=train_results['accuracy'], test_acc=test_results['accuracy'],
                                                            train_prec=train_results['precision'], test_prec=test_results['precision'], train_recall=train_results['recall'], test_recall=test_results['recall'])
                else:
                    solution = run_algorithm(method, X_train, y_train, param['dataset'], None, None, param['model_param'], result_dir_method, start_sol_copy['w'], start_sol_copy['b'], param['save_log'], param['console_log'])
                    solution.write_integrated_results(dataset_results_csv=dataset_results_csv, split=run, method=method, beta=None, X_test=X_test, y_test=y_test)

                



def run_binary_experiment(method_dict):
    dataset_list_ = DATASET_LIST

    for dataset in dataset_list_:
        dataset_dir = f"binary_new/results/{dataset}"
        os.makedirs(dataset_dir, exist_ok=True)

        dataset_results_csv = os.path.join(dataset_dir, f'binary_{dataset}_results.csv')

        X, y = sample_data(dataset=dataset)
        data_splits = {}
        start_sol = {}
        start_sol_copy = {}
        for run in range(1,5):
            data_splits[run] = split_data(X, y, random_state = 42 + run)
            X_train = data_splits[run]['X_train']
            y_train = data_splits[run]['y_train']
            start_sol[run] = train_lr_model(X_train, y_train)
            start_sol_copy[run] = copy.deepcopy(start_sol[run])
        param = {
            'dataset': dataset,
            'data_splits': data_splits,
            'method': method_dict,
            'start_sol': start_sol_copy,
            'model_param': MODEL_PARAM,
            'threshold': THRESHOLD_GRID[dataset],
            'save_log': True,
            'console_log': True
        }
        solve_binary_classification_prob(param, dataset_results_csv, dataset_dir)

method_dict = {'Full MIP': False, 'Fixed PIP': False, 'Shrinkage PIP':True, 'U-PIP': True}
run_binary_experiment(method_dict)