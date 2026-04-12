import os
from algorithm import *
from parameter import *
from utils import *
import pprint
import copy


# =========== Settings to Solve Multi-class Classification Problem with Precision Constraint ==========
def solve_tree_classification_prob(param, dataset_results_csv, dataset_dir, pareto = False):

    # --------------------------
    # Inner Function: Solve TCC with specific algorithm
    # --------------------------
    def run_algorithm(method, X_train, y_train, dataset, depth, tau_0, beta, epsilon, model_params, result_dir, a_start, b_start, c_start,
                      save_log=False, console_log=False):
        
        # Get class restrictions from beta parameters
        class_restrict = list(beta.keys())

        # Full MIP
        if method == "Full MIP":
            mip = Model(X=X_train, y=y_train, depth=depth, tau_0=tau_0, class_restrict=class_restrict, epsilon=epsilon, beta=beta,
                        model_type='full', ell=None, delta_plus=None, delta_minus=None, model_params=model_params,
                        model_dir=result_dir, model_name='FullModel', save_log=save_log, console_log=console_log)
            mip.formulate_model(a_start, b_start, c_start)
            mip.solve_model()
            return mip

        # PIP
        elif method == "PIP":
            pip = PIP(
                X_train, y_train, dataset, depth, tau_0, class_restrict, epsilon, beta,
                model_params, None, ALG_PARAM, result_dir,
                save_log, console_log
            )
            pip.main_computation(pip.iteration_process_fixed_piece, method, a_start, b_start, c_start)
            return pip

        # ISA-PIP
        elif method == "ISA-PIP":
            max_out_iter = SHRINKAGE_MAX_OUT_ITER
            epsilon_ = generate_epsilon(max_out_iter)
            shrinkage = []

            for iteration in range(max_out_iter):
                pip_dir = os.path.join(result_dir, f'ISA-PIP_out_iter_{iteration}')
                pip = PIP(
                    X_train, y_train, dataset, depth, tau_0, class_restrict, epsilon_[iteration], beta,
                    model_params, None, ALG_PARAM, pip_dir,
                    save_log, console_log
                )
                alg_name = f'ISA-PIP_out_iter_{iteration}'
                pip.main_computation(pip.iteration_process_fixed_piece, alg_name, a_start, b_start, c_start)
                shrinkage.append(pip)

                # Update warm start if algorithm succeeded
                if pip.algorithm_state >= 0:
                    a_start, b_start, c_start = pip.output['a'], pip.output['b'], pip.output['c']
                else:
                    break
            return shrinkage

        # D4-PIP: Use PA-decomposition in the basic PIP algorithm and select at most 4 arbitrary pieces in one iteration
        elif method == "D4-PIP":
            fixed_simplified = PIP(
                X_train, y_train, dataset, depth, tau_0, class_restrict, epsilon, beta,
                model_params, None, ALG_PARAM, result_dir,
                save_log, console_log
            )
            fixed_simplified.main_computation(
                fixed_simplified.iteration_process_enhanced_arbitrary_4,
                method, a_start, b_start, c_start
            )
            return fixed_simplified

        # D-PIP: Use PA-decomposition in the basic PIP algorithm and select 1 arbitrary piece in one iteration
        elif method == "D-PIP":
            fixed_simplified_arbitrary = PIP(
                X_train, y_train, dataset, depth, tau_0, class_restrict, epsilon, beta,
                model_params, None, ALG_PARAM, result_dir,
                save_log, console_log
            )
            fixed_simplified_arbitrary.main_computation(
                fixed_simplified_arbitrary.iteration_process_enhanced_arbitrary_1,
                method, a_start, b_start, c_start
            )
            return fixed_simplified_arbitrary

        # IDSA4-PIP: A variant of IDSA-PIP where at most 4 arbitrary pieces will be selected in one outer iteration
        elif method == "IDSA4-PIP":
            shrinkage_enhanced_outer = IterativeShrinkage(
                X_train, y_train, dataset, depth, tau_0, class_restrict, beta, model_params,
                ALG_PARAM, result_dir, save_log, console_log
            )
            shrinkage_enhanced_outer.main_computation(
                shrinkage_enhanced_outer.iteration_process_enhanced_arbitrary_4,
                method, a_start, b_start, c_start
            )
            return shrinkage_enhanced_outer

        # IDSA-PIP
        elif method == "IDSA-PIP":
            shrinkage_enhanced_arb_outer = IterativeShrinkage(
                X_train, y_train, dataset, depth, tau_0, class_restrict, beta, model_params,
                ALG_PARAM, result_dir, save_log, console_log
            )
            shrinkage_enhanced_arb_outer.main_computation(
                shrinkage_enhanced_arb_outer.iteration_process_enhanced_arbitrary_1,
                method, a_start, b_start, c_start
            )
            return shrinkage_enhanced_arb_outer

        elif method == "U-PIP":
            unconstrained = PIP(
                X_train, y_train, dataset, depth, tau_0, class_restrict, None, None,
                model_params, None, ALG_PARAM, result_dir,
                save_log, console_log
            )
            unconstrained.main_computation_unconstrained(unconstrained.iteration_process_unconstrained, method, a_start, b_start, c_start)
            return unconstrained
        # Return None for unsupported methods
        return None
    

    # --------------------------
    # Main Execution Pipeline
    # --------------------------
    # Configuration parameters
    epsilon = EPSILON  # Approximation parameter
    time_str = time.strftime('%y%m%d-%H%M%S')  # Timestamp for result directory

    for run in range(1,5):
        X_train = param['data_splits'][run]['X_train']
        y_train = param['data_splits'][run]['y_train']
        X_test = param['data_splits'][run]['X_test']
        y_test = param['data_splits'][run]['y_test']

        if pareto == False:
            for method, state in param['method'].items():
                if state is True:
                    start_sol_copy = copy.deepcopy(param['start_sol'][run])
                    result_dir = os.path.join(dataset_dir, f"{param['dataset']}_depth-{param['depth']}_run-{run}")
                    solution = run_algorithm(method, X_train, y_train, param['dataset'], param['depth'], param['tau_0'][run], param['beta'], epsilon, param['model_param'], 
                                            result_dir, start_sol_copy['a'], start_sol_copy['b'], start_sol_copy['c'], param['save_log'], param['console_log'])
                # write_results
        else:
            for method, state in param['method'].items():
                if state is True:
                    start_sol_copy = copy.deepcopy(param['start_sol'][run])
                    result_dir = os.path.join(dataset_dir, f"{param['dataset']}_depth-{param['depth']}_run-{run}")
                    if method == 'IDSA-PIP':
                        for threshold in THRESHOLD_GRID[param['dataset']][param['depth']]:
                            beta = {param['key_beta']: threshold}
                            solution = run_algorithm(method, X_train, y_train, param['dataset'], param['depth'], param['tau_0'][run], beta, epsilon, param['model_param'], 
                                                result_dir, start_sol_copy['a'], start_sol_copy['b'], start_sol_copy['c'], param['save_log'], param['console_log'])
                    elif method == 'U-PIP':
                        beta = {param['key_beta']: None}
                        solution = run_algorithm(method, X_train, y_train, param['dataset'], param['depth'], param['tau_0'][run], beta, None, param['model_param'], 
                                                result_dir, start_sol_copy['a'], start_sol_copy['b'], start_sol_copy['c'], param['save_log'], param['console_log'])
                # write_results


    

        

def run_tree_experiment(method_dict, depth_list, pareto = False):
    
    for dataset in DATASET_LIST:
        dataset_dir = f"results/{dataset}"
        os.makedirs(dataset_dir, exist_ok=True)

        dataset_results_csv = os.path.join(dataset_dir, f'tree_{dataset}_results.csv')

        for depth in depth_list:
            X, y = sample_data(dataset=dataset)
            y = y.values + 1
            counter_result = Counter(y)
            max_count = max(counter_result.values())
            most_common_classes = [cls for cls, count in counter_result.items() if count == max_count]
            key_beta = min(most_common_classes).item()
            data_splits = {}
            initial_train = {}
            start_sol = {}
            start_sol_copy = {}
            initial_train_results = {}
            tau_0 = {}
            for run in range(1,5):
                data_splits[run] = split_data(X, y, random_state = 42 + run)
                X_train = data_splits[run]['X_train']
                y_train = data_splits[run]['y_train']
                start_sol[run] = train_decisiontree_model(X_train, y_train, max_depth = depth)
                start_sol_copy[run] = copy.deepcopy(start_sol[run])
                initial_train_results[run] = evaluate_tree(X_train, y_train, start_sol[run]['a'], start_sol[run]['b'], start_sol[run]['c'], depth)
                initial_train[run] = initial_train_results[run]['frac']

                p = X_train.shape[1]
                if p>5:
                    if REUSE_TAU_0 == False:                                            
                        tau_lb = max(2,math.ceil(p/2)-3)                        # Upper bound: \lceil p/2 \rceil +3
                        tau_ub = min(p,math.ceil(p/2)+3)                        # Lower bound: \lceil p/2 \rceil -3
                        random.seed(42)                                         # Random seeds depend on the realtime timestamp
                        tau_0[run] =  random.choice(range(tau_lb, tau_ub + 1))  # Randomly select \tau_0 \in [\lceil p/2 \rceil -3, \lceil p/2 \rceil +3]
                        random.seed()
                    else:
                        if pareto == False:
                            df = pd.read_csv(f"tree/dataset/decisiontree_tau_0.csv")
                        else:
                            df = pd.read_csv(f"tree/dataset/decisiontree_pareto_tau_0.csv")
                        df["dataset"] = df["dataset"].astype(str)
                        df["depth"]   = df["depth"].astype(int)
                        df["split"]     = df["split"].astype(int)
                        tau_map = df.set_index(["dataset", "depth", "split"])["tau_0"].to_dict()
                        tau_0[run] = tau_map[(dataset, depth, run)]
                else:
                    tau_0[run] = None
            
            if pareto == False:
                prec_values = [initial_train[run][f'prec{key_beta}'] for run in initial_train]
                if -1 in prec_values:
                    threshold = (np.ceil(np.mean(y == 1)*100)/100).item()
                elif all(p == 1 for p in prec_values):
                    threshold = 1
                elif 1 in prec_values:
                    threshold = (np.ceil(max(p for p in prec_values if p != 1)*100)/100).item()
                else:
                    threshold = (np.ceil(max(prec_values)*100)/100).item()

                beta = {key_beta: threshold}

                param = {'dataset': dataset,
                         'depth': depth,
                         'tau_0': tau_0,
                         'data_splits': data_splits,
                         'method': method_dict,
                         'start_sol': start_sol_copy,
                         'model_param': MODEL_PARAM,
                         'beta': beta,
                         'save_log': False, 
                         'console_log': True
                         }
                
                solve_tree_classification_prob(param, dataset_results_csv, dataset_dir, pareto)

            
            # ========== Grid thresholds for pareto comparison ==========
            if pareto == True:
                param = {'dataset': dataset,
                    'depth': depth,
                    'tau_0': tau_0,
                    'data_splits': data_splits,
                    'method': method_dict,
                    'start_sol': start_sol_copy,
                    'model_param': MODEL_PARAM,
                    'key_beta': key_beta,
                    'threshold': THRESHOLD_GRID[dataset],
                    'save_log': False, 
                    'console_log': True
                    }
        
                solve_tree_classification_prob(param, dataset_results_csv, dataset_dir, pareto)



method = {'Full MIP': False, 'IDSA-PIP': False, 'U-PIP': True}
depth_list = [2, 3, 4]
run_tree_experiment(method, depth_list, pareto = True)
