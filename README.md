# HSCOP_classification
This repository contains implementations of multi-class classification in "Solving Constrained Affine Heaviside Composite Optimiation Problem by a Progressive IP Approach", by Ke Zheng, Junyi Liu, Yurui Wang, and Jong-Shi Pang.

This project consists of two parts: 
- Score-based multi-class classification, and 
- Tree-based multi-class classification.

## Setup
- **Python 3.11**

    All computations are coded in **Python 3.11**, except `decisiontree/decisiontree_pareto/binoct-master/run_exp.py`, which needs **Python 3.10** to support the `cplex==22.1.2.0`.

- **Python packages**

    Listed in `requirements.txt`, including

    - Basic data processing and computation: `numpy`, `pandas` and `scipy`.

    - **Solvers**: 
        - `gurobipy==11.0.3`: the main solver to implement PIP methods. A valid Gurobi Academic License is required.

        - `cplex==22.1.2.0`: the solver of `BinOCT`, NOT required for PIP methods. If encountering an Error like  ```ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found```, please enter the following in terminal (replace `/home/zhengke` by your own path to `anaconda`):

            ```bash
            strings ~/anaconda3/lib/libstdc++.so.6.0.29 | grep CXXABI
            export LD_LIBRARY_PATH="/home/zhengke/anaconda3/envs/python3.10/lib:$LD_LIBRARY_PATH"
            ```

        - `scikit-learn==1.5.2`: the solver of `CART`, which is required for generating the initial solution of PIP methods.


    
## Overview
The content of this repository is as follows:
- `score_based`

    - `dataset`: 6 datasets used in experiments.

    - `results`: Output results directory.

        - `our_results/score_based_run/{data_set}`: Experiment results organized by dataset, each containing:

            - `parameters.txt`: Precision configurations for the dataset. Records hyperparameters (epsilon, method, model_param) and all precision_threshold combinations used in experiments.

            - `score_based_{data_set}_results.csv`: Experiment results presented in the main text.

            - `{data_set}_{precision}_{timestamp}/`: (Optional) LogFile subdirectories per precision setting. Only present when `save_log=True` is enabled.

    - `algorithm.py`: Core algorithm classes.

        - `PIP`: Solves one $\varepsilon$-approximation problem via the PIP algorithm, supporting fixed-piece and arbitrary-piece selection strategies.

        - `IterativeShrinkage`: Solves the original problem via iterative shrinkage algorithms:

            - `ISA-PIP`: Iterative Shrinkage Algorithm

            - `IDSA-PIP`: Iterative Shrinkage Algorithm with PA-decomposition

    - `callback.py`: Gurobi callback for tracking optimality gap, improvement time, and early termination.

    - `model.py`: `Model` class for constructing and solving full or partial $\varepsilon$-approximation MIP problems via Gurobi.

    - `score_based_run.py`: Main entry point. Configure datasets, algorithms, and precision thresholds in `run_score_based_classification_experiment()`, then run this file. Available methods:
        `Full MIP`, `PIP`, `ISA-PIP`, `IDSA-PIP`

    - `score_based_pareto_run.py`: Trains baseline classifiers (LogisticRegression, LinearSVM, Perceptron, Ridge) via cross-validation. Can be used to generate Pareto curves comparing baseline methods against our algorithms (results not included in main text).

    - `parameter.py`: Centrally configures Gurobi parameters, algorithm hyperparameters, dataset paths, and precomputed MIP warm start values (`MIP_START_SOL`).

    - `utils.py`: Data loading, big-M computation, metric calculations, and result writing utilities.

- `decisiontree`

    - `dataset`: 8 datasets used in experiments.

    - `results`: Output results directory.

        - `decisiontree_results.csv`: results of tree-based classification, 8 datasets, depth-2,3,4, method `Full MIP` and `IDSA-PIP`.

        - `decisiontree_pareto_blsc_results.csv`, `decisiontree_pareto_ctmc_results.csv`: combined results of `C-PIP`, `FlowOCT`, `C-BinOCT`, `U-PIP`, `BendersOCT`, `U-BinOCT`, `CART`, which we present in Pareto curves. For their raw output, see the folder `decisiontree_pareto_raw_output`.

        - `depth2`, `depth3`, `depth4`: empty folders to store (future) detailed results.

    - `decisiontree_pareto`: Existing methods in literature, including

        - `binoct-master`: `BinOCT` [Learning optimal classification trees using a binary linear program formulation](https://ojs.aaai.org/index.php/AAAI/article/view/3978)

            - `run_exp.py`: **run** this script to reproduce results of `U-BinOCT` and `C-BinOCT`.

            - `results`: store the results.

        - `StrongTree-master`: `BendersOCT` and `FlowOCT` [Strong Optimal Classification Trees](https://sites.google.com/view/sina-aghaei/home)

            - `Code/run_exp.py`: **run** this script to reproduce results of `BendersOCT` and `FlowOCT`.

            - `Results`: store the results

    - `decisiontree_run.py`: **run** this script to reproduce results of `Full MIP` and `IDSA-PIP`(`C-PIP`) and `U-PIP`.  
        
    - `CART_run.py`: **run** this script to reproduce results of `CART`.

    - `model.py`: build and solve an MIP for a single PIP partial problem in a tree-based classification problem. 

    - `algorithm.py`: determine whether to continue or stop, and, if continuing, decide whether to enlarge or shrink the in-between sets ${\cal J}$. 

    - `callback.py`: `callback` mechanism for solving MIP problems. The parameter used in `MIP_tree_callback.py` are stored in `callback_data_tree.py`.

    - `utils.py`: utility file to store some commonly used functions in this project. 

## Usage
### Multi-class classification `score_based`
- `score_based_run.py`: Main entry point. Run this script to execute experiments.

    **Output structure per dataset:**
    ```
    results/our_results/score_based_run/{data_set}/
    ├── parameters.txt                      # Experiment configurations
    ├── score_based_{data_set}_results.csv  # Combined results CSV
    └── {data_set}_{precision}_{timestamp}/ # Optional LogFile subdirectories (save_log=True only)
    ```

    **CSV columns:** `Precision_threshold, Fold, method, obj, time, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall`

    **Configuration in `run_score_based_classification_experiment()`:**
        - `dataset_list`: List of datasets to run. Available: `['wine', 'fish', 'robo', 'segm', 'vehi', 'wave']`
        - `method`: Dictionary of algorithms to execute. Set to `True` to enable:
            ```python
            method = {'Full MIP': True, 'PIP': True, 'ISA-PIP': True, 'IDSA-PIP': True}
            ```
        - `precision_threshold`: Dictionary of precision configurations per dataset. Example:
            ```python
            precision_threshold = {
                'wine': [{1: 0.85}, {1: 0.90}, {1: 0.85, 2: 0.65}, {1: 0.90, 2: 0.65}],
                # ... other datasets
            }
            ```
        - `n_splits`: Number of cross-validation folds (default: 4)
        - `folds`: Specific folds to run (`None` = run all folds)
        - `save_log`: Whether to save Gurobi log files (default: `False`)
        - `console_log`: Whether to output Gurobi logs to console (default: `False`)

    **Global parameters in `parameter.py`:**
        - `EPSILON`: Approximation parameter (default: 1e-5) used in PIP
        - `RHO`: Penalty coefficient for infeasibility
        - `MODEL_PARAM`: Gurobi solver parameters
        - `ALG_PARAM`: Algorithm control parameters (iteration limits, ratios, etc.)
        - `SHRINKAGE_MAX_OUT_ITER`: Maximum outer iterations for shrinkage algorithms
        - `MIP_START_SOL`: Precomputed warm start solutions per dataset

    **Output path configuration:**
        - Modify line 284 in `score_based_run.py` to change results directory:
            ```python
            dataset_dir = f"results/our_results/score_based_run/{data_set}"  # Default path
            # Change to your custom path, e.g.:
            # dataset_dir = f"my_results/{data_set}"
            ```

### Tree-based classification `decisiontree`
- `decisiontree_run.py`: run it to produce the results of `Full MIP`,   `IDSA-PIP`(`C-PIP`) and `U-PIP`.

    - At the bottom of this file, we can adjust the following parameters in the function `decisiontree_run`:
        - `dataset_list`: a list of names of datasets to train decision tree classifier.
        
        - `method_list`: a list of methods. `1` for `Full MIP`, `7` for `IDSA-PIP`(`C-PIP`), `8` for `U-PIP`. `2~6` are not presented in article.  
        
        - `pareto`: `True` if run `C-PIP` with a series of precision thresholds, or run `U-PIP` without a precision threshold; `False` otherwise. 
        
        - `reuse_tau_0`: `True` if use $\tau_0$ read from existing files, `False` otherwise. Set as `True` when reproduce the results or run `C-PIP` to read the 
        same $\tau_0$ as `U-PIP`. 

    - In the function `decisiontree_constraint`, we can adjust the following dictionaries:
        - `settings`: 
            - `'epsilon'`: $\varepsilon$ used in `Full MIP` and `PIP with fixed` $\varepsilon$.
            - `'epsilon_nu'`: initial $\varepsilon_{\nu}\mid_{\nu=0}$.
            - `'rho'`: penality coefficient $\lambda$ for infeasbility.  
        - `stop_rule`:
            - `'timelimit'`: TimeLimit for `Full MIP`, we set as 3600s. 
            - `'base_rate'`: initial integer ratio $r^0$. 
            - `'pip_max_rate'`: maximal integer ratio $r_{\max}$. 
            - `'expansion_rate'`: expansion ratio $r_{\Delta}$.
            - `'unchanged_iters'`: maximal unchanged iteration number $\tilde{\mu}_{\max}$. 
            - `'max_iteration'`: maximal iteration number $\mu_{\max}$. 
            - `'max_outer_iter'`: maximal $\nu$ in `Algorithm II: Iterative decomposed shrinkage algorithm`.
        - `file_path`: the path of input and output files.

    - If you want to modify `Time limit` of `PIP`, please go to `callback_data_tree.py`, which is a separate file for `MIP_tree_callback.py` to call.
        - `timelimit = 30`: when solving PIP partial problems, if the objective value remains unchanged for 30s, Gurobi procedure will be terminated. If you want to modify it for some method, for example, method `7`(`IDSA-PIP`), as `timelimit = 60`, please then go to `MIP_tree.py`, modify:
            ```Python
            if method == 7:  # 'IDSA-PIP'
                callback_data_tree.timelimit = 60
            ```
        - `mip_timelimit = 300`: when solving PIP partial problems, if the running time exceeds 300s,  Gurobi procedure will be terminated. If you want to modify it for some method, for example, method `7`(`IDSA-PIP`), as `mip_timelimit = 600`, please then go to `MIP_tree.py`, modify:
            ```Python
            if method == 7:  # 'IDSA-PIP'
                callback_data_tree.mip_timelimit = 600
            ```
- `decisiontree_pareto/binoct-master/run_exp.py`: run it to produce the results of `U-BinOCT` and `C-BinOCT`.

- `decisiontree_pareto/StrongTree-master/Code/StrongTree/run_exp.py`: run it to produce the results of `BendersOCT` and `FlowOCT`. 
    ```Python
    for dataset in ['blsc', 'ctmc']:
        validation(dataset)                                 # For validate \lambda
        run_strongOCT(result_dir, dataset, 'BendersOCT')    # With \lambda, run BendersOCT
        run_strongOCT(result_dir, dataset, 'FlowOCT')       # With \lambda, run FlowOCT
    ```

- `CART_run.py`: run it to produce the results of `CART`. The results will be stored in `decisiontree/results/CART_results.csv`. 


## Notes
- For `C-PIP`, in `decisiontree_pareto_blsc_results.csv`, the column `restricted_class`=`0`, but in `decisiontree/results/decisiontree_pareto_raw_output/blsc/C-PIP.csv`, the column  `key_beta_p`=`1`, they are actually **the same** class, which is class `0` in the dataset `decisiontree/dataset/balance_scale.csv`, but in `decisiontree_run.py`, we shifted the class by 1 with 
    ```Python
    y = y.values + 1
    ``` 
    to align with $j\in [J]\triangleq \{1,2,\dots,J\}$, i.e., `key_beta_p`= `restricted_class`+1.