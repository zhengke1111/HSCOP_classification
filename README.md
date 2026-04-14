# HSCOP_classification
This repository contains implementations of multi-class classification in "Solving Constrained Affine Heaviside Composite Optimiation Problem by a Progressive IP Approach", by Ke Zheng, Junyi Liu, Yurui Wang, and Jong-Shi Pang.

This project consists of two parts: 
- Score-based multi-class classification, and 
- Tree-based multi-class classification.

## Setup
- **Python 3.11**

    All computations are coded in **Python 3.11**, except `tree/tree_pareto/binoct-master/run_exp.py`, which needs **Python 3.10** to support the `cplex==22.1.2.0`.

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


    

## Usage
### Score-based multi-class classification `score_based`
- `score_based_run.py`: Main entry point. **Run** this script to execute experiments.

    - **Output structure per dataset:**
        ```
        results/our_results/score_based_run/{data_set}/
        ├── parameters.txt                      # Experiment configurations
        ├── score_based_{data_set}_results.csv  # Combined results CSV
        └── {data_set}_{precision}_{timestamp}/ # Optional LogFile subdirectories (save_log=True only)
        ```

    - **CSV columns:** `Precision_threshold, Fold, method, obj, time, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall`

    - **Configuration in `run_score_based_classification_experiment()`:**
        
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
        - `n_splits`: Number of folds (default: 4)

        - `folds`: Specific folds to run (`None` = run all folds)

        - `save_log`: Whether to save Gurobi log files (default: `False`)

        - `console_log`: Whether to output Gurobi logs to console (default: `False`)

    - **Global parameters in `parameter.py`:**
        - `EPSILON`: Approximation parameter (default: 1e-5) used in PIP

        - `RHO`: Penalty coefficient for infeasibility

        - `MODEL_PARAM`: Gurobi solver parameters

        - `ALG_PARAM`: Algorithm control parameters (initial integer ratio $r^0$, maximal integer ratio $r_{\max}$, expansion ratio $r_{\Delta}$, maximal iteration number $\mu_{\max}$, maximal unchanged iteration number $\tilde{\mu}_{\max}$)

        - `SHRINKAGE_MAX_OUT_ITER`: Maximum outer iterations for shrinkage algorithms

        - `MIP_START_SOL`: Precomputed warm start solutions per dataset

### Tree-based multi-class classification `tree`
- `tree_run.py`: Main entry point. **Run** this script to execute experiments.

    - **Output structure per dataset:**

        ```
        results/{data_set}/
        ├── tree_{data_set}_results.csv        # Combined results CSV
        └── {dataset}_depth-{depth}_run-{run}/ # Optional LogFile subdirectories (save_log=True only)
        ```

    - **CSV columns:** `dataset, depth, split, method, tau_0, key_beta, beta, objective_value, optimality_gap (Full MIP), time, actual_time (Full MIP), gamma, train_acc_margin, test_acc_margin, train_acc, test_acc, train_prec, test_prec`

    - **Configuration in `run_tree_experiment(method_dict, depth_list, pareto = False)`:**

        - `method_dict`: Dictionary of algorithms to execute. Set to `True` to enable:

            ```python
            method = {'Full MIP': True, 'PIP': False, 'ISA-PIP': False, 'IDSA-PIP': True}
            ```

        - `depth_list`: List of depth(s) of decision trees, e.g., `[2, 3, 4]`.

        - `pareto`: Whether to run the Pareto comparison experiments. If `pareto = True`, `method_dict` should be set as 

            ```python
            method = {'IDSA-PIP': True, 'U-PIP': True}
            ```

            If `'IDSA-PIP': True`, a series of precision thresholds will be set for `IDSA-PIP`.
            
            If `'U-PIP': True`, it will run the tree-based classification model without precision constraints. 
        
        - `tau_0`: $\ell_0$ norm upper bound $\tau_0$ of each branching coefficient $a_k$ at branch nodes $k\in {\cal T}_{\cal B}$.

        - `data_splits`: Split datasets.

        - `beta`: Dictionary of precision configurations per dataset. `{key_beta: threshold}`. 

            `key_beta` and `threshold` are chosen as **Appendix B.2, Model parameter**.

        - `save_log`: Whether to save Gurobi log files (default: `False`)

        - `console_log`: Whether to output Gurobi logs to console (default: `False`)

    - **Global parameters in `parameters.py`:**

        - `EPSILON`: Approximation parameter (default: 1e-4) used in PIP

        - `RHO`: Penalty coefficient for infeasibility

        - `MODEL_PARAM`: Gurobi solver parameters

        - `ALG_PARAM`: Algorithm control parameters (initial integer ratio $r^0$, maximal integer ratio $r_{\max}$, expansion ratio $r_{\Delta}$, maximal iteration number $\mu_{\max}$, maximal unchanged iteration number $\tilde{\mu}_{\max}$)

        - `SHRINKAGE_MAX_OUT_ITER`: Maximum outer iterations for shrinkage algorithms

        - `THRESHOLD_GRID`: precision thresholds set for `IDSA-PIP`(i.e., `C-PIP`) for Pareto comparison 
        


    
- `tree_pareto/binoct-master/run_exp.py`: run it to produce the results of `U-BinOCT` and `C-BinOCT`.

- `tree_pareto/StrongTree-master/Code/StrongTree/run_exp.py`: run it to produce the results of `BendersOCT` and `FlowOCT`. 
    ```Python
    for dataset in ['blsc', 'ctmc']:
        validation(dataset)                                 # For validate \lambda
        run_strongOCT(result_dir, dataset, 'BendersOCT')    # With \lambda, run BendersOCT
        run_strongOCT(result_dir, dataset, 'FlowOCT')       # With \lambda, run FlowOCT
    ```

- `CART_run.py`: run it to produce the results of `CART`. The results will be stored in `tree/results/CART_results.csv`. 



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

- `tree`

    - `dataset`: 8 datasets used in experiments.

    - `results`: Output results directory.

        - `tree_results.csv`: results of tree-based classification, 8 datasets, depth-2,3,4, method `Full MIP` and `IDSA-PIP`.

        - `tree_pareto_blsc_results.csv`, `tree_pareto_ctmc_results.csv`: combined results of `C-PIP`, `FlowOCT`, `C-BinOCT`, `U-PIP`, `BendersOCT`, `U-BinOCT`, `CART`, which we present in Pareto curves. For their raw output, see the folder `tree_pareto_raw_output`.

        - `depth2`, `depth3`, `depth4`: empty folders to store (future) detailed results.

    - `tree_pareto`: Existing methods in literature, including

        - `binoct-master`: `BinOCT` [Learning optimal classification trees using a binary linear program formulation](https://ojs.aaai.org/index.php/AAAI/article/view/3978)

            - `run_exp.py`: **run** this script to reproduce results of `U-BinOCT` and `C-BinOCT`.

            - `results`: store the results.

        - `StrongTree-master`: `BendersOCT` and `FlowOCT` [Strong Optimal Classification Trees](https://sites.google.com/view/sina-aghaei/home)

            - `Code/run_exp.py`: **run** this script to reproduce results of `BendersOCT` and `FlowOCT`.

            - `Results`: store the results

    - `tree_run.py`: **run** this script to reproduce results of `Full MIP` and `IDSA-PIP`(`C-PIP`) and `U-PIP`.  
        
    - `CART_run.py`: **run** this script to reproduce results of `CART`.

    - `model.py`: build and solve an MIP for a single PIP partial problem in a tree-based classification problem. 

    - `algorithm.py`: determine whether to continue or stop, and, if continuing, decide whether to enlarge or shrink the in-between sets ${\cal J}$. 

    - `callback.py`: `callback` mechanism for solving MIP problems. The parameter used in `MIP_tree_callback.py` are stored in `callback_data_tree.py`.

    - `utils.py`: utility file to store some commonly used functions in this project. 



## Notes
- For `C-PIP`, in `tree_pareto_blsc_results.csv`, the column `restricted_class`=`0`, but in `tree/results/tree_pareto_raw_output/blsc/C-PIP.csv`, the column  `key_beta_p`=`1`, they are actually **the same** class, which is class `0` in the dataset `tree/dataset/balance_scale.csv`, but in `tree_run.py`, we shifted the class by 1 with 
    ```Python
    y = y.values + 1
    ``` 
    to align with $j\in [J]\triangleq \{1,2,\dots,J\}$, i.e., `key_beta_p`= `restricted_class`+1.