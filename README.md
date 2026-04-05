# HSCOP_classification
Source code of HSCOP classification.

This project consists of two parts: score-based multi-class classification and tree-based classification. 

## Requirements
- Python 3.11 (Except `decisiontree/decisiontree_pareto/binoct-master/run_exp.py`, which needs Python 3.10 to support the `cplex` version)

- When you run `decisiontree/decisiontree_pareto/binoct-master/run_exp.py`, if encountering an Error like  ```ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found```, you can enter the following in terminal (please replace '/home/zhengke' by your own path to anaconda):

```bash
strings ~/anaconda3/lib/libstdc++.so.6.0.29 | grep CXXABI
export LD_LIBRARY_PATH="/home/zhengke/anaconda3/envs/python3.10/lib:$LD_LIBRARY_PATH"
```

- The required Python packages are listed in `requirements.txt`.

- This project requires `gurobipy==11.0.3` and a valid Gurobi license. Our experiments were conducted using the Gurobi Academic License.
To run the optimization code, you need to install Gurobi and activate your
own valid license.

## Overview 
The content of this repository is as follows:
- `multiclass`  

- `decisiontree`

    - `dataset` includes the required datasets

    - `results` store the output results

        - `decisiontree_results.csv`: results of tree-based classification, 8 datasets, depth-2,3,4, method `Full MIP` and `IDSA-PIP`.

        - `decisiontree_pareto_blsc_results.csv`, `decisiontree_pareto_ctmc_results.csv`: combined results of `C-PIP`, `FlowOCT`, `C-BinOCT`, `U-PIP`, `BendersOCT`, `U-BinOCT`, `CART`, which we present in Pareto curves. For their raw output, see the folder `decisiontree_pareto_raw_output`.

        - `depth2`, `depth3`, `depth4`: empty folders to store (future) detailed results.

    - `decisiontree_pareto`: existing methods in literature, including

        - `binoct-master`: `BinOCT` [Learning optimal classification trees using a binary linear program formulation](https://ojs.aaai.org/index.php/AAAI/article/view/3978)

            - `run_exp.py`: **run** this script to reproduce results of `U-BinOCT` and `C-BinOCT`.

            - `results`: store the results.

        - `StrongTree-master`: `BendersOCT` and `FlowOCT` [Strong Optimal Classification Trees](https://sites.google.com/view/sina-aghaei/home)

            - `Code/run_exp.py`: **run** this script to reproduce results of `BendersOCT` and `FlowOCT`.

            - `Results`: store the results

    - `decisiontree_run.py`: **run** this script to reproduce results of `Full MIP` and `IDSA-PIP`(`C-PIP`) and `U-PIP`.  
        
    - `CART_run.py`: **run** this script to reproduce results of `CART`.

    - `PIP_tree_solve_partial_problem.py`: build and solve an MIP for a single PIP partial problem in a tree-based classification problem. 

    - `PIP_tree_control_termination.py`: determine whether to continue or stop, and, if continuing, decide whether to enlarge or shrink the in-between sets ${\cal J}$. 
    
    - `MIP_tree.py`: set input and output paths and formats for various methods. 

    - `MIP_tree_callback.py`: `callback` mechanism for solving MIP problems. The parameter used in `MIP_tree_callback.py` are stored in `callback_data_tree.py`.

    - `utils.py`: utility file to store some commonly used functions in this project. 

## Usage 
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