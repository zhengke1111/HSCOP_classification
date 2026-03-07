# HSCOP_classification
Source code of HSCOP classification.

This project consists of three parts: binary classification, multi-class classification, and decision tree. 

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
- `binary`
    - `dataset` includes the required datasets
    - `results` store the output results
        - `our_results`: the original results files of what we present in article.  
    - `binary_run.py`: run this python file to conduct experiments with `Full MIP` and `PIP`, specifically, find ```for method in range(1,7):``` and modify the range of to run the method(s) you want to run
        - 1: `Full MIP`
        - 2: `Early MIP (T300)`
        - 3: `Early MIP (T600)`
        - 4: $\varepsilon$`-fixed PIP`
        - 5: $\varepsilon$`-shrinkage PIP`
        - 6: `Unconstrained PIP`
    - `binary_pareto_run.py`: run this python file to conduct experiments with `LogisticRegression`, `Linear SVM`, `Perceptron` and `Ridge Regression`, the results will be stored in `binary/results/results_for_pareto_comparison.csv`.
    - `PIP_binary.py`: build and solve an MIP for a single PIP iteration in a precision constrained problem. For unconstrained problem, see `PIP_binary_unconstrained.py`. For `Full MIP`, `Early MIP (T300)` and `Early MIP (T600)`, see `Full_MIP_binary.py`.
    - `PIP_binary_iterations.py`: run the PIP iterations in a precision constrained problem, determine whether to continue or stop, and, if continuing, decide whether to enlarge or shrink the in-between sets ${\cal J}$. For unconstrained problem, see `PIP_binary_unconstrained_iterations.py`.
    - `MIP_binary.py`: set input and output paths and formats for various methods. 
    - `MIP_binary_callback.py`: `callback` mechanism for solving MIP problems built in `PIP_binary.py`. The parameter used in `MIP_binary_callback.py` are stored in `callback_data_binary.py`.
    - `utils.py`: utility file to store some commonly used functions in this project. 
- `multiclass`  
- `decisiontree`
    - `dataset` includes the required datasets
    - `results` store the output results
        - `our_results`: the original results files of what we present in article.
    - `decisiontree_pareto`: existing methods in literature, including
        - binoct-master: `BinOCT` [Learning optimal classification trees using a binary linear program formulation](https://ojs.aaai.org/index.php/AAAI/article/view/3978)
        - StrongTree-master: `BendersOCT` and `FlowOCT` [Strong Optimal Classification Trees](https://sites.google.com/view/sina-aghaei/home)
    - `decisiontree_run.py`: run this python file to conduct experiments with `Full MIP` and `PIP`, specifically, find ```for method in range(1,9):``` and modify the range of to run the method(s) you want to run
        - 1: `Full MIP`
        - 2: $\varepsilon$`-fixed PIP`
        - 3: $\varepsilon$`-shrinkage PIP`
        - 4: $\varepsilon$`-fixed-arbitrary4 PIP`
        - 5: $\varepsilon$`-fixed-arbitrary1 PIP`
        - 6: $\varepsilon$`-shrinkage-arbitrary4 PIP`
        - 7: $\varepsilon$`-shrinkage-arbitrary1 PIP`
        - 8: `Unconstrained PIP`
    - `cart_run.py`: run this file to conduct experiments with `DecisionTreeClassifier` in `scikit-learn`, and the results will be stored in `decisiontree/results/cart_results.csv`. 
    - `PIP_single_iter_tree.py`: build and solve an MIP for a single PIP iteration in a (precision constrained) decision tree classification problem. For the unconstrained problem, see `PIP_unconstrained_single_iter_tree.py`. For `Full MIP`, see `Full_MIP_tree.py`.
    - `PIP_iterations_tree.py`: run the PIP iterations in a (precision constrained) decision tree classification problem, determine whether to continue or stop, and, if continuing, decide whether to enlarge or shrink the in-between sets ${\cal J}$. For unconstrained problem, see `PIP_unconstrained_iterations_tree.py`.
    - `MIP_tree.py`: set input and output paths and formats for various methods. 
    - `MIP_tree_callback.py`: `callback` mechanism for solving MIP problems built in `PIP_single_iter_tree.py`. The parameter used in `MIP_tree_callback.py` are stored in `callback_data_tree.py`.
    - `utils.py`: utility file to store some commonly used functions in this project. 