from utils import *
from parameter import *
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_score, accuracy_score, recall_score
import csv
import os
import pandas as pd
import copy
import time
import warnings
import pandas as pd
import gurobipy as gp


# =========== Run binary by models in sklearn (LR, SVM, Perceptron, Ridge) ==========
def binary_pareto_run_sklearn():
    dataset_list = DATASET_LIST
    results = []
    # Define the tuning range of the hyperparameters
    param_grid = {'LogisticRegression': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2', 'l1', 'elasticnet']},
        'SVM': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2', 'l1'], 'loss': ['squared_hinge', 'hinge']},
        'Perceptron': {'alpha': [0.0001, 0.01, 0.1, 1, 10], 'penalty': ['l2', 'l1', 'elasticnet']},
        'Ridge': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}}

    for dataset in dataset_list:
        X, y = sample_data(dataset=dataset)
        for run in range(1, 5):
            data_splits = split_data(X, y, random_state=42 + run)
            X_train, y_train, X_test, y_test = data_splits['X_train'], data_splits['y_train'], data_splits['X_test'], data_splits['y_test']

            # Four methods and their corresponding tuning functions
            methods = {'LogisticRegression': LogisticRegression(), 'SVM': LinearSVC(), 'Perceptron': Perceptron(), 'Ridge': RidgeClassifier()}

            # Tune the hyperparameters for each method
            for method_name, model in methods.items():
                # Retrieve the hyperparameter grid 
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid[method_name], cv=5, n_jobs=-1, verbose=1)

                # Record warnings during training
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")             # Trigger all warnings
                    start_time = time.time()                    # Record the training start time
                    grid_search.fit(X_train, y_train)           # Execute the grid search training
                    end_time = time.time()                      # Record the training end time
                    training_time = end_time - start_time       # Calculate the training time
                    best_model = grid_search.best_estimator_    # Obtain the optimal model
                    if hasattr(best_model, 'converged_'):       # Check convergence during the training process
                        converged = best_model.converged_
                    else:
                        converged = 'N/A'  # For models without a convergence, set it to 'N/A'
                    # Capture any convergence warnings
                    convergence_warning = any(isinstance(warning.message, ConvergenceWarning) for warning in caught_warnings)
                    # Evaluate on the training set and the test set
                    train_predictions, test_predictions = best_model.predict(X_train), best_model.predict(X_test)
                    train_precision, train_accuracy, train_recall = precision_score(y_train, train_predictions), accuracy_score(y_train, train_predictions), recall_score(y_train, train_predictions)
                    test_precision, test_accuracy, test_recall = precision_score(y_test, test_predictions), accuracy_score(y_test, test_predictions), recall_score(y_test, test_predictions)
                    # Record the results
                    result_row = {
                        'dataset': dataset, 'run': run, 'method': method_name, 'train_precision': train_precision, 'train_accuracy': train_accuracy, 'train_recall': train_recall,
                        'test_precision': test_precision, 'test_accuracy': test_accuracy, 'test_recall': test_recall, 'time': training_time, 'convergence_warning': convergence_warning}
                    results.append(result_row)

    df = pd.DataFrame(results)
    df.to_csv('binary/results/results_for_pareto_comparison.csv', index=False)


binary_pareto_run_sklearn()