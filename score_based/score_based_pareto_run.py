import os
import time
import pandas as pd
from utils import *
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


def run_sklearn_classifiers(param):
    """
    Run experiments for all classifiers with cross-validation and hyperparameter tuning.
    Saves results to a CSV file in experiment_results/.

    Args:
        param (dict): Dictionary containing:
            - data_path: Path to the dataset
            - sample_size: Number of samples to use
            - n_splits: Number of cross-validation folds
            - data_set: Name of the dataset (used in output filename)

    Returns:
        pd.DataFrame: Summary of all classification results
    """
    # Fix random seed for reproducibility
    np.random.seed(42)

    # --------------------------
    # Inner function: train and evaluate one classifier
    # --------------------------
    def run_single_classifier(classifier_name, classifier, param_grid, data_folds):
        """
        Train and evaluate a single classifier with grid search hyperparameter tuning

        Args:
            classifier_name (str): Name of classifier (e.g., 'LogisticRegression')
            classifier (sklearn.base.BaseEstimator): Untrained sklearn classifier instance
            param_grid (dict): Hyperparameter grid for GridSearchCV
            data_folds (dict): Cross-validation folds (from split_folds)

        Returns:
            list: List of dictionaries with fold-level results (accuracy, precision, time, etc.)
        """
        results = []

        for fold_id, fold_data in data_folds.items():
            print(f"  Fold {fold_id}...")
            fold_start_time = time.time()

            X_train = fold_data['X_train']
            X_test = fold_data['X_test']
            y_train = fold_data['y_train']
            y_test = fold_data['y_test']

            # Grid search for hyperparameter optimization
            grid_search = GridSearchCV(
                classifier,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=1,
                verbose=0,
                error_score=np.nan
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Predict on training and test sets
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Overall accuracy (rounded to 3 decimal places)
            train_accuracy = round(accuracy_score(y_train, y_train_pred), 3)
            test_accuracy = round(accuracy_score(y_test, y_test_pred), 3)

            # Detailed classification report
            train_report = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)
            test_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

            fold_end_time = time.time()
            fold_time = round(fold_end_time - fold_start_time, 3)

            # Store basic results
            result = {
                'classifier': classifier_name,
                'fold': fold_id,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'fold_time_seconds': fold_time,
                'best_params': str(grid_search.best_params_)
            }

            # Collect all unique class labels
            all_classes = set(np.unique(y_train)) | set(np.unique(y_test))

            # Add precision for each class
            for class_label in all_classes:
                class_str = str(class_label)

                # Extract training class precision
                train_class_metrics = train_report.get(class_str, {})
                if isinstance(train_class_metrics, dict):
                    train_precision = round(train_class_metrics.get('precision', 0.0), 3)
                else:
                    train_precision = 0.0
                result[f'train_class_{class_label}_precision'] = train_precision

                # Extract test class precision
                test_class_metrics = test_report.get(class_str, {})
                if isinstance(test_class_metrics, dict):
                    test_precision = round(test_class_metrics.get('precision', 0.0), 3)
                else:
                    test_precision = 0.0
                result[f'test_class_{class_label}_precision'] = test_precision

            # Append fold results to list
            results.append(result)
        return pd.DataFrame(results)

    # --------------------------
    # Main experiment pipeline
    # --------------------------
    # Load and split data into k folds
    print("Loading and splitting data...")
    data_folds = split_folds(param['data_set'], param['sample_size'], param['n_splits'])

    # Define all classifiers and their parameter grids
    classifiers_config = {
        # Logistic Regression (supports multi-class natively)
        'LogisticRegression': {
            'classifier': LogisticRegression(random_state=42, max_iter=51000),
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'solver': ['lbfgs', 'saga'],
                'max_iter': [5000]
            }
        },

        # Linear SVM (One-vs-Rest for multi-class)
        'LinearSVM': {
            'classifier': LinearSVC(random_state=42, max_iter=10000, dual='auto'),
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'loss': ['squared_hinge', 'hinge'],
                'multi_class': ['ovr']
            }
        },

        # Linear SVM with Crammer & Singer formulation
        'LinearSVM_CS': {
            'classifier': LinearSVC(
                random_state=42,
                max_iter=10000,
                dual='auto',
                multi_class='crammer_singer'
            ),
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'loss': ['squared_hinge']
            }
        },

        # Perceptron (native multi-class support)
        'Perceptron': {
            'classifier': Perceptron(random_state=42, max_iter=10000, n_jobs=1),
            'param_grid': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2', 'l1', 'elasticnet']
            }
        },

        # Ridge Classifier (native multi-class support)
        'Ridge': {
            'classifier': RidgeClassifier(random_state=42),
            'param_grid': {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
        },
    }

    # Run all classifiers
    all_results = []
    print(f"Starting experiments with {len(classifiers_config)} classifiers...")

    for clf_name, clf_config in classifiers_config.items():
        print(f"\nRunning {clf_name}...")
        clf_results = run_single_classifier(
            classifier_name=clf_name,
            classifier=clf_config['classifier'],
            param_grid=clf_config['param_grid'],
            data_folds=data_folds
        )
        all_results.append(clf_results)

    # Combine results into one DataFrame
    summary_df = pd.concat(all_results, ignore_index=True)

    # Create output directory and save CSV
    output_dir = 'score_based/results/score_based_pareto_run'
    os.makedirs(output_dir, exist_ok=True)
    data_set_name = param['data_set']
    summary_file = os.path.join(output_dir, f'{data_set_name}_sklearn_classifiers_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\nResults saved to: {summary_file}")

    return summary_df


def score_based_pareto_run():
    for data_set in ['wine', 'fish', 'robo', 'segm', 'vehi', 'wave']:
        param = {'data_set': data_set, 'sample_size': None, 'n_splits': 4, 'folds': None}
        run_sklearn_classifiers(param)

score_based_pareto_run()