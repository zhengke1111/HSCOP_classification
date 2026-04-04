from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score
import numpy as np
import utils
import pandas as pd
import time


def split_data(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
        X_train, y_train, test_size=1/3, stratify=y_train, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_train = scaler.fit_transform(X_train_train)
    X_train_val = scaler.transform(X_train_val)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    data_splits = {
        'X_train': X_train, 'y_train': y_train,
        'X_train_train': X_train_train, 'y_train_train': y_train_train,
        'X_train_val': X_train_val, 'y_train_val': y_train_val,
        'X_test': X_test, 'y_test': y_test
    }

    return data_splits


def tune_cart_on_val(max_depth, data_splits, random_state=42):
    X_trtr = data_splits['X_train_train']
    y_trtr = data_splits['y_train_train']
    X_val  = data_splits['X_train_val']
    y_val  = data_splits['y_train_val']

    param_grid = {
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": [None, "sqrt", "log2"],
        "splitter": ["best", "random"],
    }

    best_params = None
    best_score = -np.inf
    all_results = []

    for params in ParameterGrid(param_grid):
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            **params
        )
        model.fit(X_trtr, y_trtr)
        y_pred_val = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred_val)

        all_results.append((acc, params))

        if acc > best_score:
            best_score = acc
            best_params = params

    all_results.sort(key=lambda x: x[0], reverse=True)

    return best_params, best_score, all_results


def train_final(max_depth, data_splits, best_params, random_state=42):
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']

    final_model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        **best_params
    )
    final_model.fit(X_train, y_train)

    return final_model


def class_precision(y_true, y_pred, cls):
    """
    Precision for a specific class = TP/(TP+FP) for that cls.
    Works for multiclass by restricting labels.
    """
    return precision_score(
        y_true, y_pred,
        labels=[cls],
        average=None,
        zero_division=0
    )[0]


def cart_run():
    dataset_list = ['ctmc']
    restricted_class = 1

    results = []

    for max_depth in range(2, 5):
        for dataset in dataset_list:
            for run in range(1, 5):
                # start timing (tuning + final training)
                t0 = time.perf_counter()

                X, y = utils.sample_data(dataset=dataset)
                y = y.values + 1  # your label shift

                data_splits = split_data(X, y, random_state=42 + run)

                # tuning on 50% train_train & 25% val
                best_params, best_val_acc, all_results = tune_cart_on_val(
                    max_depth, data_splits, random_state=42
                )

                # final training on 75% train
                final_model = train_final(
                    max_depth, data_splits, best_params, random_state=42
                )

                # end timing
                t1 = time.perf_counter()
                total_time = t1 - t0

                # predictions
                X_train = data_splits['X_train']
                y_train = data_splits['y_train']
                X_test  = data_splits['X_test']
                y_test  = data_splits['y_test']

                y_pred_train = final_model.predict(X_train)
                y_pred_test  = final_model.predict(X_test)

                # metrics
                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc  = accuracy_score(y_test, y_pred_test)

                train_prec = class_precision(y_train, y_pred_train, restricted_class)
                test_prec  = class_precision(y_test, y_pred_test, restricted_class)

                # record one row
                row = {
                    "method": "CART",
                    "dataset": dataset,
                    "run": run,
                    "depth": max_depth,
                    "restricted_class": restricted_class,

                    "criterion": best_params.get("criterion"),
                    "min_samples_split": best_params.get("min_samples_split"),
                    "min_samples_leaf": best_params.get("min_samples_leaf"),
                    "max_features": best_params.get("max_features"),
                    "splitter": best_params.get("splitter"),

                    "time": total_time,

                    "train_acc": train_acc,
                    "test_acc": test_acc,

                    "train_prec": train_prec,
                    "test_prec": test_prec,
                }

                results.append(row)

                # optional prints (keep if you still want console output)
                print("Best params on validation:")
                print(best_params)
                print(f"Validation accuracy = {best_val_acc:.4f}")
                print(f"Train accuracy = {train_acc:.4f}, Test accuracy = {test_acc:.4f}")
                print(f"Train precision (class {restricted_class}) = {train_prec:.4f}")
                print(f"Test precision  (class {restricted_class}) = {test_prec:.4f}")
                print(f"Time (tune+train) = {total_time:.4f}s")

                print("\nClassification report on test:")
                print(classification_report(y_test, y_pred_test))
                print("-" * 60)

    df = pd.DataFrame(results)

    # You can customize filename
    out_path = "decisiontree/results/cart_results.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved results to: {out_path}")
    print(df.head())


cart_run()