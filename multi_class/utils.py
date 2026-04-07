import csv
import sys
import numpy as np
import pandas as pd
import random
from math import prod
from typing import Union
from model import Model
from parameter import *
from algorithm import PIP, IterativeShrinkage
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def sample_data(data_set, sample_size=None):
    """
    Load dataset and sample specified number of samples per class (stratified sampling)

    Args:
        data_set (str): Dataset name (e.g., 'wine', 'fish', 'vehi')
        sample_size (list/None): List of sample counts per class (None = use all samples)

    Returns:
        tuple: (X_sampled, y_sampled)
            X_sampled (pd.DataFrame): Sampled feature matrix
            y_sampled (pd.DataFrame): Sampled label vector (single column 'Class')
    """
    # Map dataset name to file path
    if data_set == 'wine':
        data_path = RED_WINE_PATH
    elif data_set == 'fish':
        data_path = FISH_PATH
    elif data_set == 'vehi':
        data_path = VEHICLE_PATH
    elif data_set == 'robo':
        data_path = ROBOT_2_PATH
    elif data_set == 'wave':
        data_path = WAVE_PATH
    elif data_set == 'segm':
        data_path = SEGMENTATION_PATH
    else:
        print(f"Error: Dataset '{data_set}' is not supported! Please add dataset information in parameter.py")
        sys.exit(1)

    # Load dataset
    df = pd.read_csv(data_path, encoding='utf-8')

    # Map categorical labels to numeric 'Class' column for specific datasets
    if data_path == FISH_PATH:
        df['species'] = df['species'].map({
            'Setipinna taty': 0,
            'Anabas testudineus': 1,
            'Pethia conchonius': 2,
            'Otolithoides biauritus': 3,
            'Polynemus paradiseus': 4,
            'Sillaginopsis panijus': 5,
            'Otolithoides pama': 6,
            'Puntius lateristriga': 7,
            'Coilia dussumieri': 8
        })
        df.rename(columns={'species': 'Class'}, inplace=True)
    elif data_path == RED_WINE_PATH:
        df['quality'] = df['quality'].map({
            3: 0,
            4: 0,
            5: 1,
            6: 1,
            7: 2,
            8: 2,
        })
        df.rename(columns={'quality': 'Class'}, inplace=True)
    elif data_path == VEHICLE_PATH:
        df['target'] = df['target'].map({
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        })
        df.rename(columns={'target': 'Class'}, inplace=True)
    elif data_path == ROBOT_2_PATH:
        df['Class'] = df['Class'].map({
            'Move-Forward': 0,
            'Sharp-Right-Turn': 1,
            'Slight-Right-Turn': 2,
            'Slight-Left-Turn': 3
        })

    # Perform stratified sampling if sample_size is specified
    if sample_size is None:
        X_sampled = df.drop('Class', axis=1)
        y_sampled = df['Class'].to_frame()
    else:
        total_class_num = len(sample_size)
        # Get indices for each class
        class_index = {cls: np.where(df['Class'] == cls)[0] for cls in range(total_class_num)}
        # Ensure sample size does not exceed available samples per class
        num_samples = {cls: int(min(sample_size[cls], len(class_index[cls]))) for cls in range(total_class_num)}

        # Fixed random seed for reproducibility
        np.random.seed(42)
        # Sample indices for each class
        sample_index_set = [np.random.choice(class_index[cls], size=num_samples[cls], replace=False) for cls in
                            range(total_class_num)]
        sample_index_ = [index.tolist() for index in sample_index_set]
        sample_index = sum(sample_index_, [])
        sampled_data = df.loc[sample_index].reset_index(drop=True)

        # Extract features and labels
        X_sampled = sampled_data.drop('Class', axis=1)
        y_sampled = sampled_data['Class'].to_frame()
    return X_sampled, y_sampled


def split_folds(data_set, sample_size=None, n_splits=4):
    """
    Split dataset into stratified cross-validation folds

    Args:
        data_set (str): Name of dataset to split (same as sample_data)
        sample_size (list/None): Sample size per class (passed to sample_data)
        n_splits (int): Number of cross-validation folds (default=4)

    Returns:
        dict: Dictionary of folds (key=fold number, value=dict with X_train/X_test/y_train/y_test)
    """
    X, y = sample_data(data_set, sample_size)
    # Initialize stratified K-fold splitter (preserve class distribution)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    data_folds = {}

    # Split data into folds and store train/test sets
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        data_folds[fold] = {}
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Reset indices for consistency
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        y_column_name = y.columns[0]
        y_train = y_train[y_column_name].to_numpy()
        y_test = y_test[y_column_name].to_numpy()

        # the scaler process will reset index of X
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train))
        X_test = pd.DataFrame(scaler.transform(X_test))

        data_folds[fold]['X_train'] = X_train
        data_folds[fold]['X_test'] = X_test
        data_folds[fold]['y_train'] = y_train
        data_folds[fold]['y_test'] = y_test

    return data_folds

def extract_inner_values(d, values=None):
    """
    Recursively extract all non-dict values from a nested dictionary

    Args:
        d (dict): Nested dictionary to extract values from
        values (list/None): List to collect values (initialized as empty if None)

    Returns:
        list: All non-dict values from the nested dictionary
    """
    if values is None:
        values = []
    for key, value in d.items():
        if isinstance(value, dict):
            extract_inner_values(value, values)
        else:
            values.append(value)
    return values


def generate_data_info(X, y):
    """
    Generate basic data indices and class information for multi-class classification

    Args:
        X (pd.DataFrame/np.ndarray): Feature matrix (samples x features)
        y (pd.DataFrame/np.ndarray): Label vector (samples x 1)

    Returns:
        tuple: (N, p, I, class_index, class_index_rest)
            N (range): Sample indices (0 to n_samples-1)
            p (range): Feature indices (0 to n_features-1)
            I (range): Class indices (0 to n_classes-1)
            class_index (dict): Indices of samples for each class (key=class, value=array of indices)
            class_index_rest (dict): Indices of samples NOT in each class (key=class, value=array of indices)
    """
    N = range(X.shape[0])  # samples
    p = range(X.shape[1])  # dimension
    I = range(len(np.unique(y)))  # classes

    class_index = {cls: np.where(y == cls)[0] for cls in I}  # use to select samples of certain class
    class_index_rest = {cls: np.where(y != cls)[0] for cls in I}

    return N, p, I, class_index, class_index_rest


def generate_bigM(X):
    """
    Calculate Big-M constant for MIP constraint formulation (upper bound for |<w,x>+b|)

    Args:
        X (pd.DataFrame/np.ndarray): Feature matrix (samples x features)

    Returns:
        int: Big-M constant (rounded up to nearest 100 for safety)
    """
    data_abs = X.abs()
    abs_max = data_abs.max().max()
    M = (int((abs_max * 20 + 20) // 100) + 1) * 100

    return M


def generate_logistic_start(X_train, y_train, n_iters=10000):
    """
    Generate warm start values (W: weights, b: biases) using Logistic Regression
    Scales values to [-9.9, 9.9] to comply with model variable bounds

    Args:
        X_train (pd.DataFrame/np.ndarray): Training feature matrix
        y_train (pd.DataFrame/np.ndarray): Training label vector
        n_iters (int): Maximum iterations for Logistic Regression (default=10000)

    Returns:
        dict: {'W': scaled weight dict, 'b': scaled bias dict}
            W keys: (class, feature), values: scaled weight
            b keys: class, values: scaled bias
    """
    # Train Logistic Regression model
    clf = LogisticRegression(max_iter=n_iters, C=1)
    clf.fit(X_train, y_train)
    W = clf.coef_
    b = clf.intercept_
    K = len(clf.coef_)
    n_features = len(clf.coef_[0])

    # Convert weights to dictionary (key=(class, feature))
    W_dict = {}
    b_dict = {}
    for k in range(K):
        for n in range(n_features):
            position = (k, n)
            value = W[k, n]
            W_dict[position] = value

    # Convert biases to dictionary (key=class)
    for k in range(len(b)):
        b_dict[k] = b[k]

    # Calculate L1 norm of weights per class (for scaling)
    W_l1 = {}
    for k in range(K):
        W_l1.update({k: sum([abs(W_dict[k, d]) for d in range(n_features)])})
    W_l1_max = max(W_l1.values())
    b_max = max(b_dict.values())
    max_val = max(W_l1_max, b_max)

    # Scale weights/biases to [-9.9, 9.9] if max value exceeds 10
    if (W_l1_max > 10) or (b_max > 10):
        W_regular = {k: v * 9.9 / max_val for k, v in W_dict.items()}
        b_regular = {k: v * 9.9 / max_val for k, v in b_dict.items()}
    else:
        W_regular = W_dict
        b_regular = b_dict
    result = {'W': W_regular, 'b': b_regular}
    return result


def generate_svm_start(X_train, y_train, C=1.0, n_iters=200000):
    """
    Generate warm start values (W: weights, b: biases) using Linear SVM
    Scales values to [-9.9, 9.9] to comply with model variable bounds

    Args:
        X_train (pd.DataFrame/np.ndarray): Training feature matrix
        y_train (pd.DataFrame/np.ndarray): Training label vector
        C (float): SVM regularization parameter (default=1.0)
        n_iters (int): Maximum iterations for SVM (default=200000)

    Returns:
        dict: {'W': scaled weight dict, 'b': scaled bias dict}
            W keys: (class, feature), values: scaled weight
            b keys: class, values: scaled bias
    """
    # Train Linear SVM model (hinge loss, fixed random seed)
    clf = LinearSVC(
        C=C,
        max_iter=n_iters,
        loss='hinge',
        random_state=42
    )
    clf.fit(X_train, y_train)
    W = clf.coef_
    b = clf.intercept_
    I, n_features = W.shape

    # Convert weights to dictionary (key=(class, feature))
    W_dict = {(k, n): W[k, n] for k in range(I) for n in range(n_features)}
    # Convert biases to dictionary (key=class)
    b_dict = {k: b[k] for k in range(I)}

    # Calculate L1 norm of weights per class (for scaling)
    W_l1 = {k: np.sum(np.abs(W[k])) for k in range(I)}
    W_l1_max = max(W_l1.values())
    b_max = max(b_dict.values())
    max_val = max(W_l1_max, b_max)

    # Scale weights/biases to [-9.9, 9.9] if max value exceeds 10
    if max_val > 10:
        scale = 9.9 / max_val
        W_regular = {key: val * scale for key, val in W_dict.items()}
        b_regular = {key: val * scale for key, val in b_dict.items()}
    else:
        W_regular = W_dict
        b_regular = b_dict

    result = {'W': W_regular, 'b': b_regular}
    return result


def inner_function(X, y, class_restrict, epsilon, W, b, ell):
    """
    Calculate phi values (inner function) for partial model constraint formulation
    Phi measures the margin between predicted and true class scores

    Args:
        X (pd.DataFrame/np.ndarray): Feature matrix
        y (pd.DataFrame/np.ndarray): Label vector
        class_restrict (list): Classes with precision constraints
        epsilon (float): Approximation parameter
        W (dict): Weight dictionary (key=(class, feature), value=weight)
        b (dict): Bias dictionary (key=class, value=bias)
        ell (dict): Piece selection for partial model

    Returns:
        dict: Phi values (key=('obj'/class, sample), value=phi score)
    """
    # Generate basic data indices/class info
    N, p, K, class_index, class_index_rest = generate_data_info(X, y)
    # Initialize phi dictionary (obj + constrained classes x samples)
    phi = {(m, s): 0 for m in (['obj'] + class_restrict) for s in N}

    # Calculate phi for each sample
    for s in N:
        score_ms = sum(W[y[s], d] * X.iloc[s, d] for d in p) + b[y[s]]
        h_msj = []
        for j in (i for i in K if i != y[s]):
            score_j = sum(W[j, d] * X.iloc[s, d] for d in p) + b[j]
            h_msj.append(score_ms - score_j - 1)
        phi['obj', s] = min(h_msj)

    # Calculate phi for constrained classes (truncated for partial model)
    for m in class_restrict:
        for s in N:
            if (ell is None) or (s in class_index[m]):
                h_ms = []
                for l_ms in (j for j in K if j < m):
                    h_ml = ((sum(W[m, d] * X.iloc[s, d] for d in p) + b[m])
                            - (sum(W[l_ms, d] * X.iloc[s, d] for d in p) + b[l_ms])) - epsilon
                    h_ms.append(h_ml)
                for l_ms in (j for j in K if j > m):
                    h_ml = ((sum(W[m, d] * X.iloc[s, d] for d in p) + b[m])
                            - (sum(W[l_ms, d] * X.iloc[s, d] for d in p) + b[l_ms]))
                    h_ms.append(h_ml)
                phi[m, s] = min(h_ms)
            else:  # ell is not None and s in class_index_rest
                phi[m, s] = ((sum(W[m, d] * X.iloc[s, d] for d in p) + b[m])
                             - (sum(W[ell[m, s], d] * X.iloc[s, d] for d in p) + b[ell[m, s]]))
                if ell[m, s] < m:
                    phi[m, s] -= epsilon
    return phi

def generate_z_start(X, y, W, b, epsilon, class_restrict, ell):
    """
    Generate initial values for z variables (z_plus_0, z_plus, z_minus) based on phi scores.
    Z variables indicate whether margin constraints are satisfied for each sample/class.

    Args:
        X (pd.DataFrame/np.ndarray): Feature matrix.
        y (pd.DataFrame/np.ndarray): Label vector.
        W (dict): Weight dictionary (key = (class, feature), value = weight).
        b (dict): Bias dictionary (key = class, value = bias).
        epsilon (float): Approximation parameter for constraint relaxation.
        class_restrict (list): List of classes with precision constraints.
        ell (dict/None): Piece selection dictionary for partial model.

    Returns:
        tuple: Initial values for z variables:
            z_plus_0_start (dict): Key = sample index, Value = 1 (constraint satisfied) or 0 (not satisfied).
            z_plus_start (dict): Key = (class, sample), Value = 1/0 (for samples in the class).
            z_minus_start (dict): Key = (class, sample), Value = 1/0 (for samples NOT in the class).
    """
    N, p, K, class_index, class_index_rest = generate_data_info(X, y)
    z_plus_0_start = {s: 0 for s in N}
    z_plus_start = {(m, s): 0 for m in class_restrict for s in class_index[m]}
    z_minus_start = {(m, s): 0 for m in class_restrict for s in class_index_rest[m]}
    phi = inner_function(X, y, class_restrict, epsilon, W, b, ell)
    for s in N:
        z_plus_0_start[s] = (1 if phi['obj', s] >= 0 else 0)

    # calculate z_start
    for m in class_restrict:
        for s in class_index[m]:
            z_plus_start[m, s] = (1 if phi[m, s] >= 0 else 0)
        for s in class_index_rest[m]:
            z_minus_start[m, s] = (1 if -phi[m, s] - epsilon >= 0 else 0)
    return z_plus_0_start, z_plus_start, z_minus_start


def generate_gamma_start(X, y, W, b, beta, epsilon, class_restrict, ell):
    """
    Generate initial values for gamma variables (constraint violation penalties) based on phi scores.

    Args:
        X (pd.DataFrame/np.ndarray): Feature matrix.
        y (pd.DataFrame/np.ndarray): Label vector.
        W (dict): Weight dictionary (key = (class, feature), value = weight).
        b (dict): Bias dictionary (key = class, value = bias).
        beta (dict): Weighting parameter for precision/recall trade-off (key = class).
        epsilon (float): Approximation parameter for constraint relaxation.
        class_restrict (list): List of classes with precision constraints.
        ell (dict/None): Piece selection dictionary for partial model.

    Returns:
        dict: Initial gamma values (key = class, value = non-negative penalty value).
    """
    gamma_start = {m: 0 for m in class_restrict}
    phi = inner_function(X, y, class_restrict, epsilon, W, b, ell)

    K = range(len(np.unique(y)))  # classes
    class_index = {cls: np.where(y == cls)[0] for cls in K}  # use to select samples of certain class
    class_index_rest = {cls: np.where(y != cls)[0] for cls in K}

    for m in class_restrict:
        sum_positive_signed = sum(1 if phi[m, s] >= 0 else 0 for s in class_index[m])
        sum_negative_signed = sum(1 if phi[m, s] > -epsilon else 0 for s in class_index_rest[m])
        gamma_precision = beta[m] * sum_negative_signed - (1 - beta[m]) * sum_positive_signed
        gamma_recall = 0.1 * len(class_index[m]) - sum_positive_signed
        gamma_start[m] = max([0, gamma_precision, gamma_recall])
    return gamma_start


def generate_epsilon(max_iter):
    """
    Generate a sequence of epsilon values for iterative shrinkage (decays exponentially).
    Epsilon controls the relaxation of constraints during optimization.

    Args:
        max_iter (int): Total number of iterations for shrinkage.

    Returns:
        list: Sequence of epsilon values (length = max_iter).
    """
    epsilon_0 = 0.01
    shrinkage_rate = 0.1
    epsilon = [epsilon_0 * (shrinkage_rate ** (i - 1)) for i in range(1, max_iter + 1)]
    return epsilon


def precision_in_constraint(y, z_plus, z_minus, class_restrict):
    """
    Calculate precision values enforced by the optimization constraints (not empirical precision).

    Args:
        y (pd.DataFrame/np.ndarray): True label vector.
        z_plus (dict): z_plus variables (key = (class, sample), value = 1/0).
        z_minus (dict): z_minus variables (key = (class, sample), value = 1/0).
        class_restrict (list): List of classes with precision constraints.

    Returns:
        dict: Constrained precision values (key = class, value = rounded precision to 3 decimal places; -1 if division by zero).
    """
    K = range(len(np.unique(y)))
    class_index = {cls: np.where(y == cls)[0] for cls in K}  # use to select samples of certain class
    class_index_rest = {cls: np.where(y != cls)[0] for cls in K}
    precision_in_constr = {}
    for m in class_restrict:
        TP = sum(z_plus[m, s] for s in class_index[m])
        FP = sum((1 - z_minus[m, s]) for s in class_index_rest[m])
        precision = TP / (TP + FP) if (TP + FP) > 0 else -1
        precision_in_constr.update({m: np.round(precision, 3)})
    return precision_in_constr


def violations(X, y, class_restrict, epsilon, W, b, ell, z_plus_0, z_plus, z_minus):
    """
    Calculate constraint violations (feasibility and approximation errors) for the model.

    Args:
        X (pd.DataFrame/np.ndarray): Feature matrix.
        y (pd.DataFrame/np.ndarray): Label vector.
        class_restrict (list): List of classes with precision constraints.
        epsilon (float): Approximation parameter for constraint relaxation.
        W (dict): Weight dictionary (key = (class, feature), value = weight).
        b (dict): Bias dictionary (key = class, value = bias).
        ell (dict/None): Piece selection dictionary for partial model.
        z_plus_0 (dict): z_plus_0 variables (key = sample, value = 1/0).
        z_plus (dict): z_plus variables (key = (class, sample), value = 1/0).
        z_minus (dict): z_minus variables (key = (class, sample), value = 1/0).

    Returns:
        tuple: Two dictionaries:
            vio_feasibility_tol: Feasibility violations (key = 'obj'/'positive'/'negative', value = violation details).
            vio_approximation: Approximation errors (key = 'positive'/'negative', value = error count per class).
    """
    N, p, I, class_index, class_index_rest = generate_data_info(X, y)
    # Initialize violation trackers
    vio_feasibility_tol = {
        'obj': {},
        'positive': {j: {} for j in class_restrict},
        'negative': {j: {} for j in class_restrict}
    }
    vio_approximation = {
        'positive': {i: 0 for i in class_restrict},
        'negative': {i: 0 for i in class_restrict}
    }

    # Calculate phi scores for violation checks
    phi = inner_function(X, y, class_restrict, epsilon, W, b, ell)

    # Check feasibility violations for objective (obj)
    for j in N:
        if (z_plus_0[j] != 0) and (phi['obj', j] < 0):
            vio_feasibility_tol['obj'][j] = {'z': z_plus_0[j], 'phi': phi['obj', j]}

    # Check violations for constrained classes
    for i in class_restrict:
        for j in N:
            # Calculate margin for class i vs. classes with index < i
            h_in = min(
                [((sum(W[i, d] * X.iloc[j, d] for d in p) + b[i]) - (sum(W[n, d] * X.iloc[j, d] for d in p) + b[n]))
                 for n in (n for n in I if n < i)] or [None])
            # Calculate margin for class i vs. classes with index > i
            h_im = min(
                [((sum(W[i, d] * X.iloc[j, d] for d in p) + b[i]) - (sum(W[m, d] * X.iloc[j, d] for d in p) + b[m]))
                 for m in (m for m in I if m > i)] or [None])

            # Heaviside step function (1 = constraint satisfied, 0 = violated)
            if h_in is None:
                heaviside = 1 if (h_im is not None and h_im >= 0) else 0
            elif h_im is None:
                heaviside = 1 if (h_in > 0) else 0
            else:
                heaviside = 1 if (h_im >= 0 and h_in > 0) else 0

            # Check positive class violations (samples in class i)
            if j in class_index[i]:
                # Count approximation errors (h_in in (0, epsilon))
                if h_in is not None and (h_in > 0) and (h_in < epsilon):
                    if (h_im is None) or ((h_im is not None) and (h_im >= 0)):
                        vio_approximation['positive'][i] += 1
                # Feasibility violation (heaviside = 0 but z_plus = 1)
                if (heaviside == 0) and (z_plus[i, j] != 0):
                    vio_feasibility_tol['positive'][i].update({'z': z_plus[i, j], 'h': h_in})
            # Check negative class violations (samples not in class i)
            else:
                # Count approximation errors (h_im in (-epsilon, 0))
                if h_im is not None and (h_im < 0) and (h_im > -epsilon):
                    if (h_in is None) or ((h_in is not None) and (h_in > 0)):
                        vio_approximation['negative'][i] += 1
                # Feasibility violation (heaviside = 1 but z_minus = 1)
                if (heaviside == 1) and (z_minus[i, j] != 0):
                    vio_feasibility_tol['negative'][i].update({'z': z_minus[i, j], 'h': h_im})
    return vio_feasibility_tol, vio_approximation


def delta_of_J(X, y, W, b, class_restrict, epsilon, ratio, min_ratio, ell):
    """
    Calculate delta values (delta_1, delta_2) for adaptive constraint tightening.
    Delta controls the threshold for feasible/infeasible sample classification.

    Args:
        X (pd.DataFrame/np.ndarray): Feature matrix.
        y (pd.DataFrame/np.ndarray): Label vector.
        W (dict): Weight dictionary (key = (class, feature), value = weight).
        b (dict): Bias dictionary (key = class, value = bias).
        class_restrict (list): List of classes with precision constraints.
        epsilon (float): Approximation parameter for constraint relaxation.
        ratio (float): Percentile ratio for delta calculation.
        min_ratio (float): Minimum percentile ratio (prevents extreme values).
        ell (dict/None): Piece selection dictionary for partial model.

    Returns:
        tuple: Two dictionaries:
            delta_1 (dict): Upper delta threshold (key = 'obj'/class, value = threshold).
            delta_2 (dict): Lower delta threshold (key = 'obj'/class, value = threshold).
    """
    N, p, K, class_index, class_index_rest = generate_data_info(X, y)
    delta_1 = {m: np.inf for m in ['obj'] + class_restrict}
    delta_2 = {m: np.inf for m in ['obj'] + class_restrict}

    # Track values >= 0 (value_ge) and < 0 (value_le)
    value_ge = {m: [] for m in ['obj'] + class_restrict}
    value_le = {m: [] for m in ['obj'] + class_restrict}
    zero_value_counter = 0  # Alternate zero values between ge/le to avoid bias

    # Calculate phi scores and categorize values
    phi = inner_function(X, y, class_restrict, epsilon, W, b, ell)
    for m in ['obj'] + class_restrict:
        for s in N:
            if (m in ['obj']) or (s in class_index[m]):
                val = phi[m, s]
            else:
                val = -phi[m, s] - epsilon

            # Categorize value (handle zeros by alternating)
            if val > 0:
                value_ge[m].append(val)
            elif val < 0:
                value_le[m].append(val)
            else:
                if zero_value_counter % 2 == 0:
                    value_ge[m].append(val)
                else:
                    value_le[m].append(val)
                zero_value_counter += 1

        # Calculate delta_1 (upper threshold: percentile of positive values)
        if len(value_ge[m]) == 0:
            delta_1[m] = np.inf
        else:
            delta_1[m] = max(np.percentile(value_ge[m], max(ratio, min_ratio)), 1e-04)

        # Calculate delta_2 (lower threshold: percentile of negative values)
        if len(value_le[m]) == 0:
            delta_2[m] = np.inf
        else:
            delta_2[m] = max(-np.percentile(value_le[m], 100 - max(ratio, min_ratio)), 1e-04)
    return delta_1, delta_2

def piece_set(X, y, W, b, class_restrict, delta, epsilon):
    """
    Generate piece sets (ELL) for partial model formulation (active set selection).
    ELL contains the list of candidate classes for each (class, sample) pair in the partial model.

    Args:
        X (pd.DataFrame/np.ndarray): Feature matrix.
        y (pd.DataFrame/np.ndarray): Label vector.
        W (dict): Weight dictionary (key = (class, feature), value = weight).
        b (dict): Bias dictionary (key = class, value = bias).
        class_restrict (list): List of classes with precision constraints.
        delta (float): Threshold for including classes in the piece set.
        epsilon (float): Approximation parameter for constraint relaxation.

    Returns:
        dict: Piece set ELL (key = (class, sample), value = list of candidate classes).
    """
    N, p, K, class_index, class_index_rest = generate_data_info(X, y)
    ELL = {(m, s): [] for m in class_restrict for s in class_index_rest[m]}

    # Calculate candidate classes for each (class, sample) pair
    for m in class_restrict:
        for s in class_index_rest[m]:
            # Calculate score for each class
            score = {cls: sum(W[cls, d] * X.iloc[s, d] for d in p) + b[cls] for cls in K}

            # Calculate margin for class m vs. all other classes
            h = {}
            for ell in (j for j in K if j < m):
                h[ell] = score[m] - score[ell] - epsilon
            for ell in (j for j in K if j > m):
                h[ell] = score[m] - score[ell]

            # Minimum margin (phi)
            phi = min(h.values())
            # Select classes within phi + delta (candidate piece set)
            ell_list = []
            for ell in (j for j in K if j < m):
                if score[m] - score[ell] - epsilon <= phi + delta:
                    ell_list.append(ell)
            for ell in (j for j in K if j > m):
                if score[m] - score[ell] - epsilon <= phi + delta:
                    ell_list.append(ell)

            ELL[m, s] = ell_list
    return ELL


def arbitrary_choose_piece_combination(set_of_piece, num=4):
    """
    Randomly select a fixed number of piece combinations from the piece set.
    Ensures no duplicate combinations (uses set for uniqueness).

    Args:
        set_of_piece (dict): Piece set (key = (class, sample), value = list of candidate classes).
        num (int): Number of combinations to select (default: 4).

    Returns:
        list: List of piece combination dictionaries (each dict = one valid combination).
    """
    keys = list(set_of_piece.keys())
    piece_lists = [set_of_piece[k] for k in keys]

    # Total number of possible combinations
    num_piece_comb = prod([len(piece) for piece in piece_lists])
    # Select up to `num` combinations (or all if fewer exist)
    n = min(num, num_piece_comb) if num is not None else num

    # Randomly sample unique combinations
    results = set()
    while len(results) < n:
        comb = tuple(random.choice(piece) for piece in piece_lists)
        results.add(comb)

    # Convert tuples to dictionaries (map keys to selected pieces)
    dict_results = [dict(zip(keys, comb)) for comb in results]
    return dict_results


def predict(W, b, X):
    """
    Predict class labels for input samples using trained weights (W) and biases (b).
    Ties are broken by selecting the class with the smallest index.

    Args:
        W (dict): Weight dictionary (key = (class, feature), value = weight).
        b (dict): Bias dictionary (key = class, value = bias).
        X (pd.DataFrame/np.ndarray): Feature matrix of samples to predict.

    Returns:
        list: Predicted class labels for each sample (length = number of samples).
    """
    y_pred = []
    p = range(X.shape[1])
    K = sorted(set(key[0] for key in W.keys()))

    # Predict label for each sample
    for s in range(X.shape[0]):
        max_score = float('-inf')
        predict_class = None
        # Calculate score for each class
        for m in K:
            score = sum(W[m, d] * X.iloc[s, d] for d in p) + b[m]
            # Update max score (tie-break: smaller class index)
            if score > max_score:
                max_score = score
                predict_class = m
            elif score == max_score and m < predict_class:
                predict_class = m
        y_pred.append(predict_class)
    return y_pred


def classification_metric(X, y, W, b):
    """
    Calculate classification metrics (accuracy, per-class precision, per-class recall).
    Metrics are rounded to 3 decimal places for readability.

    Args:
        X (pd.DataFrame/np.ndarray): Feature matrix of test samples.
        y (pd.DataFrame/np.ndarray): True label vector for test samples.
        W (dict): Weight dictionary (key = (class, feature), value = weight).
        b (dict): Bias dictionary (key = class, value = bias).

    Returns:
        tuple: Classification metrics:
            accuracy (float): Overall classification accuracy.
            precision_dict (dict): Per-class precision (key = class, value = precision).
            recall_dict (dict): Per-class recall (key = class, value = recall).
    """
    y_pred = predict(W, b, X)
    accuracy = accuracy_score(y, y_pred)
    classes = np.unique(y)
    precision = precision_score(y, y_pred, labels=classes, average=None)
    precision_dict = {cls: precision for cls, precision in zip(classes, np.floor(precision * 1000) / 1000)}

    recall = recall_score(y, y_pred, labels=classes, average=None)
    recall_dict = {cls: recall for cls, recall in zip(classes, np.floor(recall * 1000) / 1000)}

    N = range(X.shape[0])
    p = range(X.shape[1])
    K = range(len(np.unique(y)))  # classes
    acc_margined = 0
    for s in N:
        h_msj = []
        score_ms = sum(W[y[s], d] * X.iloc[s, d] for d in p) + b[y[s]]
        for j in (i for i in K if i != y[s]):
            score_j = sum(W[j, d] * X.iloc[s, d] for d in p) + b[j]
            h_msj.append(score_ms - score_j - 1)
        if min(h_msj) >= 0:
            acc_margined += 1
    acc_margined = acc_margined / X.shape[0]

    result = {
        'acc_margined': np.round(acc_margined, 3),
        'accuracy': np.round(accuracy, 3),
        'precision': precision_dict,
        'recall': recall_dict
    }
    return result


def write_single_result_partial_model(file, title, num_int, obj_value, execution_time, model_time, num_kl,
                                      time_for_feasible, W, b):
    """
    Write results from partial Mixed Integer Programming models to a CSV file.
    Creates header row if file is empty, otherwise appends results to existing file.

    Args:
        file (str): Path to output CSV file (absolute or relative).
        title (str): Unique identifier for the experiment (e.g., "MIP").
        num_int (int): Number of integer/binary variables in the model (for complexity tracking).
        obj_value (float/None): Optimized objective function value.
        execution_time (float/None): Total execution time in seconds.
        model_time (float/None): Gurobi solver runtime in seconds.
        num_kl (int/None): Number of piece combinations used in partial model.
        time_for_feasible (float/None): Time to find first feasible solution in seconds.
        W (dict): Optimized weight dictionary (key = (class, feature), value = weight).
        b (dict): Optimized bias dictionary (key = class, value = bias).

    Returns:
        None: Writes results to CSV file and returns nothing.
    """
    fieldnames = [
        'Experiment', 'num_int', 'obj_value', 'execution_time', 'model_time', 'num_kl', 'time_for_feasible',
        'W', 'b', ]
    with open(file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        row = {'Experiment': title, 'num_int': num_int,
               'obj_value': np.round(obj_value, 3) if obj_value is not None else None,
               'execution_time': np.round(execution_time) if execution_time is not None else None,
               'model_time': np.round(model_time) if model_time is not None else None,
               'num_kl': num_kl,
               'time_for_feasible': np.round(time_for_feasible) if time_for_feasible is not None else None,
               'W': W, 'b': b, }
        writer.writerow(row)


def write_single_integrated_result(integrated_csv, title, execution_time, model_time, final_improvement_time, obj_val,
                                   train_acc_margined, train_acc, train_precision, train_recall, test_acc_margined,
                                   test_acc, test_precision, test_recall, precision_in_constr, opt_gap, num_kl, W, b,
                                   precision_threshold=None, fold=None):
    """
    Write integrated classification results (training + test metrics) to a CSV file for cross-method comparison.
    Handles both Full MIP and PIP-based algorithm results with specialized time metric handling for Full MIP models.

    Args:
        integrated_csv (str): Path to integrated results CSV file (absolute or relative).
        title (str): Unique experiment identifier (e.g., 'Full MIP', 'PIP', 'ISA-PIP', 'IDSA-PIP').
        execution_time (float/None): Total algorithm execution time in seconds.
        model_time (float/None): Gurobi solver runtime in seconds.
        final_improvement_time (float/None): Time of last objective improvement for MIP models.
        obj_val (float/None): Optimized objective function value.
        train_acc_margined (float/None): Training set margin-based accuracy.
        train_acc (float/None): Training set standard accuracy.
        train_precision (dict/None): Training set per-class precision.
        train_recall (dict/None): Training set per-class recall.
        test_acc_margined (float/None): Test set margin-based accuracy.
        test_acc (float/None): Test set standard accuracy.
        test_precision (dict/None): Test set per-class precision.
        test_recall (dict/None): Test set per-class recall.
        precision_in_constr (dict/None): Constraint-enforced precision values.
        opt_gap (float/None): MIP optimality gap percentage.
        num_kl (int/None): Number of piece combinations (None if not applicable).
        W (dict): Optimized weight dictionary (key = (class, feature), value = weight).
        b (dict): Optimized bias dictionary (key = class, value = bias).
        precision_threshold (dict/None): Precision threshold for constrained classes.
        fold(int): Fold number.

    Returns:
        None: Writes integrated results to CSV file and returns nothing.
    """
    fieldnames = [
        'Precision_threshold', 'Fold', 'method', 'obj', 'time',
        'train_accuracy', 'test_accuracy',
        'train_precision', 'test_precision',
        'train_recall', 'test_recall',
    ]
    if title == 'Full MIP':
        # For MIP: use final improvement time as time
        time_ = np.round(final_improvement_time) if final_improvement_time is not None else None
    else:
        # For PIP-based algorithms: use model solver time
        time_ = np.round(model_time) if model_time is not None else None

    with open(integrated_csv, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        row = {
            'Precision_threshold': precision_threshold,
            'Fold': fold,
            'method': title,
            'obj': np.round(obj_val, 3) if obj_val is not None else None,
            'time': time_,
            'train_accuracy': np.round(train_acc, 3) if train_acc is not None else None,
            'test_accuracy': np.round(test_acc, 3) if test_acc is not None else None,
            'train_precision': train_precision if train_precision is not None else None,
            'test_precision': test_precision if test_precision is not None else None,
            'train_recall': train_recall if train_recall is not None else None,
            'test_recall': test_recall if test_recall is not None else None,
        }
        writer.writerow(row)

def write_results(method: str, solution: Union[Model, PIP, IterativeShrinkage, list[PIP]], integrated_csv,
                  X_test, y_test, precision_threshold=None, fold=None):
    """
    Dispatch function to write results for different classification methods to an integrated CSV file.
    Supports Full MIP models and various PIP-based algorithms (PIP, IterativeShrinkage) with specialized handling
    for single vs iterative solutions.

    Supported methods:
        - 'Full MIP': Full MIP approach
        - 'PIP': PIP approach
        - 'ISA-PIP': ISA-PIP approach
        - 'D4-PIP': D4-PIP approach
        - 'D-PIP': D-PIP approach
        - 'IDSA4-PIP': IDSA4-PIP approach
        - 'IDSA-PIP': IDSA-PIP approach

    Args:
        method (str): Method identifier (see supported methods above).
        solution (Union[Model, PIP, IterativeShrinkage, List[PIP]]):
            Solution object(s) containing optimized weights/biases and metrics:
            - Model: Full MIP model instance
            - PIP/IterativeShrinkage: Single algorithm instance
            - List[PIP]: Multiple iterative PIP instances
        integrated_csv (str): Path to integrated results CSV file.
        X_test (pd.DataFrame/np.ndarray): Test set feature matrix (shape: n_samples × n_features).
        y_test (pd.DataFrame/np.ndarray): Test set label vector (shape: n_samples × 1).
        precision_threshold (dict/None): Precision threshold for constrained classes.
        fold(int): Fold number.

    Returns:
        None: Writes results to CSV file and returns nothing.
    """
    if method == "Full MIP":
        solution.write_integrated_results(integrated_csv, X_test, y_test, precision_threshold=precision_threshold, fold=fold)
    elif method == "PIP":
        solution.write_integrated_results(integrated_csv, method, X_test, y_test, precision_threshold=precision_threshold, fold=fold)
    elif method == "ISA-PIP":
        pip = solution[-1]
        pip.write_integrated_results(integrated_csv, method, X_test, y_test, precision_threshold=precision_threshold, fold=fold)
    elif method == "D4-PIP":
        solution.write_integrated_results(integrated_csv, method, X_test, y_test, precision_threshold=precision_threshold, fold=fold)
    elif method == "D-PIP":
        solution.write_integrated_results(integrated_csv, method, X_test, y_test, precision_threshold=precision_threshold, fold=fold)
    elif method == "IDSA4-PIP":
        solution.write_integrated_results(integrated_csv, method, X_test, y_test, precision_threshold=precision_threshold, fold=fold)
    elif method == "IDSA-PIP":
        solution.write_integrated_results(integrated_csv, method, X_test, y_test, precision_threshold=precision_threshold, fold=fold)