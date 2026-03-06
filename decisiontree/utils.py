'''
Author: zhengke 1604367740@qq.com
Date: 2024-12-01 12:24:26
LastEditors: zhengke 1604367740@qq.com
LastEditTime: 2024-12-05 09:19:48
FilePath: /AHC_max_accuracy/decisiontree/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import itertools
import random
from datetime import datetime
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from collections import deque
import math
import os
import random


def ancestors(D):
    A_L = {}
    A_R = {}
    for t in range(2**D):
        A_L[t] = []
        A_R[t] = []
        real_t = t + 2 ** D - 1
        current_node = real_t
        while current_node != 0:
            parent_node = (current_node - 1) // 2
            if current_node == 2 * parent_node + 1:
                A_L[t].append(parent_node)
            else:
                A_R[t].append(parent_node)
            current_node = parent_node
    return A_L, A_R



def heaviside_closed(lb, x):
    if x >= lb:
        return 1
    else:
        return 0


def heaviside_open(lb, x):
    if x > lb:
        return 1
    else:
        return 0


def calculate_gamma(X_train, y_train, a, b, c, D, selected_piece, beta_p, epsilon, class_restricted):
    N = X_train.shape[0]
    p = X_train.shape[1]
    A_L, A_R = ancestors(D)
    total_class_num = len(Counter(y_train))
    J = range(1, total_class_num+1)  # classes
    class_index = {cls: np.where(y_train == cls)[0] for cls in J}  # use to select samples of certain class
    z_plus_0_start = {}
    z_plus_start = {}
    z_minus_start = {}
    gamma = {}
    for s in range(N):
        for t in range(2**D):
            z_plus_0_start[s, t] = heaviside_closed(0, min([sum(a[k][i]*X_train[s][i] for i in range(p)) - b[k] - 1 for k in A_R[t]] + [-sum(a[k][i]*X_train[s][i] for i in range(p)) + b[k] - 1 for k in A_L[t]]))
            z_plus_start[s, t] = heaviside_closed(0, min([sum(a[k][i]*X_train[s][i] for i in range(p)) - b[k] for k in A_R[t]]+ [-sum(a[k][i]*X_train[s][i] for i in range(p)) + b[k] - epsilon for k in A_L[t]]))
            if selected_piece is None:
                z_minus_start[s, t] = 1 - heaviside_open(0, min([sum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k] for k in A_L[t]] + [sum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] + epsilon for k in A_R[t]])) 
            else:
                k = selected_piece[s][t]
                if k in A_L[t]:
                    z_minus_start[s, t] = 1 - heaviside_open(0, sum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k]) 
                if k in A_R[t]:
                    z_minus_start[s, t] = 1 - heaviside_open(0, sum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] + epsilon)
    for j in class_restricted:
        class_true = heaviside_closed(0, sum(c[j,t]*z_plus_start[s, t] for s in range(N) for t in range(2**D)) - 1)
        gamma[j] = max(0, -(sum(c[j,t]*z_plus_start[s, t] for s in class_index[j] for t in range(2**D)) - beta_p[j]*sum(c[j,t]*(1-z_minus_start[s, t]) for s in range(N) for t in range(2**D))), 100*(1-class_true))
    return gamma, z_plus_0_start, z_plus_start, z_minus_start


def calculate_z_plus_0(X_train, a, b, D):
    N = X_train.shape[0]
    p = X_train.shape[1]
    A_L, A_R = ancestors(D)
    z_plus_0_start = {}
   
    for s in range(N):
        for t in range(2**D):
            z_plus_0_start[s, t] = heaviside_closed(0, min([sum(a[k][i]*X_train[s][i] for i in range(p)) - b[k] - 1 for k in A_R[t]] + [-sum(a[k][i]*X_train[s][i] for i in range(p)) + b[k] - 1 for k in A_L[t]]))
            
    return z_plus_0_start


def calculate_delta(X_train, a, b, D, selected_piece, epsilon, base_rate):
    N = X_train.shape[0]
    p = X_train.shape[1]
    if epsilon is not None:  # In unconstrained case, epsilon is None
        value_ge = {0:[], 1:[]}
        value_le = {0:[], 1:[]}
        delta_1 = {0:1, 1:1}
        delta_2 = {0:1, 1:1}
    else:
        value_ge = {0:[]}
        value_le = {0:[]}
        delta_1 = {0:1}
        delta_2 = {0:1}
    index_odd = 0
    
    A_L, A_R = ancestors(D)
            
    for s in range(N):
        for t in range(2**D):
            phi_plus_0 = min([sum(a[k][i]*X_train[s][i] for i in range(p)) - b[k] - 1 for k in A_R[t]] + [-sum(a[k][i]*X_train[s][i] for i in range(p)) + b[k] - 1 for k in A_L[t]])
            if phi_plus_0 > 0:
                value_ge[0].append(phi_plus_0)
            elif phi_plus_0 < 0:
                value_le[0].append(phi_plus_0)
            else:
                if index_odd == 0:
                    value_ge[0].append(phi_plus_0)
                else:
                    value_le[0].append(phi_plus_0)
                index_odd += 1
            
            if epsilon is not None:  # In unconstrained case, epsilon is None
                phi_plus = min([sum(a[k][i]*X_train[s][i] for i in range(p)) - b[k] for k in A_R[t]]+[-sum(a[k][i]*X_train[s][i] for i in range(p)) + b[k] - epsilon for k in A_L[t]])
                if phi_plus > 0:
                    value_ge[1].append(phi_plus)
                elif phi_plus < 0:
                    value_le[1].append(phi_plus)
                else:
                    if index_odd == 0:
                        value_ge[1].append(phi_plus)
                    else:
                        value_le[1].append(phi_plus)
                    index_odd += 1
                
                if selected_piece is None:
                    underline_phi = max([sum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] for k in A_L[t]] + [sum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k] - epsilon for k in A_R[t]])
                    if underline_phi > 0:
                        value_ge[1].append(underline_phi)
                    elif underline_phi < 0:
                        value_le[1].append(underline_phi)
                    else:
                        if index_odd == 0:
                            value_ge[1].append(underline_phi)
                        else:
                            value_le[1].append(underline_phi)
                        index_odd += 1
                else:
                    k = selected_piece[s][t]
                    if k in A_L[t]:
                        phi_minus = sum(-a[k][i] * X_train[s][i] for i in range(p)) + b[k]
                    if k in A_R[t]:
                        phi_minus = sum(a[k][i] * X_train[s][i] for i in range(p)) - b[k] + epsilon
                    if -phi_minus > 0:
                        value_ge[1].append(-phi_minus)
                    elif -phi_minus < 0:
                        value_le[1].append(-phi_minus)
                    else:
                        if index_odd == 0:
                            value_ge[1].append(-phi_minus)
                        else:
                            value_le[1].append(-phi_minus)
                        index_odd += 1
    if epsilon is not None:
        if base_rate < 100:
            delta_1[0], delta_1[1]= max(1e-5, np.percentile(value_ge[0], base_rate) if len(value_ge[0])>0 else 1e-5), max(1e-5, np.percentile(value_ge[1], base_rate) if len(value_ge[1])>0 else 1e-5)
            delta_2[0], delta_2[1] = max(1e-5, -np.percentile(value_le[0], 100 - base_rate) if len(value_le[0])>0 else 1e-5), max(1e-5, -np.percentile(value_le[1], 100 - base_rate) if len(value_le[1])>0 else 1e-5) 
        else:
            delta_1[0], delta_1[1] = np.max(value_ge[0]) if len(value_ge[0])>0 else 1e-5, np.max(value_ge[1]) if len(value_ge[1])>0 else 1e-5
            delta_2[0], delta_2[1] = -np.min(value_le[0]) if len(value_le[0])>0 else 1e-5, -np.min(value_le[1]) if len(value_le[1])>0 else 1e-5
    else:
        if base_rate < 100:
            delta_1[0] = max(1e-5, np.percentile(value_ge[0], base_rate) if len(value_ge[0])>0 else 1e-5)
            delta_2[0] = max(1e-5, -np.percentile(value_le[0], 100 - base_rate) if len(value_le[0])>0 else 1e-5)
        else:
            delta_1[0] = np.max(value_ge[0]) if len(value_ge[0])>0 else 1e-5
            delta_2[0] = -np.min(value_le[0]) if len(value_le[0])>0 else 1e-5
        
    return delta_1, delta_2


def calculate_eta_zeta_L(X_train, y_train, c, D, z_plus_0, z_plus, z_minus):
    total_class_num = len(Counter(y_train))
    N = X_train.shape[0]
    J = range(1, total_class_num+1)  
    class_index = {cls: np.where(y_train == cls)[0] for cls in J}
    M_eta = {cls: sum(1 for _ in class_index[cls]) for cls in J}
    L = {}
    eta = {}
    zeta = {}
    for j in J:
        for t in range(2**D):
            if c[j, t] > 1/2:
                L[t] = sum(z_plus_0[s, t] for s in class_index[j]) + N*(1-c[j,t])
                eta[j,t] = sum(z_plus[s,t] for s in class_index[j]) + M_eta[j]*(1-c[j,t])
                zeta[j,t] = sum((1-z_minus[s,t]) for s in range(N)) - N*(1-c[j,t])
            else:
                eta[j,t] = M_eta[j]*c[j,t]
                zeta[j,t] = 0
    return eta, zeta, L


def calculate_L(X_train, y_train, c, D, z_plus_0):
    total_class_num = len(Counter(y_train))
    N = X_train.shape[0]
    J = range(1, total_class_num+1)  
    class_index = {cls: np.where(y_train == cls)[0] for cls in J}
    L = {}
    
    for j in J:
        for t in range(2**D):
            if c[j, t] > 1/2:
                L[t] = sum(z_plus_0[s, t] for s in class_index[j]) + N*(1-c[j,t])
            
    return L


def get_positions_in_complete_binary_tree(children_left, children_right, depth):
    total_nodes = 2 ** (depth + 1) - 1
    level_order_positions = [-1] * total_nodes
    queue = deque([(0, 0)])  
    position_counter = 0  
    while queue:
        node, position = queue.popleft() 
        level_order_positions[position] = node
        if children_left[node] != -1:  
            queue.append((children_left[node], 2 * position + 1))  
        if children_right[node] != -1:  
            queue.append((children_right[node], 2 * position + 2))  
    return level_order_positions



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
    

def train_decisiontree_model(X_train, y_train, max_depth):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    tree_structure = clf.tree_
    children_left = tree_structure.children_left
    children_right = tree_structure.children_right
    tree_level_order_positions = get_positions_in_complete_binary_tree(children_right, children_left, max_depth)
    D = math.floor(math.log(len(tree_level_order_positions),2))
    J = list(set(y_train))
    feature_dim = X_train.shape[1]
    a_start = {}
    b_start = {}
    for k in range(2**D-1):
        a_start[k] = np.zeros(feature_dim)
        b_start[k] = 0
    keys_c = [(j, t) for j in J for t in range(2**D)]
    c_start = {key: 0 for key in keys_c}

    parent_features = {}
    for i in range(len(tree_level_order_positions)):
        node_id = tree_level_order_positions[i]
        if i > 0:
            parent = (i-1) // 2
            parent_features[i] = parent_features[parent].copy()
        else:
            parent = None
            parent_features[i] = set()

        if node_id == -1:
            parent_node = i
            while tree_level_order_positions[parent_node] == -1:
                parent_node = (parent_node - 1) // 2
            class_label = np.argmax(tree_structure.value[tree_level_order_positions[parent_node]][0])+1
            if i >= 2**D -1:
                j = class_label
                t = i - (2**D - 1)
                c_start[j, t] = 1
            else:
                k = i
                available_features = list(set(range(feature_dim)) - parent_features[parent])
                if available_features == []:
                    b_start[k] = i
                else:
                    rng = np.random.default_rng(seed=42)
                    chosen_feature = rng.choice(available_features)
                    a_start[k][chosen_feature] = -100
                    b_start[k] = 0
                    parent_features[i].add(chosen_feature)
            # Set a_start[k][arbitrary_feature] = -10 instead of 0, set b_start[k] = i to ensure \mathcal M is a singelton
        elif tree_structure.feature[node_id]== -2:
            class_label = np.argmax(tree_structure.value[node_id][0]) +1
            if i >= 2**D -1:
                j = class_label
                t = i - (2**D - 1)
                c_start[j, t] = 1
            else: 
                k = i
                available_features = list(set(range(feature_dim)) - parent_features[parent])
                if available_features == []:
                    b_start[k] = i
                else:
                    rng = np.random.default_rng(seed=42)
                    chosen_feature = rng.choice(available_features)
                    a_start[k][chosen_feature] = -100
                    b_start[k] = 0
                    parent_features[i].add(chosen_feature)
            # Set a_start[k][arbitrary_feature] = -10 instead of 0, set b_start[k] = i to ensure \mathcal M is a singelton
        else:
            feature_index = tree_structure.feature[node_id]
            threshold = tree_structure.threshold[node_id]
            k = i
            a_start[k][feature_index] = -100
            b_start[k] = -100*threshold 
            parent_features[i].add(feature_index)
            # 10*original a_start, b_start to ensure there exists \phi_plus_0 > 0 
    result = {'a': a_start, 'b': b_start, 'c': c_start}
    return result


def generate_M_delta(X_train, enhanced_size, a_start, b_start, D, epsilon, integer_rate): 
    N = X_train.shape[0]
    p = X_train.shape[1]
    phi_max = {}
    value_list = []
    A_L, A_R = ancestors(D)
    selected_piece = generate_random_combination(X_train, a_start, b_start, D, epsilon)
    delta_1, delta_2 = calculate_delta(X_train=X_train, a=a_start, b=b_start, D=D, selected_piece=selected_piece, epsilon=epsilon, base_rate=integer_rate)
    for s in range(N):
        phi_max[s] = {}
        for t in range(2**D):
            # the max value in each (s,t)
            phi_max[s][t] = max([-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon for k in A_R[t]]+[sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k] for k in A_L[t]])
            if phi_max[s][t] >= -delta_2[1]:
                max_index = 0 # to avoid multiply max values
                for k in A_R[t]:
                    if (-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon) < phi_max[s][t]:
                        value_list.append([phi_max[s][t] - (-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon),(s,t)])
                    elif (-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon) == phi_max[s][t]:
                        max_index += 1
                        if max_index > 1:
                            value_list.append([0,(s,t)])
                for k in A_L[t]:
                    if (sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k]) < phi_max[s][t]:
                        value_list.append([phi_max[s][t] - (sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k]),(s,t)])
                    elif (sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k]) == phi_max[s][t]:
                        max_index += 1
                        if max_index > 1:
                            value_list.append([0,(s,t)])
    value_list.sort(key=lambda x: x[0])
    if enhanced_size == 1:
        delta, key = 0, value_list[0]
    else:
        index = int(math.log(enhanced_size, 2))
        delta, key = value_list[index-1][0], value_list[0:index]
        
    M_set_index = {}
    for s in range(N):
        M_set_index[s] = {}
        for t in range(2**D):
            M_set_index[s][t] = []
            max_index = 0 # to avoid multiply max values
            for k in A_R[t]:
                if phi_max[s][t] - (-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon) == 0 and max_index == 0:
                    max_index += 1
                    M_set_index[s][t].append(k)
                elif phi_max[s][t] - (-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon) <= delta:
                    if (s,t) in [row[1] for row in key]:
                        M_set_index[s][t].append(k)
            for k in A_L[t]:
                if phi_max[s][t] - (sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k]) == 0 and max_index == 0:
                    max_index += 1
                    M_set_index[s][t].append(k)
                elif phi_max[s][t] - (sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k]) <= delta:
                    if (s,t) in [row[1] for row in key]:
                        M_set_index[s][t].append(k)
    return M_set_index


def generate_M(X_train, a_start, b_start, D, epsilon, integer_rate):
    N = X_train.shape[0]
    p = X_train.shape[1]
    phi_max = {}
    M_set_index = {}
    A_L, A_R = ancestors(D)
    rng = np.random.default_rng(seed=42)
    
    for s in range(N):
        phi_max[s] = {}
        M_set_index[s] = {}
        for t in range(2**D):
            M_set_index[s][t] = []
            phi_max[s][t] = max([-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon for k in A_R[t]]+[sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k] for k in A_L[t]])
            for k in A_R[t]:
                if phi_max[s][t] - (-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon) == 0:
                    M_set_index[s][t].append(k)
            for k in A_L[t]:
                if phi_max[s][t] - (sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k]) == 0:
                    M_set_index[s][t].append(k)
    selected_piece = generate_random_combination(X_train, a_start, b_start, D, epsilon)
    delta_1, delta_2 = calculate_delta(X_train=X_train, a=a_start, b=b_start, D=D, selected_piece=selected_piece, epsilon=epsilon, base_rate=integer_rate)
    multi_piece = 0
    for s in range(N):
        for t in range(2**D):
            if len(M_set_index[s][t])>1:
                if phi_max[s][t] < -delta_2[1]:
                    M_set_index[s][t] = [rng.choice(M_set_index[s][t])]
                else:
                    multi_piece += 1
                    if multi_piece <= 2:
                        M_set_index[s][t] = rng.choice(M_set_index[s][t], size=2, replace=False).tolist()
                    if multi_piece > 2:
                        M_set_index[s][t] = [rng.choice(M_set_index[s][t])]
    return M_set_index, multi_piece


def generate_combinations(nested_dict):
    outer_keys = list(nested_dict.keys())
    inner_keys = {s: list(nested_dict[s].keys()) for s in outer_keys}
    
    value_combinations = []
    for s in outer_keys:
        value_combinations.append(
            list(itertools.product(*[nested_dict[s][t] for t in inner_keys[s]]))
        )
    all_combinations = itertools.product(*value_combinations)
    
    result = []
    for combination in all_combinations:
        new_dict = {}
        for idx, outer_key in enumerate(outer_keys):
            new_dict[outer_key] = {
                inner_keys[outer_key][i]: value
                for i, value in enumerate(combination[idx])
            }
        result.append(new_dict)
    return result


def generate_random_combination(X_train, a_start, b_start, D, epsilon):
    N = X_train.shape[0]
    p = X_train.shape[1]
    phi_max = {}
    M_set_index = {}
    selected_piece = {}
    A_L, A_R = ancestors(D)
    for s in range(N):
        phi_max[s] = {}
        M_set_index[s] = {}
        selected_piece[s] = {}
        for t in range(2**D):
            M_set_index[s][t] = []
            phi_max[s][t] = max([-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon for k in A_R[t]]+[sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k] for k in A_L[t]])
            for k in A_R[t]:
                if phi_max[s][t] - (-sum(a_start[k][i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon) == 0:
                    M_set_index[s][t].append(k)
            for k in A_L[t]:
                if phi_max[s][t] - (sum(a_start[k][i]*X_train[s][i] for i in range(p)) - b_start[k]) == 0:
                    M_set_index[s][t].append(k)
            rng = np.random.default_rng(seed=42)
            selected_piece[s][t] = rng.choice(M_set_index[s][t]) 
    return selected_piece


def evaluate_tree(X, y, a, b, c, D): 
    J = list(set(y))
    N = X.shape[0]
    p = X.shape[1]
    A_L, A_R = ancestors(D)
    heaviside_sets = {}
    leaf_node = {}
    nm_accuracy = 0
    nm_accuracy_margin = 0
    nm_precision = {key: 0 for key in J}
    dm_precision = {key: 0 for key in J}

    frac = {}
    counts = {}

    for s in range(N):
        for t in range(2**D):
            if round(c[y[s],t],0) == 1:
                nm_accuracy_margin += heaviside_closed(0, min([sum(a[k][i]*X[s][i] for i in range(p)) - b[k] - 1 for k in A_R[t]] + [-sum(a[k][i]*X[s][i] for i in range(p)) + b[k] - 1 for k in A_L[t]]))
    
    for s in range(N):
        heaviside_sets[s]={}
        leaf_node[s] = -1
        for t in range(2**D):
            heaviside_sets[s][t]=[]
            for k in A_R[t]:
                heaviside_sets[s][t].append(heaviside_closed(0,sum(a[k][i]*X[s][i] for i in range(p))-b[k]))
            for k in A_L[t]:
                heaviside_sets[s][t].append(heaviside_open(0,-sum(a[k][i]*X[s][i] for i in range(p))+b[k]))
            if math.prod(heaviside_sets[s][t]) == 1:
                leaf_node[s] = t

    for s in range(N):
        if leaf_node[s] >= 0:
            if round(c[y[s], leaf_node[s]],0) == 1:
                nm_accuracy += 1
                nm_precision[y[s]] += 1
                dm_precision[y[s]] += 1
            else:
                for j in J:
                    if round(c[j, leaf_node[s]],0) == 1:
                        dm_precision[j] += 1
            
    frac['acc'] = round(nm_accuracy/N,3)
    frac['acc_margin'] = round(nm_accuracy_margin/N,3)
    counts['acc'] = {'nm': nm_accuracy, 'dm': N}
    counts['acc_margin'] = {'nm': nm_accuracy_margin, 'dm':N}
    for j in J:  
        frac[f'prec{j}'] = round(nm_precision[j]/dm_precision[j],3) if dm_precision[j]>0 else -1
        counts[f'prec{j}'] = {'nm': nm_precision[j],'dm':dm_precision[j]}

    result = {'frac': frac, 'counts': counts}
    return result


def train_test_results(X_train, y_train, X_test, y_test, solution, D, J, beta_p, class_restricted):
    train_result = evaluate_tree(X_train, y_train, solution['a'], solution['b'], solution['c'], D)
    train_constraint_gap = {}

    test_result = evaluate_tree(X_test, y_test, solution['a'], solution['b'], solution['c'], D)
    test_constraint_gap = {}

    test_train_gap = {}
    test_train_gap['acc'] = round(test_result['frac']['acc'] - train_result['frac']['acc'], 3)
    test_train_gap['acc_margin'] = round(test_result['frac']['acc_margin'] - train_result['frac']['acc_margin'],3)

    if beta_p is not None:
        for j in class_restricted:
            if train_result['frac'][f'prec{j}'] >= 0:
                train_constraint_gap[f'prec{j}'] = min(0, round(train_result['frac'][f'prec{j}'] - beta_p[j],3))
            else:
                train_constraint_gap[f'prec{j}'] = -2
    
        for j in class_restricted:
            if test_result['frac'][f'prec{j}'] >= 0:
                test_constraint_gap[f'prec{j}'] = min(0, round(test_result['frac'][f'prec{j}'] - beta_p[j],3))
            else:
                test_constraint_gap[f'prec{j}'] = -2
    
        for j in J:
            if train_result['frac'][f'prec{j}'] >= 0 and test_result['frac'][f'prec{j}'] >= 0:
                test_train_gap[f'prec{j}'] = round(test_result['frac'][f'prec{j}'] - train_result['frac'][f'prec{j}'], 3)
            elif train_result['frac'][f'prec{j}'] == -1:
                test_train_gap[f'prec{j}'] = 999
            elif test_result['frac'][f'prec{j}'] == -1:
                test_train_gap[f'prec{j}'] = -999
    else:
        train_constraint_gap, test_constraint_gap = None, None

    return train_result, test_result, train_constraint_gap, test_constraint_gap, test_train_gap

def format_output(data):
    """
    Formats the input data:
    - Converts arrays to lists
    - Converts numpy-specific types (e.g., np.float64, np.int64) to Python native types
    - Rounds floating-point numbers to 4 decimal places
    """
    if isinstance(data, dict):
        formatted_dict = {}
        for key, value in data.items():
            # Ensure keys are hashable
            if isinstance(key, (list, np.ndarray, tuple)):
                key = tuple(format_output(item) for item in key)
            elif isinstance(key, (np.integer, int)):
                key = int(key)  # Convert numpy int to Python int
            formatted_dict[key] = format_output(value)
        return formatted_dict
    elif isinstance(data, (list, tuple, np.ndarray)):
        return [format_output(item) for item in data]
    elif isinstance(data, (float, np.float32, np.float64)):
        return round(float(data), 4)  # Round to 4 decimal places
    elif isinstance(data, (int, np.integer)):
        return int(data)  # Convert numpy int to Python int
    else:
        return data
    

def get_class_distribution(y):
    """
    Returns the class distribution as a dictionary.
    - y: array-like, target labels
    """
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


def sample_data(dataset=None):
    if dataset == 'anth':
        df = pd.read_csv('dataset/ann_thyroid.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'blsc':
        df = pd.read_csv('dataset/balance_scale.csv', encoding='utf-8')
        # Extract features and Classs
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'ceva':
        df = pd.read_csv('dataset/car_evaluation.csv', encoding='utf-8')
        # Extract features and Classs
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'ctmc':
        df = pd.read_csv('dataset/contraceptive_method_choice.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('contraceptive_method', axis=1)
        y_sampled = df['contraceptive_method']
    if dataset == 'dmtl':
        df = pd.read_csv('dataset/dermatology.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'dryb':
        df = pd.read_csv('dataset/dry_bean.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('Class', axis=1)
        y_sampled = df['Class']
    if dataset == 'fish':
        df = pd.read_csv('dataset/fish.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'htds':
        df = pd.read_csv('dataset/heart_disease.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('num', axis=1)
        y_sampled = df['num']
    if dataset == 'imsg':
        df = pd.read_csv('dataset/image_segmentation.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'iris':
        df = pd.read_csv('dataset/iris.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'nwth':
        df = pd.read_csv('dataset/new_thyroid.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'orhd':
        df = pd.read_csv('dataset/optical_recognition_of_handwritten_digits.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'seed':
        df = pd.read_csv('dataset/seeds.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'taev':
        df = pd.read_csv('dataset/tae_onehot.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'wine':
        df = pd.read_csv('dataset/wine.csv', encoding='utf-8')
        # Extract features and Classs
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']

    return X_sampled, y_sampled


