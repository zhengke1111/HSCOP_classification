from collections import deque
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import itertools
import csv
import math
from typing import Union
# from model import Model
# from algorithm import PIP, IterativeShrinkage

def generate_epsilon(max_iter):
    """
    Generate a sequence of epsilon values for iterative shrinkage (decays exponentially).
    Epsilon controls the relaxation of constraints during optimization.

    Args:
        max_iter (int): Total number of iterations for shrinkage.

    Returns:
        list: Sequence of epsilon values (length = max_iter).
    """
    epsilon_0 = 0.1
    shrinkage_rate = 0.1
    epsilon = [epsilon_0 * (shrinkage_rate ** (i - 1)) for i in range(1, max_iter + 1)]
    return epsilon


def ancestors(D):
    """
    Compute the left-ancestor and right-ancestor sets for each leaf node in a complete binary decision tree of depth `D`.

    The tree is indexed using the standard array-based binary tree convention:
    - the root node has index 0,
    - for an internal node `p`, its left child is `2*p + 1`,
    - its right child is `2*p + 2`.

    Leaves are indexed externally by `t = 0, ..., 2**D - 1`. Internally, these correspond to node indices `2**D - 1, ..., 2**(D+1) - 2` in the complete binary tree.

    For each leaf `t`, the function traces the path from that leaf back to the root and records:
    - `A_L[t]`: the set of ancestor nodes at which the path moves through the left child,
    - `A_R[t]`: the set of ancestor nodes at which the path moves through the right child.

    Args:
        D: An integer representing the depth of the decision tree.

    Returns:
        tuple:
            - `A_L` (dict): a dictionary where `A_L[t]` is the list of ancestor node indices for leaf `t` corresponding to left branches;
            - `A_R` (dict): a dictionary where `A_R[t]` is the list of ancestor node indices for leaf `t` corresponding to right branches.

    Notes:
        - The tree is assumed to be a complete binary tree.
        - There are exactly `2**D` leaves.
        - The returned ancestor lists are ordered from the leaf upward toward the root.
    """
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
    """
    Calculate the value of Heaviside function \mathbf{1}_{[lb,\infty)}(x)

    Args:
        lb (float): the left endpoint of the interval of the Heaviside function
        x (float): the variable of the Heaviside function

    Returns:
        int: 0 or 1
    """
    if x >= lb:
        return 1
    else:
        return 0


def heaviside_open(lb, x):
    """
    Calculate the value of Heaviside function \mathbf{1}_{(lb,\infty)}(x)

    Args:
        lb (float): the left endpoint of the interval of the Heaviside function
        x (float): the variable of the Heaviside function

    Returns:
        int: 0 or 1
    """
    if x > lb:
        return 1
    else:
        return 0


def calculate_gamma(X_train, y_train, a, b, c, D, selected_piece, beta_p, epsilon, class_restricted):
    r"""
    Compute the violation amount \gamma and the initial solutions of binary variables \xi, z^+, z^- 
    Used in precision constrained problem

    Args:
        X_train (ndarray): training set: X
        y_train (ndarray): training set: y
        a (ndarray): parameter a = \{a_k\}_{k\in {\cal T}_{\cal B}} in classification score a_k @ X - b_k at node k
        b (ndarray): parameter b = \{b_k\}_{k\in {\cal T}_{\cal B}} in classification score a_k @ X - b_k at node k
        c (ndarray): parameter c = \{c_{jt}\}_{j\in [J], t\in {\cal T}_{\ell}}, assign class j to leaf node t
        beta_p (dict): {class: precision threshold}
        epsilon (float): epsilon
        class_restricted (int): the class(es) that have precision lower bound(s) 

    Returns:
        tuple[int, dict, dict, dict]: gamma,  z_plus_0_start, z_plus_start, z_minus_start
                                      \gamma, \xi,            z^+,          z^-
    """
    N = X_train.shape[0]
    p = X_train.shape[1]
    A_L, A_R = ancestors(D)
    total_class_num = len(Counter(y_train))
    J = range(1, total_class_num+1)  # classes
    class_index = {cls: np.where(y_train == cls)[0] for cls in J}  # use to select samples of certain class
    z_plus_0_start, z_plus_start, z_minus_start = {}, {}, {}
    gamma = {}
    for s in range(N):
        for t in range(2**D):
            z_plus_0_start[s, t] = heaviside_closed(0, min([sum(a[k, i]*X_train[s][i] for i in range(p)) - b[k] - 1 for k in A_R[t]] + [-sum(a[k, i]*X_train[s][i] for i in range(p)) + b[k] - 1 for k in A_L[t]]))
            z_plus_start[s, t] = heaviside_closed(0, min([sum(a[k, i]*X_train[s][i] for i in range(p)) - b[k] for k in A_R[t]]+ [-sum(a[k, i]*X_train[s][i] for i in range(p)) + b[k] - epsilon for k in A_L[t]]))
            if selected_piece is None:
                z_minus_start[s, t] = 1 - heaviside_open(0, min([sum(-a[k, i] * X_train[s][i] for i in range(p)) + b[k] for k in A_L[t]] + [sum(a[k, i] * X_train[s][i] for i in range(p)) - b[k] + epsilon for k in A_R[t]])) 
            else:
                k = selected_piece[s][t]
                if k in A_L[t]:
                    z_minus_start[s, t] = 1 - heaviside_open(0, sum(-a[k, i] * X_train[s][i] for i in range(p)) + b[k]) 
                if k in A_R[t]:
                    z_minus_start[s, t] = 1 - heaviside_open(0, sum(a[k, i] * X_train[s][i] for i in range(p)) - b[k] + epsilon)
    for j in class_restricted:
        class_true = heaviside_closed(0, sum(c[j,t]*z_plus_start[s, t] for s in range(N) for t in range(2**D)) - 1)
        gamma[j] = max(0, -(sum(c[j,t]*z_plus_start[s, t] for s in class_index[j] for t in range(2**D)) - beta_p[j]*sum(c[j,t]*(1-z_minus_start[s, t]) for s in range(N) for t in range(2**D))), 100*(1-class_true))
    return gamma, z_plus_0_start, z_plus_start, z_minus_start


def calculate_z_plus_0(X_train, a, b, D):
    r"""
    Compute the initial solution of binary variable \xi
    Used in unconstrained problem, in which there are only \xi, no z^+, z^- and precision constraint violation \gamma

    Args:
        X_train (ndarray): training set: X
        a (ndarray): parameter a = \{a_k\}_{k\in {\cal T}_{\cal B}} in classification score a_k @ X - b_k at node k
        b (ndarray): parameter b = \{b_k\}_{k\in {\cal T}_{\cal B}} in classification score a_k @ X - b_k at node k
        c (ndarray): parameter c = \{c_{jt}\}_{j\in [J], t\in {\cal T}_{\ell}}, assign class j to leaf node t

    Returns:
        dict: z_plus_0_start (intial solution of \xi)
    """
    N = X_train.shape[0]
    p = X_train.shape[1]
    A_L, A_R = ancestors(D)
    z_plus_0_start = {}
   
    for s in range(N):
        for t in range(2**D):
            z_plus_0_start[s, t] = heaviside_closed(0, min([sum(a[k, i]*X_train[s][i] for i in range(p)) - b[k] - 1 for k in A_R[t]] + [-sum(a[k, i]*X_train[s][i] for i in range(p)) + b[k] - 1 for k in A_L[t]]))
            
    return z_plus_0_start


def calculate_delta(X_train, a, b, D, selected_piece, epsilon, base_rate):
    """
    The index sets {\cal J}_{0;<}, {\cal J}_{0;>}, {\cal J}_{0;0} and {\cal J}_{1;<}, {\cal J}_{1;>}, {\cal J}_{1;0} are determined by the
    values of the corresponding path-based functions \phi:

        {\cal J}^+_{0;<} = \{ (s,t) \mid \phi_{0;st}(a,b) < -\delta_2[0] \}
        {\cal J}^+_{0;>} = \{ (s,t) \mid \phi_{0;st}(a,b) > \delta_1[0] \}
        {\cal J}^+_{0;0} = \{ (s,t) \mid -\delta_2[0] \le \phi_{0;st}(a,b) \le \delta_1[0] \}

        {\cal J}^+_{1;<} = \{ (s,t) \mid \phi_{1;st}(a,b) < -\delta_2[1] \}
        {\cal J}^+_{1;>} = \{ (s,t) \mid \phi_{1;st}(a,b) > \delta_1[1] \}
        {\cal J}^+_{1;0} = \{ (s,t) \mid -\delta_2[1] \le \phi_{1;st}(a,b) \le \delta_1[1] \}

        {\cal J}^-_{1;<} = \{ (s,t) \mid -\phi^{PA or \ell_{st};-}_{st}(a,b) - \varepsilon < -\delta_2[1] \}
        {\cal J}^+_{1;>} = \{ (s,t) \mid -\phi^{PA or \ell_{st};-}_{st}(a,b) - \varepsilon > \delta_1[1] \}
        {\cal J}^+_{1;0} = \{ (s,t) \mid -\delta_2[1] \le -\phi^{PA or \ell_{st};-}_{st}(a,b) - \varepsilon \le \delta_1[1] \}

    More precisely:
    - `delta_1` is determined from the positive \phi-values;
    - `delta_2` is determined from the negative \phi-values;
    - if `base_rate < 100`, quantiles are used;
    - if `base_rate == 100`, extreme values are used instead.

    In the unconstrained case, `epsilon` is `None`, so only the index sets associated with type `0` are computed.

    Args:
        X_train (ndarray): training set: X
        a (ndarray): parameter a = \{a_k\}_{k\in {\cal T}_{\cal B}} in classification score a_k @ X - b_k at node k
        b (ndarray): parameter b = \{b_k\}_{k\in {\cal T}_{\cal B}} in classification score a_k @ X - b_k at node k
        D (int): Depth of the decision tree.
        selected_piece: Selected piece used in the piecewise decomposition; if `None`, the full form is used.
        epsilon (float or None): Margin parameter. \varepsilon
        base_rate (float): Integer ratio as the quantile level used to determine the thresholds defining the in-between sets.

    Returns:
        tuple[dict, dict]:
            - `delta_1`: dictionary containing the upper thresholds;
            - `delta_2`: dictionary containing the lower thresholds.

            If `epsilon is not None`, then
                `delta_1 = {0: \delta_1[0], 1: \delta_1[1]}`
                `delta_2 = {0: \delta_2[0], 1: \delta_2[1]}`.

            If `epsilon is None`, then only
                `delta_1 = {0: \delta_1[0]}`
                `delta_2 = {0: \delta_2[0]}`
            are returned.
    """
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
            phi_plus_0 = min([sum(a[k, i]*X_train[s][i] for i in range(p)) - b[k] - 1 for k in A_R[t]] + [-sum(a[k, i]*X_train[s][i] for i in range(p)) + b[k] - 1 for k in A_L[t]])
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
                phi_plus = min([sum(a[k, i]*X_train[s][i] for i in range(p)) - b[k] for k in A_R[t]]+[-sum(a[k, i]*X_train[s][i] for i in range(p)) + b[k] - epsilon for k in A_L[t]])
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
                    underline_phi = max([sum(a[k, i] * X_train[s][i] for i in range(p)) - b[k] for k in A_L[t]] + [sum(-a[k, i] * X_train[s][i] for i in range(p)) + b[k] - epsilon for k in A_R[t]])
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
                        phi_minus = sum(-a[k, i] * X_train[s][i] for i in range(p)) + b[k]
                    if k in A_R[t]:
                        phi_minus = sum(a[k, i] * X_train[s][i] for i in range(p)) - b[k] + epsilon
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
    r"""
    Compute the quantities `eta`, `zeta`, and `L` for each class-leaf pair in the precision constrained decision tree model.

    For each class `j` and leaf node `t`, this function aggregates the corresponding `z_plus_0`, `z_plus`, and `z_minus` values over the relevant
    samples. These quantities are used to evaluate how samples of each class are assigned to leaves under the current leaf-class assignment `c`.

    Args:
        X_train (ndarray): training set: X
        y_train (ndarray): training set: y
        c (ndarray): parameter c = \{c_{jt}\}_{j\in [J], t\in {\cal T}_{\ell}}, assign class j to leaf node t
        D (int): Depth of the decision tree.
        z_plus_0 (dict): Values of `z_plus_0[(s, t)]` (\xi_{st}) for sample-leaf pairs.
        z_plus (dict): Values of `z_plus[(s, t)]`     (z^+_{st}) for sample-leaf pairs.
        z_minus (dict): Values of `z_minus[(s, t)]`   (z^-_{st}) for sample-leaf pairs.

    Returns:
        tuple[dict, dict, dict]:
            - `eta[(j, t)]`: sum of `z_plus` over samples in class `j` for leaf `t`,
            - `zeta[(j, t)]`: sum of `1 - z_minus` over all samples for leaf `t`,
            - `L[t]`: sum of `z_plus_0` over samples of the class assigned to leaf `t`.
    """
    total_class_num = len(Counter(y_train))
    N = X_train.shape[0]
    J = range(1, total_class_num+1)  
    class_index = {cls: np.where(y_train == cls)[0] for cls in J}
    M_eta = {cls: sum(1 for _ in class_index[cls]) for cls in J}
    L , eta, zeta = {}, {}, {}
    for j in J:
        for t in range(2**D):
            if c[j, t] > 1/2:  # Instead of == 1 to avoid the numerical issues that some c[j, t] = 0.99999...
                L[t] = sum(z_plus_0[s, t] for s in class_index[j]) + N*(1-c[j,t])
                eta[j,t] = sum(z_plus[s,t] for s in class_index[j]) + M_eta[j]*(1-c[j,t])
                zeta[j,t] = sum((1-z_minus[s,t]) for s in range(N)) - N*(1-c[j,t])
            else:
                eta[j,t] = M_eta[j]*c[j,t]
                zeta[j,t] = 0
    return eta, zeta, L


def calculate_L(X_train, y_train, c, D, z_plus_0):
    r"""
    Compute `L` for each class-leaf pair in the unconstrained decision tree model.

    Args:
        X_train (ndarray): training set: X
        y_train (ndarray): training set: y
        c (ndarray): parameter c = \{c_{jt}\}_{j\in [J], t\in {\cal T}_{\ell}}, assign class j to leaf node t
        D (int): Depth of the decision tree.
        z_plus_0 (dict): Values of `z_plus_0[(s, t)]` (\xi_{st}) for sample-leaf pairs.

    Returns:
        dict:
            - `L[t]`: sum of `z_plus_0` over samples of the class assigned to leaf `t`.
    """
    total_class_num = len(Counter(y_train))
    N = X_train.shape[0]
    J = range(1, total_class_num+1)  
    class_index = {cls: np.where(y_train == cls)[0] for cls in J}
    L = {}    
    for j in J:
        for t in range(2**D):
            if c[j, t] > 1/2:  # Instead of == 1 to avoid the numerical issues that some c[j, t] = 0.99999...
                L[t] = sum(z_plus_0[s, t] for s in class_index[j]) + N*(1-c[j,t])
    return L



def split_data(X, y, random_state = 42):
    """
    Split the dataset into training, validation, and test sets, and standardize the features using the training subset.

    The data is first split into training and test sets with stratification. 
    The training set is then further split into a smaller training subset and a validation set, also with stratification. A `StandardScaler` is fitted on
    `X_train_train`, and the same scaling is applied to the validation, full training, and test sets.

    Args:
        X: Feature matrix.
        y: Label vector.
        random_state (int): Random seed used for reproducible splitting. Dedault as 42.

    Returns:
        dict: A dictionary containing the standardized data splits:
            - `X_train`, `y_train`                  Training set used in non-tuning parameter experiments
            - `X_train_train`, `y_train_train`      Training set used in tuning parameter experiments
            - `X_train_val`, `y_train_val`          Evaluation set used in tuning parameter experiments
            - `X_test`, `y_test`                    Test set used in non-tuning parameter experiments
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = random_state)

    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
        X_train, y_train, test_size=1/3, stratify=y_train, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_train = scaler.fit_transform(X_train_train)
    X_train_val = scaler.transform(X_train_val)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    data_splits = {'X_train': X_train, 'y_train': y_train,
        'X_train_train': X_train_train, 'y_train_train': y_train_train,
        'X_train_val': X_train_val, 'y_train_val': y_train_val,
        'X_test': X_test, 'y_test': y_test}

    return data_splits
    

def get_positions_in_complete_binary_tree(children_left, children_right, depth):
    """
    Map each node of a given binary tree to its corresponding position in the array representation of a complete binary tree of depth `depth`.

    This function assumes that the input tree is described by two arrays:
    - `children_left[node]`: the index of the left child of `node`,
    - `children_right[node]`: the index of the right child of `node`.

    If a node does not have a left or right child, the corresponding value is `-1`.

    The function starts from the root node `0` and performs a level-order (breadth-first) traversal. During this traversal, each existing node is placed
    into the position it would occupy in a complete binary tree stored as an array:

    - the root is placed at position `0`,
    - if a node is placed at position `p`, then
    - its left child is placed at position `2*p + 1`,
    - its right child is placed at position `2*p + 2`.

    The output is a list of length `2 ** (depth + 1) - 1`, corresponding to all positions in a complete binary tree of depth `depth`. If some positions do not
    contain any actual node from the input tree, they remain `-1`.

    Example:
        Suppose the input tree is
             0
            / \
           1   2
              /
             3
        Then we can represent it by
            children_left  = [1, -1, 3, -1]
            children_right = [2, -1, -1, -1]

        If `depth = 2`, then a complete binary tree of depth 2 has
        `2 ** (2 + 1) - 1 = 7` positions:
                   position 0
                  /          \
           position 1       position 2
             /    \           /    \
            3      4         5      6

        The nodes are placed as follows:
        - node `0` goes to position `0`,
        - node `1` goes to position `1`,
        - node `2` goes to position `2`,
        - node `3` is the left child of node `2`, so it goes to position `5`.

        Therefore, the returned list is
            [0, 1, 2, -1, -1, 3, -1]

        This means:
        - position 0 contains node 0,
        - position 1 contains node 1,
        - position 2 contains node 2,
        - position 5 contains node 3,
        - all other positions are empty.

    Args:
        children_left (array-like):
            A list or array such that `children_left[node]` is the index of the left child of `node`, or `-1` if the node has no left child.

        children_right (array-like):
            A list or array such that `children_right[node]` is the index of the right child of `node`, or `-1` if the node has no right child.

        depth (int):
            The depth of the target complete binary tree. The returned list has length `2 ** (depth + 1) - 1`.

    Returns:
        list[int]:
            A list representing the positions of the actual tree nodes inside the complete binary tree array. The value at each position is the node
            index of the original tree, and positions with no corresponding node are filled with `-1`.
    """
    total_nodes = 2 ** (depth + 1) - 1
    level_order_positions = [-1] * total_nodes
    queue = deque([(0, 0)])  
    while queue:
        node, position = queue.popleft() 
        level_order_positions[position] = node
        if children_left[node] != -1:  
            queue.append((children_left[node], 2 * position + 1))  
        if children_right[node] != -1:  
            queue.append((children_right[node], 2 * position + 2))  
    return level_order_positions


def train_decisiontree_model(X_train, y_train, max_depth):
    """
    Train a decision tree classifier and convert it into an initialization `(a, b, c)` for the complete binary tree formulation of depth `max_depth`.

    Args:
        X_train (ndarray): training set: X
        y_train (ndarray): training set: y
        max_depth (int): Depth of the decision tree.

    Returns:
            dict: trained (a, b, c)
    """
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
        for f in range(feature_dim):
            a_start[k, f] = 0
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
                    a_start[k, feature_index] = -100
                    b_start[k] = 0
                    parent_features[i].add(chosen_feature)
            # Set a_start[k][arbitrary_feature] = -100 instead of 0, set b_start[k] = i to ensure \mathcal M is a singelton
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
                    a_start[k, feature_index] = -100
                    b_start[k] = 0
                    parent_features[i].add(chosen_feature)
            # Set a_start[k][arbitrary_feature] = -100 instead of 0, set b_start[k] = i to ensure \mathcal M is a singelton
        else:
            feature_index = tree_structure.feature[node_id]
            threshold = tree_structure.threshold[node_id]
            k = i
            a_start[k, feature_index] = -100
            b_start[k] = -100*threshold 
            parent_features[i].add(feature_index)
            # 100 * original a_start, b_start to ensure there exists \phi_plus_0 > 0 
    result = {'a': a_start, 'b': b_start, 'c': c_start}
    return result


def generate_M(X_train, a_start, b_start, D, epsilon, integer_rate, enhanced_size):
    """
    Generate the candidate index set `M_set_index` for each sample-leaf pair.

    Args:
        X_train: Training feature matrix.
        a_start: Initial split-coefficient parameters.
        b_start: Initial split-threshold parameters.
        D: Depth of the decision tree.
        epsilon: Margin parameter.
        integer_rate: Quantile level used in `calculate_delta`.
        enhanced_size: Parameter controlling the allowed number of enlarged multi-piece selections.

    Returns:
        tuple:
            - `M_set_index`: nested dictionary of candidate indices, for example, if s\in \{0,1\}, t\in \{0\}, then 
            M_set_index = {
                            0: {0: [5, 6]},
                            1: {0: [8, 9]}                            
                                            }
            - `multi_piece`: number of sample-leaf pairs with multiple retained candidate indices.
    """
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
            phi_max[s][t] = max([-sum(a_start[k, i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon for k in A_R[t]]+[sum(a_start[k, i]*X_train[s][i] for i in range(p)) - b_start[k] for k in A_L[t]])
            for k in A_R[t]:
                if phi_max[s][t] - (-sum(a_start[k, i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon) == 0:
                    M_set_index[s][t].append(k)
            for k in A_L[t]:
                if phi_max[s][t] - (sum(a_start[k, i]*X_train[s][i] for i in range(p)) - b_start[k]) == 0:
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
                    if multi_piece <= int(math.log2(enhanced_size)):
                        M_set_index[s][t] = rng.choice(M_set_index[s][t], size=int(math.log2(enhanced_size)), replace=False).tolist()
                    if multi_piece > int(math.log2(enhanced_size)):
                        M_set_index[s][t] = [rng.choice(M_set_index[s][t])]
    return M_set_index, multi_piece


def generate_combinations(nested_dict):
    """
    Generate all possible combinations from a nested dictionary of candidate values.

    The input is assumed to have the form `nested_dict[s][t] = [candidates]`. The function enumerates every possible selection of one candidate for each
    `(s, t)` pair and returns the full list of resulting nested dictionaries.

    Args:
        nested_dict: Nested dictionary whose entries are candidate lists. For example, 
        M_set_index = {
                        0: {0: [5, 6]},
                        1: {0: [8, 9]}                            
                                        }

    Returns:
        list: A list of nested dictionaries, where each dictionary corresponds to one complete combination of selected values. For example, 
        [
            {0: {0: 5}, 1: {0: 8}},
            {0: {0: 5}, 1: {0: 9}},
            {0: {0: 6}, 1: {0: 8}},
            {0: {0: 6}, 1: {0: 9}}
        ]
    """
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
    """
    Generate one random piece selection for all sample-leaf pairs.

    For each `(s, t)`, the function computes the set of indices attaining the maximal path value and randomly selects one of them. The result is a single
    nested dictionary representing one piecewise choice.

    Args:
        X_train: Training feature matrix.
        a_start: Initial split-coefficient parameters.
        b_start: Initial split-threshold parameters.
        D: Depth of the decision tree.
        epsilon: Margin parameter.

    Returns:
        dict: Nested dictionary `selected_piece[s][t]` containing one selected index for each sample-leaf pair.
    """
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
            phi_max[s][t] = max([-sum(a_start[k, i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon for k in A_R[t]]+[sum(a_start[k, i]*X_train[s][i] for i in range(p)) - b_start[k] for k in A_L[t]])
            for k in A_R[t]:
                if phi_max[s][t] - (-sum(a_start[k, i]*X_train[s][i] for i in range(p)) + b_start[k] - epsilon) == 0:
                    M_set_index[s][t].append(k)
            for k in A_L[t]:
                if phi_max[s][t] - (sum(a_start[k, i]*X_train[s][i] for i in range(p)) - b_start[k]) == 0:
                    M_set_index[s][t].append(k)
            rng = np.random.default_rng(seed=42)
            selected_piece[s][t] = rng.choice(M_set_index[s][t]) 
    return selected_piece


def evaluate_tree(X, y, a, b, c, depth): 
    """
    Evaluate a decision tree solution on a dataset.

    Given the tree parameters `(a, b, c)` and tree depth `D`, this function computes the leaf assignment of each sample and evaluates the resulting
    classification performance. It returns both overall accuracy and class-specific precision, together with their corresponding counts.

    In addition, the function computes a margin-based accuracy measure, namely the proportion of samples that are correctly classified and satisfy the
    unit-margin condition along the assigned leaf.

    Args:
        X: Feature matrix.
        y: True class labels.
        a: Split-coefficient parameters of the tree.
        b: Split-threshold parameters of the tree.
        c: Leaf-class assignment variables.
        D: Depth of the decision tree.

    Returns:
        dict: A dictionary with two entries:
            - `frac`: fractional performance measures, including overall accuracy (`acc`), margin-based accuracy (`acc_margin`), and class-specific precisions (`precj`);
            - `counts`: corresponding numerator/denominator counts for each metric.

    Notes:
        - A sample is assigned to leaf `t` if it satisfies all branching conditions along the path to `t`.
        - `acc_margin` is stricter than ordinary accuracy, since it requires satisfaction of the margin condition.
        - If no sample is predicted as class `j`, then `precj` is set to `-1`.
    """
    J = list(set(y))
    N = X.shape[0]
    p = X.shape[1]
    A_L, A_R = ancestors(depth)
    heaviside_sets = {}
    leaf_node = {}
    nm_accuracy = 0
    nm_accuracy_margin = 0
    nm_precision = {key: 0 for key in J}
    dm_precision = {key: 0 for key in J}

    frac = {}
    counts = {}

    for s in range(N):
        for t in range(2**depth):
            if round(c[y[s],t],0) == 1:
                nm_accuracy_margin += heaviside_closed(0, min([sum(a[k, i]*X[s][i] for i in range(p)) - b[k] - 1 for k in A_R[t]] + [-sum(a[k, i]*X[s][i] for i in range(p)) + b[k] - 1 for k in A_L[t]]))
    
    for s in range(N):
        heaviside_sets[s]={}
        leaf_node[s] = -1
        for t in range(2**depth):
            heaviside_sets[s][t]=[]
            for k in A_R[t]:
                heaviside_sets[s][t].append(heaviside_closed(0,sum(a[k, i]*X[s][i] for i in range(p))-b[k]))
            for k in A_L[t]:
                heaviside_sets[s][t].append(heaviside_open(0,-sum(a[k, i]*X[s][i] for i in range(p))+b[k]))
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
    """
    Evaluate a tree solution on both the training and test sets, and compute the corresponding constraint gaps and train-test performance differences.

    The function first evaluates the solution on the training and test data using `evaluate_tree`. It then compares the resulting metrics, including
    accuracy, margin-based accuracy, and class-specific precision.

    If precision thresholds `beta_p` are provided, the function also computes the precision constraint gap for each restricted class:
    - `min(0, prec_j - beta_p[j])` if precision is defined,
    - `-2` if precision is undefined (i.e., no sample is predicted as class `j`).

    In addition, it computes the difference between test and training performance. For class-specific precision:
    - `999` means training precision is undefined,
    - `-999` means test precision is undefined.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_test: Test feature matrix.
        y_test: Test labels.
        solution: Dictionary containing the tree solution, with keys `a`, `b`, and `c`.
        D: Depth of the decision tree.
        J: List of class labels.
        beta_p: Dictionary of class-specific precision thresholds, or `None`.
        class_restricted: Classes for which precision constraints are imposed.

    Returns:
        tuple:
            - `train_result`: evaluation result on the training set,
            - `test_result`: evaluation result on the test set,
            - `train_constraint_gap`: training precision constraint gaps,
            - `test_constraint_gap`: test precision constraint gaps,
            - `test_train_gap`: test minus training performance differences.
    """
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
    if dataset == 'blsc':
        df = pd.read_csv('decisiontree/dataset/balance_scale.csv', encoding='utf-8')
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'ceva':
        df = pd.read_csv('decisiontree/dataset/car_evaluation.csv', encoding='utf-8')
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'ctmc':
        df = pd.read_csv('decisiontree/dataset/contraceptive_method_choice.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('contraceptive_method', axis=1)
        y_sampled = df['contraceptive_method']
    if dataset == 'dmtl':
        df = pd.read_csv('decisiontree/dataset/dermatology.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'fish':
        df = pd.read_csv('decisiontree/dataset/fish.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'htds':
        df = pd.read_csv('decisiontree/dataset/heart_disease.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('num', axis=1)
        y_sampled = df['num']
    if dataset == 'nwth':
        df = pd.read_csv('decisiontree/dataset/new_thyroid.csv', encoding='utf-8')
        df.dropna(inplace=True)
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']
    if dataset == 'wine':
        df = pd.read_csv('decisiontree/dataset/wine.csv', encoding='utf-8')
        X_sampled = df.drop('class', axis=1)
        y_sampled = df['class']

    return X_sampled, y_sampled

def write_results(split, method, tau_0, beta, solution, dataset_results_csv, X_test, y_test):
    if method == "Full MIP":
        pass
    if method == "IDSA PIP":
        solution.write_integrated_results(dataset_results_csv=dataset_results_csv, split=split, method=method, tau_0=tau_0, beta=beta, X_test=X_test, y_test=y_test)
        

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

def write_single_integrated_result(results_csv, dataset, depth, split, method, tau_0, beta, objective_value, optimality_gap, time, actual_time, gamma, train_acc, test_acc, train_prec, test_prec):
    
    with open(results_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                'dataset',
                'depth',
                'split',
                'method',
                'tau_0',
                'key_beta_p',
                'beta_p',
                'objective_value',
                'optimality_gap (Full MIP)',
                'time',
                'actual_time (Full MIP)',
                'gamma',
                'train_acc',
                'test_acc',
                'train_prec',
                'test_prec'
            ])
        writer.writerow([
            dataset, depth, split, method, tau_0, 
            next(iter(beta)) if beta is not None else None,
            next(iter(beta.values())) if beta is not None else None,
            objective_value,
            optimality_gap,
            time,
            actual_time,
            gamma,
            train_acc, test_acc, train_prec, test_prec
        ])