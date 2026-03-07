import cplex
from cplex.exceptions import CplexError

import sys
import csv
import io
import math
import getopt
import subprocess
import os

import importlib
import imp

import ntpath

import regtrees as tr
from learn_tree_funcs2 import *

# boolean relaxing all binary and integer variables to be continuous
relaxation = 0
# boolean whether node constraints are relaxed using penalties
relax_node_constraints = 0
# boolean whether to use scores or errors
use_score = 0
# computes all predictions in addition to errors
extra_error_obj = 0
# fix first x layers of tree using CART
fix_layers = 0
# make all non-decision variables boolean
bool_decision = 0
# ===== make default precision constraints =====
use_precision_constraint = 0    # 0 = off, 1 = on
precision_min = 0.75             # required training precision for positive class
min_predicted_positives = 1     # minimum number of predicted positives
positive_label = 0              # value of the positive class label in the data

count_max = -1
maxi_depth = 10

forest_bounds = -1

# priority ordering
PRIO = []
FIXED_NODES = dict()

ITER_VARS = dict()
ITER_LEAF = dict()

# create the ILP variables, add to VARS dictionary
def create_variables(depth):
    global VARS, PRIO
    var_value = 0
    VARS = dict()
    PRIO = []

    var_names = []
    var_types = ""
    var_lb = []
    var_ub = []
    var_obj = []
    
    num_features = get_num_features()
    data_size = get_data_size()
    
    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    for l in range(num_leafs):
        for l2 in range(num_leafs):
            VARS["pair_leafsize_" + str(l) + "_" + str(l2)] = var_value
            var_names.append("#" + str(var_value))
            var_types = var_types + "C"
            var_lb.append(0)
            var_ub.append(data_size)
            var_obj.append(0.01)
            var_value = var_value + 1

    # leaf l predicts type s, boolean
    for s in range(get_num_targets()):
        for l in range(num_leafs):
            VARS["prediction_type_" + str(s) + "_" + str(l)] = var_value
            PRIO.append((var_value,10,0))
            var_names.append("#" + str(var_value))
            if not relaxation:
                var_types = var_types + "B"
            else:
                var_types = var_types + "C"
            var_lb.append(0)
            var_ub.append(1)
            var_obj.append(0)
            var_value = var_value + 1

    for n in range(num_nodes):
        # node n had a boolean test on feature f, boolean
        for f in range(num_features):
            VARS["node_feature_" + str(n) + "_" + str(f)] = var_value
            PRIO.append((var_value,101 - 2*get_num_parents(n,num_nodes),0))
            var_names.append("#" + str(var_value))
            if not relaxation:
                var_types = var_types + "B"
            else:
                var_types = var_types + "C"
            var_lb.append(0)
            var_ub.append(1)
            var_obj.append(0)
            var_value = var_value + 1

    for n in range(num_nodes):
        for i in range(1+int(math.log(max(1,get_max_num_constants())) / math.log(2.))):
        #for i in range(get_max_num_constants()-1):
            VARS["node_constant_bin_" + str(n) + "_" + str(i)] = var_value
            PRIO.append((var_value,100 - 2*get_num_parents(n,num_nodes),0))
            var_names.append("#" + str(var_value))
            if not relaxation:
                var_types = var_types + "B"
            else:
                var_types = var_types + "C"
            var_lb.append(0)
            var_ub.append(1)
            var_obj.append(0)
            var_value = var_value + 1

    if relax_node_constraints:
        for n in range(num_nodes):
            for f in range(num_features):
                VARS["node_feature_error_" + str(n) + "_" + str(f)] = var_value
                var_names.append("#" + str(var_value))
                if not bool_decision:
                    var_types = var_types + "C"
                else:
                    var_types = var_types + "I"
                var_lb.append(0)
                var_ub.append(data_size)
                var_obj.append(1)
                var_value = var_value + 1

        for n in range(num_nodes):
            for i in range(1+int(math.log(get_max_num_constants()) / math.log(2.))):
                VARS["node_constant_error_" + str(n) + "_" + str(i)] = var_value
                var_names.append("#" + str(var_value))
                if not bool_decision:
                    var_types = var_types + "C"
                else:
                    var_types = var_types + "I"
                var_lb.append(0)
                var_ub.append(data_size)
                var_obj.append(1)
                var_value = var_value + 1

    if not use_score:
        for d in range(data_size):
            for l in range(num_leafs):
                VARS["leaf_" + str(d) + "_" + str(l)] = var_value
                var_names.append("#" + str(var_value))
                if not bool_decision:
                    var_types = var_types + "C"
                else:
                    var_types = var_types + "B"
                var_lb.append(0)
                var_ub.append(1)
                var_obj.append(0)
                var_value = var_value + 1
    
        for d in range(data_size):
            for l in range(num_leafs):
                VARS["row_error_" + str(d) + "_" + str(l)] = var_value
                var_names.append("#" + str(var_value))
                if not bool_decision:
                    var_types = var_types + "C"
                else:
                    var_types = var_types + "B"
                var_lb.append(0)
                var_ub.append(1)
                var_obj.append(1)
                var_value = var_value + 1

        if extra_error_obj != 1:
            for s in range(get_num_targets()):
                for l in range(num_leafs):
                    VARS["error_leaf_" + str(s) + "_" + str(l)] = var_value
                    var_names.append("#" + str(var_value))
                    if not bool_decision:
                        var_types = var_types + "C"
                    else:
                        var_types = var_types + "I"
                    var_lb.append(0)
                    #var_ub.append(0)
                    var_ub.append(max([sum([1 for d in range(data_size) if get_target(d) != TARGETS[s2]]) for s2 in range(get_num_targets())]))
                    var_obj.append(1)
                    var_value = var_value + 1
        else:
            for l in range(num_leafs):
                for s1 in range(get_num_targets()):
                    for s2 in range(get_num_targets()):
                        VARS["target_count_" + str(l) + "_" + str(s1) + "_" + str(s2)] = var_value
                        var_names.append("#" + str(var_value))
                        if not bool_decision:
                            var_types = var_types + "C"
                        else:
                            var_types = var_types + "I"
                        var_lb.append(0)
                        var_ub.append(get_data_size())
                        if s1 != s2:
                            var_obj.append(1)
                        else:
                            var_obj.append(0)
                        var_value = var_value + 1

    if use_score:
        for d in range(data_size):
            for l in range(num_leafs):
                VARS["score_" + str(d) + "_" + str(l)] = var_value
                var_names.append("#" + str(var_value))
                if not bool_decision:
                    var_types = var_types + "C"
                else:
                    var_types = var_types + "B"
                var_lb.append(0)
                var_ub.append(1)
                var_obj.append(-1)
                var_value = var_value + 1

        for d in range(data_size):
            VARS["scoreC_" + str(d)] = var_value
            var_names.append("#" + str(var_value))
            if not bool_decision:
                var_types = var_types + "C"
            else:
                var_types = var_types + "B"
            var_lb.append(1)
            var_ub.append(1)
            var_obj.append(1)
            var_value = var_value + 1
    
    # =========== precision constraint ===================
    if use_precision_constraint:
        N_pos = 0
        for d in range(data_size):
            if get_target(d) == positive_label:
                N_pos += 1

        for l in range(num_leafs):
            VARS[f"eta_{l}"] = var_value
            var_names.append("#" + str(var_value))
            var_types += "C"
            var_lb.append(0)
            var_ub.append(N_pos)
            var_obj.append(0)
            var_value = var_value + 1

            VARS[f"zeta_{l}"] = var_value
            var_names.append("#" + str(var_value))
            var_types += "C"
            var_lb.append(0)
            var_ub.append(data_size)
            var_obj.append(0)
            var_value = var_value + 1

        # ---- gamma: slack for global precision to ensure feasibility ----
        # gamma >= 0
        VARS["gamma"] = var_value
        var_names.append("#" + str(var_value))
        var_types += "C"
        var_lb.append(0)
        # a safe upper bound; you can also use data_size if you want tighter
        var_ub.append(data_size)
        # IMPORTANT: penalize gamma so precision is not ignored
        # choose a large weight relative to your error objective
        var_obj.append(data_size * 1e4)
        var_value = var_value + 1
    # ====================================================

    return var_names, var_types, var_lb, var_ub, var_obj

BIN_MAP = dict()

def get_binbin_ranges(min, max):
    if max <= min:
        return []
    if max - min <= 1:
        return [[[], [min, min], [max, max]]]
    
    if min == 0 and max in BIN_MAP:
        return BIN_MAP[max]
    
    #print min, max
    mid = int(float(max - min) / 2.0)
    #if mid - min >= max - mid - 1:
    #    mid = mid - 1
    result = [[[], [min, min + mid], [min + mid + 1, max]]]
    for i in get_binbin_ranges(min, min + mid):
        result.extend([[[0] + i[0], i[1], i[2]]])
    for i in get_binbin_ranges(min + mid + 1, max):
        result.extend([[[1] + i[0], i[1], i[2]]])

    if min == 0:
        BIN_MAP[max] = result

    return result

def create_score_binbin_rows(depth, row_value):
    num_features = get_num_features()
    data_size = get_data_size()
    
    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = ""
    
    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    for d in range(data_size):
        col_names = [VARS["score_" + str(d) + "_" + str(l)] for l in range(num_leafs)]
        col_values = [1 for l in range(num_leafs)]

        row_names.append("#" + str(row_value))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "L"
        row_value = row_value + 1

    for n in range(num_nodes):
        if n in FIXED_NODES:
            f = FIXED_NODES[n][0]
            v = FIXED_NODES[n][1]
            
            lower_leafs = get_right_leafs(n, num_nodes)

            col_names = []
            col_values = []
                
            for d in range(data_size):
                if get_feature_value(d,f) <= v:
                    col_names.extend([VARS["score_" + str(d) + "_" + str(l)] for l in lower_leafs])
                    col_values.extend([1 for l in lower_leafs])
                
                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(0)
                row_senses = row_senses + "L"
                row_value = row_value + 1
        
            continue
    
        for f in range(num_features):
            lower_leafs = get_right_leafs(n, num_nodes)
            
            #for bin in get_binbin_ranges(0, get_num_constants(f)-1):
            #    print bin
            for bin in get_binbin_ranges(0, get_num_constants(f)-1):
                col_names = []
                col_values = []
                
                #print "testing: [",get_constant_val(f, bin[1][0]),",",get_constant_val(f, bin[2][0]),"]"
                
                max_val = 0
                for d in range(data_size):
                    if get_feature_value(d,f) >= get_constant_val(f, bin[1][0]) and get_feature_value(d,f) <= get_constant_val(f, bin[2][0]):
                        col_names.extend([VARS["score_" + str(d) + "_" + str(l)] for l in lower_leafs])
                        col_values.extend([1 for l in lower_leafs])
                        max_val = max_val + 1
            
                col_names.extend([VARS["node_feature_"  + str(n) + "_" + str(f)]])
                col_values.extend([max_val])
                
                num_bin = 0
                for i in range(len(bin[0])):
                    #if bin[0][i] is 0:
                    #    col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                    #    col_values.extend([max_val])
                    #    num_bin = num_bin + 1
                    if bin[0][i] is 1:
                        col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                        col_values.extend([-max_val])
                
                col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]])
                col_values.extend([-max_val])

                if relax_node_constraints:
                    col_names.extend([VARS["node_constant_error_" + str(n) + "_" + str(len(bin[0]))]])
                    col_values.extend([-1])
                
                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(max_val + max_val * num_bin)
                row_senses = row_senses + "L"
                row_value = row_value + 1


    for n in range(num_nodes):
        if n in FIXED_NODES:
            f = FIXED_NODES[n][0]
            v = FIXED_NODES[n][1]
            
            lower_leafs = get_left_leafs(n, num_nodes)

            col_names = []
            col_values = []
                
            for d in range(data_size):
                if get_feature_value(d,f) >= v:
                    col_names.extend([VARS["score_" + str(d) + "_" + str(l)] for l in lower_leafs])
                    col_values.extend([1 for l in lower_leafs])
                
                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(0)
                row_senses = row_senses + "L"
                row_value = row_value + 1
        
            continue

        for f in range(num_features):
            lower_leafs = get_left_leafs(n, num_nodes)
            
            for bin in get_binbin_ranges(0, get_num_constants(f)-1):
                col_names = []
                col_values = []

                #print "testing: [",get_constant_val(f, bin[1][1]),",",get_constant_val(f, bin[2][1]),"]"

                max_val = 0
                for d in range(data_size):
                    if get_feature_value(d,f) >= get_constant_val(f, bin[1][1]) and get_feature_value(d,f) <= get_constant_val(f, bin[2][1]):
                        col_names.extend([VARS["score_" + str(d) + "_" + str(l)] for l in lower_leafs])
                        col_values.extend([1 for l in lower_leafs])
                        max_val = max_val + 1
            
                col_names.extend([VARS["node_feature_"  + str(n) + "_" + str(f)]])
                col_values.extend([max_val])
                
                num_bin = 0
                for i in range(len(bin[0])):
                    if bin[0][i] is 0:
                        col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                        col_values.extend([max_val])
                        num_bin = num_bin + 1
                    #if bin[0][i] is 1:
                    #    col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                    #    col_values.extend([-max_val])
                
                col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]])
                col_values.extend([max_val])

                if relax_node_constraints:
                    col_names.extend([VARS["node_constant_error_" + str(n) + "_" + str(len(bin[0]))]])
                    col_values.extend([-1])
                
                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(2*max_val + max_val * num_bin)
                row_senses = row_senses + "L"
                row_value = row_value + 1

    for n in range(num_nodes):
        if n in FIXED_NODES:
            continue
        
        for f in range(num_features):
            lower_leafs = get_right_leafs(n, num_nodes)
            col_names = []
            col_values = []
            
            max_val = 0
            for d in range(data_size):
                if get_feature_value(d,f) <= get_min_constant_val(f):
                    col_names.extend([VARS["score_" + str(d) + "_" + str(l)] for l in lower_leafs])
                    col_values.extend([1 for l in lower_leafs])
                    max_val = max_val + 1

            col_names.extend([VARS["node_feature_"  + str(n) + "_" + str(f)]])
            col_values.extend([max_val])
            
            if relax_node_constraints:
                col_names.extend([VARS["node_feature_error_"  + str(n) + "_" + str(f)]])
                col_values.extend([-1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(max_val)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    for n in range(num_nodes):
        if n in FIXED_NODES:
            continue
        
        for f in range(num_features):
            lower_leafs = get_left_leafs(n, num_nodes)

            col_names = []
            col_values = []
            
            max_val = 0
            for d in range(data_size):
                if get_feature_value(d,f) >= get_max_constant_val(f):
                    col_names.extend([VARS["score_" + str(d) + "_" + str(l)] for l in lower_leafs])
                    col_values.extend([1 for l in lower_leafs])
                    max_val = max_val + 1
            
            col_names.extend([VARS["node_feature_"  + str(n) + "_" + str(f)]])
            col_values.extend([max_val])

            if relax_node_constraints:
                col_names.extend([VARS["node_feature_error_"  + str(n) + "_" + str(f)]])
                col_values.extend([-1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(max_val)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    return row_names, row_values, row_right_sides, row_senses, row_value

def create_error_binbin_rows(depth, row_value):
    num_features = get_num_features()
    data_size = get_data_size()
    
    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = ""
    
    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    for d in range(data_size):
        col_names = [VARS["leaf_" + str(d) + "_" + str(l)] for l in range(num_leafs)]
        col_values = [1 for l in range(num_leafs)]

        row_names.append("#" + str(row_value))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "E"
        row_value = row_value + 1
    
    for n in range(num_nodes):
        if n in FIXED_NODES:
            f = FIXED_NODES[n][0]
            v = FIXED_NODES[n][1]
            
            lower_leafs = get_right_leafs(n, num_nodes)

            col_names = []
            col_values = []
                
            for d in range(data_size):
                if get_feature_value(d,f) <= v:
                    col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in lower_leafs])
                    col_values.extend([1 for l in lower_leafs])
                
            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1
        
            continue
        
        for f in range(num_features):
            lower_leafs = get_right_leafs(n, num_nodes)
            
            #for bin in get_binbin_ranges(0, get_num_constants(f)-1):
            #    print bin
            for bin in get_binbin_ranges(0, get_num_constants(f)-1):
                col_names = []
                col_values = []
                
                #print "testing: [",get_constant_val(f, bin[1][0]),",",get_constant_val(f, bin[2][0]),"]"
                
                max_val = 0
                for d in range(data_size):
                    if get_feature_value(d,f) >= get_constant_val(f, bin[1][0]) and get_feature_value(d,f) <= get_constant_val(f, bin[2][0]):
                        col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in lower_leafs])
                        col_values.extend([1 for l in lower_leafs])
                        max_val = max_val + 1
            
                #for v in range(bin[1][0], bin[2][0]):
                    #col_names.extend([VARS["leafnum_" + str(l) + "_" + str(f) + "_" + str(v)] for l in lower_leafs])
                    #col_values.extend([1 for l in lower_leafs])
            
                col_names.extend([VARS["node_feature_"  + str(n) + "_" + str(f)]])
                col_values.extend([max_val])
                
                num_bin = 0
                for i in range(len(bin[0])):
                    #if bin[0][i] is 0:
                    #    col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                    #    col_values.extend([max_val])
                    #    num_bin = num_bin + 1
                    if bin[0][i] is 1:
                        col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                        col_values.extend([-max_val])
                
                col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]])
                col_values.extend([-max_val])

                if relax_node_constraints:
                    col_names.extend([VARS["node_constant_error_" + str(n) + "_" + str(len(bin[0]))]])
                    col_values.extend([-1])
                
                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(max_val + max_val * num_bin)
                row_senses = row_senses + "L"
                row_value = row_value + 1


    for n in range(num_nodes):
        if n in FIXED_NODES:
            f = FIXED_NODES[n][0]
            v = FIXED_NODES[n][1]
            
            lower_leafs = get_left_leafs(n, num_nodes)

            col_names = []
            col_values = []
                
            for d in range(data_size):
                if get_feature_value(d,f) >= v:
                    col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in lower_leafs])
                    col_values.extend([1 for l in lower_leafs])
                
            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1
        
            continue

        for f in range(num_features):
            lower_leafs = get_left_leafs(n, num_nodes)
            
            for bin in get_binbin_ranges(0, get_num_constants(f)-1):
                col_names = []
                col_values = []

                #print "testing: [",get_constant_val(f, bin[1][1]),",",get_constant_val(f, bin[2][1]),"]"

                max_val = 0
                for d in range(data_size):
                    if get_feature_value(d,f) >= get_constant_val(f, bin[1][1]) and get_feature_value(d,f) <= get_constant_val(f, bin[2][1]):
                        col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in lower_leafs])
                        col_values.extend([1 for l in lower_leafs])
                        max_val = max_val + 1
            
                #for v in range(bin[1][1], bin[2][1]):
                #    col_names.extend([VARS["leafnum_" + str(l) + "_" + str(f) + "_" + str(v)] for l in lower_leafs])
                #    col_values.extend([1 for l in lower_leafs])

                col_names.extend([VARS["node_feature_"  + str(n) + "_" + str(f)]])
                col_values.extend([max_val])
                
                num_bin = 0
                for i in range(len(bin[0])):
                    if bin[0][i] is 0:
                        col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                        col_values.extend([max_val])
                        num_bin = num_bin + 1
                    #if bin[0][i] is 1:
                    #    col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                    #    col_values.extend([-max_val])
                
                col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]])
                col_values.extend([max_val])

                if relax_node_constraints:
                    col_names.extend([VARS["node_constant_error_" + str(n) + "_" + str(len(bin[0]))]])
                    col_values.extend([-1])
                
                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(2*max_val + max_val * num_bin)
                row_senses = row_senses + "L"
                row_value = row_value + 1

    for n in range(num_nodes):
        if n in FIXED_NODES:
            continue

        for f in range(num_features):
            left_leafs = get_left_leafs(n, num_nodes)
            right_leafs = get_right_leafs(n, num_nodes)

            col_names = []
            col_values = []
            
            max_val = 0 
            for d in range(data_size):
                if get_feature_value(d,f) >= get_max_constant_val(f):
                    col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in left_leafs])
                    col_values.extend([1 for l in left_leafs])
                    max_val = max_val + 1
                if get_feature_value(d,f) <= get_min_constant_val(f):
                    col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in right_leafs])
                    col_values.extend([1 for l in right_leafs])
                    max_val = max_val + 1
            
            #col_names.extend([VARS["leafmax_" + str(l) + "_" + str(f)] for l in lower_leafs])
            #col_values.extend([1 for l in lower_leafs])

            col_names.extend([VARS["node_feature_"  + str(n) + "_" + str(f)]])
            col_values.extend([max_val])

            if relax_node_constraints:
                col_names.extend([VARS["node_feature_error_"  + str(n) + "_" + str(f)]])
                col_values.extend([-1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(max_val)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    return row_names, row_values, row_right_sides, row_senses, row_value

def create_error_binbin_rows2(depth, row_value):
    num_features = get_num_features()
    data_size = get_data_size()
    
    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = ""
    
    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    for d in range(data_size):
        col_names = [VARS["leaf_" + str(d) + "_" + str(l)] for l in range(num_leafs)]
        col_values = [1 for l in range(num_leafs)]

        row_names.append("#" + str(row_value))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "E"
        row_value = row_value + 1
    
    for n in range(num_nodes):
        if n in FIXED_NODES:
            f = FIXED_NODES[n][0]
            v = FIXED_NODES[n][1]
            
            lower_leafs = get_right_leafs(n, num_nodes)

            col_names = []
            col_values = []
                
            for d in range(data_size):
                if get_feature_value(d,f) <= v:
                    col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in lower_leafs])
                    col_values.extend([1 for l in lower_leafs])
                
            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1
        
            continue
        
        for f in range(num_features):
            lower_leafs = get_right_leafs(n, num_nodes)
            
            #for bin in get_binbin_ranges(0, get_num_constants(f)-1):
            #    print bin
            for bin in get_binbin_ranges(0, get_num_constants(f)-1):
                col_names = []
                col_values = []
                
                #print "testing: [",get_constant_val(f, bin[1][0]),",",get_constant_val(f, bin[2][0]),"]"
                
                max_val = 0
                for d in range(data_size):
                    if get_feature_value(d,f) >= get_constant_val(f, bin[1][0]) and get_feature_value(d,f) <= get_constant_val(f, bin[2][0]):
                        col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in lower_leafs])
                        col_values.extend([1 for l in lower_leafs])
                        max_val = max_val + 1
            
                #for v in range(bin[1][0], bin[2][0]):
                    #col_names.extend([VARS["leafnum_" + str(l) + "_" + str(f) + "_" + str(v)] for l in lower_leafs])
                    #col_values.extend([1 for l in lower_leafs])
            
                col_names.extend([VARS["node_feature_"  + str(n) + "_" + str(f)]])
                col_values.extend([max_val])
                
                num_bin = 0
                for i in range(len(bin[0])):
                    #if bin[0][i] is 0:
                    #    col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                    #    col_values.extend([max_val])
                    #    num_bin = num_bin + 1
                    if bin[0][i] is 1:
                        col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                        col_values.extend([-max_val])
                
                col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]])
                col_values.extend([-max_val])

                if relax_node_constraints:
                    col_names.extend([VARS["node_constant_error_" + str(n) + "_" + str(len(bin[0]))]])
                    col_values.extend([-1])
                
                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(max_val + max_val * num_bin)
                row_senses = row_senses + "L"
                row_value = row_value + 1


    for n in range(num_nodes):
        if n in FIXED_NODES:
            f = FIXED_NODES[n][0]
            v = FIXED_NODES[n][1]
            
            lower_leafs = get_left_leafs(n, num_nodes)

            col_names = []
            col_values = []
                
            for d in range(data_size):
                if get_feature_value(d,f) >= v:
                    col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in lower_leafs])
                    col_values.extend([1 for l in lower_leafs])
                
            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1
        
            continue

        for f in range(num_features):
            lower_leafs = get_left_leafs(n, num_nodes)
            
            for bin in get_binbin_ranges(0, get_num_constants(f)-1):
                col_names = []
                col_values = []

                #print "testing: [",get_constant_val(f, bin[1][1]),",",get_constant_val(f, bin[2][1]),"]"

                max_val = 0
                for d in range(data_size):
                    if get_feature_value(d,f) >= get_constant_val(f, bin[1][1]) and get_feature_value(d,f) <= get_constant_val(f, bin[2][1]):
                        col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in lower_leafs])
                        col_values.extend([1 for l in lower_leafs])
                        max_val = max_val + 1
            
                #for v in range(bin[1][1], bin[2][1]):
                #    col_names.extend([VARS["leafnum_" + str(l) + "_" + str(f) + "_" + str(v)] for l in lower_leafs])
                #    col_values.extend([1 for l in lower_leafs])

                col_names.extend([VARS["node_feature_"  + str(n) + "_" + str(f)]])
                col_values.extend([max_val])
                
                num_bin = 0
                for i in range(len(bin[0])):
                    if bin[0][i] is 0:
                        col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                        col_values.extend([max_val])
                        num_bin = num_bin + 1
                    #if bin[0][i] is 1:
                    #    col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                    #    col_values.extend([-max_val])
                
                col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]])
                col_values.extend([max_val])

                if relax_node_constraints:
                    col_names.extend([VARS["node_constant_error_" + str(n) + "_" + str(len(bin[0]))]])
                    col_values.extend([-1])
                
                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(2*max_val + max_val * num_bin)
                row_senses = row_senses + "L"
                row_value = row_value + 1

    for n in range(num_nodes):
        if n in FIXED_NODES:
            continue
        
        for f in range(num_features):
            lower_leafs = get_right_leafs(n, num_nodes)
            col_names = []
            col_values = []
            
            max_val = 0
            for d in range(data_size):
                if get_feature_value(d,f) <= get_min_constant_val(f):
                    col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in lower_leafs])
                    col_values.extend([1 for l in lower_leafs])
                    max_val = max_val + 1

            #col_names.extend([VARS["leafmin_" + str(l) + "_" + str(f)] for l in lower_leafs])
            #col_values.extend([1 for l in lower_leafs])

            col_names.extend([VARS["node_feature_"  + str(n) + "_" + str(f)]])
            col_values.extend([max_val])
            
            if relax_node_constraints:
                col_names.extend([VARS["node_feature_error_"  + str(n) + "_" + str(f)]])
                col_values.extend([-1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(max_val)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    for n in range(num_nodes):
        if n in FIXED_NODES:
            continue

        for f in range(num_features):
            lower_leafs = get_left_leafs(n, num_nodes)

            col_names = []
            col_values = []
            
            max_val = 0 
            for d in range(data_size):
                if get_feature_value(d,f) >= get_max_constant_val(f):
                    col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)] for l in lower_leafs])
                    col_values.extend([1 for l in lower_leafs])
                    max_val = max_val + 1
            
            #col_names.extend([VARS["leafmax_" + str(l) + "_" + str(f)] for l in lower_leafs])
            #col_values.extend([1 for l in lower_leafs])

            col_names.extend([VARS["node_feature_"  + str(n) + "_" + str(f)]])
            col_values.extend([max_val])

            if relax_node_constraints:
                col_names.extend([VARS["node_feature_error_"  + str(n) + "_" + str(f)]])
                col_values.extend([-1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(max_val)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    return row_names, row_values, row_right_sides, row_senses, row_value

    for d in range(data_size):
        for l in range(num_leafs):
            col_names = [VARS["leaf_" + str(d) + "_" + str(l)]]
            col_values = [1]

            col_names.extend([VARS["leaf_div_" + str(l)]])
            col_values.extend([-1])

            col_names.extend([VARS["row_div_" + str(d) + "_" + str(l)]])
            col_values.extend([1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(1)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    for d in range(data_size):
        for l in range(num_leafs):
            col_names = [VARS["leaf_" + str(d) + "_" + str(l)]]
            col_values = [1]

            col_names.extend([VARS["leaf_div_" + str(l)]])
            col_values.extend([1])

            col_names.extend([VARS["row_div_" + str(d) + "_" + str(l)]])
            col_values.extend([-1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(1)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    for d in range(data_size):
        for l in range(num_leafs):
            col_names = [VARS["leaf_" + str(d) + "_" + str(l)]]
            col_values = [-1]

            col_names.extend([VARS["row_div_" + str(d) + "_" + str(l)]])
            col_values.extend([1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    for l in range(num_leafs):
        col_names = []
        col_values = []

        col_names.extend([VARS["leaf_prob_" + str(l)]])
        col_values.extend([1])

        col_names.extend([VARS["row_div_" + str(d) + "_" + str(l)] for d in range(data_size)])
        col_values.extend([-1 for d in range(data_size)])

        row_names.append("#" + str(row_value))
        row_values.append([col_names,col_values])
        row_right_sides.append(0)
        row_senses = row_senses + "E"
        row_value = row_value + 1

    for d in range(data_size):
        for l in range(num_leafs):
            col_names = [VARS["leaf_" + str(d) + "_" + str(l)]]
            col_values = [1]

            col_names.extend([VARS["leaf_prob_" + str(l)]])
            col_values.extend([-1])

            col_names.extend([VARS["row_prob_" + str(d) + "_" + str(l)]])
            col_values.extend([1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(1)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    for d in range(data_size):
        for l in range(num_leafs):
            col_names = [VARS["leaf_" + str(d) + "_" + str(l)]]
            col_values = [1]

            col_names.extend([VARS["leaf_prob_" + str(l)]])
            col_values.extend([1])

            col_names.extend([VARS["row_prob_" + str(d) + "_" + str(l)]])
            col_values.extend([-1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(1)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    for d in range(data_size):
        for l in range(num_leafs):
            col_names = [VARS["leaf_" + str(d) + "_" + str(l)]]
            col_values = [-1]

            col_names.extend([VARS["row_prob_" + str(d) + "_" + str(l)]])
            col_values.extend([1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    for l in range(num_leafs):
        col_names = []
        col_values = []

        col_names.extend([VARS["row_div_" + str(d) + "_" + str(l)] for d in range(data_size)])
        col_values.extend([1 for d in range(data_size)])

        col_names.extend([VARS["leaf_div_" + str(l)]])
        col_values.extend([10])

        row_names.append("#" + str(row_value))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "E"
        row_value = row_value + 1

    for l in range(num_leafs):
        for f in range(num_features):
            for v in range(get_num_constants(f)-1):
                col_names = []
                col_values = []
                for d in range(data_size):
                    if get_feature_value(d,f) >= get_constant_val(f,v) and get_feature_value(d,f) < get_constant_val(f,v+1):
                        col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)]])
                        col_values.extend([1])
                
                col_names.extend([VARS["leafnum_" + str(l) + "_" + str(f) + "_" + str(v)]])
                col_values.extend([-1])

                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(0)
                row_senses = row_senses + "L"
                row_value = row_value + 1

            col_names = []
            col_values = []
            for d in range(data_size):
                if get_feature_value(d,f) <= get_min_constant_val(f):
                    col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)]])
                    col_values.extend([1])
                
            col_names.extend([VARS["leafmin_" + str(l) + "_" + str(f)]])
            col_values.extend([-1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1

            col_names = []
            col_values = []
            for d in range(data_size):
                if get_feature_value(d,f) >= get_max_constant_val(f):
                    col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)]])
                    col_values.extend([1])
                
            col_names.extend([VARS["leafmax_" + str(l) + "_" + str(f)]])
            col_values.extend([-1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1

def create_row_score_rows(depth, row_value):
    num_features = get_num_features()
    data_size = get_data_size()
    
    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = ""
    
    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    num_features = get_num_features()
    data_size = get_data_size()
    
    for s in range(get_num_targets()):
        for l in range(num_leafs):
            col_names = []
            col_values = []
            
            max_val = 0
            for d in range(data_size):
                if TARGETS[s] == get_target(d):
                    col_names.extend([VARS["score_" + str(d) + "_" + str(l)]])
                    col_values.extend([1])
                    max_val = max_val + 1
            
            col_names.extend([VARS["prediction_type_" + str(s) + "_" + str(l)]])
            col_values.extend([-max_val])

            #row_names.append("error_" + str(d) + "_" + str(l))
            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    return row_names, row_values, row_right_sides, row_senses, row_value
    
    for d in range(data_size):
        for l in range(num_leafs):
            col_names = []
            col_values = []
            
            for s in range(get_num_targets()):
                if TARGETS[s] == get_target(d):
                    col_names.extend([VARS["prediction_type_" + str(s) + "_" + str(l)]])
                    col_values.extend([-1])

            col_names.extend([VARS["score_" + str(d) + "_" + str(l)]])
            col_values.extend([1])

            #row_names.append("error_" + str(d) + "_" + str(l))
            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1
    
    return row_names, row_values, row_right_sides, row_senses, row_value


def create_leaf_error_rows(depth, row_value):
    num_features = get_num_features()
    data_size = get_data_size()

    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = ""
    
    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    for s in range(get_num_targets()):
        for l in range(num_leafs):
            col_names = [VARS["error_leaf_" + str(s) + "_" + str(l)]]
            col_values = [-1]
            
            max_val = 0
            for d in range(data_size):
                if get_target(d) == TARGETS[s]:
                    col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)]])
                    #col_names.extend([VARS["row_prob_" + str(d) + "_" + str(l)]])
                    col_values.extend([1])
                    max_val = max_val + 1
            
            col_names.extend([VARS["prediction_type_" + str(s) + "_" + str(l)]])
            col_values.extend([-max_val])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    return row_names, row_values, row_right_sides, row_senses, row_value
    
    for d in range(data_size):
        for l in range(num_leafs):
            col_names = [VARS["leaf_" + str(d) + "_" + str(l)]]
            col_values = [1]
            
            for s in range(get_num_targets()):
                if TARGETS[s] != get_target(d):
                    col_names.extend([VARS["prediction_type_" + str(s) + "_" + str(l)]])
                    col_values.extend([1])

            col_names.extend([VARS["row_error_" + str(d) + "_" + str(l)]])
            col_values.extend([-1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(1)
            row_senses = row_senses + "L"
            row_value = row_value + 1
    
    return row_names, row_values, row_right_sides, row_senses, row_value
    


    for l in range(num_leafs):
        for l2 in range(num_leafs):
            if l == l2:
                continue
            
            col_names = [VARS["pair_leafsize_" + str(l) + "_" + str(l2)]]
            col_values = [-1]
            
            for d in range(data_size):
                col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)]])
                col_values.extend([1])

                col_names.extend([VARS["leaf_" + str(d) + "_" + str(l2)]])
                col_values.extend([-1])

            row_names.append("#" + str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(0)
            row_senses = row_senses + "L"
            row_value = row_value + 1

    return row_names, row_values, row_right_sides, row_senses, row_value

def fix_start_solutions(depth, layers, row_value):
    global inputsym
    global FIXED_NODES

    num_features = get_num_features()
    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = ""

    num_features = get_num_features()
    data_size = get_data_size()
    
    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    values = tr.dt.tree_.value
    
    for n in range(num_nodes):
        if get_depth(n, num_nodes) > layers: continue
        
        #print "fixing node: ", n
    
        feat = sget_feature(tr.dt, convert_node(tr.dt, n, num_nodes))
        
        if feat < 0:
            feat = 0

        for f in range(num_features):
            if f == feat:
                col_names = [VARS["node_feature_" + str(n) + "_" + str(f)]]
                col_values = [1]

                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(1)
                row_senses = row_senses + "E"
                row_value = row_value + 1
            else:
                col_names = [VARS["node_feature_" + str(n) + "_" + str(f)]]
                col_values = [1]

                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(0)
                row_senses = row_senses + "E"
                row_value = row_value + 1

        val = sget_node_constant(tr.dt, convert_node(tr.dt, n, num_nodes))
        
        for i in range(get_num_constants(feat)-2):
            if val >= get_constant_val(feat,i) and val < get_constant_val(feat,i+1):
                val = get_constant_val(feat,i)
        
        FIXED_NODES[n] = [feat, val]
    
        bins = get_binbin_ranges(0, get_num_constants(feat)-1)
        for index in range(len(bins)):
            bin = bins[index]
            #print bin, val, get_constant_val(feat, bin[1][0]), get_constant_val(feat, bin[2][0])
            if bin[1][0] == bin[1][1] and val == get_constant_val(feat, bin[1][0]):
                for i in range(len(bin[0])):
                    col_names = [VARS["node_constant_bin_" + str(n) + "_" + str(i)]]
                    col_values = [1]
                    
                    row_names.append("#" + str(row_value))
                    row_values.append([col_names,col_values])
                    row_right_sides.append(1 - bin[0][i])
                    row_senses = row_senses + "E"
                    row_value = row_value + 1

                col_names = [VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]]
                col_values = [1]
                    
                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(1)
                row_senses = row_senses + "E"
                row_value = row_value + 1
                    
            if bin[2][0] == bin[2][1] and val == get_constant_val(feat, bin[2][0]):
                for i in range(len(bin[0])):
                    col_names = [VARS["node_constant_bin_" + str(n) + "_" + str(i)]]
                    col_values = [1]

                    row_names.append("#" + str(row_value))
                    row_values.append([col_names,col_values])
                    row_right_sides.append(1 - bin[0][i])
                    row_senses = row_senses + "E"
                    row_value = row_value + 1

                col_names = [VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]]
                col_values = [0]

                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(0)
                row_senses = row_senses + "E"
                row_value = row_value + 1
                    
        if get_min_constant_val(feat) > val:
            for i in range(1+int(math.log(get_max_num_constants()) / math.log(2.))):
                col_names = [VARS["node_constant_bin_" + str(n) + "_" + str(i)]]
                col_values = [1]
                
                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(1)
                row_senses = row_senses + "E"
                row_value = row_value + 1
                    
        if get_max_constant_val(feat) < val:
            for i in range(1+int(math.log(get_max_num_constants()) / math.log(2.))):
                col_names = [VARS["node_constant_bin_" + str(n) + "_" + str(i)]]
                col_values = [0]

                row_names.append("#" + str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(0)
                row_senses = row_senses + "E"
                row_value = row_value + 1

    return row_names, row_values, row_right_sides, row_senses, row_value

def extra_objective_rows(depth, row_value):
    
    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = ""

    num_features = get_num_features()
    data_size = get_data_size()
    
    num_leafs = 2**depth
    num_nodes = num_leafs-1

    for l in range(num_leafs):
        for s1 in range(get_num_targets()):
            for s2 in range(get_num_targets()):
                #if s1 != s2:
                    col_names = [VARS["target_count_" + str(l) + "_" + str(s1) + "_" + str(s2)]]
                    col_values = [-1]
                    
                    max_val = 0
                    for d in range(data_size):
                        if get_target(d) == TARGETS[s1]:
                            col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)]])
                            col_values.extend([1])
                            max_val = max_val + 1
                
                    col_names.extend([VARS["prediction_type_" + str(s2) + "_" + str(l)]])
                    col_values.extend([max_val])
    
                    row_names.append("#" + str(row_value))
                    row_values.append([col_names,col_values])
                    row_right_sides.append(max_val)
                    row_senses = row_senses + "L"
                    row_value = row_value + 1

    for l in range(num_leafs):
        for s1 in range(get_num_targets()):
            for s2 in range(get_num_targets()):

                    col_names = [VARS["target_count_" + str(l) + "_" + str(s1) + "_" + str(s2)]]
                    col_values = [1]
                    
                    max_val = 0
                    for d in range(data_size):
                        if get_target(d) == TARGETS[s1]:
                            col_names.extend([VARS["leaf_" + str(d) + "_" + str(l)]])
                            col_values.extend([-1])
                            max_val = max_val + 1
                
                    col_names.extend([VARS["prediction_type_" + str(s2) + "_" + str(l)]])
                    col_values.extend([max_val])
    
                    row_names.append("#" + str(row_value))
                    row_values.append([col_names,col_values])
                    row_right_sides.append(max_val)
                    row_senses = row_senses + "L"
                    row_value = row_value + 1

    for l in range(num_leafs):
        for s1 in range(get_num_targets()):
            for s2 in range(get_num_targets()):

                    col_names = [VARS["target_count_" + str(l) + "_" + str(s1) + "_" + str(s2)]]
                    col_values = [1]
                    
                    max_val = 0
                    for d in range(data_size):
                        if get_target(d) == TARGETS[s1]:
                           max_val = max_val + 1
                
                    col_names.extend([VARS["prediction_type_" + str(s2) + "_" + str(l)]])
                    col_values.extend([-max_val])
    
                    row_names.append("#" + str(row_value))
                    row_values.append([col_names,col_values])
                    row_right_sides.append(0)
                    row_senses = row_senses + "L"
                    row_value = row_value + 1

    return row_names, row_values, row_right_sides, row_senses, row_value


def create_precision_rows(depth, row_value):
    """
    Add global precision constraints for the positive class.

    This implements the Big-M linearization consistent with the screenshot:

        precision >= precision_min

    using leaf assignment variables:
        leaf_{d,l}  (only if not use_score)

    and leaf prediction variables:
        prediction_type_{s,l}

    New continuous vars required (create in create_variables):
        eta_{l}  ~ true positives contributed by leaf l if it predicts positive
        zeta_{l} ~ predicted positives contributed by leaf l if it predicts positive

    Constraints:
        eta_l <= M * p_l
        eta_l <= sum_{d: y_d=pos} leaf_{d,l} + M * (1 - p_l)

        zeta_l >= -M' * p_l
        zeta_l >= sum_{d} leaf_{d,l} - M' * (1 - p_l)

        sum_l eta_l - precision_min * sum_l zeta_l >= 0
        sum_l zeta_l >= min_predicted_positives
    """

    global VARS
    global use_precision_constraint, precision_min
    global min_predicted_positives, positive_label

    # respect the global flag
    if not use_precision_constraint:
        return [], [], [], "", row_value

    # This precision linearization relies on leaf assignment variables
    if use_score:
        # If you want precision with score-mode, you'd need a different formulation.
        return [], [], [], "", row_value

    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = ""

    data_size = get_data_size()
    num_leafs = 2 ** depth

    # locate the target index for the positive label
    pos_s = None
    for s in range(get_num_targets()):
        if TARGETS[s] == positive_label:
            pos_s = s
            break
    if pos_s is None:
        # positive_label not found in TARGETS
        return [], [], [], "", row_value

    # Big-M values
    N_pos = 0
    pos_rows = []
    for d in range(data_size):
        if get_target(d) == positive_label:
            N_pos += 1
            pos_rows.append(d)

    M = N_pos
    M_prime = data_size  # corresponds to R in the screenshot

    # Edge case: no positive samples in training data
    # In that case, the precision constraint is ill-posed.
    if M == 0:
        return [], [], [], "", row_value

    # Per-leaf linearization constraints
    for l in range(num_leafs):
        p_var = VARS["prediction_type_" + str(pos_s) + "_" + str(l)]
        eta_var = VARS["eta_" + str(l)]
        zeta_var = VARS["zeta_" + str(l)]

        # (1) eta_l <= M * p_l
        # eta_l - M p_l <= 0
        col_names = [eta_var, p_var]
        col_values = [1, -M]
        row_names.append("#" + str(row_value))
        row_values.append([col_names, col_values])
        row_right_sides.append(0)
        row_senses += "L"
        row_value += 1

        # (2) eta_l <= sum_pos leaf_{d,l} + M(1 - p_l)
        # eta_l - sum_pos leaf_{d,l} + M p_l <= M
        col_names = [eta_var, p_var]
        col_values = [1, M]
        for d in pos_rows:
            col_names.append(VARS["leaf_" + str(d) + "_" + str(l)])
            col_values.append(-1)

        row_names.append("#" + str(row_value))
        row_values.append([col_names, col_values])
        row_right_sides.append(M)
        row_senses += "L"
        row_value += 1

        # (3) zeta_l >= -M' * p_l
        # zeta_l + M' p_l >= 0
        col_names = [zeta_var, p_var]
        col_values = [1, M_prime]
        row_names.append("#" + str(row_value))
        row_values.append([col_names, col_values])
        row_right_sides.append(0)
        row_senses += "G"
        row_value += 1

        # (4) zeta_l >= sum_all leaf_{d,l} - M'(1 - p_l)
        # zeta_l - sum_all leaf_{d,l} - M' p_l >= -M'
        col_names = [zeta_var, p_var]
        col_values = [1, -M_prime]
        for d in range(data_size):
            col_names.append(VARS["leaf_" + str(d) + "_" + str(l)])
            col_values.append(-1)

        row_names.append("#" + str(row_value))
        row_values.append([col_names, col_values])
        row_right_sides.append(-M_prime)
        row_senses += "G"
        row_value += 1

    # (5) Global precision:
    # sum_l eta_l - precision_min * sum_l zeta_l >= 0
    col_names = []
    col_values = []
    for l in range(num_leafs):
        col_names.append(VARS["eta_" + str(l)])
        col_values.append(1)
    for l in range(num_leafs):
        col_names.append(VARS["zeta_" + str(l)])
        col_values.append(-precision_min)

        # add gamma slack to guarantee feasibility
    col_names.append(VARS["gamma"])
    col_values.append(1)

    row_names.append("#" + str(row_value))
    row_values.append([col_names, col_values])
    row_right_sides.append(0)
    row_senses += "G"
    row_value += 1

    # (6) Minimum predicted positives:
    # sum_l zeta_l >= min_predicted_positives
    if min_predicted_positives is not None and min_predicted_positives > 0:
        col_names = [VARS["zeta_" + str(l)] for l in range(num_leafs)]
        col_values = [1 for _ in range(num_leafs)]
        # add the same gamma slack
        col_names.append(VARS["gamma"])
        col_values.append(1)
        row_names.append("#" + str(row_value))
        row_values.append([col_names, col_values])
        row_right_sides.append(min_predicted_positives)
        row_senses += "G"
        row_value += 1

    return row_names, row_values, row_right_sides, row_senses, row_value


def create_rows(depth):
    global VARS
    row_value = 0

    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = ""

    num_features = get_num_features()
    data_size = get_data_size()

    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    for n in range(num_nodes):
        col_names = []
        col_values = []

        count = 0
        for f in range(num_features):
            if get_num_constants(f) > 0:
                col_names.extend([VARS["node_feature_" + str(n) + "_" + str(f)]])
                col_values.extend([1])
                count = count + 1

        if count == 0:
            continue

        row_names.append("#"+str(row_value))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "E"
        row_value = row_value + 1

    for n in range(num_nodes):
        col_names = []
        col_values = []
        for f in range(num_features):
            if get_num_constants(f) == 0:
                col_names.extend([VARS["node_feature_" + str(n) + "_" + str(f)]])
                col_values.extend([1])

        row_names.append("#"+str(row_value))
        row_values.append([col_names,col_values])
        row_right_sides.append(0)
        row_senses = row_senses + "E"
        row_value = row_value + 1

    for l in range(num_leafs):
        col_names = [VARS["prediction_type_" + str(s) + "_" + str(l)] for s in range(get_num_targets())]
        col_values = [1 for s in range(get_num_targets())]
        row_names.append("#" + str(row_value))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "E"
        row_value = row_value + 1
    
    if use_score:
        nrow_names, nrow_values, nrow_right_sides, nrow_senses, nrow_value = create_score_binbin_rows(depth, row_value)
        row_names.extend(nrow_names)
        row_values.extend(nrow_values)
        row_right_sides.extend(nrow_right_sides)
        row_senses += nrow_senses
        row_value = nrow_value
        
        nrow_names, nrow_values, nrow_right_sides, nrow_senses, nrow_value = create_row_score_rows(depth, row_value)
        row_names.extend(nrow_names)
        row_values.extend(nrow_values)
        row_right_sides.extend(nrow_right_sides)
        row_senses += nrow_senses
        row_value = nrow_value

    if not use_score:
        nrow_names, nrow_values, nrow_right_sides, nrow_senses, nrow_value = create_error_binbin_rows(depth, row_value)
        row_names.extend(nrow_names)
        row_values.extend(nrow_values)
        row_right_sides.extend(nrow_right_sides)
        row_senses += nrow_senses
        row_value = nrow_value

        # ================================= add precision rows =============================================================
        nrow_names, nrow_values, nrow_right_sides, nrow_senses, nrow_value = create_precision_rows(depth, row_value)
        row_names.extend(nrow_names)
        row_values.extend(nrow_values)
        row_right_sides.extend(nrow_right_sides)
        row_senses += nrow_senses
        row_value = nrow_value
        # ==================================================================================================================

        if extra_error_obj != 1:
            nrow_names, nrow_values, nrow_right_sides, nrow_senses, nrow_value = create_leaf_error_rows(depth, row_value)
            row_names.extend(nrow_names)
            row_values.extend(nrow_values)
            row_right_sides.extend(nrow_right_sides)
            row_senses += nrow_senses
            row_value = nrow_value
       
        else:
            nrow_names, nrow_values, nrow_right_sides, nrow_senses, nrow_value = extra_objective_rows(depth, row_value)
            row_names.extend(nrow_names)
            row_values.extend(nrow_values)
            row_right_sides.extend(nrow_right_sides)
            row_senses += nrow_senses
            row_value = nrow_value
        
    if fix_layers != 0:
        nrow_names, nrow_values, nrow_right_sides, nrow_senses, nrow_value = fix_start_solutions(depth, fix_layers, row_value)
        row_names.extend(nrow_names)
        row_values.extend(nrow_values)
        row_right_sides.extend(nrow_right_sides)
        row_senses += nrow_senses
        row_value = nrow_value

    return row_names, row_values, row_right_sides, row_senses

    for n in range(num_nodes):
        for f in range(num_features):
            max_bin = 1+int(math.log(get_max_num_constants()) / math.log(2.))
            col_values = []
            col_names = []

            bins = get_binbin_ranges(0, get_num_constants(f)-1)
            bin_len = -1
            
            for b in bins:
                if len(b[0]) > bin_len:
                   bin_len = len(b[0])
            
            max_val = 0
            for i in range(bin_len + 1, max_bin):
                col_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                col_values.extend([1])
                max_val = max_val + 1
        
            col_names.extend([VARS["node_feature_" + str(n) + "_" + str(f)]])
            col_values.extend([max_val])

            row_names.append("#"+str(row_value))
            row_values.append([col_names,col_values])
            row_right_sides.append(max_val)
            row_senses = row_senses + "L"
            row_value = row_value + 1

def create_leaf_fixed_rows(depth):
    global VARS
    row_value = 0

    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = ""

    num_features = get_num_features()
    data_size = get_data_size()

    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    for l in range(num_leafs):
        col_names = [VARS["prediction_type_" + str(s) + "_" + str(l)] for s in range(get_num_targets())]
        col_values = [1 for s in range(get_num_targets())]
        row_names.append("#" + str(row_value))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "E"
        row_value = row_value + 1
    
    for d in range(data_size):
        col_names = [VARS["leaf_" + str(d) + "_" + str(l)] for l in range(num_leafs)]
        col_values = [1 for l in range(num_leafs)]

        row_names.append("#" + str(row_value))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "E"
        row_value = row_value + 1

    print(set([ITER_LEAF[d] for d in range(len(ITER_LEAF))]))
    for d in range(len(ITER_LEAF)):
        l = ITER_LEAF[d]
        
        col_names = [VARS["leaf_" + str(d) + "_" + str(l)]]
        col_values = [1]
        row_names.append("#" + str(row_value))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "E"
        row_value = row_value + 1
    
    if use_score:
        nrow_names, nrow_values, nrow_right_sides, nrow_senses, nrow_value = create_row_score_rows(depth, row_value)
        row_names.extend(nrow_names)
        row_values.extend(nrow_values)
        row_right_sides.extend(nrow_right_sides)
        row_senses += nrow_senses
        row_value = nrow_value

    if not use_score:
        if extra_error_obj != 1:
            nrow_names, nrow_values, nrow_right_sides, nrow_senses, nrow_value = create_leaf_error_rows(depth, row_value)
            row_names.extend(nrow_names)
            row_values.extend(nrow_values)
            row_right_sides.extend(nrow_right_sides)
            row_senses += nrow_senses
            row_value = nrow_value
        else:
            nrow_names, nrow_values, nrow_right_sides, nrow_senses, nrow_value = extra_objective_rows(depth, row_value)
            row_names.extend(nrow_names)
            row_values.extend(nrow_values)
            row_right_sides.extend(nrow_right_sides)
            row_senses += nrow_senses
            row_value = nrow_value

    for n in range(num_nodes):
        if "node_feature_" + str(n) not in ITER_VARS:
            continue
            
        node_vars = ITER_VARS["node_feature_" + str(n)]
        print(n, ":", node_vars)
        for f in range(num_features):
            if get_feature(f) == node_vars[0]:
                col_names = [VARS["node_feature_" + str(n) + "_" + str(f)]]
                col_values = [1]
                
                row_names.append("#"+str(row_value))
                row_values.append([col_names,col_values])
                row_right_sides.append(1)
                row_senses = row_senses + "E"
                row_value = row_value + 1
            
                val = node_vars[1]
                for i in range(get_num_constants(f)-2):
                    if val >= get_constant_val(f,i) and val < get_constant_val(f,i+1):
                        val = get_constant_val(f,i)
                if get_min_constant_val(f) > val:
                    val = get_min_constant_val(f)
                if get_max_constant_val(f) < val:
                    val = get_max_constant_val(f)
            
                bins = get_binbin_ranges(0, get_num_constants(f)-1)
                for index in range(len(bins)):
                    bin = bins[index]
                    #print bin, val, get_constant_val(feat, bin[1][0]), get_constant_val(feat, bin[2][0])
                    if bin[1][0] == bin[1][1] and val == get_constant_val(f, bin[1][0]):
                        for i in range(len(bin[0])):
                            col_names = [VARS["node_constant_bin_" + str(n) + "_" + str(i)]]
                            col_values = [1]
                    
                            row_names.append("#" + str(row_value))
                            row_values.append([col_names,col_values])
                            row_right_sides.append(1 - bin[0][i])
                            row_senses = row_senses + "E"
                            row_value = row_value + 1

                        col_names = [VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]]
                        col_values = [1]
                    
                        row_names.append("#" + str(row_value))
                        row_values.append([col_names,col_values])
                        row_right_sides.append(1)
                        row_senses = row_senses + "E"
                        row_value = row_value + 1
                    
                    if bin[2][0] == bin[2][1] and val == get_constant_val(f, bin[2][0]):
                        for i in range(len(bin[0])):
                            col_names = [VARS["node_constant_bin_" + str(n) + "_" + str(i)]]
                            col_values = [1]
                            
                            row_names.append("#" + str(row_value))
                            row_values.append([col_names,col_values])
                            row_right_sides.append(1 - bin[0][i])
                            row_senses = row_senses + "E"
                            row_value = row_value + 1

                        col_names = [VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]]
                        col_values = [0]

                        row_names.append("#" + str(row_value))
                        row_values.append([col_names,col_values])
                        row_right_sides.append(0)
                        row_senses = row_senses + "E"
                        row_value = row_value + 1

    return row_names, row_values, row_right_sides, row_senses

def print_nodes(node, diff, depth, solution_values, num_nodes, num_features):
    constant = 0
    feat = 0
    for f in range(num_features):
        if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
            feat = f
            break

    bins = get_binbin_ranges(0, get_num_constants(feat)-1)
    for index in range(len(bins)):
        bin = bins[index]
        if(bin[1][0] == bin[1][1]):
            if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(len(bin[0]))]] == 1:
                found = 0
                for i in range(len(bin[0])):
                    if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(i)]] == bin[0][i]:
                        found = 1
                if not found:
                    constant = get_constant_val(feat, bin[1][0])
        if(bin[2][0] == bin[2][1]):
            if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(len(bin[0]))]] == 0:
                found = 0
                for i in range(len(bin[0])):
                    if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(i)]] == bin[0][i]:
                        found = 1
                if not found:
                    constant = get_constant_val(feat, bin[2][0])
    if len(bins) == 0:
        constant = (get_max_value_f(feat) + get_min_value_f(feat)) / 2.0

    if diff > 0:
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                print("  " * depth, node, get_feature(f), "<=", constant)
        print_nodes(int(float(node)-diff), int(diff/2), depth+1, solution_values, num_nodes, num_features)
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                print("  " * depth, node, get_feature(f), ">", constant)
        print_nodes(int(float(node)+diff), int(diff/2), depth+1, solution_values, num_nodes, num_features)
    else:
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                print("  " * depth, node, get_feature(f), "<=", constant)
        for l in get_left_leafs(node, num_nodes):
            for s in range(get_num_targets()):
                if solution_values[VARS["prediction_type_" + str(s) + "_" + str(l)]] > 0.5:
                    print("  " * (depth+1), l, TARGETS[s])
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                print("  " * depth, node, get_feature(f), ">", constant)
        for l in get_right_leafs(node, num_nodes):
            for s in range(get_num_targets()):
                if solution_values[VARS["prediction_type_" + str(s) + "_" + str(l)]] > 0.5:
                    print("  " * (depth+1), l, TARGETS[s])

def print_nodes_to_file(node, diff, depth, solution_values, num_nodes, num_features, output):
    constant = 0
    feat = 0
    for f in range(num_features):
        if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
            feat = f
            break

    bins = get_binbin_ranges(0, get_num_constants(feat)-1)
    for index in range(len(bins)):
        bin = bins[index]
        if(bin[1][0] == bin[1][1]):
            if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(len(bin[0]))]] == 1:
                found = 0
                for i in range(len(bin[0])):
                    if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(i)]] == bin[0][i]:
                        found = 1
                if not found:
                    constant = get_constant_val(feat, bin[1][0])
        if(bin[2][0] == bin[2][1]):
            if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(len(bin[0]))]] == 0:
                found = 0
                for i in range(len(bin[0])):
                    if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(i)]] == bin[0][i]:
                        found = 1
                if not found:
                    constant = get_constant_val(feat, bin[2][0])
    if len(bins) == 0:
        constant = (get_max_value_f(feat) + get_min_value_f(feat)) / 2.0

    if diff > 0:
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                output.write("  " * depth + "if " + "float(row[header[\"" + get_feature(f) + "\"]])" + " <= " + str(constant) + ":\n")
        print_nodes_to_file(int(float(node)-diff), int(diff/2), depth+1, solution_values, num_nodes, num_features, output)
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                output.write("  " * depth + "if " + "float(row[header[\"" + get_feature(f) + "\"]])" + " > " + str(constant) + ":\n")
        print_nodes_to_file(int(float(node)+diff), int(diff/2), depth+1, solution_values, num_nodes, num_features, output)
    else:
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                output.write("  " * depth + "if " + "float(row[header[\"" + get_feature(f) + "\"]])" + " <= " + str(constant) + ":\n")
        for l in get_left_leafs(node, num_nodes):
            for s in range(get_num_targets()):
                if solution_values[VARS["prediction_type_" + str(s) + "_" + str(l)]] > 0.5:
                    output.write("  " * (depth+1) + "return " + str(TARGETS[s]) + "\n")
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                output.write("  " * depth + "if " + "float(row[header[\"" + get_feature(f) + "\"]])" + " > " + str(constant) + ":\n")
        for l in get_right_leafs(node, num_nodes):
            for s in range(get_num_targets()):
                if solution_values[VARS["prediction_type_" + str(s) + "_" + str(l)]] > 0.5:
                    output.write("  " * (depth+1) + "return " + str(TARGETS[s]) + "\n")

def print_tree(num_nodes, solution_values, num_features):
    print("tree:")
    diff = (num_nodes + 1) / 2
    print_nodes(int(diff-1), int(diff / 2), 0, solution_values, num_nodes, num_features)
    #print "discrimination difference", solution_values[VARS["discrimination_diff_1"]] + solution_values[VARS["discrimination_diff_2"]]

def print_solution_to_file(num_nodes, solution_values, num_features, filename, ifile, tfile):
    f = open(filename, "w")
    f.write("import csv\nimport sys\nimport os\n\n")
    f.write("train = \"/home/zhengke/AHC_max_accuracy/decisiontree_pareto/binoct-master/data_sets_aaaipaper/" + ifile + "\"\n")
    f.write("test = \"/home/zhengke/AHC_max_accuracy/decisiontree_pareto/binoct-master/data_sets_aaaipaper/" + tfile + "\"\n")
    f.write("def predict(row,header):\n")
    diff = (num_nodes + 1) / 2
    print_nodes_to_file(int(diff-1), int(diff / 2), 1, solution_values, num_nodes, num_features, f)
    f.write("\n\ndef main(argv):\n")
    f.write("  header = 0\n")
    f.write("  num_correct = 0\n")
    f.write("  num_total = -1\n")
    f.write("  preds = dict()\n")
    f.write("  with open(argv[0], 'rt') as csvfile:\n")
    f.write("    reader = csv.reader(csvfile, delimiter=',')\n")
    f.write("    for row in reader:\n")
    f.write("      num_total = num_total + 1\n")
    f.write("      if header == 0:\n")
    f.write("        header = dict()\n")
    f.write("        for i in range(len(row)):\n")
    f.write("          header[row[i]] = i\n")
    f.write("      else:\n")
    f.write("        pred = predict(row,header)\n")
    f.write("        orig = row[len(row)-1]\n")
    f.write("        #print(row, pred, orig)\n")
    f.write("        if int(float(pred)) == int(float(orig)):\n")
    f.write("          num_correct = num_correct + 1\n")
    f.write("        if str(pred) + \" - \" + str(orig) not in preds:\n")
    f.write("          preds[str(pred) + \" - \" + str(orig)] = 0\n")
    f.write("        preds[str(pred) + \" - \" + str(orig)] = preds[str(pred) + \" - \" + str(orig)] + 1\n")
    f.write("  print(\"num_correct\", num_correct)\n")
    f.write("  print(\"accuracy\", float(num_correct) / float(num_total))\n")
    f.write("  print(\"crosstable\", preds)\n")
    f.write("  f = open(\"/home/zhengke/AHC_max_accuracy/decisiontree_pareto/binoct-master/data_sets_aaaipaper\" + \"/result.txt\",\"w\")\n")
    f.write("  f.write(\"accuracy \" + str(num_correct / float(num_total)))\n")
    f.write("  f.write(\"crosstable \" + str(preds))\n")
    f.write("\n")
    f.write("print(\"train\")\n")
    f.write("main([train])\n")
    f.write("print(\"test\")\n")
    f.write("main([test])\n")
    f.close()
    #write results to a file
#    f1 = fopen(filename 

def build_predict_function(num_nodes, solution_values, num_features):
    """
    Use your existing code generator (print_nodes_to_file)
    to build a predict(row, header) function in-memory.
    """
    buf = io.StringIO()

    # function signature
    buf.write("def predict(row, header):\n")

    diff = (num_nodes + 1) / 2
    # This is your existing codegen call
    print_nodes_to_file(
        int(diff - 1),
        int(diff / 2),
        1,
        solution_values,
        num_nodes,
        num_features,
        buf
    )

    code = buf.getvalue()

    # Compile into a namespace
    ns = {}
    exec(code, ns, ns)

    return ns["predict"]


def evaluate_csv(file_path, predict_func, pos_label=None):
    """
    Return:
        acc: accuracy
        preds: crosstable dict {"pred - orig": count}
        prec_pos: precision for pos_label (TP/(TP+FP)), or None if pos_label is None
    """
    num_correct = 0
    num_total = 0
    preds = {}

    TP = 0
    FP = 0

    with open(file_path, "rt") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        header_map = None

        for row_idx, row in enumerate(reader):
            if row_idx == 0:
                header_map = {name: i for i, name in enumerate(row)}
                continue

            pred = predict_func(row, header_map)
            orig = row[len(row) - 1]

            # 原来的 accuracy 逻辑
            if int(float(pred)) == int(float(orig)):
                num_correct += 1

            key = f"{pred} - {orig}"
            preds[key] = preds.get(key, 0) + 1

            # 统计 TP / FP（针对指定的 pos_label）
            if pos_label is not None:
                p = int(float(pred))
                o = int(float(orig))
                if p == pos_label and o == pos_label:
                    TP += 1
                elif p == pos_label and o != pos_label:
                    FP += 1

            num_total += 1

    acc = num_correct / float(num_total) if num_total > 0 else 0.0

    if pos_label is not None and (TP + FP) > 0:
        prec_pos = TP / float(TP + FP)
    elif pos_label is not None:
        prec_pos = 0.0   # 没有预测为正类时，precision 置 0（也可以换成 None）
    else:
        prec_pos = None

    return acc, preds, prec_pos


def evaluate_solution(
    num_nodes, solution_values, num_features,
    ifile, tfile,
    pos_label,
    base_path= 'decisiontree/decisiontree_pareto/binoct-master/dataset/'
):
    train_path = base_path + ifile
    test_path  = base_path + tfile

    # 1) build predict(row, header) in memory
    predict_func = build_predict_function(num_nodes, solution_values, num_features)

    # 2) evaluate（现在多传一个 pos_label）
    train_acc, train_ct, train_prec = evaluate_csv(train_path, predict_func, pos_label=pos_label)
    test_acc,  test_ct,  test_prec  = evaluate_csv(test_path,  predict_func, pos_label=pos_label)

    # 3) print 结果（包含 precision）
    print("train")
    print("accuracy", train_acc)
    print("precision (class {}):".format(pos_label), train_prec)
    print("crosstable", train_ct)

    print("test")
    print("accuracy", test_acc)
    print("precision (class {}):".format(pos_label), test_prec)
    print("crosstable", test_ct)

    return {
        "train_acc": train_acc,
        "train_crosstable": train_ct,
        "train_precision_pos": train_prec,

        "test_acc": test_acc,
        "test_crosstable": test_ct,
        "test_precision_pos": test_prec,
    }
    

def get_start_solutions(depth):
    global inputsym

    col_var_names = []
    col_var_values = []
    num_features = get_num_features()
    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    num_features = get_num_features()
    data_size = get_data_size()
    
    num_leafs = 2**depth
    num_nodes = num_leafs-1
    
    values = tr.dt.tree_.value
    
    for n in range(num_nodes):
        feat = sget_feature(tr.dt, convert_node(tr.dt, n, num_nodes))
        
        if feat < 0:
            feat = 0

        for f in range(num_features):
            if f == feat:
                col_var_names.extend([VARS["node_feature_" + str(n) + "_" + str(f)]])
                col_var_values.extend([1])
            else:
                col_var_names.extend([VARS["node_feature_" + str(n) + "_" + str(f)]])
                col_var_values.extend([0])
        
        val = sget_node_constant(tr.dt, convert_node(tr.dt, n, num_nodes))
        
        print(val, get_feature(feat))
        
        for i in range(get_num_constants(feat)-2):
            if val >= get_constant_val(feat,i) and val < get_constant_val(feat,i+1):
                val = get_constant_val(feat,i)

        bins = get_binbin_ranges(0, get_num_constants(feat)-1)
        for index in range(len(bins)):
            bin = bins[index]
            #print bin, val, get_constant_val(feat, bin[1][0]), get_constant_val(feat, bin[2][0])
            if bin[1][0] == bin[1][1] and val == get_constant_val(feat, bin[1][0]):
                for i in range(len(bin[0])):
                    col_var_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                    col_var_values.extend([1 - bin[0][i]])
                col_var_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]])
                col_var_values.extend([1])
                print(bin, val)
            if bin[2][0] == bin[2][1] and val == get_constant_val(feat, bin[2][0]):
                for i in range(len(bin[0])):
                    col_var_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                    col_var_values.extend([1 - bin[0][i]])
                col_var_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(len(bin[0]))]])
                col_var_values.extend([0])
                print(bin, val)
        if get_min_constant_val(feat) > val:
            for i in range(1+int(math.log(get_max_num_constants()) / math.log(2.))):
                col_var_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                col_var_values.extend([1])
        if get_max_constant_val(feat) < val:
            for i in range(1+int(math.log(get_max_num_constants()) / math.log(2.))):
                col_var_names.extend([VARS["node_constant_bin_" + str(n) + "_" + str(i)]])
                col_var_values.extend([0])

    for l in reversed(list(range(num_leafs))):
        predictions = values[convert_leaf(tr.dt, l, num_nodes)].tolist()[0]
        max_index = predictions.index(max(predictions))
        max_class = tr.dt.classes_[max_index]
        
        for s in range(get_num_targets()):
            if TARGETS[s] == max_class:
                col_var_names.extend([VARS["prediction_type_" + str(s) + "_" + str(num_leafs - 1 - l)]])
                col_var_values.extend([1])
            else:
                col_var_names.extend([VARS["prediction_type_" + str(s) + "_" + str(num_leafs - 1 - l)]])
                col_var_values.extend([0])
        
        prev_max_class = max_class
    
    return col_var_names, col_var_values


def lpdtree(depth):
    global SORTED_FEATURE, inputstart, inputtime, inputpolish

    prob = cplex.Cplex()

    num_features = get_num_features()
    data_size = get_data_size()

    num_leafs = 2**depth
    num_nodes = num_leafs-1

    try:
        prob.objective.set_sense(prob.objective.sense.minimize)

        var_names, var_types, var_lb, var_ub, var_obj = create_variables(depth)

        print("num_vars", len(var_names))

        prob.variables.add(obj = var_obj, lb = var_lb, ub = var_ub, types = var_types)#, names = var_names)

        row_names, row_values, row_right_sides, row_senses = create_rows(depth)

        print("num_rows", len(row_names))

        prob.linear_constraints.add(lin_expr = row_values, senses = row_senses, rhs = row_right_sides, names = row_names)
       
        prob.order.set(PRIO)

        if inputstart == 1:
            col_var_names, col_var_values = get_start_solutions(depth)
            start_solution = [0 for i in range(len(VARS))]
            for i in range(len(col_var_values)):
                start_solution[col_var_names[i]] = col_var_values[i]
            print_tree(num_nodes, start_solution, num_features)
            prob.MIP_starts.add([col_var_names, col_var_values], prob.MIP_starts.effort_level.auto)
    
        # prob.write("test.lp")

        #prob.parameters.emphasis.mip.set(4)
        #prob.parameters.emphasis.memory.set(1)
        #prob.parameters.emphasis.numerical.set(1)
        #prob.parameters.mip.strategy.bbinterval.set(0)
        #prob.parameters.mip.strategy.heuristicfreq.set(3)
        #prob.parameters.benders.strategy.set(3)
        #prob.parameters.advance.set(2)
        #prob.parameters.mip.strategy.branch.set(1)
        #prob.parameters.mip.strategy.backtrack.set(0.8)
        #prob.parameters.mip.strategy.nodeselect.set(2)
        #prob.parameters.mip.strategy.variableselect.set(4)
        #prob.parameters.mip.strategy.bbinterval.set(0)
        #prob.parameters.mip.strategy.rinsheur.set(20)
        #prob.parameters.mip.strategy.lbheur.set(1)
        #prob.parameters.mip.strategy.probe.set(3)
        #prob.parameters.preprocessing.presolve.set(1)
        #prob.parameters.lpmethod.set(5)
        #prob.parameters.mip.strategy.startalgorithm.set(3)
        #prob.parameters.mip.strategy.subalgorithm.set(3)
        #prob.parameters.barrier.algorithm.set(3)
        #prob.parameters.workmem.set(8000)
        prob.parameters.mip.polishafter.time.set(inputtime - inputpolish)
        prob.parameters.timelimit.set(inputtime)
        
        start_time = prob.get_time()
        prob.solve()
        end_time = prob.get_time()
        obj = None
        gap = None
        if prob.solution.is_primal_feasible():
            obj = prob.solution.get_objective_value()

        # optimality gap（只对 MIP/MIQP 等有意义）
        try:
            gap = prob.solution.MIP.get_mip_relative_gap()
        except:
            gap = None

        solve_time = end_time - start_time


    except CplexError as exc:
        print(exc)
        return []
    
    print()
    # print("Solution status = " , prob.solution.get_status(), ":", prob.solution.status[prob.solution.get_status()])

    if "infeasible" in prob.solution.status[prob.solution.get_status()]:
        return []
    
    # print("Solution value  = ", prob.solution.get_objective_value())
    
    num_features = get_num_features()
    data_size = get_data_size()
    
    num_leafs = 2**depth
    num_nodes = num_leafs-1

    solution = []
    solution_values = prob.solution.get_values()

    solution_information = {'obj': obj, 'gap': gap, 'time': solve_time}

    # print_tree(num_nodes, solution_values, num_features)

    #for d in range(data_size):
    #    print([get_feature_value(d,f) for f in range(num_features)], [solution_values[VARS["leaf_" + str(d) + "_" + str(l)]] for l in range(num_leafs)])

    return solution_values, solution_information

    for n in range(num_nodes):
        for f in range(num_features):
            print([solution_values[VARS["node_constant_bin_" + str(n) + "_" + str(f) + "_" + str(i)]] for i in range(int(get_min_value_f(f)),int(get_max_value_f(f)))])

    for n in range(num_nodes):
        for f in range(num_features):
            SORTED_FEATURE = f
            all_rows = sorted(list(range(data_size)), key=get_sorted_feature_value)
            if solution_values[VARS["node_feature_" + str(n) + "_" + str(f)]] == 1.0:
                print(solution_values[VARS["node_constant_" + str(n)]])
                print([get_feature_value(d,f) for d in all_rows if sum([int(solution_values[VARS["score_" + str(d) + "_" + str(l)]]) for l in get_left_leafs(n, num_nodes)]) > 0])
                print([get_feature_value(d,f) for d in all_rows if sum([int(solution_values[VARS["score_" + str(d) + "_" + str(l)]]) for l in get_right_leafs(n, num_nodes)]) > 0])

    for d in range(data_size):
        print(get_target(d), \
              [int(round(100.0 * get_target(d) * solution_values[VARS["row_error_" + str(d)]]))] , \
              [[int(round(100.0 * solution_values[VARS["prediction_type_" + str(s) + "_" + str(l)]])) for s in range(get_num_targets())] for l in range(num_leafs)], \
              [int(round(100.0 * (solution_values[VARS["score_" + str(d) + "_" + str(l)]]))) for l in range(num_leafs)])

    return solution


def node_to_string(node, diff, depth, solution_values, num_nodes, num_features, current_node):
    global ITER_VARS
    
    left_result = ""
    right_result = ""
    constant = 0
    feat = 0

    for f in range(num_features):
        if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
            feat = f
            break

    bins = get_binbin_ranges(0, get_num_constants(feat)-1)
    for index in range(len(bins)):
        bin = bins[index]
        if(bin[1][0] == bin[1][1]):
            if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(len(bin[0]))]] == 1:
                found = 0
                for i in range(len(bin[0])):
                    if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(i)]] == bin[0][i]:
                        found = 1
                if not found:
                    constant = get_constant_val(feat, bin[1][0])
        if(bin[2][0] == bin[2][1]):
            if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(len(bin[0]))]] == 0:
                found = 0
                for i in range(len(bin[0])):
                    if solution_values[VARS["node_constant_bin_" + str(node) + "_" + str(i)]] == bin[0][i]:
                        found = 1
                if not found:
                    constant = get_constant_val(feat, bin[2][0])
    if len(bins) == 0:
        constant = (get_max_value_f(feat) + get_min_value_f(feat)) / 2.0

    if diff > 0:
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                left_result = left_result + "  " * depth + " " + str(node) + " " + str(get_feature(f)) + " " + "<=" + " " + str(constant) + "\n"
                ITER_VARS["node_feature_" + str(current_node)] = [str(get_feature(f)), constant]
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                right_result = right_result + "  " * depth + " " + str(node) + " " + str(get_feature(f)) + " " + ">" + " " + str(constant) + "\n"
    else:
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                ITER_VARS["node_feature_" + str(current_node)] = [str(get_feature(f)), constant]
                left_result = left_result + "  " * depth + " " + str(node) + " " + str(get_feature(f)) + " <= " + str(constant) + "\n"
        for l in get_left_leafs(node, num_nodes):
            for s in range(get_num_targets()):
                if solution_values[VARS["prediction_type_" + str(s) + "_" + str(l)]] > 0.5:
                    left_result = left_result + "  " * (depth+1) + " " + str(l) + " " + str(TARGETS[s]) + "\n"
        for f in range(num_features):
            if solution_values[VARS["node_feature_" + str(node) + "_" + str(f)]] > 0.5:
                right_result = right_result + "  " * depth + " " + str(node) + " " + str(get_feature(f)) + " > " + str(constant) + "\n"
        for l in get_right_leafs(node, num_nodes):
            for s in range(get_num_targets()):
                if solution_values[VARS["prediction_type_" + str(s) + "_" + str(l)]] > 0.5:
                    left_result = left_result + "  " * (depth+1) + " " + str(l) + " " + str(TARGETS[s]) + "\n"

    return left_result, right_result

def iterative(rows, current_depth, current_node, max_depth, tree_depth, num_examples, inputfile):
    global ITER_VARS
    global ITER_PREDICT
    
    if current_depth < 0:
        return ""
    
    read_file_rows(inputfile,set(rows))

    pure = True
    tar = get_target(0)
    for r in range(len(rows)):
        if get_target(r) != tar:
            pure = False

    if pure:
        return ""
    if len(rows) < num_examples:
        return ""

    find_constants()
    
    depth = min(tree_depth, max_depth - current_depth)
    
    if depth < 1:
        return ""

    write_file(inputfile+".iter")
    tr.df = tr.get_data(inputfile+".iter")
    tr.learnTrees(depth)
    tr.get_code()

    num_features = get_num_features()
    data_size = get_data_size()

    num_leafs = 2**depth
    num_nodes = num_leafs-1
    num_nodes_max = (2**max_depth)-1

    solution_values = lpdtree(depth)
    
    for d in range(data_size):
        for l in range(num_leafs):
            if solution_values[VARS["leaf_" + str(d) + "_" + str(l)]] > 0.5:
                ITER_LEAF[rows[d]] = l + min(get_left_leafs(current_node, num_nodes_max))
                if l + min(get_left_leafs(current_node, num_nodes_max)) > 31:
                    print("here")
                    print(l)
                    print(current_node)
                    print(depth)
    

    for n in range(num_nodes):
        if get_num_parents(n,num_nodes) == 0:
            
            left_string, right_string = node_to_string(n, depth - 1, current_depth, solution_values, num_nodes, num_features, current_node)
            
            lower_leafs = get_left_leafs(n, num_nodes)
            rows_left = []
            rows_right = []
                
            for d in range(data_size):
                if sum([solution_values[VARS["leaf_" + str(d) + "_" + str(l)]] for l in lower_leafs]) > 0.5:
                    rows_left.append(rows[d])

            lower_leafs = get_right_leafs(n, num_nodes)
            row_set_right = ()
            for d in range(data_size):
                if sum([solution_values[VARS["leaf_" + str(d) + "_" + str(l)]] for l in lower_leafs]) > 0.5:
                    rows_right.append(rows[d])

            left_string = left_string + iterative(rows_left, current_depth + 1, get_left_node(current_node, num_nodes_max), max_depth, tree_depth, num_examples, inputfile)

            right_string = right_string + iterative(rows_right, current_depth + 1, get_right_node(current_node, num_nodes_max), max_depth, tree_depth, num_examples, inputfile)

            return left_string + right_string
    return ""

def lpdtree_iter(max_depth, tree_depth, count_max, inputfile):
    global ITER_VARS
    global FIXED_NODES
    global ITER_PREDICT
    
    string_result = ""
    
    num_nodes_max = (2**max_depth)-1
    for n in range(num_nodes_max):
        if get_num_parents(n,num_nodes_max) == 0:
            string_result = iterative(list(range(get_data_size())), 0, n, max_depth, tree_depth, count_max, inputfile)
    
    read_file(inputfile)
    find_constants()

    num_features = get_num_features()
    data_size = get_data_size()

    num_leafs = 2**max_depth
    num_nodes = num_leafs-1

    try:
        prob = cplex.Cplex()

        prob.objective.set_sense(prob.objective.sense.minimize)

        var_names, var_types, var_lb, var_ub, var_obj = create_variables(max_depth)

        prob.variables.add(obj = var_obj, lb = var_lb, ub = var_ub, types = var_types)#, names = var_names)

        row_names, row_values, row_right_sides, row_senses = create_leaf_fixed_rows(max_depth)

        # print("num_rows", len(row_names))

        prob.linear_constraints.add(lin_expr = row_values, senses = row_senses, rhs = row_right_sides, names = row_names)

        prob.parameters.emphasis.mip.set(4)
        prob.parameters.mip.polishafter.time.set(inputtime - inputpolish)
        prob.parameters.timelimit.set(inputtime)
                
        prob.solve()

    except CplexError as exc:
        print(exc)
        return []
    # print()
    # print("Solution status = " , prob.solution.get_status(), ":", prob.solution.status[prob.solution.get_status()])

    if "infeasible" in prob.solution.status[prob.solution.get_status()]:
        return []
    
    # print("Solution value  = ", prob.solution.get_objective_value())
    solution_values = prob.solution.get_values()


    # print_tree(num_nodes, solution_values, num_features)

    # print(string_result)

    return solution_values

def main(argv):
   global relaxation, relax_node_constraints, use_score, extra_error_obj, fix_layers, bool_decision, count_max, maxi_depth, forest_bounds, PRIO, FIXED_NODES, ITER_VARS, ITER_LEAF
   global inputstart
   global inputsym
   global inputtime
   global inputpolish
   global double_data
   global ctype
   global continuousconstant
   global relaxation
   global fix_layers
   global extra_error_obj
   global relax_node_constraints
   global bool_decision
   global max_depth
   global count_max
   global use_score
   global forest_bounds
   
   # boolean relaxing all binary and integer variables to be continuous
   relaxation = 0
   # boolean whether node constraints are relaxed using penalties
   relax_node_constraints = 0
   # boolean whether to use scores or errors
   use_score = 0
   # computes all predictions in addition to errors
   extra_error_obj = 0
   # fix first x layers of tree using CART
   fix_layers = 0
   # make all non-decision variables boolean
   bool_decision = 0
   
   inputstart = 0

   count_max = -1
   maxi_depth = 10

   forest_bounds = -1

   # priority ordering
   PRIO = []
   FIXED_NODES = dict()

   ITER_VARS = dict()
   ITER_LEAF = dict()

   # print(argv)

   inputfile = ''
   
   try:
      opts, args = getopt.getopt(argv,"f:y:z:d:u:s:t:p:r:e:o:p:l:b:m:c:x:",["ifile=", "ofile=", "dataset=", "depth=", "run=", "start=","timelimit=","polishtime=","relax=","error=","objective=","penalty=","layer=","boolean=","max_depth=","count_max=","forest_bounds="])
   except getopt.GetoptError:
      sys.exit(2)
   for opt, arg in opts:
      if opt in ("-f", "--ifile"):
         inputfile = arg
      elif opt in ("-y", "--ofile"): # output file
         outputfile = arg
      elif opt in ("-z", "--dataset"): # dataset name
         dataset = arg
      elif opt in ("-d", "--depth"):
         inputdepth = int(arg)
      elif opt in ("-u", "--run"):
          run = int(arg)
      elif opt in ("-s", "--start"):
         inputstart = int(arg)
      elif opt in ("-t", "--timelimit"):
         inputtime = int(arg)
      elif opt in ("-p", "--polishtime"):
         inputpolish = int(arg)
      elif opt in ("-r", "--relax"):
         relaxation = int(arg)
      elif opt in ("-e", "--error"):
         use_score = 1 - int(arg)
      elif opt in ("-o", "--objective"):
         extra_error_obj = int(arg)
      elif opt in ("-p", "--penalty"):
         relax_node_constraints = int(arg)
      elif opt in ("-l", "--layer"):
         fix_layers = int(arg)
      elif opt in ("-b", "--boolean"):
         bool_decision = int(arg)
      elif opt in ("-m", "--max_depth"):
         maxi_depth = int(arg)
      elif opt in ("-c", "--count_max"):
         count_max = int(arg)
      elif opt in ("-x", "--forest_bounds"):
         forest_bounds = int(arg)

   # print(opts, args, inputfile)
 
   read_file(inputfile)
   find_constants()
   tr.df = tr.get_data(inputfile)
   tr.learnTrees(inputdepth)
   tr.get_code()

#    print("num features:")
#    print(get_num_features())
#    print("constants:")
   sum = 0
   for f in range(get_num_features()):
      # print(get_num_constants(f));
      sum = sum+get_num_constants(f);
#    print("sum:")
#    print(sum)

   # return;

   if forest_bounds > 0:
      clear_constants()
      add_constants_from_forest(forest_bounds, inputdepth)

   add_constants_from_tree(inputdepth)

   solution_values = []
   if count_max != -1:
      solution_values = lpdtree_iter(maxi_depth, inputdepth, count_max, inputfile)
   else:
      solution_values, solution_information = lpdtree(inputdepth)

   num_features = get_num_features()
   num_leafs = 2**inputdepth
   num_nodes = num_leafs-1
   
   testfile = inputfile
   testfile = testfile.replace("train", "test")

   # print_solution_to_file(num_nodes, solution_values, num_features, inputfile + "-" + str(inputdepth) + "-" + str(count_max) + "-" + str(forest_bounds) + "-" + str(inputstart) + ".sol.py", ntpath.basename(inputfile), ntpath.basename(testfile))
   results = evaluate_solution(num_nodes=num_nodes, solution_values=solution_values, num_features=num_features, ifile=ntpath.basename(inputfile), tfile=ntpath.basename(testfile), pos_label=positive_label
)
   with open(outputfile, mode='a', newline='') as all_result:
    method = 'U-BinOCT' if precision_min is None else 'C-BinOCT'
    writer = csv.writer(all_result)
    obj = solution_information['obj']
    gap = solution_information['gap']
    time = solution_information['time']
    gamma = solution_values[VARS["gamma"]] if precision_min is not None else None
    writer.writerow([method, dataset, run, inputdepth, positive_label, precision_min, obj, gap, time, gamma, results['train_acc'], results['test_acc'], results['train_precision_pos'], results['test_precision_pos']])
   
###################################################################################################################
   # Print to a file and run this file to get the results, we need to split the train set and the test set by ourselves
###################################################################################################################
   # IndexError: list index out of range: it may due to the Cplex cannot be used here, so no solution is obtaiend.
###################################################################################################################
   #importlib.invalidate_caches()
   #path = ntpath.dirname(inputfile)
   #fil  = ntpath.basename(inputfile)

   #print(path, fil)
   
   #print(inputfile + "-" + str(inputdepth) + "-" + str(count_max) + "-" + str(forest_bounds) + "-" + str(inputstart) + ".sol.py")
   #mod = importlib.import_module(inputfile + "-" + str(inputdepth) + "-" + str(count_max) + "-" + str(forest_bounds) + "-" + str(inputstart) + ".sol.py")
   #mod = imp.load_source(inputfile + "-" + str(inputdepth) + "-" + str(count_max) + "-" + str(forest_bounds) + "-" + str(inputstart) + ".sol.py", path)
   #mod.main([inputfile])
   #mod.main([testfile])

   #os.system(inputfile + "-" + str(inputdepth) + "-" + str(count_max) + "-" + str(forest_bounds) + "-" + str(inputstart) + ".sol.py "+ testfile)
   #subprocess.call([inputfile + "-" + str(inputdepth) + "-" + str(count_max) + "-" + str(forest_bounds) + "-" + str(inputstart) + ".sol.py", testfile])

   #write results to file
#   write_results_to_file(inputfile+ + "-" + str(inputdepth) + "-" + str(count_max != 1) + "-" + str(forest_bounds) + "-" + str(inputstart) + ".results")

if __name__ == "__main__":
   main(sys.argv[1:])



