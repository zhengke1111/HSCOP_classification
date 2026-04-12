FEASIBILITYTOL = 1e-09

MODEL_PARAM = {'MIPFocus': 1,
               'IntegralityFocus': 1,
               'NumericFocus': 3,
               'FeasibilityTol': FEASIBILITYTOL,
               'Threads': 32,
               }
RHO = 1e04
TAU_1 = 100

FULL_MODEL_TIME_LIMIT = 3600

UNPA_PARTIAL_MODEL_TIME_LIMIT = 600
UNPA_UNCHANGED_TOLERANCE = 100

PARTIAL_MODEL_TIME_LIMIT = 300
UNCHANGED_TOLERANCE = 30

ENHANCED_SIZE = 4

SHRINKAGE_MAX_OUT_ITER = 4

ALG_PARAM = {'iteration': {'unchanged_iter': 3, 'max_iter': 10},
             'ratio': {
                 'max_ratio': 60, 
                 'base_ratio': {2: {'wine': 40, 'nwth': 40, 'htds': 20, 'dmtl': 20, 'blsc': 20, 'ctmc': 20, 'ceva': 20, 'fish': 20},
                                3: {'wine': 20, 'nwth': 20, 'htds': 10, 'dmtl': 10, 'blsc': 10, 'ctmc': 10, 'ceva': 10, 'fish': 10},
                                4: {'wine': 10, 'nwth': 10, 'htds': 10, 'dmtl': 10, 'blsc': 10, 'ctmc': 5, 'ceva': 5, 'fish': 5}}, 
                 'change_ratio': 10}}

EPSILON = 1e-4

DATASET_LIST = ['wine', 'nwth', 'htds', 'dmtl', 'blsc', 'ctmc', 'ceva', 'fish']
DATASET_LIST_PARETO = ['blsc', 'ctmc']


THRESHOLD_GRID = {'blsc': {2: [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97], 
                        3: [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97], 
                        4: [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]},
                  'ctmc': {2: [0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72], 
                        3: [0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82], 
                        4: [0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82]}}


REUSE_TAU_0 = True