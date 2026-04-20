TAU_1 = 10

FEASIBILITY_TOL = 1e-09
MODEL_PARAM = {'MIPFocus': 1,
               'IntegralityFocus': 1,
               'FeasibilityTol': FEASIBILITY_TOL,
               'Threads': 4,
               }

RHO = 1e4
EPSILON = 1e-4

FULL_MODEL_TIME_LIMIT = 3600
PARTIAL_MODEL_TIME_LIMIT = 300
UNCHANGED_TOLERANCE = 30

SHRINKAGE_MAX_OUT_ITER = 4

ALG_PARAM = {'iteration': {'unchanged_iter': 3, 'max_iter': 10},
             'ratio': {
                 'max_ratio': {'rice': 80, 'hmeq': 80, 'ospi': 60}, 
                 'base_ratio': {'rice': 20, 'hmeq': 20, 'ospi': 10}, 
                 'initial_ratio_prime': {'rice': 50, 'hmeq': 50, 'ospi': 20}, 
                 'change_ratio': 10}}

DATASET_LIST = ['rice', 'hmeq', 'ospi']
# DATASET_LIST = ['rice'] # For test

THRESHOLD_GRID = {'rice': [0.96, 0.97, 0.98], 'hmeq': [0.80, 0.85, 0.90], 'ospi': [0.77, 0.79, 0.81]}
# THRESHOLD_GRID = {'rice': [0.96]} # For test