TAU_1 = 10
# TAU_1 = 100  # In pareto comparison, we set TAU_1 as 100

FEASIBILITY_TOL = 1e-09
MODEL_PARAM = {'MIPFocus': 1,
               'IntegralityFocus': 1,
               'FeasibilityTol': FEASIBILITY_TOL,
               'Threads': 4,
               }

RHO = 1e4
EPSILON = 1e-4

FULL_MODEL_TIME_LIMIT = 3600
FULL_EARLY_TERMINATION_UNCHANGED_TOLERANCE = 300
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

THRESHOLD_GRID = {'rice': [0.96, 0.97, 0.98], 'hmeq': [0.80, 0.85, 0.90], 'ospi': [0.77, 0.79, 0.81]}

THRESHOLD_GRID_PARETO = {'rice': [0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97],
                         'hmeq': [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98],
                         'ospi': [0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85]}
