import FlowOCTReplication
import BendersOCTReplication
import time
import pandas as pd
import random
import os
import csv
from datetime import datetime

depths = [2, 3, 4]
samples = [1, 2, 3, 4]
_lambdas = [round(0.1 * k, 1) for k in range(10)]

dataset = 'blsc'
current_datetime = datetime.now()
result_dir = os.getcwd() + '/decisiontree/decisiontree_pareto/StrongTree-master/Results'
result_csv = f'{result_dir}/'+ f'{dataset}_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+'.csv'
with open(result_csv, mode='a', newline='') as all_result:
    writer = csv.writer(all_result)
    writer.writerow(['dataset', 'depth', 'split', 'nrow', 'method', 'restricted_class', 'beta_p', 'lambda', 
                     'time_limit', 'status', 'obj_value', 'gamma', 'time', 'gap',
                     'train_acc', 'eval_acc','test_acc', 'train_prec', 'eval_prec','test_prec',
                     'node_count','cb_time_int','cb_time_int_suc','cb_counter_int','cb_counter_int_suc'])
    

# ===================================== BendersOCT (Train on training set, validate on validation set) =====================================================
def validation(dataset, result_csv):
    for depth in depths:
        for sample in samples:
            for _lambda in _lambdas:
                BendersOCTReplication.main(["-f", dataset +'.csv', "-o", result_csv, "-d", depth, "-t", 3600, "-l", _lambda, "-i", sample, "-c", 1, "-r", 0])
    # Rename the result file as f'{dataset}_' + 'validation.csv'


# ============================ Benders & FlowOCT ============================
def select_lambda_from_validation(validation_csv_path: str, seed: int | None = None):
    """
    Returns: dict mapping (dataset_base, depth, sample) -> tuned_lambda
    dataset_base is dataset filename WITHOUT extension, e.g. 'balance_scale_onehot'
    """
    df = pd.read_csv(validation_csv_path)
    # make sure types are consistent
    df["depth"] = df["depth"].astype(int)
    df["sample"] = df["sample"].astype(int)
    df["eval_acc"] = pd.to_numeric(df["eval_acc"], errors="coerce")
    df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")

    rng = random.Random(seed)

    tuned = {}
    for (dset, dep, samp), g in df.groupby(["dataset", "depth", "sample"], dropna=True):
        g = g.dropna(subset=["eval_acc", "lambda"])
        if g.empty:
            continue

        best = g["eval_acc"].max()
        candidates = g.loc[g["eval_acc"] == best, "lambda"].unique().tolist()
        if not candidates:
            continue

        tuned[(dset, dep, samp)] = rng.choice(candidates)  # random tie-break

    return tuned


def run_strongOCT(result_dir, dataset, method = 'BendersOCT'):
    validation_csv = result_dir + "/" + dataset + "_validation.csv"
    tuned_lambda_map = select_lambda_from_validation(validation_csv, seed=None)

    for depth in depths:
        for sample in samples:
            tuned_lambda = tuned_lambda_map.get((dataset, depth, sample), 42)
            if tuned_lambda is None:
                print(f"[WARN] No tuned lambda for (dataset={dataset}, depth={depth}, sample={sample}). Fallback to 0.0")
                tuned_lambda = 0.0
            if method == 'BendersOCT':
                BendersOCTReplication.main(["-f", dataset +'.csv', "-o", result_csv, "-d", depth, "-t", 3600, "-l", tuned_lambda, "-i", sample, "-c", 0, "-r", 0])
            if method == 'FlowOCT':
                if dataset == 'blsc':
                    threshold_dict = {2: [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97], 
                                      3: [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97], 
                                      4: [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]}
                if dataset == 'ctmc':
                    threshold_dict = {2: [0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72], 
                                      3: [0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82], 
                                      4: [0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82]}
                for precision in threshold_dict[depth]:
                    FlowOCTReplication.main(["-f", dataset +'.csv', "-o", result_csv, "-d", depth, "-t", 3600, "-l", tuned_lambda, "-i", sample, "-c", 0, "-p", precision, "-r", 0])
                    

run_strongOCT(result_dir, dataset, 'FlowOCT')