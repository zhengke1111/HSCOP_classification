import os
import glob
import learn_class_bin as lcb
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import csv


def ensure_train_test_files(
    path,
    dataset,
    run,
    base_random_state=42,
    test_size=0.25,
    label_col=None  # None means "use last column"
):
    """
    Check if {dataset}_run{run}_train.csv and _test.csv exist under path1.
    If not, generate them from {dataset}.csv using stratified train_test_split.

    Returns:
        train_file, test_file
    """
    train_file = os.path.join(path, f"{dataset}_run{run}_train.csv")
    test_file  = os.path.join(path, f"{dataset}_run{run}_test.csv")
    raw_file   = os.path.join(path, f"{dataset}.csv")

    # If both exist, do nothing
    if os.path.exists(train_file) and os.path.exists(test_file):
        return train_file, test_file

    # Otherwise generate
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw dataset file not found: {raw_file}")

    df = pd.read_csv(raw_file)

    # Decide label column
    if label_col is None:
        label_col = df.columns[-1]

    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in columns: {list(df.columns)}")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=base_random_state + run
    )

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test  = pd.concat([X_test,  y_test],  axis=1)

    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)

    return train_file, test_file


#path1 = 'D:\Sicco\Dropbox\Dropbox\INFORMS-J-Optimization\dataset-cg-paper'
path1 = 'decisiontree/decisiontree_pareto/binoct-master/dataset/'
current_datetime = datetime.now()    
output_filename = path1 + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+'.csv'
with open(output_filename, mode='a', newline='') as all_result:
    writer = csv.writer(all_result)
    writer.writerow(['method', 'dataset', 'run', 'depth', 'restricted_class', 'beta_p', 'objective_value', 'optimality_gap', 'time', 'gamma', 
                    'train_acc','test_acc','train_prec','test_prec'])
#for filename in os.listdir(path2):

# === Precision constraint configuration ===
USE_PRECISION      = True   # True = enforce precision constraint, False = ignore
# precision      = 0.75    # required training precision for positive class
MIN_PRED_POS       = 1      # minimum number of predicted positives
POS_LABEL          = 0      # label value of the positive class in your dataset

# path1 = sys.argv[1]
dataset_list = ['ctmc']
for depth in range(2,5):
    for dataset in dataset_list:
        for run in range(1,5):
            train_file, test_file = ensure_train_test_files(
                path=path1,
                dataset=dataset,
                run=run,
                base_random_state=42,
                test_size=0.25,
                label_col='target' 
            )
            # for precision in [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]:
            threshold_dict = {2: [0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72], 
                          3: [0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82], 
                          4: [0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82]}
            for precision in threshold_dict[depth]:
                input_filename = train_file
                lcb.use_precision_constraint  = 1 if USE_PRECISION else 0
                lcb.precision_min             = precision if USE_PRECISION else None
                lcb.min_predicted_positives   = MIN_PRED_POS
                lcb.positive_label            = POS_LABEL

                lcb.main(["-f", input_filename, "-y", output_filename, "-z", dataset, "-d", depth, "-u", run, "-t", 3600, "-p", 600])









# for filename in glob.glob(os.path.join(path1, '*train.csv')):
#     print(filename)
#      #lcb.main(["-f",filename, "-d", 1, "-t", 900, "-p", 300])

#     #for depth in range(1,5):
#     #lcb.main(["-f",filename, "-d", depth, "-t", 600, "-p", 150])

#     # === configure precision constraint for this run ===
#     lcb.use_precision_constraint  = 1 if USE_PRECISION else 0
#     lcb.precision_min             = PRECISION_MIN
#     lcb.min_predicted_positives   = MIN_PRED_POS
#     lcb.positive_label            = POS_LABEL

#     lcb.main(["-f",filename, "-d", 2, "-t", 3600, "-p", 600])

    #for depth in range(1,5):
    #lcb.main(["-f",filename, "-d", 4, "-t", 600, "-p", 150, "-x", 20, "-s", 1])
    #lcb.main(["-f",filename, "-d", depth, "-t", 600, "-p", 150, "-x", 10, "-s", 1])
