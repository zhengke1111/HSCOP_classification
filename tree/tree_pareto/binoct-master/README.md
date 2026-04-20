# binoct (Original)
Binary Programming Formulation for Learning Classification Trees Using Cplex


# Modification for precision constrained multi-class classification 
In "Solving Constrained Affine Heaviside Composite Optimiation Problem by a Progressive IP Approach", we add the precision constraint in `learn_class_bin.py`, see `BinOCT Code Modification.pdf`. 

- For running an instance of C-BinOCT,

    ```Python
    lcb.use_precision_constraint  = 1  # Enable the precision constraint
    lcb.precision_min             = precision  # Set the precision threshold as a predefined precision
    lcb.min_predicted_positives   = MIN_PRED_POS  # Minimum number of predicted positives (positive: the class that we add the precision constraint)
    lcb.positive_label            = POS_LABEL  # The label of the positive class in your dataset

    lcb.main(["-f", input_filename, "-y", output_filename, "-z", dataset, "-d", depth, "-u", run, "-t", 3600, "-p", 600])
    ```
    
    - f : name of the dataset (training set)
    - y : output results file
    - z : dataset, e.g. `'blsc'`
    - d : maximum depth of the tree
    - u : sample (for shuffling and splitting data)
    - t : time limit
    - p : time of solution polishing in `Cplex`

- For running an instance of U-BinOCT,

    ```Python
    lcb.use_precision_constraint  = 0  # Disable the precision constraint 
    lcb.precision_min             = None  # There is no precision threshold
    lcb.min_predicted_positives   = MIN_PRED_POS  
    lcb.positive_label            = POS_LABEL  # The label of the positive class in your dataset, although there is no precision constraint, we can still set a POS_LABEL to ensure at least MIN_PRED_POS samples are classified as POS_LABEL
    lcb.main(["-f", input_filename, "-y", output_filename, "-z", dataset, "-d", depth, "-u", run, "-t", 3600, "-p", 600])
    ```

    - f : name of the dataset (training set)
    - y : output results file
    - z : dataset, e.g. `'blsc'`
    - d : maximum depth of the tree
    - u : sample (for shuffling and splitting data)
    - t : time limit
    - p : time of solution polishing in `Cplex`

   