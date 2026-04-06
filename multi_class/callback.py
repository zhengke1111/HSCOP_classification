from gurobipy import GRB


def full_model_callback(model, where):
    """
    Callback function for monitoring the full MIP optimization process.

    This callback is triggered during the Gurobi optimization procedure and is intended to:
    1. record incumbent objective value, best bound, optimality gap, and runtime at 3-second intervals,
    2. track the time of the most recent objective improvement,
    3. record the runtime at which a feasible solution (obj >= 0) is first detected,
    4. update key optimization metrics (optimality gap, last objective value) for persistent tracking.

    Specifically, at MIP and MIPSOL callback points, the function:
    - retrieves the current incumbent objective value, best bound, and runtime,
    - computes the optimality gap (avoiding division by zero),
    - updates the "final improvement time" if the objective changes significantly (>=1e-06),
    - logs the runtime used for finding the first feasible solution (obj >= 0),
    - maintains persistent state in `model.__dict__` for cross-callback tracking.

    Args:
        model (gurobipy.Model): Gurobi model object passed automatically to the callback.
            The function uses `model.__dict__` to store persistent information across calls:
            - `last_time`: the last runtime checkpoint (for 3-second interval logging),
            - `last_obj`: the last recorded incumbent objective value,
            - `final_improvement_time`: runtime of the most recent objective improvement,
            - `time_for_feasible`: runtime used to find the first feasible solution (obj >= 0),
            - `optimality_gap`: current optimality gap between incumbent and best bound.
        where (int): Integer callback code provided by Gurobi indicating the current context.
            This function acts on `GRB.Callback.MIP` (MIP progress) and `GRB.Callback.MIPSOL` (new MIP solution).

    Side Effects:
        - Updates fields in `model.__dict__` for persistent metric tracking,
        - Sets `model.__dict__['time_for_feasible']` when an infeasible solution (obj ≤ 0) is detected,
        - Updates `model.__dict__['final_improvement_time']` on significant objective changes.

    Notes:
        - Optimality gap is calculated as |obj_bst - obj_bnd| / |obj_bst| (returns -1 if obj_bst = 0),
        - Progress metrics are updated every 3 seconds of solver runtime,
        - Objective improvement is defined as a change ≥1e-06 in the incumbent value.
    """
    if where == GRB.Callback.MIP:
        run_time = model.cbGet(GRB.Callback.RUNTIME)

        # Update information every 3 seconds
        if run_time - model.__dict__['last_time'] > 3:
            obj_bst = model.cbGet(GRB.Callback.MIP_OBJBST)
            obj_bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            obj_variation = abs(obj_bst - model.__dict__['last_obj'])

            # Calculate optimality gap (avoid division by zero)
            obj_gap = (abs(obj_bst - obj_bnd) / abs(obj_bst)) if abs(
                obj_bst) != 0 else -1

            # Update final improvement time if objective changes significantly
            if obj_variation >= 1e-06:
                model.__dict__['final_improvement_time'] = run_time

            # Update model metrics in dictionary
            model.__dict__['optimality_gap'] = obj_gap
            model.__dict__['last_time'] = run_time
            model.__dict__['last_obj'] = obj_bst
            # Record time used for finding the first feasible solution (obj>=0)
            if obj_bst <= 0:
                model.__dict__['time_for_feasible'] = run_time
    if where == GRB.Callback.MIPSOL:
        run_time = model.cbGet(GRB.Callback.RUNTIME)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        obj_bst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        obj_bnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        # Calculate optimality gap for MIP solution
        obj_gap = (abs(obj - obj_bnd) / abs(obj)) if abs(
            obj) != 0 else -1
        # Update improvement time if objective improves
        if obj >= obj_bst + 1e-06:
            # update info
            model.__dict__['final_improvement_time'] = run_time
        # Update model metrics
        model.__dict__['optimality_gap'] = obj_gap
        model.__dict__['last_obj'] = obj
        model.__dict__['last_time'] = run_time
        # Record time used for finding the first feasible solution (obj>=0)
        if obj <= 0:
            model.__dict__['time_for_feasible'] = run_time


def partial_model_callback(model, where):
    """
    Callback function for monitoring the partial MIP model optimization process with early termination.

    This callback is triggered during the Gurobi optimization procedure and is intended to:
    1. track objective stability at 3-second intervals,
    2. record the runtime used for the first feasible solution (obj >= 0),
    3. terminate the optimization early if the objective value remains unchanged for a prescribed tolerance period,
    4. log early termination events to the model's log file.

    Specifically, at MIP callback points, the function:
    - checks for significant objective changes (≥1e-06) to update improvement time,
    - terminates optimization if no improvement is observed for `model.__dict__['unchanged_tolerance']`,
    - logs termination reason to the model's log file,
    - records the runtime used for finding the first feasible solution (obj >= 0).

    Args:
        model (gurobipy.Model): Gurobi model object passed automatically to the callback.
            The function uses `model.__dict__` to store persistent information across calls:
            - `last_time`: the last runtime checkpoint (for 3-second interval checks),
            - `last_obj`: the last recorded incumbent objective value,
            - `final_improvement_time`: runtime of the most recent objective improvement,
            - `unchanged_tolerance`: maximum allowed duration (seconds) for unchanged objective,
            - `time_for_feasible`: runtime when first feasible solution (obj ≤ 0) is found.
        where (int): Integer callback code provided by Gurobi indicating the current context.
            This function only acts on `GRB.Callback.MIP`.

    Side Effects:
        - Updates fields in `model.__dict__` for persistent metric tracking,
        - Sets `model.__dict__['time_for_feasible']` when a feasible solution (obj ≤ 0) is detected,
        - Writes termination logs to `model.Params.LogFile` if early termination is triggered,
        - May terminate the optimization by calling `model.terminate()` (implied by log, add if needed).

    Notes:
        - Objective stability is defined as a change <1e-06 in the incumbent value,
        - Early termination is triggered if stability duration exceeds `unchanged_tolerance`,
        - Progress checks are performed every 3 seconds of solver runtime.
    """
    if where == GRB.Callback.MIP:
        run_time = model.cbGet(GRB.Callback.RUNTIME)
        # Update information every 3 seconds
        if run_time - model.__dict__['last_time'] > 3:
            obj_bst = model.cbGet(GRB.Callback.MIP_OBJBST)
            obj_variation = abs(obj_bst - model.__dict__['last_obj'])
            # Update improvement time if objective changes significantly
            if obj_variation >= 1e-06:
                model.__dict__['final_improvement_time'] = run_time
            else:
                # Update runtime used for finding feasible solution
                if obj_bst <= 0:
                    model.__dict__['time_for_feasible'] = run_time

                # Terminate optimization if objective unchanged for tolerance period
                if run_time - model.__dict__['final_improvement_time'] >= model.__dict__['unchanged_tolerance']:
                    logfile = model.Params.LogFile
                    with open(logfile, 'a') as f:
                        print('======================================================================================\n'
                              'Terminating optimization due to unchanged objective.\n'
                              '======================================================================================\n',
                              file=f)
                    model.terminate()
            model.__dict__['last_time'] = run_time
            model.__dict__['last_obj'] = obj_bst
