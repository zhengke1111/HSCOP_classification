from gurobipy import GRB


def full_model_callback(model, where):
    """
    Gurobi callback for full MIP: tracks optimality gap, improvement time, and first feasible time.
    Updates metrics every 3 seconds via model.__dict__.
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
    Gurobi callback for partial MIP: tracks improvement time and terminates early if objective is unchanged.
    Checks every 3 seconds; terminates if unchanged for longer than unchanged_tolerance.
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