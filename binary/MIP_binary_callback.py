from binary import callback_data_binary
import gurobipy as grb
import time 

def mip_binary_callback(model, where):
    """
    Callback function for monitoring a binary MIP optimization process.

    This callback is triggered during the Gurobi optimization procedure and is
    intended to:
    1. record incumbent objective value, best bound, optimality gap, and runtime,
    2. detect when a feasible solution is first found,
    3. terminate the optimization early if the incumbent objective value remains
       unchanged for longer than a prescribed time limit.

    Specifically, at MIP callback points, the function:
    - retrieves the current incumbent objective value and best bound,
    - computes the current optimality gap,
    - appends progress information to `callback_data_binary.log_data`,
    - stores the time at which a feasible solution is first detected,
    - checks whether the incumbent objective has stagnated, and if so,
      terminates the optimization once the stagnation duration exceeds the
      prescribed callback time limit.

    Args:
        model: A Gurobi model object passed automatically to the callback.
            The function also uses `model.__dict__` to store persistent
            information across callback calls, including:
            - `last_time`: the last runtime checkpoint,
            - `last_obj`: the last recorded incumbent objective value,
            - `start_time`: the wall-clock time when the current incumbent
              value started being monitored for stagnation,
            - `time_limit`: the allowed stagnation duration before termination.

        where: An integer callback code provided by Gurobi indicating the
            current callback context. This function only acts when
            `where == grb.GRB.Callback.MIP`.

    Side Effects:
        - Updates fields in the global object `callback_data_binary`.
        - Appends progress records to `callback_data_binary.log_data`.
        - May set `callback_data_binary.time_for_finding_feasible_solution`.
        - May terminate the optimization by calling `model.terminate()`.

    Notes:
        - The stagnation check is based on the incumbent objective value.
        - Progress is logged approximately every 5 seconds of solver runtime.
        - The termination criterion uses wall-clock time (`time.time()`) rather
          than solver runtime.
    """
    if where == grb.GRB.Callback.MIP:
        # Obtain current objective value and runtime
        current_obj = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        current_best_bd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        current_optimality_gap = abs(current_obj-current_best_bd)/abs(current_obj) if current_obj!=0 else -1
        current_time = model.cbGet(grb.GRB.Callback.RUNTIME)
        # Initialize time record if not already set
        if 'last_time' not in model.__dict__:
            model.__dict__['last_time'] = current_time
            model.__dict__['last_obj'] = current_obj
            model.__dict__['start_time'] = time.time()
            model.__dict__['time_limit'] = callback_data_binary.timelimit
            callback_data_binary.log_data.append({
                'Incumbent': current_obj,
                'BestBd': current_best_bd,
                'Gap': current_optimality_gap,
                'Time': current_time
                })
        if current_obj <= 0:
            callback_data_binary.time_for_finding_feasible_solution = current_time
        
        # Check if objective value is unchanged
        if current_time - model.__dict__['last_time'] > 5:
            callback_data_binary.log_data.append({
            'Incumbent': current_obj,
            'BestBd': current_best_bd,
            'Gap': current_optimality_gap,
            'Time': current_time
            })
            
            if abs(current_obj - model.__dict__['last_obj']) < 1e-4:
                if time.time() - model.__dict__['start_time'] > model.__dict__['time_limit']:
                    print("Terminating optimization due to stagnant objective value.")
                    model.terminate()
            
            else:
                # Reset the time record if objective value changes
                model.__dict__['last_time'] = current_time
                model.__dict__['last_obj'] = current_obj
                model.__dict__['start_time'] = time.time()



def full_mip_callback(model,where):
    """
    Callback function for monitoring the full MIP optimization process.

    This callback is triggered during the Gurobi optimization procedure and is
    intended to:
    1. record incumbent objective value, best bound, optimality gap, and runtime,
    2. track the time and gap corresponding to the most recent objective
       improvement,
    3. record when a feasible solution is first found,
    4. optionally terminate the optimization immediately after a feasible
       solution is detected.

    Specifically, at MIP callback points, the function:
    - retrieves the current incumbent objective value and best bound,
    - computes the current optimality gap,
    - appends progress information to `callback_data_binary.log_data`,
    - updates the recorded "final improvement" time and gap whenever the
      incumbent objective value changes,
    - stores the runtime at which a feasible solution is first detected,
    - optionally terminates the optimization once a feasible solution is found
      if `callback_data_binary.full_mip_for_feasible` is set to `True`.

    Args:
        model: A Gurobi model object passed automatically to the callback.
            The function uses `model.__dict__` to store persistent information
            across callback calls, including:
            - `last_obj`: the last recorded incumbent objective value,
            - `last_time`: the last runtime checkpoint,
            - `final_improvement_time`: the runtime at which the most recent
              incumbent improvement occurred,
            - `final_improvement_gap`: the optimality gap at the most recent
              incumbent improvement.

        where: An integer callback code provided by Gurobi indicating the
            current callback context. This function only acts when
            `where == grb.GRB.Callback.MIP`.

    Side Effects:
        - Updates fields in the global object `callback_data_binary`.
        - Appends progress records to `callback_data_binary.log_data`.
        - May set `callback_data_binary.time_for_finding_feasible_solution`.
        - May terminate the optimization by calling `model.terminate()`.

    Notes:
        - Progress is logged approximately every 5 seconds of solver runtime.
        - The "final improvement" information refers to the most recent change
          in the incumbent objective value.
        - If `callback_data_binary.full_mip_for_feasible` is `True`, the solver
          is terminated as soon as a feasible solution is detected.
    """
    if where == grb.GRB.Callback.MIP:
        # Obtain current objective value and runtime
        current_obj = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        current_best_bd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        current_optimality_gap = abs(current_obj-current_best_bd)/abs(current_obj)
        current_time = model.cbGet(grb.GRB.Callback.RUNTIME)
        
        # Initialize time record if not already set
        if 'last_time' not in model.__dict__:
            model.__dict__['last_obj'] = current_obj
            model.__dict__['last_time'] = current_time
            model.__dict__['final_improvement_time'] = current_time
            model.__dict__['final_improvement_gap'] = current_optimality_gap

            callback_data_binary.log_data.append({
            'Incumbent': current_obj,
            'BestBd': current_best_bd,
            'Gap': current_optimality_gap,
            'Time': current_time,
            'final_improvement_time': model.__dict__['final_improvement_time'],
            'final_improvement_gap': model.__dict__['final_improvement_gap']
            })

        # Check if objective value is unchanged
        if current_time - model.__dict__['last_time'] > 5:
            if abs(current_obj - model.__dict__['last_obj']) >= 1e-4:
                model.__dict__['final_improvement_time'] = current_time
                model.__dict__['final_improvement_gap'] = current_optimality_gap

            callback_data_binary.log_data.append({
            'Incumbent': current_obj,
            'BestBd': current_best_bd,
            'Gap': current_optimality_gap,
            'Time': current_time,
            'final_improvement_time': model.__dict__['final_improvement_time'],
            'final_improvement_gap': model.__dict__['final_improvement_gap']
            })
            model.__dict__['last_time'] = current_time
            model.__dict__['last_obj'] = current_obj
            

        if current_obj <= 0:
            callback_data_binary.time_for_finding_feasible_solution = current_time

        if callback_data_binary.full_mip_for_feasible == True:
            if current_obj > 0:
                print("Feasible Solution Found.")
                model.terminate()


def full_mip_callback_t300(model,where):
    """
    Callback function for monitoring the full MIP optimization process with
    early termination based on incumbent stagnation.

    This callback is triggered during the Gurobi optimization procedure and is
    intended to:
    1. record incumbent objective value, best bound, optimality gap, and runtime,
    2. track the time and gap corresponding to the most recent objective
       improvement,
    3. detect prolonged stagnation of the incumbent objective value,
    4. terminate the optimization early if no improvement is observed for longer
       than a prescribed time limit,
    5. record when a feasible solution is first found.

    Specifically, at MIP callback points, the function:
    - retrieves the current incumbent objective value and best bound,
    - computes the current optimality gap,
    - appends progress information to `callback_data_binary.log_data`,
    - updates the recorded "final improvement" time and gap whenever the
      incumbent objective value changes,
    - resets the stagnation timer when an improvement is detected,
    - terminates the optimization if the incumbent objective remains unchanged
      for longer than `callback_data_binary.full_mip_timelimit_t300`,
    - stores the runtime at which a feasible solution is first detected.

    Args:
        model: A Gurobi model object passed automatically to the callback.
            The function uses `model.__dict__` to store persistent information
            across callback calls, including:
            - `last_obj`: the last recorded incumbent objective value,
            - `last_time`: the last runtime checkpoint,
            - `final_improvement_time`: the runtime at which the most recent
              incumbent improvement occurred,
            - `final_improvement_gap`: the optimality gap at the most recent
              incumbent improvement,
            - `start_time`: the wall-clock time when the current stagnation
              period began,
            - `time_limit`: the maximum allowed stagnation duration before
              termination.

        where: An integer callback code provided by Gurobi indicating the
            current callback context. This function only acts when
            `where == grb.GRB.Callback.MIP`.

    Side Effects:
        - Updates fields in the global object `callback_data_binary`.
        - Appends progress records to `callback_data_binary.log_data`.
        - May set `callback_data_binary.time_for_finding_feasible_solution`.
        - May terminate the optimization by calling `model.terminate()`.

    Notes:
        - Progress is logged approximately every 5 seconds of solver runtime.
        - The "final improvement" information refers to the most recent change
          in the incumbent objective value.
        - The stagnation test is based on wall-clock time (`time.time()`),
          rather than solver runtime.
        - The stagnation threshold is given by
          `callback_data_binary.full_mip_timelimit_t300`.
    """
    if where == grb.GRB.Callback.MIP:
        # Obtain current objective value and runtime
        current_obj = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        current_best_bd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        current_optimality_gap = abs(current_obj-current_best_bd)/abs(current_obj)
        current_time = model.cbGet(grb.GRB.Callback.RUNTIME)
        
        # Initialize time record if not already set
        if 'last_time' not in model.__dict__:
            model.__dict__['last_obj'] = current_obj
            model.__dict__['last_time'] = current_time
            model.__dict__['final_improvement_time'] = current_time
            model.__dict__['final_improvement_gap'] = current_optimality_gap
            model.__dict__['start_time'] = time.time()
            model.__dict__['time_limit'] = callback_data_binary.full_mip_timelimit_t300

            callback_data_binary.log_data.append({
            'Incumbent': current_obj,
            'BestBd': current_best_bd,
            'Gap': current_optimality_gap,
            'Time': current_time,
            'final_improvement_time': model.__dict__['final_improvement_time'],
            'final_improvement_gap': model.__dict__['final_improvement_gap']
            })

        # Check if objective value is unchanged
        if current_time - model.__dict__['last_time'] > 5:
            if abs(current_obj - model.__dict__['last_obj']) >= 1e-4:
                model.__dict__['final_improvement_time'] = current_time
                model.__dict__['final_improvement_gap'] = current_optimality_gap
                model.__dict__['start_time'] = time.time()

            else:
                if time.time() - model.__dict__['start_time'] > model.__dict__['time_limit']:
                    print("Terminating optimization due to stagnant objective value.")
                    model.terminate()

            callback_data_binary.log_data.append({
            'Incumbent': current_obj,
            'BestBd': current_best_bd,
            'Gap': current_optimality_gap,
            'Time': current_time,
            'final_improvement_time': model.__dict__['final_improvement_time'],
            'final_improvement_gap': model.__dict__['final_improvement_gap']
            })
            model.__dict__['last_time'] = current_time
            model.__dict__['last_obj'] = current_obj
            

        if current_obj <= 0:
            callback_data_binary.time_for_finding_feasible_solution = current_time


def full_mip_callback_t600(model,where):
    """
    Callback function for monitoring the full MIP optimization process with
    early termination based on incumbent stagnation.
    Similar to full_mip_callback_t300

    Args:
        model: A Gurobi model object passed automatically to the callback.

        where: An integer callback code provided by Gurobi indicating the
            current callback context. This function only acts when
            `where == grb.GRB.Callback.MIP`.
    """
    if where == grb.GRB.Callback.MIP:
        # Obtain current objective value and runtime
        current_obj = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        current_best_bd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        current_optimality_gap = abs(current_obj-current_best_bd)/abs(current_obj)
        current_time = model.cbGet(grb.GRB.Callback.RUNTIME)
        
        # Initialize time record if not already set
        if 'last_time' not in model.__dict__:
            model.__dict__['last_obj'] = current_obj
            model.__dict__['last_time'] = current_time
            model.__dict__['final_improvement_time'] = current_time
            model.__dict__['final_improvement_gap'] = current_optimality_gap
            model.__dict__['start_time'] = time.time()
            model.__dict__['time_limit'] = callback_data_binary.full_mip_timelimit_t600

            callback_data_binary.log_data.append({
            'Incumbent': current_obj,
            'BestBd': current_best_bd,
            'Gap': current_optimality_gap,
            'Time': current_time,
            'final_improvement_time': model.__dict__['final_improvement_time'],
            'final_improvement_gap': model.__dict__['final_improvement_gap']
            })

        # Check if objective value is unchanged
        if current_time - model.__dict__['last_time'] > 5:
            if abs(current_obj - model.__dict__['last_obj']) >= 1e-4:
                model.__dict__['final_improvement_time'] = current_time
                model.__dict__['final_improvement_gap'] = current_optimality_gap
                model.__dict__['start_time'] = time.time()

            else:
                if time.time() - model.__dict__['start_time'] > model.__dict__['time_limit']:
                    print("Terminating optimization due to stagnant objective value.")
                    model.terminate()

            callback_data_binary.log_data.append({
            'Incumbent': current_obj,
            'BestBd': current_best_bd,
            'Gap': current_optimality_gap,
            'Time': current_time,
            'final_improvement_time': model.__dict__['final_improvement_time'],
            'final_improvement_gap': model.__dict__['final_improvement_gap']
            })
            model.__dict__['last_time'] = current_time
            model.__dict__['last_obj'] = current_obj
            

        if current_obj <= 0:
            callback_data_binary.time_for_finding_feasible_solution = current_time