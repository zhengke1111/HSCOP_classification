import gurobipy as grb
import time
import callback_data_tree

def partial_model_callback(model, where):
    """ callback function for MIP including PIP and full_MIP(if necesary), mainly used to record the solver log and control terminating time

    Args:
        model (gurobipy model): _description_
        where (grb.GRB.Callback.MIP): _description_
    """
    if where == grb.GRB.Callback.MIP:
        # Obtain current objective value and runtime
        current_obj = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        current_best_bd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        current_optimality_gap = abs(current_obj-current_best_bd)/abs(current_obj) if current_obj!=0 else -1
        current_time = model.cbGet(grb.GRB.Callback.RUNTIME)
        
        # Initialize time record if not already set
        if current_time > callback_data_tree.mip_timelimit:
            model.terminate()
        if 'last_time' not in model.__dict__:
            model.__dict__['last_time'] = current_time
            model.__dict__['last_obj'] = current_obj
            model.__dict__['start_time'] = current_time
            model.__dict__['time_limit'] = callback_data_tree.timelimit
            callback_data_tree.log_data.append({
                'Incumbent': current_obj,
                'BestBd': current_best_bd,
                'Gap': current_optimality_gap,
                'Time': current_time
                })
        if current_obj <= 0:
            callback_data_tree.time_for_finding_feasible_solution = current_time
        if current_time - model.__dict__['last_time'] > 5:
            callback_data_tree.log_data.append({
            'Incumbent': current_obj,
            'BestBd': current_best_bd,
            'Gap': current_optimality_gap,
            'Time': current_time
            })
            if current_obj>0:
                if abs(current_obj - model.__dict__['last_obj']) < 1e-4:
                    if current_time - model.__dict__['start_time'] > model.__dict__['time_limit']:
                        print("Terminating optimization due to stagnant objective value.")
                        model.terminate()
                else:
                    # Reset the time record if objective value changes
                    model.__dict__['last_time'] = current_time
                    model.__dict__['last_obj'] = current_obj
                    model.__dict__['start_time'] = current_time
            else:   
                if abs(current_obj - model.__dict__['last_obj']) < 1e4:
                    if current_time - model.__dict__['start_time'] > model.__dict__['time_limit']:
                        print("Terminating optimization due to stagnant objective value.")
                        model.terminate()
                else:
                    # Reset the time record if objective value changes
                    model.__dict__['last_time'] = current_time
                    model.__dict__['last_obj'] = current_obj
                    model.__dict__['start_time'] = current_time

    if where == grb.GRB.Callback.MIPSOL:
        x_values = model.cbGetSolution(model._vars)
        for i, x in enumerate(x_values):
            if model._vars[i].vtype in [grb.GRB.BINARY]:
                if abs(x - round(x)) > 1e-9:
                    model.cbLazy(model._vars[i] <= 0)


def full_model_callback(model, where):
    """
    :param model: model well constructed before optimize()
    :param where: just call in this way: model.optimize(mip_callback)
    :return: 
    """
    if where == grb.GRB.Callback.MIP:
        # Obtain current objective value and runtime
        current_obj = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        current_best_bd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        current_optimality_gap = abs(current_obj-current_best_bd)/abs(current_obj)
        current_time = model.cbGet(grb.GRB.Callback.RUNTIME)
        if current_time > callback_data_tree.full_mip_timelimit:
            model.terminate()
        # Initialize time record if not already set
        if 'last_time' not in model.__dict__:
            model.__dict__['last_obj'] = current_obj
            model.__dict__['last_time'] = current_time
            model.__dict__['final_improvement_time'] = current_time
            model.__dict__['final_improvement_gap'] = current_optimality_gap
            callback_data_tree.log_data.append({
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
            callback_data_tree.log_data.append({
            'Incumbent': current_obj,
            'BestBd': current_best_bd,
            'Gap': current_optimality_gap,
            'Time': current_time,
            'final_improvement_time': model.__dict__['final_improvement_time'],
            'final_improvement_gap': model.__dict__['final_improvement_gap']
            })
            model.__dict__['last_time'] = current_time
            model.__dict__['last_obj'] = current_obj

        if callback_data_tree.full_mip_for_feasible == True:
            if current_obj > 0:
                print("Feasible Solution Found.")
                model.terminate()

    if where == grb.GRB.Callback.MIPSOL:
        x_values = model.cbGetSolution(model._vars)
        for i, x in enumerate(x_values):
            if model._vars[i].vtype in [grb.GRB.BINARY]:
                if abs(x - round(x)) > 1e-9:
                    model.cbLazy(model._vars[i] <= 0)