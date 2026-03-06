from binary import callback_data_binary
import gurobipy as grb
import time 

def mip_binary_callback(model, where):
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


def full_mip_callback1(model,where):
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
            model.__dict__['time_limit'] = callback_data_binary.full_mip_timelimit1

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


def full_mip_callback2(model,where):
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
            model.__dict__['time_limit'] = callback_data_binary.full_mip_timelimit2

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