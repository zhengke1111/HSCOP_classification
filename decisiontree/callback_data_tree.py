'''
Author: zhengke 1604367740@qq.com
Date: 2024-11-11 04:56:57
LastEditors: zhengke 1604367740@qq.com
LastEditTime: 2024-12-01 14:18:26
FilePath: /AHC_max_accuracy/ahc_packages/call_back_data.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# log_data: list of dictionary, each dictionary contains: 
# - 'Incumbent': current_obj,
# - 'BestBd': current_best_bd,
# - 'Gap': current_optimality_gap,
# - 'Time': current_time,
# - (Full MIP)'final_improvement_time': model.__dict__['final_improvement_time'],
# - (Full MIP)'final_improvement_gap': model.__dict__['final_improvement_gap'] 
log_data = []

# time_limit: integer, an single PIP iteration breaks if the objective value doesn't change for timelimit
timelimit=30
full_mip_timelimit = 3600
mip_timelimit = 300
time_for_finding_feasible_solution = 0
full_mip_for_feasible = False
