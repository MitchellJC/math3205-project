# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:19:35 2023

@author: mitch
"""
import time
import pandas as pd
from models import MIPScheduler, BendersLoopScheduler, BendersCallbackScheduler
from data_gen import generate_data

NUM_PATIENTS = (20,)
SEEDS = (42, 831, 316)
NUM_OR = 5

mip_obj_vals = []
mip_times = []

loop_obj_vals = []
loop_times = []

callback_obj_vals = []
callback_times = []

for i in range(len(NUM_PATIENTS)):
    num_patients = NUM_PATIENTS[i]
    seed = SEEDS[i]
    
    P, mand_P, H, R, D, rho, alpha, health_status, B, T, G, F = generate_data(
        num_patients, NUM_OR, output_dict=True, verbose=False, seed=seed)
    
    # MIP
    mip = MIPScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P, gurobi_log=False)
    start_time = time.time()
    mip.run_model()
    run_time = time.time() - start_time
    mip_obj_vals.append(mip.model.objVal)
    mip_times.append(run_time)
    
    # Benders' Loop
    benders_loop = BendersLoopScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P,
                                        verbose=False, gurobi_log=False)
    start_time = time.time()
    benders_loop.run_model()
    run_time = time.time() - start_time
    loop_obj_vals.append(benders_loop.model.objVal)
    loop_times.append(run_time)
    
    # Benders' Callback
    benders_callback = BendersCallbackScheduler(P, H, R, D, G, F, B, T, rho, alpha, 
                                                mand_P, verbose=False, 
                                                gurobi_log=False)
    start_time = time.time()
    benders_callback.run_model()
    run_time = time.time() - start_time
    callback_obj_vals.append(benders_callback.model.objVal)
    callback_times.append(run_time)
    
columns = {'seed': SEEDS[:len(NUM_PATIENTS)], 'num_patients': NUM_PATIENTS, 'MIP': mip_times, 
           'loop': loop_times, 'callback': callback_times}
df = pd.DataFrame(columns)
print(df)