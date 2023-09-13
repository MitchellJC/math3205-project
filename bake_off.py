# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:19:35 2023

@author: mitch
"""
import time
import pandas as pd
import statistics
from models import MIPScheduler, BendersLoopScheduler, BendersCallbackScheduler
from data_gen import generate_data
from constants import UNDERLINE, LBBD_PLUS

SEEDS = (42, 831, 306, 542, 1)
NUM_PATIENTS = (20,)
NUM_OR = 5
GAP = 0.00 # 0.01

instance_nums = []
seeds = []
num_patients_full = []

mip_obj_vals = []
mip_times = []

loop_obj_vals = []
loop_times = []

callback_obj_vals = []
callback_times = []

for num_patients in NUM_PATIENTS:
    print(UNDERLINE, "\n Num patients", num_patients, UNDERLINE)
    for instance, seed in enumerate(SEEDS):
        print(UNDERLINE, "\nInstance", instance, UNDERLINE)
    
        instance_nums.append(instance)
        seeds.append(seed)
        num_patients_full.append(num_patients)
        
        P, mand_P, H, R, D, rho, alpha, health_status, B, T, G, F = generate_data(
            num_patients, NUM_OR, output_dict=True, verbose=False, seed=seed)
        
        # MIP
        print(f"Solving with pure MIP")
        mip = MIPScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P, 
                           gurobi_log=False, gap=GAP)
        start_time = time.time()
        mip.run_model()
        run_time = time.time() - start_time
        mip_obj_vals.append(mip.model.objVal)
        mip_times.append(run_time)
        
        # Benders' Loop
        print(f"Solving with Benders' loop")
        benders_loop = BendersLoopScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P,
                                            verbose=False, gurobi_log=False, gap=GAP,
                                            chosen_lbbd=LBBD_PLUS)
        start_time = time.time()
        benders_loop.run_model()
        run_time = time.time() - start_time
        loop_obj_vals.append(benders_loop.model.objVal)
        loop_times.append(run_time)
        # loop_obj_vals.append(0)
        # loop_times.append(0)
        
        # Benders' Callback
        print(f"Solving with Benders' callback")
        benders_callback = BendersCallbackScheduler(P, H, R, D, G, F, B, T, rho, alpha, 
                                                    mand_P, verbose=False, 
                                                    gurobi_log=False, gap=GAP)
        start_time = time.time()
        benders_callback.run_model()
        run_time = time.time() - start_time
        callback_obj_vals.append(benders_callback.model.objVal)
        callback_times.append(run_time)

# Time output
print("\nTime Output", UNDERLINE)
columns = {'instance': instance_nums, 'seed': seeds, 'num_patients': num_patients_full, 
           'MIP': mip_times, 
           'loop': loop_times, 'callback': callback_times}
df = pd.DataFrame(columns)
print(df)

# Objective output
print("\nObjective Output", UNDERLINE)
columns = {'instance': instance_nums, 'seed': seeds, 'num_patients': num_patients_full, 
           'MIP': mip_obj_vals, 
           'loop': loop_obj_vals, 'callback': callback_obj_vals}
df = pd.DataFrame(columns)
print(df)

# Average time output
print("\nAverage Time Output", UNDERLINE)
columns = {'num_patients': NUM_PATIENTS, 'MIP': statistics.mean(mip_times),
           'loop': statistics.mean(loop_times), 
           'callback': statistics.mean(callback_times)}
df = pd.DataFrame(columns)
print(df)
