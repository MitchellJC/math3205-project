# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:19:35 2023

@author: mitch
"""
import sys
import time
import pandas as pd
import statistics
from datetime import datetime
from models import MIPScheduler, BendersLoopScheduler, BendersCallbackScheduler
from data_gen import generate_data
from constants import UNDERLINE, LBBD_PLUS, LBBD_1, TIME_LIMIT

SEEDS = (42, 831, 306, 542, 1)
NUM_PATIENTS = (20,)
NUM_OR = 5
GAP = 0.00 # 0.01
FILE_OUTPUT = True # WARNING! Set to false for use in IPython console (Spyder)
TOL = 0.00001

if FILE_OUTPUT:
    now = datetime.now()
    formatted_now = now.strftime("%d-%m-%Y_%H-%M-%S")
    sys.stdout = open(f'logs/{formatted_now}.log', 'w')


instance_nums = []
seeds = []
num_patients_full = []

# MIP data
mip_obj_vals = []
mip_times = []
mip_gaps = []

# Loop data
loop_obj_vals = []
loop_times = []
loop_gaps = []

# Callback data
callback_obj_vals = []
callback_times = []
callback_gaps = []

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
        mip_gaps.append(mip.model.MIPGap)
        
        # Benders' Loop
        print(f"Solving with Benders' loop")
        benders_loop = BendersLoopScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P,
                                            verbose=False, gurobi_log=False, gap=GAP,
                                            chosen_lbbd=LBBD_1,
                                            use_propagation=False)
        start_time = time.time()
        benders_loop.run_model()
        run_time = time.time() - start_time
        loop_obj_vals.append(benders_loop.model.objVal)
        loop_times.append(run_time)
        loop_gaps.append(benders_loop.model.MIPGap)
        
        # Benders' Callback
        print(f"Solving with Benders' callback")
        benders_callback = BendersCallbackScheduler(P, H, R, D, G, F, B, T, rho, alpha, 
                                                    mand_P, verbose=False, 
                                                    gurobi_log=False, gap=GAP,
                                                    chosen_lbbd=LBBD_1,
                                                    use_propagation=False)
        start_time = time.time()
        benders_callback.run_model()
        run_time = time.time() - start_time
        callback_obj_vals.append(benders_callback.model.objVal)
        callback_times.append(run_time)
        callback_gaps.append(benders_callback.model.MIPGap)

# Time output
print("\nTime Output", UNDERLINE)
columns = {'instance': instance_nums, 'seed': seeds, 'num_patients': num_patients_full, 
           'MIP': mip_times, 'loop': loop_times, 'callback': callback_times}
df = pd.DataFrame(columns)
print(df)

# Objective output
print("\nObjective Output", UNDERLINE)
columns = {'instance': instance_nums, 'seed': seeds, 'num_patients': num_patients_full, 
           'MIP': mip_obj_vals, 'loop': loop_obj_vals, 'callback': callback_obj_vals}
df = pd.DataFrame(columns)
print(df)

# Gap output
print("\nGap Output", UNDERLINE)
columns = {'instance': instance_nums, 'seed': seeds, 'num_patients': num_patients_full, 
           'MIP': mip_gaps, 'loop': loop_gaps, 'callback': callback_gaps}
df = pd.DataFrame(columns)
print(df)

# Average time output
print("\nAverage Time Output", UNDERLINE)
columns = {'num_patients': NUM_PATIENTS, 
           'MIP': statistics.mean([t for t in mip_times if t < TIME_LIMIT]),
           'loop': statistics.mean([t for t in loop_times if t < TIME_LIMIT]), 
           'callback': statistics.mean([t for t in callback_times if t < TIME_LIMIT])}
df = pd.DataFrame(columns)
print(df)

if FILE_OUTPUT:
    sys.stdout.close()