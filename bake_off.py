# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:19:35 2023

@author: mitch
"""
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from datetime import datetime
from models import MIPScheduler, NetworkScheduler, BendersLoopScheduler, \
    BendersCallbackScheduler
from data_gen import generate_data
from constants import UNDERLINE, LBBD_PLUS, LBBD_1, LBBD_2, TIME_LIMIT


SEEDS = (42, 831, 306, 542, 1)
NUM_PATIENTS = (80,)
NUM_OR = 5
GAP = 0.01 # 0.01
FILE_OUTPUT = True # WARNING! Set to false for use in IPython console (Spyder)
SAVE_RESULTS = True
MAKE_PLOTS = True

def run_model(model, obj_vals, times, gaps, mp_time, sp_time):
    start_time = time.time()
    model.run_model()
    run_time = time.time() - start_time
    
    obj_vals.append(model.model.objVal)
    times.append(run_time)
    gaps.append(model.run_gap)
    mp_time.append(model.mp_time)
    sp_time.append(model.sp_time)

if FILE_OUTPUT:
    now = datetime.now()
    formatted_now = now.strftime("%d-%m-%Y_%H-%M-%S")
    sys.stdout = open(f'logs/{formatted_now}.log', 'w')

instance_nums = []
seeds = []
num_patients_full = []

model_names = ('MIP', 'Network', 'iLBBD1', 'cLBBD1', 'iLBBD2p', 'cLBBD2p', 
                'cLBBD4p')
models = lambda P, H, R, D, G, F, B, T, rho, alpha, mand_P: {
    'MIP': lambda: MIPScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P, 
                                gurobi_log=False, gap=GAP),
    'Network': lambda: NetworkScheduler(P, H, R, D, G, F, B, T, rho, alpha, 
                                        mand_P, gurobi_log=False, gap=GAP),
    'iLBBD1': lambda: BendersLoopScheduler(P, H, R, D, G, F, B, T, rho, alpha, 
                                            mand_P, verbose=False, 
                                            gurobi_log=False, gap=GAP, 
                                            chosen_lbbd=LBBD_1, 
                                            use_propagation=False),
    'cLBBD1': lambda: BendersCallbackScheduler(P, H, R, D, G, F, B, T, rho, alpha, 
                                                mand_P, verbose=False, 
                                                gurobi_log=False, gap=GAP,
                                                chosen_lbbd=LBBD_1,
                                                use_propagation=False),
    'iLBBD2p': lambda: BendersLoopScheduler(P, H, R, D, G, F, B, T, rho, alpha, 
                                            mand_P, verbose=False, 
                                            gurobi_log=False, gap=GAP, 
                                            chosen_lbbd=LBBD_2, 
                                            use_propagation=True),
    'cLBBD2p': lambda: BendersCallbackScheduler(P, H, R, D, G, F, B, T, rho, alpha, 
                                                mand_P, verbose=False, 
                                                gurobi_log=False, gap=GAP,
                                                chosen_lbbd=LBBD_2,
                                                use_propagation=True),
    'cLBBD4p': lambda: BendersCallbackScheduler(P, H, R, D, G, F, B, T, rho, alpha, 
                                                mand_P, verbose=False, 
                                                gurobi_log=False, gap=GAP,
                                                chosen_lbbd=LBBD_PLUS,
                                                use_propagation=True),
}

data = {}
for model_name in model_names:
    data[model_name] = {'obj_vals': [], 'times': [], 'gaps': [], 'mp_time': [],
                        'sp_time': []}

# Run models
for num_patients in NUM_PATIENTS:
    print(UNDERLINE, "\n Num patients", num_patients, UNDERLINE)
    for instance, seed in enumerate(SEEDS):
        print(UNDERLINE, "\nInstance", instance, UNDERLINE)
    
        instance_nums.append(instance)
        seeds.append(seed)
        num_patients_full.append(num_patients)
        
        P, mand_P, H, R, D, rho, alpha, health_status, B, T, G, F = generate_data(
            num_patients, NUM_OR, output_dict=True, verbose=False, seed=seed)
        
        for model_name in model_names:
            print(f"Solving with {model_name}")
            model = models(P, H, R, D, G, F, B, T, rho, alpha, mand_P)[model_name]()
            run_model(model, data[model_name]['obj_vals'], data[model_name]['times'], 
                      data[model_name]['gaps'], data[model_name]['mp_time'],
                      data[model_name]['sp_time'])

now = datetime.now()
formatted_now = now.strftime("%d-%m-%Y_%H-%M-%S")

# Time output
print("\nTime Output", UNDERLINE)
columns = {'instance': instance_nums, 'seed': seeds, 
           'num_patients': num_patients_full}
for model_name in model_names:
    columns[model_name] = data[model_name]['times']
df = pd.DataFrame(columns)
print(df.to_string())
if SAVE_RESULTS:
    df.to_csv(f'results_data/{formatted_now}_time.csv', index=False)

# Objective output
print("\nObjective Output", UNDERLINE)
columns = {'instance': instance_nums, 'seed': seeds, 
           'num_patients': num_patients_full}
for model_name in model_names:
    columns[model_name] = data[model_name]['obj_vals']
df = pd.DataFrame(columns)
print(df.to_string())
if SAVE_RESULTS:
    df.to_csv(f'results_data/{formatted_now}_obj.csv', index=False)

# Gap output
print("\nGap Output", UNDERLINE)
columns = {'instance': instance_nums, 'seed': seeds, 
           'num_patients': num_patients_full}
for model_name in model_names:
    columns[model_name] = data[model_name]['gaps']
df = pd.DataFrame(columns)
print(df.to_string())
if SAVE_RESULTS:
    df.to_csv(f'results_data/{formatted_now}_gap.csv', index=False)

# Average gap output
print("\nAverage Gap Output", UNDERLINE)
columns = {'num_patients': NUM_PATIENTS}
for model_name in model_names:
    columns[model_name] = statistics.mean(data[model_name]['gaps'])
df = pd.DataFrame(columns)
print(df.to_string())
if SAVE_RESULTS:
    df.to_csv(f'results_data/{formatted_now}_avg_gap.csv', index=False)

# Average time output
print("\nAverage Time Output", UNDERLINE)
columns = {'num_patients': NUM_PATIENTS}
for model_name in model_names:
    qual_times = [t for t in data[model_name]['times'] if t <= TIME_LIMIT]
    columns[model_name] = None if len(qual_times) == 0 else statistics.mean(qual_times)
df = pd.DataFrame(columns)
print(df.to_string())
if SAVE_RESULTS:
    df.to_csv(f'results_data/{formatted_now}_avg_time.csv', index=False)
        
# Time in MP output
print("\nMP Time Output", UNDERLINE)
columns = {'instance': instance_nums, 'seed': seeds, 
           'num_patients': num_patients_full}
for model_name in model_names:
    columns[model_name] = data[model_name]['mp_time']
df = pd.DataFrame(columns)
print(df.to_string())
if SAVE_RESULTS:
    df.to_csv(f'results_data/{formatted_now}_mp_time.csv', index=False)
    
# Time in SP output
print("\nSP Time Output", UNDERLINE)
columns = {'instance': instance_nums, 'seed': seeds, 
           'num_patients': num_patients_full}
for model_name in model_names:
    columns[model_name] = data[model_name]['sp_time']
df = pd.DataFrame(columns)
print(df.to_string())
if SAVE_RESULTS:
    df.to_csv(f'results_data/{formatted_now}_sp_time.csv', index=False)

# Make time in MP-SP bar plots
if MAKE_PLOTS:
    x = model_names[2:]
    y1 = [statistics.mean(data[model_name]['mp_time']) for model_name in model_names[2:]]
    y2 = [statistics.mean(data[model_name]['sp_time']) for model_name in model_names[2:]]
    plt.bar(x, y1, label='Time in MP')
    plt.bar(x, y2, bottom=y1, label='Time in SP')
    plt.legend()
    plt.title(f'Average Time in MP and SP for {NUM_PATIENTS[0]} patients')
    plt.xlabel('Model')
    plt.ylabel('Total time (seconds)')
    plt.ylim([0, plt.gca().get_ylim()[1]*1.25])
    plt.savefig(f'plots/{formatted_now}({NUM_PATIENTS[0]}_timeinMPSP)')

if FILE_OUTPUT:
    sys.stdout.close()