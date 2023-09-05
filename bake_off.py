# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:19:35 2023

@author: mitch
"""
from models import MIPScheduler, BendersLoopScheduler, BendersCallbackScheduler
from data_gen import generate_data

NUM_PATIENTS = 20
NUM_OR = 5

P, mand_P, H, R, D, rho, alpha, health_status, B, T, G, F = generate_data(
    NUM_PATIENTS, NUM_OR, output_dict=True, verbose=False, seed=42)

mip = MIPScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P)
mip.run_model()

benders_loop = BendersLoopScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P,
                                    verbose=False, gurobi_log=False)
benders_loop.run_model()


benders_callback = BendersCallbackScheduler(P, H, R, D, G, F, B, T, rho, alpha, 
                                            mand_P, verbose=False, 
                                            gurobi_log=False)
benders_callback.run_model()

print(mip.model.objVal)
print(benders_loop.model.objVal)
print(benders_callback.model.objVal)