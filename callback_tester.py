# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:49:19 2023

@author: mitch
"""

import time
import pandas as pd
import statistics
from models import MIPScheduler, BendersLoopScheduler, BendersCallbackScheduler
from data_gen import generate_data
from constants import UNDERLINE, LBBD_2, LBBD_PLUS

SEED = 42
NUM_PATIENTS = 20
NUM_OR = 5
GAP = 0.00
DISPLAY_ALLOCS = True

P, mand_P, H, R, D, rho, alpha, health_status, B, T, G, F = generate_data(
    NUM_PATIENTS, NUM_OR, output_dict=True, verbose=False, seed=SEED)

callback = BendersCallbackScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P, 
                   gurobi_log=True, gap=GAP, verbose=False, chosen_lbbd=LBBD_2,
                   bend_gap=True, use_propagation=True)

callback.run_model()

print("Objective value:", callback.model.objVal)
print("Time spent in master problem:", callback.mp_time, "seconds")
print("Time spent in sub problems:", callback.sp_time, "seconds")

if DISPLAY_ALLOCS:
    print("Mand patients", mand_P)
    for d in D:
        print("Day", d, ":")
        for h in H:
            if callback.u[h, d].x > 0.5:
                print("\tHospital", h, "open time", B[h, d])
                for r in R:
                    if callback.sub_room_allocs[h, d][r] > 0.5:
                        print("\t\tRoom", r, "total used time", 
                              sum(callback.sub_patient_allocs[h, d][p, r]*T[p] 
                                  for p in P))
                        print("\t\tMandatory patients")
                        for p in mand_P:
                            if callback.sub_patient_allocs[h, d][p, r] > 0.5:
                                print("\t\t", p, "time", T[p])
                                
                        print("\t\tOptional patients")
                        for p in P:
                            if (p not in mand_P 
                                and callback.sub_patient_allocs[h, d][p, r] > 0.5):
                                print("\t\t",p, "time", T[p])
