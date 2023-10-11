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
from constants import UNDERLINE, LBBD_1, LBBD_2, LBBD_PLUS

SEEDS = (42, 831, 306, 542, 1)
NUM_PATIENTS = 40
NUM_OR = 5
GAP = 0.00
DISPLAY_ALLOCS = True

P, mand_P, H, R, D, rho, alpha, health_status, B, T, G, F = generate_data(
    NUM_PATIENTS, NUM_OR, output_dict=True, verbose=False, seed=SEEDS[0])

loop = BendersLoopScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P, 
                   gurobi_log=False, gap=GAP, verbose=False, chosen_lbbd=LBBD_1,
                   bend_gap=True, use_propagation=True)

loop.run_model()

print("Objective value:", loop.model.objVal)
print("Time spent in master problem:", loop.mp_time, "seconds")
print("Time spent in sub problems:", loop.sp_time, "seconds")

if DISPLAY_ALLOCS:
    print("Mand patients", mand_P)
    for d in D:
        print("Day", d, ":")
        for h in H:
            if loop.u[h, d].x > 0.5:
                print("\tHospital", h, "open time", B[h, d])
                for r in R:
                    if loop.sub_room_allocs[h, d][r] > 0.5:
                        print("\t\tRoom", r, "total used time", 
                              sum(loop.sub_patient_allocs[h, d][p, r]*T[p] for p in P))
                        print("\t\tMandatory patients")
                        for p in mand_P:
                            if loop.sub_patient_allocs[h, d][p, r] > 0.5:
                                print("\t\t", p, "time", T[p])
                                
                        print("\t\tOptional patients")
                        for p in P:
                            if (p not in mand_P 
                                and loop.sub_patient_allocs[h, d][p, r] > 0.5):
                                print("\t\t",p, "time", T[p])
