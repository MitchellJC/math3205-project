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
from constants import UNDERLINE

SEEDS = (42, 831, 306, 542, 1)
NUM_PATIENTS = 20
NUM_OR = 5
GAP = 0.00
DISPLAY_ALLOCS = True

P, mand_P, H, R, D, rho, alpha, health_status, B, T, G, F = generate_data(
    NUM_PATIENTS, NUM_OR, output_dict=True, verbose=False, seed=SEEDS[0])

mip = MIPScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P, 
                   gurobi_log=True, gap=GAP)

mip.run_model()

print("Objective value:", mip.model.objVal)

if DISPLAY_ALLOCS:
    print("Mand patients", mand_P)
    for d in D:
        print("Day", d, ":")
        for h in H:
            if mip.u[h, d].x > 0.5:
                print("\tHospital", h, "open time", B[h, d])
                for r in R:
                    if mip.y[h, d, r].x > 0.5:
                        print("\t\tRoom", r, "total used time", 
                              sum(mip.x[h, d, p, r].x*T[p] for p in P))
                        print("\t\tMandatory patients")
                        for p in mand_P:
                            if mip.x[h, d, p, r].x > 0.5:
                                print("\t\t", p, "time", T[p])
                                
                        print("\t\tOptional patients")
                        for p in P:
                            if p not in mand_P and mip.x[h, d, p, r].x > 0.5:
                                print("\t\t",p, "time", T[p])