# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:49:19 2023

@author: mitch
"""

import time
import pandas as pd
import statistics
from models import NetworkScheduler
from data_gen import generate_data
from constants import UNDERLINE

SEEDS = (42, 831, 306, 542, 1)
NUM_PATIENTS = 20
NUM_OR = 5
GAP = 0.00
DISPLAY_ALLOCS = True

P, mand_P, H, R, D, rho, alpha, health_status, B, T, G, F = generate_data(
    NUM_PATIENTS, NUM_OR, output_dict=True, verbose=False, seed=SEEDS[0])

m = NetworkScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P, 
                   gurobi_log=True, gap=GAP)

m.run_model()
print("Objective value:", m.model.objVal)


if DISPLAY_ALLOCS:
    print("Mand patients", mand_P)
    for d in D:
        print("Day", d, ":")
        for h in H:
            if m.u[h, d].x > 0.5:
                print("\tHospital", h, "open time", B[h, d])
                print("\t\tNum rooms open", m.y[h, d].x, 
                      "Total room time", m.y[h, d].x*B[h, d])
                print("\t\tNum Mandatory patients", 
                      sum(m.x[h, d, p].x for p in mand_P))
                print("\t\tTotal Time of Mandatory Patients", 
                      sum(T[p] for p in mand_P if m.x[h, d, p].x > 0.5))
                
                print("\t\tNum Optional patients", 
                      sum(m.x[h, d, p].x for p in P if p not in mand_P))
                print("\t\tTotal Time of Optional Patients", 
                      sum(T[p] for p in P if  p not in mand_P and m.x[h, d, p].x > 0.5))
        
                