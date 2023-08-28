# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:38:39 2023

Simple loop for Bender's Decomp.

@author: mitch
"""
import time
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum


# General constants
LBBD_1 = "LBBD_1"
LBBD_2 = "LBBD_2"

# Formatting
UNDERLINE = "\n" + 80*"="

# Script parameters
EPS = 1e6
TIME_LIMIT = 7200

NUM_ROOMS = 5
NUM_PATIENTS = 20
NUM_HOSPITALS = 3
NUM_DAYS = 5

K_1 = 50
K_2 = 5

CHOSEN_LBBD = LBBD_2

# Load csv files
patients = pd.read_csv(f'data/patients-{NUM_ROOMS}-{NUM_PATIENTS}.csv')
hospitals = pd.read_csv(f'data/hospitals-{NUM_ROOMS}-{NUM_PATIENTS}.csv')
print("Data Preview" + UNDERLINE)
print(patients.head())
print(hospitals.head())

# Sets
P = range(NUM_PATIENTS)
H = range(NUM_HOSPITALS)
R = range(NUM_ROOMS)
D = range(NUM_DAYS)

# Data
# Cost of opening hospital operating suite
G = {(h, d): float(hospitals[(hospitals['hospital_id'] == h) 
                             & (hospitals['day'] == d)]['hospital_open_cost']) 
     for h in H for d in D}

# Cost of opening operating room
F = {(h, d): float(hospitals[(hospitals['hospital_id'] == h) 
                             & (hospitals['day'] == d)]['or_open_cost']) 
     for h in H for d in D}

# Operating minutes of hospital
B = {(h, d): float(hospitals[(hospitals['hospital_id'] == h) 
                             & (hospitals['day'] == d)]['open_minutes']) 
     for h in H for d in D}

# Surgery times
T = {p: float(patients[patients['id'] == p]['surgery_time']) for p in P}

# Urgency
rho = {p: int(patients[patients['id'] == p]['urgency']) for p in P}

# Days elapsed since referral
alpha = {p: int(patients[patients['id'] == p]['wait_time']) for p in P}

mandatory_P = [p for p in P if patients.loc[p, 'is_mandatory'] == 1]

MP = gp.Model()

# Variables
# 1 if patient p assigned to OR r in hospital h on day d
x = {(h, d, p): MP.addVar(vtype=GRB.BINARY) for h in H for d in D for p in P}

# 1 if surgical suite in hospital h is opened on day d
u = {(h, d): MP.addVar(vtype=GRB.BINARY) for h in H for d in D}

# number of OR r in hospital h open on day d
y = {(h, d): MP.addVar(vtype=GRB.INTEGER, lb=0, ub=len(R)) for h in H for d in D}

# 1 if patient does not get surgery within time horizon
w = {p: MP.addVar(vtype=GRB.BINARY) for p in P if p not in mandatory_P}

# Objective
MP.setObjective(quicksum(G[h, d]*u[h, d] for h in H for d in D)
                   + quicksum(F[h, d]*y[h, d] for h in H for d in D)
                   + quicksum(K_1*rho[p]*(d - alpha[p])*x[h, d, p] 
                              for h in H for d in D for p in P)
                   + quicksum(K_2*rho[p]*(NUM_DAYS + 1 - alpha[p])*w[p]
                              for p in P if p not in mandatory_P), GRB.MINIMIZE)

# Constraints
mandatory_operations = {p: MP.addConstr(
    quicksum(x[h, d, p] for h in H for d in D) == 1) 
    for p in mandatory_P}

turn_on_w = {p: MP.addConstr(
    quicksum(x[h, d, p] for h in H for d in D) + w[p] == 1) 
    for p in P if p not in mandatory_P}

lp_strengthener = {(h, d, p): MP.addConstr(x[h, d, p] <= u[h, d]) 
                   for h in H for d in D for p in P}

time_for_ops_in_hosp = {(h, d): MP.addConstr(
    quicksum(T[p]*x[h, d, p] for p in P) <= len(R)*B[h, d]*u[h, d]) 
    for h in H for d in D}

no_single_long_op = {(h, d, p): MP.addConstr(T[p]*x[h, d, p] <= B[h, d]) 
                     for h in H for d in D for p in P}

num_or_lb = {(h, d): MP.addConstr(y[h, d]*B[h, d] 
                                  >= quicksum(T[p]*x[h, d, p] for p in P))
            for h in H for d in D}

MP.setParam('OutputFlag', 1)
MP.setParam('MIPGap', 0)
MP.setParam('MIPFocus', 3)
MP.setParam('Heuristics', 0)
MP.setParam('TimeLimit', TIME_LIMIT)

start_time = time.time()
iterations = 0
while True:
    print("Iteration", iterations)
    iterations += 1
    MP.optimize()
    print("Curr objVal", MP.objVal)
    
    cuts_added = 0
    for h in H:
        for d in D:
            print("Hospital", h, "Day", d, end=" ")
            # Set of patients assigned to this hospital and day.
            P_prime = [p for p in P if x[h, d, p].x == 1]
            
            SP = gp.Model()
            SP.setParam('OutputFlag', 0)
            SP.setParam('MIPGap', 0)
            
            
            # Variables
            y_prime = {r: SP.addVar(vtype=GRB.BINARY) for r in R}
            x_prime = {(p, r): SP.addVar(vtype=GRB.BINARY) for p in P_prime 
                       for r in R}
            
            # Objective
            SP.setObjective(quicksum(y_prime[r] for r in R), GRB.MINIMIZE)
            
            # Constraints
            patients_assigned_hosp_get_room = {
                p: SP.addConstr(quicksum(x_prime[p, r] for r in R) == 1) 
                for p in P_prime}
            
            OR_capacity = {r: SP.addConstr(quicksum(T[p]*x_prime[p, r] 
                                                    for p in P_prime) 
                                           <= B[h, d]*y_prime[r]) for r in R}
            
            sub_lp_strengthener = {(p, r): SP.addConstr(x_prime[p, r] 
                                                        <= y_prime[r]) 
                               for p in P_prime for r in R}
            
            OR_symmetries = {r: SP.addConstr(y_prime[r] <= y_prime[r - 1]) 
                             for r in R[1:]}
            
            SP.optimize()
            
            if SP.Status == GRB.OPTIMAL:
                num_open_or = sum(y_prime[r].x for r in R)
                
            if SP.Status != GRB.OPTIMAL:
                print("Infeasible, status code:", SP.Status)
                if CHOSEN_LBBD == LBBD_1:
                    MP.addConstr(quicksum(1 - x[h, d, p] for p in P_prime) >= 1)
                if CHOSEN_LBBD == LBBD_2:
                    MP.addConstr(y[h, d] >= len(R) + 1 - quicksum(1 - x[h, d, p] 
                                                                  for p in P_prime))
                
                cuts_added += 1
            elif num_open_or == y[h, d].x:
                print(f"Upper bound = Lower bound, {num_open_or} = {y[h, d].x}")
                
            elif num_open_or > y[h, d].x:
                print(f"Upper bound > Lower bound, {num_open_or} > {y[h, d].x}")
                MP.addConstr(y[h, d] >= num_open_or - quicksum(1 - x[h, d, p] 
                                                               for p in P_prime))
                cuts_added += 1
            elif num_open_or < y[h, d].x - EPS:
                raise RuntimeError(f"Sub problem < Master problem!, {num_open_or} < {y[h, d].x}")
                        
    print("Cuts added", cuts_added)
    if cuts_added == 0:
        break

end_time = time.time()

print("\n")
print("Optimal objective value:", MP.objVal)
print("Ran in", end_time - start_time, "seconds")
print(iterations, "iterations")