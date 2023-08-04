# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:38:39 2023

Script for prototyping the operating rooms allocation model with 5 ORs and
20 patients. 

############ Note d is added to alpha in objective which is contrary to paper.
############ I believe there is some inconsistent notation in the paper.
############ K_1*rho is supposed to be a fixed daily deterioration cost associated
############ time until operation occurs. In the paper they have d - alpha
############ which is negative given their data generation scheme
############ -- adding alpha seems to make more sense. 
############ This occurs in data generation aswell for determing health status 
############ and determining mandatory patients - Mitch

@author: mitch
"""
import time
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

# Formatting
UNDERLINE = "\n" + 80*"="

NUM_ROOMS = 5
NUM_PATIENTS = 160
NUM_HOSPITALS = 3
NUM_DAYS = 5

K_1 = 50
K_2 = 5

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

model = gp.Model()

# Variables
# 1 if patient p assigned to OR r in hospital h on day d
x = {(h, d, p, r): model.addVar(vtype=GRB.BINARY) for h in H for d in D for p in P
     for r in R}

# 1 if surgical suite in hospital h is opened on day d
u = {(h, d): model.addVar(vtype=GRB.BINARY) for h in H for d in D}

# 1 if OR r in hospital h open on day d
y = {(h, d, r): model.addVar(vtype=GRB.BINARY) for h in H for d in D for r in R}

# 1 if patient does not get surgery within time horizon
w = {p: model.addVar(vtype=GRB.BINARY) for p in P}

# Objective
model.setObjective(quicksum(G[h, d]*u[h, d] for h in H for d in D)
                   + quicksum(F[h, d]*y[h, d, r] for h in H for d in D for r in R)
                   + quicksum(K_1*rho[p]*(d + alpha[p])*x[h, d, p, r] 
                              for h in H for d in D for p in P for r in R)
                   + quicksum(K_2*rho[p]*(NUM_DAYS + 1 + alpha[p])*w[p] 
                              for p in P if p not in mandatory_P), GRB.MINIMIZE)

# Constraints
mandatory_operations = {p: model.addConstr(
    quicksum(x[h, d, p, r] for h in H for d in D for r in R) == 1) 
    for p in mandatory_P}

turn_on_w = {p: model.addConstr(
    quicksum(x[h, d, p, r] for h in H for d in D for r in R) + w[p] == 1) 
    for p in P if p not in mandatory_P}

time_for_op = {(h, d, r): model.addConstr(
    quicksum(T[p]*x[h, d, p, r] for p in P) <= B[h, d]*y[h, d, r]) 
    for h in H for d in D for r in R}

# Reduces amount of combinations to check
OR_symmetries = {(h, d, r): model.addConstr(y[h, d, r] <= y[h, d, r - 1]) 
                 for h in H for d in D for r in R[1:]}

suite_open_before_OR = {(h, d, r): model.addConstr(y[h, d, r] <= u[h, d]) 
                        for h in H for d in D for r in R}

lp_strengthener = {(h, d, p, r): model.addConstr(x[h, d, p, r] <= y[h, d, r]) 
                   for h in H for d in D for p in P for r in R}

start_time = time.time()
model.optimize()
end_time = time.time()

print("\n")
print("Optimal objective value:", model.objVal)
print("Ran in", end_time - start_time, "seconds")