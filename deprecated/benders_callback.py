# -*- coding: utf-8 -*-
"""
Callback for Bender's Decomp.

@author: mitch
"""
import time
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
from utilities import ffd
from typing import Union
from constants import UNDERLINE, LBBD_1, LBBD_2, EPS, TIME_LIMIT, K_1, K_2

# Script parameters
VERBOSE = False

NUM_ROOMS = 5
NUM_PATIENTS = 20
NUM_HOSPITALS = 3
NUM_DAYS = 5

CHOSEN_LBBD = LBBD_2
USE_PROPAGATION = False

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
G = {(h, d): int(hospitals[(hospitals['hospital_id'] == h) 
                             & (hospitals['day'] == d)]['hospital_open_cost']) 
     for h in H for d in D}

# Cost of opening operating room
F = {(h, d): int(hospitals[(hospitals['hospital_id'] == h) 
                             & (hospitals['day'] == d)]['or_open_cost']) 
     for h in H for d in D}

# Operating minutes of hospital day
B = {(h, d): int(hospitals[(hospitals['hospital_id'] == h) 
                             & (hospitals['day'] == d)]['open_minutes']) 
     for h in H for d in D}

# Surgery times
T = {p: int(patients[patients['id'] == p]['surgery_time']) for p in P}

# Urgency
rho = {p: int(patients[patients['id'] == p]['urgency']) for p in P}

# Days elapsed since referral
alpha = {p: int(patients[patients['id'] == p]['wait_time']) for p in P}

mandatory_P = [p for p in P if patients.loc[p, 'is_mandatory'] == 1]

MP = gp.Model()
MP.setParam('OutputFlag', 1)
MP.setParam('LazyConstraints', 1)
MP.setParam('MIPGap', 0)
MP.setParam('MIPFocus', 3)
MP.setParam('Heuristics', 0)
MP.setParam('TimeLimit', TIME_LIMIT)

# Variables
# 1 if patient p assigned to hospital h on day d
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

def precompute_ffd(Y_hat: dict, P_prime: list, h: int, d: int) -> (
        tuple[bool, Union[int, None]]):
    """Use first-fit decreasing algorithm to prove feasible upperbound for sub
    problem. 
    
    Parameters:
        Y_hat - Master problem value for num ORs to open
        P_prime - List of patients assigned to hospital h
        h - Hospital
        d - day
        
    Returns:
        Tuple. First element is bool indicating whether sub problem needs to 
        continue. Second element is number for feasible upperbound or None if
        feasible upper bound was not found.
    """
    # Get heuristic solution
    items = [(p, T[p]) for p in P_prime]
    heur_open_rooms = ffd(items, len(R), B[h, d])
    
    # Continue sub problem based on heuristic solution
    FFD_upperbound = None
    if heur_open_rooms:
        if abs(len(heur_open_rooms) - Y_hat[h, d]) < EPS:
            if VERBOSE:
                print("FFD soln same as master problem.")
            return False, None
        elif len(heur_open_rooms) < Y_hat[h, d] - EPS:
            if VERBOSE:
                print("FFD soln better than master problem."
                      +" (Non-optimal master soln)")
            return False, None
        elif len(heur_open_rooms) > Y_hat[h, d] + EPS:
            if VERBOSE:
                print("FFD soln worst as master problem.")
            FFD_upperbound = len(heur_open_rooms)
            
    return True, FFD_upperbound

def solve_sub_ip(P_prime: list, FFD_upperbound: Union[int, None], h: int, 
                 d: int) -> tuple[gp.Model, dict]:
    """Defines ands solves to sub problem to optimality as an IP.
    
    Parameters:
        P_prime - List of patients assigned to hospital h
        h - Hospital
        d - day
        
    Returns:
        Tuple. First element containing the defined IP and second element 
        containing dictionary of binary variables indicating whether or not
        to open each operating room.
    """
    SP = gp.Model()
    SP.setParam('OutputFlag', 0)
    SP.setParam('MIPGap', 0)
    
    # Variables
    y_prime = {r: SP.addVar(vtype=GRB.BINARY) for r in R}
    x_prime = {(p, r): SP.addVar(vtype=GRB.BINARY) for p in P_prime for r in R}
    
    # Objective
    SP.setObjective(quicksum(y_prime[r] for r in R), GRB.MINIMIZE)
    
    # Constraints
    patients_assigned_hosp_get_room = {
        p: SP.addConstr(quicksum(x_prime[p, r] for r in R) == 1) 
        for p in P_prime}

    OR_capacity = {r: SP.addConstr(quicksum(T[p]*x_prime[p, r] for p in P_prime) 
                                   <= B[h, d]*y_prime[r]) for r in R}
    
    sub_lp_strengthener = {(p, r): SP.addConstr(x_prime[p, r] <= y_prime[r]) 
                           for p in P_prime for r in R}
    
    OR_symmetries = {r: SP.addConstr(y_prime[r] <= y_prime[r - 1]) for r in R[1:]}
    
    if FFD_upperbound:
        FFD_tightener = SP.addConstr(quicksum(y_prime[r] for r in R)
                                     <= FFD_upperbound)
    
    SP.optimize()
    
    return SP, y_prime

def solve_sub_problem(model: gp.Model, x_hat: dict, Y_hat: dict, 
                      cuts_container: list, h: int, d: int) -> None:
    """Solve the hospital-day sub-problem. Adds Bender's cuts to master problem
    as required.
    
    Parameters:
        model - Master problem Gurobi model.
        x_hat - Dictionary containing binary variables indicating patient 
            assignments.
        Y_hat - Dictionary containing number of operating rooms to open.
        cuts_container - List containing one element, the number of cuts added
        h - Hospital
        d - day
    """
    cuts_added = cuts_container[0]
    
    if VERBOSE:
        print("Hospital", h, "Day", d, end=" ")\
            
    # Set of patients assigned to this hospital and day.
    P_prime = [p for p in P if x_hat[h, d, p] > 0.5]
    
    need_to_continue, FFD_upperbound = precompute_ffd(Y_hat, P_prime, h, d)
    if not need_to_continue:
        return
    
    SP, y_prime = solve_sub_ip(P_prime, FFD_upperbound, h, d)
    
    if SP.Status == GRB.OPTIMAL:
        num_open_or = sum(y_prime[r].x for r in R)
    
    # Feasbility cut
    if SP.Status != GRB.OPTIMAL:
        cuts_added += 1
        
        if VERBOSE:
            print("Infeasible, status code:", SP.Status)
            
        if CHOSEN_LBBD == LBBD_1:
            if USE_PROPAGATION:
                [model.cbLazy(quicksum(1 - x[h_prime, d_prime, p] for p in P_prime) 
                             >= 1) for h_prime in H for d_prime in D 
                 if B[h_prime, d_prime] <= B[h, d]]
            else:
                model.cbLazy(quicksum(1 - x[h, d, p] for p in P_prime) 
                             >= 1) 
        elif CHOSEN_LBBD == LBBD_2:
            if USE_PROPAGATION:
                [model.cbLazy(y[h_prime, d_prime] >= len(R) + 1 
                             - quicksum(1 - x[h_prime, d_prime, p] 
                                        for p in P_prime))
                 for h_prime in H for d_prime in D 
                 if B[h_prime, d_prime] <= B[h, d]]
            else:
                model.cbLazy(y[h, d] >= len(R) + 1 
                             - quicksum(1 - x[h, d, p] for p in P_prime))
                
    # Optimal, no cuts required
    elif abs(num_open_or - Y_hat[h, d]) < EPS:
        if VERBOSE:
            print(f"Upper bound = Lower bound, {num_open_or}" 
                  + f" = {Y_hat[h, d]}")
        
    # Optimality cut
    elif num_open_or > Y_hat[h, d] + EPS:
        cuts_added += 1
        if VERBOSE:
            print(f"Upper bound > Lower bound, {num_open_or}" 
                  + f" > {Y_hat[h, d]}")
        
        model.cbLazy(y[h, d] >= round(num_open_or) 
                     - quicksum(1 - x[h, d, p] for p in P_prime))
    
    # Ignore, no cut needed
    elif num_open_or < Y_hat[h, d] - EPS:
        # This branch is allowed to happen!
        # MIPSOL is just a new incumbent but not necessarily optimal.
        if VERBOSE:
            print(f"Upper bound > Lower bound, {num_open_or}" 
                  + f" < {Y_hat[h, d]}")
            
    cuts_container[0] = cuts_added

def callback(model, where):
    if where == GRB.Callback.MIPSOL:
        Y_hat = model.cbGetSolution(y)
        x_hat = model.cbGetSolution(x)
        
        cuts_added = [0]
        for h in H:
            for d in D:
                solve_sub_problem(model, x_hat, Y_hat, cuts_added, h, d)
        
        if VERBOSE:
            print("Cuts added", cuts_added[0])
        
start_time = time.time()
MP.optimize(callback)
end_time = time.time()

print("\n")
print("Optimal objective value:", MP.objVal)
print("Ran in", end_time - start_time, "seconds")

# Collect output summary
hospitals = []
days = []
num_opened = []
num_patients = []
max_times = []
total_surg_times = []
avail_times = []
for h in H:
    for d in D:
        hospitals.append(h)
        days.append(d)
        num_opened.append(y[h, d].x)
        num_patients.append(sum(x[h, d, p].x for p in P))
        avail_times.append(B[h, d]*y[h, d].x)
        max_times.append(B[h, d]*len(R))
        total_surg_times.append(sum(x[h, d, p].x*T[p] for p in P))

# Format and display output summary
columns = {'h': hospitals, 'd': days, 'num_opened': num_opened, 
           'num_patients': num_patients, 'total_surg_time': total_surg_times,
           'avail_time': avail_times, 'max_time': max_times}
results = pd.DataFrame(columns)
print(results)