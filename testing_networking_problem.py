# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:38:39 2023

Script for prototyping the operating rooms allocation model with 5 ORs and
20 patients. 

@author: mitch
"""
import time
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

# Formatting
UNDERLINE = "\n" + 80*"="

NUM_ROOMS = 3
NUM_PATIENTS = 20
NUM_HOSPITALS = 3
NUM_DAYS = 5

K_1 = 50
K_2 = 5

# Load csv files
patients = pd.read_csv(f'data/patients-{NUM_ROOMS}-{NUM_PATIENTS}.csv')
hospitals = pd.read_csv(f'data/hospitals-{NUM_ROOMS}-{NUM_PATIENTS}.csv')
#print("Data Preview" + UNDERLINE)
#print(patients.head())
#print(hospitals.head())

# Sets
P = range(NUM_PATIENTS)
H = range(NUM_HOSPITALS)
R = range(NUM_ROOMS)
D = range(NUM_DAYS)

#data

# Data
# Cost of opening hospital operating suite
G = {(h, d): int(hospitals[(hospitals['hospital_id'] == h) 
                              & (hospitals['day'] == d)]['hospital_open_cost']) 
      for h in H for d in D}

# Cost of opening operating room
F = {(h, d): int(hospitals[(hospitals['hospital_id'] == h) 
                              & (hospitals['day'] == d)]['or_open_cost']) 
      for h in H for d in D}

# Operating minutes of hospital
B = {(h, d): int(hospitals[(hospitals['hospital_id'] == h) 
                             & (hospitals['day'] == d)]['open_minutes'])
     for h in H for d in D}

# Surgery times
T = {p: int((patients[patients['id'] == p]['surgery_time'])) for p in P}

# Urgency
rho = {p: int(patients[patients['id'] == p]['urgency']) for p in P}

# Days elapsed since referral
alpha = {p: int(patients[patients['id'] == p]['wait_time']) for p in P}

mandatory_P = [p for p in P if patients.loc[p, 'is_mandatory'] == 1]

minDuration = min(T.values())

Nodes = set()
for h in H:
    for d in D:
        #starting node (start at this hospital on this day at time is 0):
        Nodes.add((h,d,0))
        #ending node (end for this hospital on this day at the closing hours of the hospital):
        Nodes.add((h,d,B[h,d]))
        #intermediate node (add a node for each possible arc start time which must be after the
        #the time required to do the quickest operation and before the time where the is not enough
        #time to complete the quickest operation):
        for t in range(minDuration,B[h,d] - minDuration + 1):
            Nodes.add((h,d,t))

#arc are of the form((from node), (to node), surgery we did/wait)
Arcs = set()
for h in H:
    for d in D:
            #waiting arcs
            #if our time is greater than the minimum surgery duration but has less than the time long enough to meet
            #minimum surgery time we can add a waiting arc (wouldnt you want to run this up to the time 
            #equal to the total possible time for the hospital - copying the example)
            for t in range(minDuration,B[h,d] - minDuration):
                Arcs.add(((h,d,t), (h,d,t+1),-1))
            #operating arcs
            for p in P:
                #can serve them at the start of the day
                Arcs.add(((h,d,0), (h,d,T[p]), p))
                #or we can serve them at the intermediate nodes
                #for every patient we can create an arc from a node that has a time greater than or equal
                #to the minimum surgery duration up to the opening hours of the hospital less the surgery time
                #of that patient, and then create arcs from the possible start times to the end times
                for t in range(minDuration,B[h,d] - T[p]):
                    if (h,d,t+T[p]) in Nodes:
                        Arcs.add(((h,d,t), (h,d,t+T[p]), p))
 
#checking code
for a in Arcs:
    if a[0] not in Nodes or a[1] not in Nodes:
        print("missing node", a)

m = gp.Model()

#add a variable to select which arcs are used
z = {a: m.addVar(vtype = gp.GRB.BINARY) for a in Arcs}
y = {(h,d): m.addVar(vtype = gp.GRB.INTEGER) for h in H for d in D}
x = {(h,d,p): m.addVar(vtype = gp.GRB.BINARY) for h in H for d in D for p in P}
u = {(h,d): m.addVar(vtype = gp.GRB.BINARY) for h in H for d in D}


    
# Objective
m.setObjective(quicksum(G[h, d]*u[h, d] for h in H for d in D)
                    + quicksum(F[h, d]*y[h, d] for h in H for d in D)
                    + quicksum(K_1*rho[p]*(d - alpha[p])*x[h, d, p] 
                              for h in H for d in D for p in P)
                    # + quicksum(K_2*rho[p]*(NUM_DAYS + 1 - alpha[p])*w[p] 
                              # for p in P if p not in mandatory_P)
                                , GRB.MINIMIZE)



#turn on operating theatres
TurnOnOperatingTheatres = {(h,d):
                  m.addConstr(quicksum(z[a] for a in Arcs if a[0] == (h,d,0)) <= y[h,d])
                  for d in D for h in H}    

#Maximum number of rooms in a hospital
MaxNumRooms = {(h,d):
               m.addConstr(y[h,d] <= NUM_ROOMS)
               for d in D for h in H}

#turn on hospitals
TurnOnHospitals = {(h,d):
                   m.addConstr(NUM_ROOMS * u[h,d] >= 
                               quicksum(z[a] for a in Arcs if a[0] == (h,d,0)))
                   for h in H for d in D}
    
#assign patients from network to problem
PatientAssignmentFromNetwork = {(h,d,p):
                                m.addConstr(x[h,d,p] == quicksum(z[a] for a in Arcs if a[0][:1] 
                                            == (h,d) and a[2] == p))
                                for h in H for d in D for p in P}
    
m.optimize()















# # Data
# # Cost of opening hospital operating suite
# G = {(h, d): float(hospitals[(hospitals['hospital_id'] == h) 
#                              & (hospitals['day'] == d)]['hospital_open_cost']) 
#      for h in H for d in D}

# # Cost of opening operating room
# F = {(h, d): float(hospitals[(hospitals['hospital_id'] == h) 
#                              & (hospitals['day'] == d)]['or_open_cost']) 
#      for h in H for d in D}

# # Operating minutes of hospital
# B = {(h, d): float(hospitals[(hospitals['hospital_id'] == h) 
#                              & (hospitals['day'] == d)]['open_minutes']) 
#      for h in H for d in D}

# # Surgery times
# T = {p: float(patients[patients['id'] == p]['surgery_time']) for p in P}

# # Urgency
# rho = {p: int(patients[patients['id'] == p]['urgency']) for p in P}

# # Days elapsed since referral
# alpha = {p: int(patients[patients['id'] == p]['wait_time']) for p in P}

# mandatory_P = [p for p in P if patients.loc[p, 'is_mandatory'] == 1]

# MP = gp.Model()

# # Variables
# # 1 if patient p assigned to OR r in hospital h on day d
# x = {(h, d, p): MP.addVar(vtype=GRB.BINARY) for h in H for d in D for p in P}

# # 1 if surgical suite in hospital h is opened on day d
# u = {(h, d): MP.addVar(vtype=GRB.BINARY) for h in H for d in D}

# # number of OR r in hospital h open on day d
# y = {(h, d): MP.addVar(vtype=GRB.INTEGER, lb=0, ub=len(R)) for h in H for d in D}

# # 1 if patient does not get surgery within time horizon
# w = {p: MP.addVar(vtype=GRB.BINARY) for p in P if p not in mandatory_P}

# # Objective
# MP.setObjective(quicksum(G[h, d]*u[h, d] for h in H for d in D)
#                    + quicksum(F[h, d]*y[h, d] for h in H for d in D)
#                    + quicksum(K_1*rho[p]*(d - alpha[p])*x[h, d, p] 
#                               for h in H for d in D for p in P)
#                    + quicksum(K_2*rho[p]*(NUM_DAYS + 1 - alpha[p])*w[p] 
#                               for p in P if p not in mandatory_P), GRB.MINIMIZE)

# # Constraints
# mandatory_operations = {p: MP.addConstr(
#     quicksum(x[h, d, p] for h in H for d in D) == 1) 
#     for p in mandatory_P}

# turn_on_w = {p: MP.addConstr(
#     quicksum(x[h, d, p] for h in H for d in D) + w[p] == 1) 
#     for p in P if p not in mandatory_P}

# lp_strengthener = {(h, d, p): MP.addConstr(x[h, d, p] <= u[h, d]) 
#                    for h in H for d in D for p in P}

# time_for_ops_in_hosp = {(h, d): MP.addConstr(
#     quicksum(T[p]*x[h, d, p] for p in P) <= len(R)*B[h, d]*u[h, d]) 
#     for h in H for d in D}

# no_single_long_op = {(h, d, p): MP.addConstr(T[p]*x[h, d, p] <= B[h, d]) 
#                      for h in H for d in D for p in P}

# num_or_lb = {(h, d): MP.addConstr(y[h, d]*B[h, d] 
#                                   >= quicksum(T[p]*x[h, d, p] for p in P))
#             for h in H for d in D}

# MP.setParam('OutputFlag', 0)
# MP.setParam('MIPGap', 0)

# start_time = time.time()
# iterations = 0
# #while true loop will make the code cycle through until it eventually finds the break statement
# # in this case we cycle through and we keep cycling through until we havent added any cuts
# #this means the cutsadded will be 0 and we will reach our break function
# while True:
#     print("Iteration", iterations)
#     iterations += 1
#     MP.optimize()
#     print("Curr objVal", MP.objVal)
        
#     cuts_added = 0
#     for h in H:
#         for d in D:
#             print("Hospital", h, "Day", d, end=" ")
#             # Set of patients assigned to this hospital and day.
#             P_prime = [p for p in P if x[h, d, p].x == 1]
            
#             SP = gp.Model()
#             SP.setParam('OutputFlag', 0)
            
            
#             # Variables
#             y_prime = {r: SP.addVar(vtype=GRB.BINARY) for r in R}
#             x_prime = {(p, r): SP.addVar(vtype=GRB.BINARY) for p in P_prime for r in R}
            
#             # Objective
#             SP.setObjective(quicksum(y_prime[r] for r in R), GRB.MINIMIZE)
            
#             # Constraints
#             patients_assigned_hosp_get_room = {
#                 p: SP.addConstr(quicksum(x_prime[p, r] for r in R) == 1) for p in P_prime}
            
#             OR_capacity = {r: SP.addConstr(quicksum(T[p]*x_prime[p, r] for p in P_prime) 
#                                            <= B[h, d]*y_prime[r]) for r in R}
            
#             sub_lp_strengthener = {(p, r): SP.addConstr(x_prime[p, r] <= y_prime[r]) 
#                                for p in P_prime for r in R}
            
#             OR_symmetries = {r: SP.addConstr(y_prime[r] <= y_prime[r - 1]) 
#                              for r in R[1:]}
            
#             SP.optimize()
#             # if we have the optimal solution then we return our answer
#             if SP.Status == GRB.OPTIMAL:
#                 num_open_or = sum(y_prime[r].x for r in R)
#             #if the solution isnt feasible then remove at least one patient from at least one hospital
#             if SP.Status != GRB.OPTIMAL:
#                 print("Infeasible, status code:", SP.Status)
#                 # TODO Infeasibility cut not working/ master problem becomes too hard
#                 # MP.addConstr(quicksum(1 - x[h, d, p] for p in P_prime) >= 1)
#                 MP.addConstr(y[h, d] >= len(R) + 1 - quicksum(1 - x[h, d, p] for p in P_prime))
                
#                 cuts_added += 1
#             #isnt the line below pointless?
#             elif num_open_or == y[h, d].x:
#                 print("Upper bound = Lower bound")
#             # if our solution is feasible but our upper bound doesnt equal our lower bound then
#             # open at least one more operating room and or subtract at least one patient
#             else:
#                 print("Upper bound != Lower bound", num_open_or, "!=", y[h, d].x)
#                 MP.addConstr(y[h, d] >= num_open_or - quicksum(1 - x[h, d, p] 
#                                                                for p in P_prime))
#                 cuts_added += 1
                        
#     print("Cuts added", cuts_added)
#     if cuts_added == 0:
#         break

# end_time = time.time()

# print("\n")
# print("Optimal objective value:", MP.objVal)
# print("Ran in", end_time - start_time, "seconds")
# print(iterations, "iterations")