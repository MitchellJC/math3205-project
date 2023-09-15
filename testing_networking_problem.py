# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:38:39 2023

Script for prototyping the operating rooms allocation model with network

Sets:
    P, The patients,
    mandatory_P, The mandatory patients
    H, The hospitals,
    R, The operating rooms
    D, The days

Data:
    G, Cost of opening hosptial operating suite
    F, Cost of opening operating room
    B, Operating minutes of hospital
    T, Surgery times
    rho, Urgency
    alpha, days elapsed since refered
    
    

@author: Ethan, Mitch & Ben
"""
import time
from gurobipy import GRB, quicksum, Model
from data_gen import generate_data
from constants import UNDERLINE, K_1, K_2
from dataclasses import dataclass, field
from itertools import count
from collections import defaultdict

NUM_ROOMS = 5
NUM_PATIENTS = 20
NUM_HOSPITALS = 3
NUM_DAYS = 5

SEED = 42
GAP = 0.00

@dataclass(eq=True, frozen=True)
class Node:
    hosp: int
    day: int
    time: int
    id_: int = field(default_factory=count().__next__, hash=False, compare=False)

@dataclass(eq=True, frozen=True)
class Arc:
    hosp: int
    day: int
    start_time: int
    end_time: int
    patient: int
    id_: int = field(default_factory=count().__next__, hash=False, compare=False)

P, mandatory_P, H, R, D, rho, alpha, health_status, B, T, G, F = generate_data(
    NUM_PATIENTS, NUM_ROOMS, output_dict=True, verbose=True, seed=SEED)

min_dur = min(T.values())
# Define nodes
print("Constructing nodes...")
nodes = set()
for h in H:
    for d in D:
        start_node = Node(h, d, 0)
        nodes.add(start_node)
        end_node = Node(h, d, B[h,d])
        nodes.add(end_node)
        
        # Add intermediate nodes.
        for t in range(min_dur, B[h,d] - min_dur + 1):
            nodes.add(Node(h, d, t))
print("Done constructing nodes.")
print("Number of Nodes: ", len(nodes))
# Define arcs
print("Constructing arcs...")
arcs = set()
from_arcs = defaultdict(lambda: set()) # Cache arcs for speed performance
to_arcs = defaultdict(lambda: set())
for h in H:
    for d in D:
            # Add waiting arcs, ignore tail time gaps
            for t in range(min_dur, B[h,d] - min_dur):
                arc = Arc(h, d, t, t + 1, -1)
                arcs.add(arc)
                from_arcs[h, d, t].add(arc)
                to_arcs[h, d, t + 1].add(arc)
            # Add operating arcs
            for p in P:
                # Can operate at start of the day
                arc = Arc(h, d, 0, T[p], p)
                arcs.add(arc)
                from_arcs[h, d, 0].add(arc)
                to_arcs[h, d, T[p]].add(arc)
               
                # Add arcs in intermediate gap.
                for t in range(min_dur, B[h, d]):   # TODO Not sure about this - Mitch
                                                    # This looks correct to me - Hart
                    if Node(h, d, t + T[p]) in nodes:
                        arc = Arc(h, d, t, t + T[p], p)
                        arcs.add(arc)
                        from_arcs[h, d, t].add(arc)
                        to_arcs[h, d, t + T[p]].add(arc)
print("Done constructing arcs.")
print("Number of Arcs: ", len(arcs))
# Checking code
for a in arcs:
    if (Node(a.hosp, a.day, a.start_time) not in nodes 
        or Node(a.hosp, a.day, a.end_time) not in nodes):
        print("missing node", a)

m = Model()

# 1 if arc a turned on
z = {a.id_: m.addVar(vtype=GRB.BINARY) for a in arcs}

# Number of op rooms opened
y = {(h, d): m.addVar(vtype=GRB.INTEGER, lb=0) for h in H for d in D}

# 1 if patient is operated on in hospital day
x = {(h, d, p): m.addVar(vtype=GRB.BINARY) for h in H for d in D for p in P}

# 1 if hospital on day is opened
u = {(h, d): m.addVar(vtype=GRB.BINARY) for h in H for d in D}

# 1 if patient not operated on in time horizon
w = {p: m.addVar(vtype=GRB.BINARY) for p in P if p not in mandatory_P}
    
# Objective
m.setObjective(quicksum(G[h, d]*u[h, d] for h in H for d in D)
                    + quicksum(F[h, d]*y[h, d] for h in H for d in D)
                    + quicksum(K_1*rho[p]*(d - alpha[p])*x[h, d, p] 
                              for h in H for d in D for p in P)
                    + quicksum(K_2*rho[p]*(NUM_DAYS + 1 - alpha[p])*w[p] 
                               for p in P if p not in mandatory_P), 
                    GRB.MINIMIZE)

# Constraints
room_flow = {n.id_: m.addConstr(
    quicksum(z[a.id_] for a in from_arcs[n.hosp, n.day, n.time]) 
    == quicksum(z[a.id_] for a in to_arcs[n.hosp, n.day, n.time])) 
    for n in nodes if min_dur <= n.time <= B[n.hosp, n.day] - min_dur} 

restrict_ops_by_ors = {(h, d): 
                        m.addConstr(quicksum(z[a.id_] for a in arcs 
                                            if (a.hosp, a.day, a.start_time) 
                                            == (h, d, 0)) 
                                    <= y[h,d]) for d in D for h in H}    

force_hosp_on = {(h, d): m.addConstr(y[h, d] <= NUM_ROOMS*u[h,d]) 
                  for h in H for d in D}

max_ors = {(h, d): m.addConstr(y[h, d] <= NUM_ROOMS) for d in D for h in H}


is_patient_operated_on = {(h, d, p): m.addConstr(
    quicksum(z[a.id_] for a in arcs if (a.hosp, a.day, a.patient) == (h, d, p)) 
    == x[h, d, p]) for h in H for d in D for p in P}
    
patient_once = {p: m.addConstr(quicksum(x[h, d, p] for h in H for d in D) <= 1)
                      for p in P}
    
must_do_mandatory = {p: m.addConstr(
    quicksum(x[h, d, p] for h in H for d in D) == 1) for p in mandatory_P}
    
turn_on_w = {p: m.addConstr(w[p] <= 1 - quicksum(x[h, d, p] for h in H for d in D)) 
              for p in P if p not in mandatory_P}

start_t = time.time()
m.optimize()
print("Ran in", time.time() - start_t, "seconds")

for d in D:
    print("Day", d)
    for h in H:
        print("Hospital", h)
        for p in P:
            if x[h, d, p].x > 0.5:
                print("Patient", p)