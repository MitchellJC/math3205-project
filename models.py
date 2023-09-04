# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:36:14 2023

@author: mitch
"""
import gurobipy as gp
from gurobipy import GRB, quicksum
from constants import K_1, K_2

TIME_LIMIT = 7200

class ORScheduler:
    def __init__(self, P, H, R, D, G, F, B, T, rho, alpha, mand_P):
        self.P = P
        self.H = H
        self.R = R
        self.D = D
        self.G = G
        self.F = F
        self.B = B
        self.T = T
        self.rho = rho
        self.alpha = alpha
        self.mand_P = mand_P


class MIPScheduler(ORScheduler):
    def __init__(self, P, H, R, D, G, F, B, T, rho, alpha, mand_P):
        super().__init__(P, H, R, D, G, F, B, T, rho, alpha, mand_P)
        self._define_model()
        
    def _define_model(self):
        # Model and parameters
        self.model = gp.Model()
        self.model.setParam('MIPGap', 0)
        self.model.setParam('MIPFocus', 2)
        self.model.setParam('Heuristics', 0)
        self.model.setParam('TimeLimit', TIME_LIMIT)
        
        # Variables
        # 1 if patient p assigned to OR r in hospital h on day d
        self.x = {(h, d, p, r): self.model.addVar(vtype=GRB.BINARY) 
                  for h in self.H for d in self.D for p in self.P for r in self.R}

        # 1 if surgical suite in hospital h is opened on day d
        self.u = {(h, d): self.model.addVar(vtype=GRB.BINARY) 
                  for h in self.H for d in self.D}

        # 1 if OR r in hospital h open on day d
        self.y = {(h, d, r): self.model.addVar(vtype=GRB.BINARY) 
                  for h in self.H for d in self.D for r in self.R}

        # 1 if patient does not get surgery within time horizon
        self.w = {p: self.model.addVar(vtype=GRB.BINARY) 
                  for p in self.P if p not in self.mand_P}
        
        # Objective
        self.model.setObjective(
            quicksum(self.G[h, d]*self.u[h, d] for h in self.H for d in self.D)
            + quicksum(self.F[h, d]*self.y[h, d, r] for h in self.H 
                       for d in self.D for r in self.R)
            + quicksum(K_1*self.rho[p]*(d - self.alpha[p])*self.x[h, d, p, r] 
                     for h in self.H for d in self.D for p in self.P for r in self.R)
            + quicksum(K_2*self.rho[p]*(len(self.D) + 1 - self.alpha[p])*self.w[p] 
                     for p in self.P if p not in self.mand_P), 
            GRB.MINIMIZE)

        # Constraints
        mandatory_operations = {p: self.model.addConstr(
            quicksum(self.x[h, d, p, r] 
                     for h in self.H for d in self.D for r in self.R) == 1) 
            for p in self.mand_P}

        turn_on_w = {p: self.model.addConstr(
            quicksum(self.x[h, d, p, r] 
                     for h in self.H for d in self.D for r in self.R) 
            + self.w[p] == 1) 
            for p in self.P if p not in self.mand_P}
        
        time_for_op = {(h, d, r): self.model.addConstr(
            quicksum(self.T[p]*self.x[h, d, p, r] for p in self.P) 
            <= self.B[h, d]*self.y[h, d, r]) 
            for h in self.H for d in self.D for r in self.R}

        # Reduces amount of combinations to check
        OR_symmetries = {(h, d, r): self.model.addConstr(self.y[h, d, r] 
                                                         <= self.y[h, d, r - 1]) 
                         for h in self.H for d in self.D for r in self.R[1:]}

        suite_open_before_OR = {(h, d, r): self.model.addConstr(
            self.y[h, d, r] <= self.u[h, d]) 
            for h in self.H for d in self.D for r in self.R}

        lp_strengthener = {(h, d, p, r): self.model.addConstr(
            self.x[h, d, p, r] <= self.y[h, d, r]) 
            for h in self.H for d in self.D for p in self.P for r in self.R}
        
    def run_model(self):
        self.model.optimize()
        