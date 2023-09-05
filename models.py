# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:36:14 2023

@author: mitch
"""
import gurobipy as gp
from gurobipy import GRB, quicksum
from constants import K_1, K_2, TIME_LIMIT, LBBD_1, LBBD_2, EPS
from typing import Union
from utilities import ffd

class ORScheduler:
    """Abstract class for OR scheduler methods."""
    def __init__(self, P: range, H: range, R: range, D: range, G: dict, F: dict, 
                 B: dict, T: dict, rho: dict, alpha: dict, mand_P: list, 
                 gurobi_log=True) -> None:
        """Initialise new object instance.
        
        Parameters:
            P - Range of patients
            H - Range of hospitals
            R - Range of operating rooms.
            D - Range of days.
            G - Dictionary of hospital opening costs.
            F - Dictionary of operating room opening costs.
            B - Dictionary of open minutes of hospital-days.
            T - Dictionary of operating times for each patient.
            rho - Dictionary of patient urgencies.
            alpha - Dictionary of patient wait times.
            mand_P - List of mandatory patients.
            gurobi_log - True to show Gurobi log.
        """
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
        self.gurobi_log = gurobi_log
        self._define_model()
        
    def _define_model(self):
        # Model and parameters
        self.model = gp.Model()
        self.model.setParam('MIPGap', 0)
        # self.model.setParam('MIPFocus', 2)
        # self.model.setParam('Heuristics', 0)
        self.model.setParam('TimeLimit', TIME_LIMIT)
        out_flag = 1 if self.gurobi_log else 0
        self.model.setParam('OutputFlag', out_flag)
        self.model.setParam('TimeLimit', TIME_LIMIT)
        
        # Variables
        # 1 if surgical suite in hospital h is opened on day d
        self.u = {(h, d): self.model.addVar(vtype=GRB.BINARY) 
                  for h in self.H for d in self.D}
        
        # 1 if patient does not get surgery within time horizon
        self.w = {p: self.model.addVar(vtype=GRB.BINARY) 
                  for p in self.P if p not in self.mand_P}
        
class BendersORScheduler(ORScheduler):
    """Abstract class for OR schedulers using Benders' Decomposition."""
    def __init__(self, P: range, H: range, R: range, D: range, G: dict, F: dict, 
                 B: dict, T: dict, rho: dict, alpha: dict, mand_P: list, 
                 chosen_lbbd: str = LBBD_2, use_propagation: bool = False, 
                 tol: float = EPS, verbose: bool = True, **kwargs):
        """Initialise new object instance.
        
        Parameters:
            P - Range of patients
            H - Range of hospitals
            R - Range of operating rooms.
            D - Range of days.
            G - Dictionary of hospital opening costs.
            F - Dictionary of operating room opening costs.
            B - Dictionary of open minutes of hospital-days.
            T - Dictionary of operating times for each patient.
            rho - Dictionary of patient urgencies.
            alpha - Dictionary of patient wait times.
            mand_P - List of mandatory patients.
            chosen_lbbd - Variant of Benders' cuts apply.
            use_propagation - True to use cut propagation.
            tol - Numerical tolerance.
            verbose - True to output algorithm logs.
            gurobi_log - True to show Gurobi log.
        """
        self.chosen_lbbd = chosen_lbbd
        self.use_propagation = use_propagation
        self.tol = tol
        self.verbose = verbose
        super().__init__(P, H, R, D, G, F, B, T, rho, alpha, mand_P, **kwargs)
        
    def _define_model(self):
        super()._define_model()

        # Variables
        # 1 if patient p assigned to OR r in hospital h on day d
        self.x = {(h, d, p): self.model.addVar(vtype=GRB.BINARY) 
             for h in self.H for d in self.D for p in self.P}


        # number of OR r in hospital h open on day d
        self.y = {(h, d): self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=len(self.R)) 
             for h in self.H for d in self.D}

        # Objective
        self.model.setObjective(
            quicksum(self.G[h, d]*self.u[h, d] 
                     for h in self.H for d in self.D) 
            + quicksum(self.F[h, d]*self.y[h, d] for h in self.H for d in self.D)
            + quicksum(K_1*self.rho[p]*(d - self.alpha[p])*self.x[h, d, p] 
                     for h in self.H for d in self.D for p in self.P)
            + quicksum(K_2*self.rho[p]*(len(self.D) + 1 - self.alpha[p])*self.w[p]
                     for p in self.P if p not in self.mand_P), GRB.MINIMIZE)

        # Constraints
        mandatory_operations = {p: self.model.addConstr(
            quicksum(self.x[h, d, p] for h in self.H for d in self.D) == 1) 
            for p in self.mand_P}

        turn_on_w = {p: self.model.addConstr(
            quicksum(self.x[h, d, p] for h in self.H for d in self.D) 
            + self.w[p] == 1) 
            for p in self.P if p not in self.mand_P}

        lp_strengthener = {(h, d, p): self.model.addConstr(self.x[h, d, p] 
                                                           <= self.u[h, d]) 
                           for h in self.H for d in self.D for p in self.P}

        time_for_ops_in_hosp = {(h, d): self.model.addConstr(
            quicksum(self.T[p]*self.x[h, d, p] for p in self.P) 
            <= len(self.R)*self.B[h, d]*self.u[h, d]) 
            for h in self.H for d in self.D}

        no_single_long_op = {(h, d, p): self.model.addConstr(
            self.T[p]*self.x[h, d, p] <= self.B[h, d]) 
                             for h in self.H for d in self.D for p in self.P}

        num_or_lb = {(h, d): self.model.addConstr(self.y[h, d]*self.B[h, d] 
                                          >= quicksum(self.T[p]*self.x[h, d, p] 
                                                      for p in self.P))
                    for h in self.H for d in self.D}

class MIPScheduler(ORScheduler):
    """OR scheduler that uses a pure mixed integer programming routine."""
    def _define_model(self):
        super()._define_model()
        
        # Variables
        # 1 if patient p assigned to OR r in hospital h on day d
        self.x = {(h, d, p, r): self.model.addVar(vtype=GRB.BINARY) 
                  for h in self.H for d in self.D for p in self.P for r in self.R}

        # 1 if OR r in hospital h open on day d
        self.y = {(h, d, r): self.model.addVar(vtype=GRB.BINARY) 
                  for h in self.H for d in self.D for r in self.R}

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
        
class BendersLoopScheduler(BendersORScheduler):
    """OR scheduler that uses Benders' decomposition in a loop."""    
    def run_model(self):
        iterations = 0
        while True:
            if self.verbose:
                print("Iteration", iterations)
            iterations += 1
            self.model.optimize()
            if self.verbose:
                print("Curr objVal", self.model.objVal)
            
            cuts_added = 0
            for h in self.H:
                for d in self.D:
                    if self.verbose:
                        print("Hospital", h, "Day", d, end=" ")
                    # Set of patients assigned to this hospital and day.
                    P_prime = [p for p in self.P if self.x[h, d, p].x > 0.5]
                    
                    SP = gp.Model()
                    SP.setParam('OutputFlag', 0)
                    SP.setParam('MIPGap', 0)
                    
                    
                    # Variables
                    y_prime = {r: SP.addVar(vtype=GRB.BINARY) for r in self.R}
                    x_prime = {(p, r): SP.addVar(vtype=GRB.BINARY) 
                               for p in P_prime for r in self.R}
                    
                    # Objective
                    SP.setObjective(quicksum(y_prime[r] for r in self.R), 
                                    GRB.MINIMIZE)
                    
                    # Constraints
                    patients_assigned_hosp_get_room = {
                        p: SP.addConstr(quicksum(x_prime[p, r] for r in self.R) 
                                        == 1) 
                        for p in P_prime}
                    
                    OR_capacity = {r: SP.addConstr(
                        quicksum(self.T[p]*x_prime[p, r] for p in P_prime) 
                        <= self.B[h, d]*y_prime[r]) for r in self.R}
                    
                    sub_lp_strengthener = {(p, r): SP.addConstr(x_prime[p, r] 
                                                                <= y_prime[r]) 
                                       for p in P_prime for r in self.R}
                    
                    OR_symmetries = {r: SP.addConstr(y_prime[r] <= y_prime[r - 1]) 
                                     for r in self.R[1:]}
                    
                    SP.optimize()
                    
                    if SP.Status == GRB.OPTIMAL:
                        num_open_or = sum(y_prime[r].x for r in self.R)
                        
                    if SP.Status != GRB.OPTIMAL:
                        if self.verbose:
                            print("Infeasible, status code:", SP.Status)
                        if self.chosen_lbbd == LBBD_1:
                            self.model.addConstr(quicksum(1 - self.x[h, d, p] 
                                                          for p in P_prime) >= 1)
                        if self.chosen_lbbd == LBBD_2:
                            self.model.addConstr(
                                self.y[h, d] >= len(self.R) + 1 
                                - quicksum(1 - self.x[h, d, p] for p in P_prime))
                        
                        cuts_added += 1
                    elif num_open_or == self.y[h, d].x:
                        if self.verbose:
                            print(f"Upper bound = Lower bound, {num_open_or} " 
                                  + f"= {self.y[h, d].x}")
                        
                    elif num_open_or > self.y[h, d].x:
                        if self.verbose:
                            print(f"Upper bound > Lower bound, {num_open_or} "
                                  + f"> {self.y[h, d].x}")
                        self.model.addConstr(
                            self.y[h, d] >= num_open_or 
                            - quicksum(1 - self.x[h, d, p] for p in P_prime))
                        cuts_added += 1
                    elif num_open_or < self.y[h, d].x - self.tol:
                        raise RuntimeError("Sub problem < Master problem!, "
                                           + f"{num_open_or} < {self.y[h, d].x}")
                        
            if self.verbose:            
                print("Cuts added", cuts_added)
            if cuts_added == 0:
                break
            
class BendersCallbackScheduler(BendersORScheduler):
    """OR scheduler that uses Benders' decomposition in a callback."""
    def _define_model(self):
        super()._define_model()
        self.model.setParam('LazyConstraints', 1)
        
    def run_model(self):
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
            items = [(p, self.T[p]) for p in P_prime]
            heur_open_rooms = ffd(items, len(self.R), self.B[h, d])
            
            # Continue sub problem based on heuristic solution
            FFD_upperbound = None
            if heur_open_rooms:
                if abs(len(heur_open_rooms) - Y_hat[h, d]) < self.tol:
                    if self.verbose:
                        print("FFD soln same as master problem.")
                    return False, None
                elif len(heur_open_rooms) < Y_hat[h, d] - self.tol:
                    if self.verbose:
                        print("FFD soln better than master problem."
                              +" (Non-optimal master soln)")
                    return False, None
                elif len(heur_open_rooms) > Y_hat[h, d] + self.tol:
                    if self.verbose:
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
            y_prime = {r: SP.addVar(vtype=GRB.BINARY) for r in self.R}
            x_prime = {(p, r): SP.addVar(vtype=GRB.BINARY) for p in P_prime 
                       for r in self.R}
            
            # Objective
            SP.setObjective(quicksum(y_prime[r] for r in self.R), GRB.MINIMIZE)
            
            # Constraints
            patients_assigned_hosp_get_room = {
                p: SP.addConstr(quicksum(x_prime[p, r] for r in self.R) == 1) 
                for p in P_prime}

            OR_capacity = {r: SP.addConstr(quicksum(self.T[p]*x_prime[p, r] 
                                                    for p in P_prime) 
                                           <= self.B[h, d]*y_prime[r]) 
                           for r in self.R}
            
            sub_lp_strengthener = {(p, r): SP.addConstr(x_prime[p, r] 
                                                        <= y_prime[r]) 
                                   for p in P_prime for r in self.R}
            
            OR_symmetries = {r: SP.addConstr(y_prime[r] <= y_prime[r - 1]) 
                             for r in self.R[1:]}
            
            if FFD_upperbound:
                FFD_tightener = SP.addConstr(quicksum(y_prime[r] for r in self.R)
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
            
            if self.verbose:
                print("Hospital", h, "Day", d, end=" ")\
                    
            # Set of patients assigned to this hospital and day.
            P_prime = [p for p in self.P if x_hat[h, d, p] > 0.5]
            
            need_to_continue, FFD_upperbound = precompute_ffd(Y_hat, P_prime, h, d)
            if not need_to_continue:
                return
            
            SP, y_prime = solve_sub_ip(P_prime, FFD_upperbound, h, d)
            
            if SP.Status == GRB.OPTIMAL:
                num_open_or = sum(y_prime[r].x for r in self.R)
            
            # Feasbility cut
            if SP.Status != GRB.OPTIMAL:
                cuts_added += 1
                
                if self.verbose:
                    print("Infeasible, status code:", SP.Status)
                    
                if self.chosen_lbbd == LBBD_1:
                    if self.use_propagation:
                        [model.cbLazy(quicksum(1 - self.x[h_prime, d_prime, p] 
                                               for p in P_prime) 
                                     >= 1) for h_prime in self.H for d_prime in self.D 
                         if self.B[h_prime, d_prime] <= self.B[h, d]]
                    else:
                        model.cbLazy(quicksum(1 - self.x[h, d, p] for p in P_prime) 
                                     >= 1) 
                elif self.chosen_lbbd == LBBD_2:
                    if self.use_propagation:
                        [model.cbLazy(self.y[h_prime, d_prime] >= len(self.R) + 1 
                                     - quicksum(1 - self.x[h_prime, d_prime, p] 
                                                for p in P_prime))
                         for h_prime in self.H for d_prime in self.D 
                         if self.B[h_prime, d_prime] <= self.B[h, d]]
                    else:
                        model.cbLazy(self.y[h, d] >= len(self.R) + 1 
                                     - quicksum(1 - self.x[h, d, p] for p in P_prime))
                        
            # Optimal, no cuts required
            elif abs(num_open_or - Y_hat[h, d]) < self.tol:
                if self.verbose:
                    print(f"Upper bound = Lower bound, {num_open_or}" 
                          + f" = {Y_hat[h, d]}")
                
            # Optimality cut
            elif num_open_or > Y_hat[h, d] + self.tol:
                cuts_added += 1
                if self.verbose:
                    print(f"Upper bound > Lower bound, {num_open_or}" 
                          + f" > {Y_hat[h, d]}")
                
                model.cbLazy(self.y[h, d] >= round(num_open_or) 
                             - quicksum(1 - self.x[h, d, p] for p in P_prime))
            
            # Ignore, no cut needed
            elif num_open_or < Y_hat[h, d] - self.tol:
                # This branch is allowed to happen!
                # MIPSOL is just a new incumbent but not necessarily optimal.
                if self.verbose:
                    print(f"Upper bound > Lower bound, {num_open_or}" 
                          + f" < {Y_hat[h, d]}")
                    
            cuts_container[0] = cuts_added
            
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                Y_hat = model.cbGetSolution(self.y)
                x_hat = model.cbGetSolution(self.x)
                
                cuts_added = [0]
                for h in self.H:
                    for d in self.D:
                        solve_sub_problem(model, x_hat, Y_hat, cuts_added, h, d)
                
                if self.verbose:
                    print("Cuts added", cuts_added[0])
                    
        self.model.optimize(callback)
        