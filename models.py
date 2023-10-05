# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:36:14 2023

@author: mitch
"""
import time
import gurobipy as gp
from gurobipy import GRB, quicksum
from constants import (K_1, K_2, TIME_LIMIT, LBBD_1, LBBD_2, LBBD_PLUS, EPS, 
                       UNDERLINE)
from typing import Union
from utilities import ffd, Bin
from dataclasses import dataclass, field
from itertools import count
from collections import defaultdict

class ORScheduler:
    """Abstract class for OR scheduler methods."""
    def __init__(self, P: range, H: range, R: range, D: range, G: dict, F: dict, 
                 B: dict, T: dict, rho: dict, alpha: dict, mand_P: list,
                 gap: int = 0, gurobi_log: bool = True) -> None:
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
        self.gap = gap
        self.mp_time = 0
        self.sp_time = 0
        self._define_model()
        
    def _define_model(self):
        # Model and parameters
        self.model = gp.Model()
        self.model.setParam('MIPGap', self.gap)
        # self.model.setParam('MIPFocus', 2)
        # self.model.setParam('Heuristics', 0)
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

class NetworkScheduler(ORScheduler):
    def _define_model(self):
        min_dur = min(self.T.values())
        
        # Define nodes
        print("Constructing nodes...")
        self.nodes = set()
        for h in self.H:
            for d in self.D:
                start_node = Node(h, d, 0)
                self.nodes.add(start_node)
                end_node = Node(h, d, self.B[h,d])
                self.nodes.add(end_node)
                
                # Add intermediate nodes.
                for t in range(min_dur, self.B[h,d] - min_dur + 1):
                    self.nodes.add(Node(h, d, t))
                    
        print("Done constructing nodes.")
        print("Number of Nodes: ", len(self.nodes))
        
        # Define arcs
        print("Constructing arcs...")
        self.arcs = set()
        self.from_arcs = defaultdict(lambda: set()) # Cache arcs for speed 
        self.to_arcs = defaultdict(lambda: set())
        for h in self.H:
            for d in self.D:
                    # Add waiting arcs, ignore tail time gaps
                    for t in range(min_dur, self.B[h,d] - min_dur):
                        arc = Arc(h, d, t, t + 1, -1)
                        self.arcs.add(arc)
                        self.from_arcs[h, d, t].add(arc)
                        self.to_arcs[h, d, t + 1].add(arc)
                    # Add operating arcs
                    for p in self.P:
                        # Can operate at start of the day
                        arc = Arc(h, d, 0, self.T[p], p)
                        self.arcs.add(arc)
                        self.from_arcs[h, d, 0].add(arc)
                        self.to_arcs[h, d, self.T[p]].add(arc)
                       
                        # Add arcs in intermediate gap.
                        for t in range(min_dur, self.B[h, d]):   
                            if Node(h, d, t + self.T[p]) in self.nodes:
                                arc = Arc(h, d, t, t + self.T[p], p)
                                self.arcs.add(arc)
                                self.from_arcs[h, d, t].add(arc)
                                self.to_arcs[h, d, t + self.T[p]].add(arc)
                                
        print("Done constructing arcs.")
        print("Number of Arcs: ", len(self.arcs))
        # Checking code
        for a in self.arcs:
            if (Node(a.hosp, a.day, a.start_time) not in self.nodes 
                or Node(a.hosp, a.day, a.end_time) not in self.nodes):
                print("missing node", a)

        self.model = gp.Model()
        self.model.setParam('MIPGap', self.gap)
        self.model.setParam('TimeLimit', TIME_LIMIT)

        # 1 if arc a turned on
        self.z = {a.id_: self.model.addVar(vtype=GRB.BINARY) for a in self.arcs}

        # Number of op rooms opened
        self.y = {(h, d): self.model.addVar(vtype=GRB.INTEGER, lb=0) 
                  for h in self.H for d in self.D}

        # 1 if patient is operated on in hospital day
        self.x = {(h, d, p): self.model.addVar(vtype=GRB.BINARY)
                  for h in self.H for d in self.D for p in self.P}

        # 1 if hospital on day is opened
        self.u = {(h, d): self.model.addVar(vtype=GRB.BINARY) 
                  for h in self.H for d in self.D}

        # 1 if patient not operated on in time horizon
        self.w = {p: self.model.addVar(vtype=GRB.BINARY) 
                  for p in self.P if p not in self.mand_P}
            
        # Objective
        self.model.setObjective(quicksum(self.G[h, d]*self.u[h, d] 
                                         for h in self.H for d in self.D)
                            + quicksum(self.F[h, d]*self.y[h, d] for h in self.H 
                                       for d in self.D)
                            + quicksum(K_1*self.rho[p]*(d - self.alpha[p])*self.x[h, d, p] 
                                      for h in self.H for d in self.D for p in self.P)
                            + quicksum(K_2*self.rho[p]*(len(self.D) + 1 
                                                        - self.alpha[p])*self.w[p] 
                                       for p in self.P if p not in self.mand_P), 
                            GRB.MINIMIZE)

        # Constraints
        self.room_flow = {n.id_: self.model.addConstr(
            quicksum(self.z[a.id_] for a in self.from_arcs[n.hosp, n.day, n.time]) 
            == quicksum(self.z[a.id_] for a in self.to_arcs[n.hosp, n.day, n.time])) 
            for n in self.nodes if min_dur <= n.time <= self.B[n.hosp, n.day] - min_dur} 

        self.restrict_ops_by_ors = {(h, d): 
                                self.model.addConstr(quicksum(self.z[a.id_] 
                                                              for a in self.arcs 
                                                if (a.hosp, a.day, a.start_time) 
                                                == (h, d, 0)) 
                                <= self.y[h,d]) for d in self.D for h in self.H}    

        self.force_hosp_on = {(h, d): self.model.addConstr(self.y[h, d] 
                                                           <= len(self.R)*self.u[h,d]) 
                          for h in self.H for d in self.D}

        self.max_ors = {(h, d): self.model.addConstr(self.y[h, d] <= len(self.R)) 
                        for d in self.D for h in self.H}


        self.is_patient_operated_on = {(h, d, p): self.model.addConstr(
            quicksum(self.z[a.id_] for a in self.arcs 
                     if (a.hosp, a.day, a.patient) == (h, d, p)) 
            == self.x[h, d, p]) for h in self.H for d in self.D for p in self.P}
            
        self.patient_once = {p: self.model.addConstr(quicksum(self.x[h, d, p] 
                                                              for h in self.H 
                                                              for d in self.D) 
                            <= 1) for p in self.P}
            
        self.must_do_mandatory = {p: self.model.addConstr(
            quicksum(self.x[h, d, p] for h in self.H for d in self.D) == 1) 
            for p in self.mand_P}
            
        self.turn_on_w = {p: self.model.addConstr(self.w[p] 
                             <= 1 - quicksum(self.x[h, d, p] 
                                             for h in self.H for d in self.D)) 
                      for p in self.P if p not in self.mand_P}
        
    def run_model(self):
        self.model.optimize()
        self.run_gap = self.model.MIPGap

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
        self.mandatory_operations = {p: self.model.addConstr(
            quicksum(self.x[h, d, p, r] 
                     for h in self.H for d in self.D for r in self.R) == 1) 
            for p in self.mand_P}

        self.turn_on_w = {p: self.model.addConstr(
            quicksum(self.x[h, d, p, r] 
                     for h in self.H for d in self.D for r in self.R) 
            + self.w[p] == 1) 
            for p in self.P if p not in self.mand_P}
        
        self.time_for_op = {(h, d, r): self.model.addConstr(
            quicksum(self.T[p]*self.x[h, d, p, r] for p in self.P) 
            <= self.B[h, d]*self.y[h, d, r]) 
            for h in self.H for d in self.D for r in self.R}

        # Reduces amount of combinations to check
        self.OR_symmetries = {(h, d, r): self.model.addConstr(self.y[h, d, r] 
                                                         <= self.y[h, d, r - 1]) 
                         for h in self.H for d in self.D for r in self.R[1:]}

        self.suite_open_before_OR = {(h, d, r): self.model.addConstr(
            self.y[h, d, r] <= self.u[h, d]) 
            for h in self.H for d in self.D for r in self.R}

        self.lp_strengthener = {(h, d, p, r): self.model.addConstr(
            self.x[h, d, p, r] <= self.y[h, d, r]) 
            for h in self.H for d in self.D for p in self.P for r in self.R}
        
    def run_model(self):
        self.model.optimize()
        self.run_gap = self.model.MIPGap
        
class BendersORScheduler(ORScheduler):
    """Abstract class for OR schedulers using Benders' Decomposition."""
    def __init__(self, P: range, H: range, R: range, D: range, G: dict, F: dict, 
                 B: dict, T: dict, rho: dict, alpha: dict, mand_P: list, 
                 chosen_lbbd: str = LBBD_2, use_propagation: bool = False, 
                 tol: float = EPS, verbose: bool = True, bend_gap: bool = True, 
                 **kwargs):
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
        self.bend_gap = bend_gap
        super().__init__(P, H, R, D, G, F, B, T, rho, alpha, mand_P, **kwargs)
        
    def precompute_ffd(self, Y_hat: dict, P_prime: list, h: int, d: int) -> (
            tuple[bool, Union[int, None], Union[None, list[Bin]]]):
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
                return False, None, heur_open_rooms
            
            elif len(heur_open_rooms) < Y_hat[h, d] - self.tol:
                if self.verbose:
                    print("FFD soln better than master problem."
                          +" (Non-optimal master soln)")
                return False, None, heur_open_rooms
            
            elif len(heur_open_rooms) > Y_hat[h, d] + self.tol:
                if self.verbose:
                    print("FFD soln worst than master problem.")
                FFD_upperbound = len(heur_open_rooms)
                
        return True, FFD_upperbound, heur_open_rooms

    def solve_sub_ip(self, P_prime: list, FFD_upperbound: Union[int, None],
                     FFD_soln: list[Bin], h: int, d: int
                     ) -> tuple[gp.Model, dict]:
        """Defines ands solves to sub problem to optimality as an IP.
        
        Parameters:
            P_prime - List of patients assigned to hospital h
            FFD_upperbound - Heuristic upperbound on number of rooms needed
            FFD_soln - Heuristic solution to use as warm start to sub-problem.
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
        
        # Use heuristic soln as warm start
        FFD_soln = [] if FFD_soln == None else FFD_soln
        for r, room in enumerate(FFD_soln):
            for p, _ in room.items:
                x_prime[p, r].Start = 1
        
        # Objective
        SP.setObjective(quicksum(y_prime[r] for r in self.R), GRB.MINIMIZE)
        
        # Constraints
        # Patients_assigned_hosp_get_room 
        {p: SP.addConstr(quicksum(x_prime[p, r] for r in self.R) == 1) 
            for p in P_prime}
        
        # OR_capacity
        {r: SP.addConstr(quicksum(self.T[p]*x_prime[p, r] for p in P_prime) 
                         <= self.B[h, d]*y_prime[r]) for r in self.R}
        
        # sub_lp_strengthener  
        {(p, r): SP.addConstr(x_prime[p, r] <= y_prime[r]) 
         for p in P_prime for r in self.R}
        
        # OR_symmetries
        {r: SP.addConstr(y_prime[r] <= y_prime[r - 1]) for r in self.R[1:]}
        
        # FFD Tightener
        if FFD_upperbound:
            SP.addConstr(quicksum(y_prime[r] for r in self.R) <= FFD_upperbound)
        
        SP.optimize()
        
        return SP, y_prime, x_prime

    def solve_sub_problem(self, model: gp.Model, cuts_container: list, h: int, 
                          d: int, lazy: bool = True, save_soln: bool = False) -> None:
        """Solve the hospital-day sub-problem. Adds Bender's cuts to master problem
        as required.
        
        Parameters:
            model - Master problem Gurobi model.
            cuts_container - List containing one element, the number of cuts added
            h - Hospital
            d - day
            lazy - True to add constraint to model as lazy constraints.
            save_soln - True to save sub problem allocations
        """        
        if self.verbose:
            print("Hospital", h, "Day", d, end=" ")
        
        # Decisions for if called from loop or callback
        if lazy:
            add_constr = model.cbLazy
            
            Y_hat = model.cbGetSolution(self.y)
            x_hat = model.cbGetSolution(self.x)
        else:
            add_constr = model.addConstr
            
            Y_hat = {(h, d): self.y[h, d].x for h in self.H for d in self.D}
            x_hat = {(h, d, p): self.x[h, d, p].x for h in self.H for d in self.D 
                     for p in self.P}            
                
        # Set of patients assigned to this hospital and day.
        P_prime = [p for p in self.P if x_hat[h, d, p] > 0.5]
        
        need_to_continue, FFD_upperbound, FFD_soln = \
            self.precompute_ffd(Y_hat, P_prime, h, d)
        if not need_to_continue:
            return
        
        SP, y_prime, x_prime = self.solve_sub_ip(P_prime, FFD_upperbound, 
                                                 FFD_soln, h, d)
        
        if SP.Status == GRB.OPTIMAL:
            num_open_or = sum(y_prime[r].x for r in self.R)      
        
        if self.verbose:
            print()
        
        # Feasbility cut
        if SP.Status != GRB.OPTIMAL:
            cuts_container[0] += 1
            
            if self.verbose:
                print("Infeasible, status code:", SP.Status)
            
            # LBBD1
            if self.chosen_lbbd == LBBD_1:
                if self.use_propagation:
                    [add_constr(quicksum(1 - self.x[h_prime, d_prime, p] 
                                           for p in P_prime) 
                                 >= 1) for h_prime in self.H for d_prime in self.D 
                     if self.B[h_prime, d_prime] <= self.B[h, d]]
                else:
                    add_constr(quicksum(1 - self.x[h, d, p] for p in P_prime) 
                                 >= 1) 
                    
            # LBBD2
            elif self.chosen_lbbd == LBBD_2:
                if self.use_propagation:
                    [add_constr(self.y[h_prime, d_prime] >= len(self.R) + 1 
                                 - quicksum(1 - self.x[h_prime, d_prime, p] 
                                            for p in P_prime))
                     for h_prime in self.H for d_prime in self.D 
                     if self.B[h_prime, d_prime] <= self.B[h, d]]
                else:
                    add_constr(self.y[h, d] >= len(self.R) + 1 
                                 - quicksum(1 - self.x[h, d, p] for p in P_prime))
                    
            # LBBD4
            elif self.chosen_lbbd == LBBD_PLUS:
                max_dur = max(self.T[curr_p] for curr_p in P_prime)
                P_longer = [p for p in self.P if p not in P_prime and
                            self.T[p] >= max_dur]
                if self.use_propagation:
                    [add_constr(self.y[h, d] >= len(self.R) + 1 
                                 - quicksum(1 - self.x[h, d, p] for p in P_prime)
                                 + quicksum(self.x[h, d, p] for p in P_longer))
                     for h_prime in self.H for d_prime in self.D 
                     if self.B[h_prime, d_prime] <= self.B[h, d]]
                else:
                    add_constr(self.y[h, d] >= len(self.R) + 1 
                                 - quicksum(1 - self.x[h, d, p] for p in P_prime)
                                 + quicksum(self.x[h, d, p] for p in P_longer))
                    
        # Optimal, no cuts required
        elif abs(num_open_or - Y_hat[h, d]) < self.tol:
            if self.verbose:
                print(f"Upper bound = Lower bound, {num_open_or}" 
                      + f" = {Y_hat[h, d]}")
            
            # Save sub problem allocation
            if save_soln:
                self.sub_solns[h, d] = num_open_or
                self.sub_room_allocs[h, d] = {r: y_prime[r].x for r in self.R}
                self.sub_patient_allocs[h, d] = {(p, r): x_prime[p, r].x if p 
                                                 in P_prime else 0
                                                 for p in self.P for r in self.R}
            
        # Optimality cut
        elif num_open_or > Y_hat[h, d] + self.tol:
            cuts_container[1] += 1
            if self.verbose:
                print(f"Upper bound > Lower bound, {num_open_or}" 
                      + f" > {Y_hat[h, d]}")
            
            add_constr(self.y[h, d] >= round(num_open_or) 
                         - quicksum(1 - self.x[h, d, p] for p in P_prime))
        
        # Ignore, no cut needed
        elif num_open_or < Y_hat[h, d] - self.tol:
            # This branch is allowed to happen for lazy!
            # MIPSOL is just a new incumbent but not necessarily optimal.
            if self.verbose:
                print(f"Upper bound > Lower bound, {num_open_or}" 
                      + f" < {Y_hat[h, d]}")
            if not lazy:
                raise RuntimeError("Sub problem < Master problem!, "
                                        + f"{num_open_or} < {self.y[h, d].x}")
                
        
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
        self.mandatory_operations = {p: self.model.addConstr(
            quicksum(self.x[h, d, p] for h in self.H for d in self.D) == 1) 
            for p in self.mand_P}

        self.turn_on_w = {p: self.model.addConstr(
            quicksum(self.x[h, d, p] for h in self.H for d in self.D) 
            + self.w[p] == 1) 
            for p in self.P if p not in self.mand_P}

        self.lp_strengthener = {(h, d, p): self.model.addConstr(self.x[h, d, p] 
                                                           <= self.u[h, d]) 
                           for h in self.H for d in self.D for p in self.P}

        self.time_for_ops_in_hosp = {(h, d): self.model.addConstr(
            quicksum(self.T[p]*self.x[h, d, p] for p in self.P) 
            <= len(self.R)*self.B[h, d]*self.u[h, d]) 
            for h in self.H for d in self.D}

        self.no_single_long_op = {(h, d, p): self.model.addConstr(
            self.T[p]*self.x[h, d, p] <= self.B[h, d]) 
                             for h in self.H for d in self.D for p in self.P}

        self.num_or_lb = {(h, d): self.model.addConstr(self.y[h, d]*self.B[h, d] 
                                          >= quicksum(self.T[p]*self.x[h, d, p] 
                                                      for p in self.P))
                    for h in self.H for d in self.D}
        
class BendersLoopScheduler(BendersORScheduler):
    """OR scheduler that uses Benders' decomposition in a loop."""    
    def run_model(self):
        self.sub_solns = {}
        self.sub_room_allocs = {}
        self.sub_patient_allocs = {}
        self.sp_time = 0
        self.mp_time = 0
        self.run_gap = GRB.INFINITY
        
        ub = GRB.INFINITY
        iterations = 0
        start_time = time.time()
        
        while time.time() - start_time <= TIME_LIMIT and self.run_gap > self.gap:
            # Iteration output
            if self.verbose:
                print("Iteration", iterations)
            iterations += 1
            
            if self.verbose:
                print(UNDERLINE)
            
            # Solve master problem
            start_master_t = time.time()
            self.model.optimize()
            end_master_t = time.time()
            self.mp_time += end_master_t - start_master_t
            
            # Master problem obj val output
            if self.verbose:
                print(UNDERLINE)
            if self.verbose:
                print("Curr objVal", self.model.objVal)
            
            # Solve each sub problem
            cuts_added = [0, 0]
            start_sub_t = time.time()
            for h in self.H:
                for d in self.D:
                    self.solve_sub_problem(self.model, cuts_added, h, d, lazy=False)
            end_sub_t = time.time()
            self.sp_time += end_sub_t - start_sub_t
            
            # Cut info outputs
            if self.verbose:            
                print("Feasibility cuts added", cuts_added[0])
                print("Optimality cuts added", cuts_added[1])
                
            # Check gap and store solution suggestion if
            if cuts_added[0] == 0:
                # Run sub problems and save
                for h in self.H:
                    for d in self.D:
                        self.solve_sub_problem(self.model, cuts_added, h, d, 
                                               lazy=False, save_soln=True)
                x_hat = self.x
                u_hat = self.u
                w_hat = self.w
                
                sub_obj_val = (
                    sum(self.G[h, d]*u_hat[h, d].x 
                         for h in self.H for d in self.D) 
                    + sum(self.F[h, d]*self.sub_solns[h, d] 
                          for h in self.H for d in self.D)
                    + sum(K_1*self.rho[p]*(d - self.alpha[p])*x_hat[h, d, p].x 
                         for h in self.H for d in self.D for p in self.P)
                    + sum(K_2*self.rho[p]*(len(self.D) + 1 - self.alpha[p])*w_hat[p].x
                         for p in self.P if p not in self.mand_P)
                )
                ub = min(ub, sub_obj_val)
                master_obj_val = self.model.objVal
                self.run_gap = abs(master_obj_val - ub)/abs(ub)
                
                # Display Benders gap output
                if self.bend_gap:
                    print(master_obj_val, ub)
                    print("Gap", self.run_gap)
               
            if cuts_added[0] + cuts_added[1] == 0:
                break
            
class BendersCallbackScheduler(BendersORScheduler):
    """OR scheduler that uses Benders' decomposition in a callback."""
    def _define_model(self):
        super()._define_model()
        self.model.setParam('LazyConstraints', 1)
        
    def run_model(self):
        def callback(model, where):
            start_time = time.time()
            if where == GRB.Callback.MIPSOL:
                cuts_added = [0, 0]
                for h in self.H:
                    for d in self.D:
                        self.solve_sub_problem(model, cuts_added, h, d, lazy=True)
                
                if self.verbose:
                    print("Feasibility cuts added", cuts_added[0])
                    print("Optimality cuts added", cuts_added[1])
                    
                end_time = time.time()
                self.sp_time += end_time - start_time
                
                # # Check gap and store solution suggestion
                # if cuts_added[0] == 0 and cuts_added[1] != 0:
                #     x_hat = model.cbGetSolution(self.x)
                #     u_hat = model.cbGetSolution(self.u)
                #     w_hat = model.cbGetSolution(self.w)
                    
                #     sub_obj_val = (
                #         sum(self.G[h, d]*u_hat[h, d] 
                #              for h in self.H for d in self.D) 
                #         + sum(self.F[h, d]*self.sub_solns[h, d] 
                #               for h in self.H for d in self.D)
                #         + sum(K_1*self.rho[p]*(d - self.alpha[p])*x_hat[h, d, p] 
                #              for h in self.H for d in self.D for p in self.P)
                #         + sum(K_2*self.rho[p]*(len(self.D) + 1 - self.alpha[p])*w_hat[p]
                #              for p in self.P if p not in self.mand_P)
                #     )
                #     self.ub = min(self.ub, sub_obj_val)
                #     master_obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    
                #     if self.bend_gap:
                #         print(master_obj_val, self.ub)
                #         print("Gap", 100*abs(master_obj_val 
                #                              - self.ub)
                #               /abs(sub_obj_val), "%")
                    
            # # Suggest solution.
            # if (where == GRB.Callback.MIPNODE 
            #     and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL 
            #     and model._best_found 
            #     < model.cbGet(GRB.Callback.MIPNODE_OBJBST) - self.tol):
            #     # TODO Can only suggest solution if all sub problems are 
            #     # feasible
            #     pass
            #     model.cbSetSolution()
            #     model.cbSetSolution()
        
        self.sub_solns = {}
        self.sub_room_allocs = {}
        self.sub_patient_allocs = {}
        self.sp_time = 0
        self.mp_time = 0
        
        self.model._best_found = GRB.INFINITY
        self.ub = GRB.INFINITY
        
        start_time = time.time()
        self.model.optimize(callback)
        end_time = time.time()
        self.mp_time = (end_time - start_time) - self.sp_time
        
        self.run_gap = self.model.MIPGap
        
        # Ensures optimal sub problem allocation is stored
        for h in self.H:
            for d in self.D:
                self.solve_sub_problem(self.model, [0, 0], h, d, lazy=False,
                                       save_soln=True)
        
        