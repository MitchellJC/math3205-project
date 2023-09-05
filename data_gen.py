# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:05:20 2023

Generates data for the operation room scheduling problem.

@author: mitch
"""
import csv
import random
import numpy as np
from scipy.stats import randint, truncnorm

SEED = 42

NUM_HOSPITALS = 3
NUM_PATIENTS = (20, 40, 60, 80, 100, 120, 140, 160)
NUM_ORS = (5, 3)
NUM_DAYS = 5

# Params
LB_SURG_TIME = 45
UB_SURG_TIME = 480
AVG_SURG_TIME = 160
STD_SURG_TIME = 40
LEFT_TIME_CLIP, RIGHT_TIME_CLIP = ( (LB_SURG_TIME - AVG_SURG_TIME) / STD_SURG_TIME, 
                                   (UB_SURG_TIME - AVG_SURG_TIME) / STD_SURG_TIME )

HEALTH_THRESHOLD = 500
URGENT_LOW = 1
URGENT_HIGH = 5

ELAPSED_WAIT_LOW = 60
ELAPSED_WAIT_HIGH = 120

POSSIBLE_REG_HOURS = range(420, 480 + 1, 15)

OPEN_HOSP_LOW = 1500
OPEN_HOSP_HIGH = 2500

OPEN_OR_LOW = 4000
OPEN_OR_HIGH = 6000

# Formats
PATIENTS_HEADER = ('id', 'surgery_time', 'urgency', 'wait_time', 'health_status', 
                   'is_mandatory')
HOSPITAL_HEADER = ('hospital_id', 'day', 'open_minutes', 'hospital_open_cost', 
                   'or_open_cost')

# Set random state
random.seed(SEED)
np.random.seed(SEED)

def generate_data(num_patients, num_or, output_dict=False, seed=None, 
                  verbose=False):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create sets
    P = range(num_patients)
    H = range(NUM_HOSPITALS)
    R = range(num_or)
    D = range(NUM_DAYS)
    
    # Create urgency data
    rho = randint.rvs(URGENT_LOW, URGENT_HIGH + 1, size=(num_patients, 1))
    
    # Create wait time data
    alpha = randint.rvs(ELAPSED_WAIT_LOW, ELAPSED_WAIT_HIGH + 1, 
                        size=(num_patients, 1))
    
    # Create health status data
    health_status = np.multiply(rho, NUM_DAYS - alpha) 
        
    # Create operating hours time data
    B = np.zeros(shape=(NUM_HOSPITALS, NUM_DAYS))
    for h in H:
        for d in D:
            B[h, d] = random.choice(POSSIBLE_REG_HOURS)
    
    # Create surgery times data
    T = truncnorm.rvs(LEFT_TIME_CLIP, RIGHT_TIME_CLIP, loc=AVG_SURG_TIME,
                      scale=STD_SURG_TIME, size=(num_patients, 1)).astype(int)
    
    # Create cost of opening hospital data
    G = randint.rvs(OPEN_HOSP_LOW, OPEN_HOSP_HIGH + 1, 
                    size=(NUM_HOSPITALS, NUM_DAYS))
    
    # Create cost of opening operating room data
    F = randint.rvs(OPEN_OR_LOW, OPEN_OR_HIGH + 1, 
                    size=(NUM_HOSPITALS, NUM_DAYS))
    
    # Create set of mandatory patients
    mandatory_P = [p for p in P if health_status[p, 0] <= -HEALTH_THRESHOLD]
    if verbose:
        print("\tPercentage of patients that are mandatory (Expected 6-10%)", 
              f"{100*len(mandatory_P) / num_patients}%")
    
    if output_dict:
        G = {(h, d): int(G[h, d]) for h in H for d in D}
        F = {(h, d): int(F[h, d]) for h in H for d in D}
        B = {(h, d): int(B[h, d]) for h in H for d in D}
        T = {p: int(T[p]) for p in P}
        rho = {p: int(rho[p]) for p in P}
        alpha = {p: int(alpha[p]) for p in P}
    
    return P, mandatory_P, H, R, D, rho, alpha, health_status, B, T, G, F

def main():
    for num_or in NUM_ORS:
        print(f"Generating data for {num_or} operating rooms")
        for num_patients in NUM_PATIENTS:
            print(f"\tGenerating data for {num_patients} patients")
            
            P, mandatory_P, H, R, D, rho, alpha, health_status, B, T, G, F = (
                generate_data(num_patients, num_or) )
            
            # Save patients data
            with open(f'data/patients-{num_or}-{num_patients}.csv', 'w', newline='', 
                      encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(PATIENTS_HEADER)
                for p in P:
                    data = [p, T[p, 0], rho[p, 0], alpha[p, 0], health_status[p, 0], 
                            1 if p in mandatory_P else 0]
                    writer.writerow(data)
                    
            # Save hospital data
            with open(f'data/hospitals-{num_or}-{num_patients}.csv', 'w', newline='', 
                      encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(HOSPITAL_HEADER)
                for h in H:
                    for d in D:
                        data = [h, d, B[h, d], G[h, d], F[h, d]]
                        writer.writerow(data)

if __name__ == '__main__':
    main()