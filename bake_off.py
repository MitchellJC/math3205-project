# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:19:35 2023

@author: mitch
"""
from models import MIPScheduler
from data_gen import generate_data

P, mand_P, H, R, D, rho, alpha, health_status, B, T, G, F = generate_data(
    20, 5, output_dict=True)
mip = MIPScheduler(P, H, R, D, G, F, B, T, rho, alpha, mand_P)
mip.run_model()