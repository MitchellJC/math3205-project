# math3205-project
## Python Files
- constants.py: Shared constants across scripts.
- data_gen.py: Contains function for randomly generating all required data for OR scheduling problem, running the script as a standalone generates data and saves in csv format.
- utilities.py: Contains extra utility functions. Contains bin-packing heuristic solver.
- pure_MIP.py: Script for solving OR scheduling problem as a pure mixed integer programming problem.
- benders_loop.py: Script for solving OR scheduling problem with Benders' Cuts in a loop.
- benders_callback.py: Script for solving OR scheduling problem with Benders' Cuts in a callback.
- models.py: Contains classes for pure MIP, Benders' loop and Benders' callback.
- bake_off.py: Contains script for coordinating the execution of experiments and saving results.
