# math3205-project
## Running the Project
To run all models and generate summary outputs of time taken and final gaps use bake_off.py, script parameters are found at the top of file.

To run an individual model and produce allocation output use the pure_MIP_tester.py, network_tester.py, loop_tester.py and callback_tester.py files.

To modify any extra parameters not found in the script parameters (such as the time limit) use the constants.py file. 

## Python Files
- **bake_off.py**: Contains script for coordinating the execution of experiments and saving results.
- **constants.py**: Shared constants across scripts.
- **data_gen.py**: Contains function for randomly generating all required data for OR scheduling problem, running the script as a standalone generates data and saves in csv format.
- **utilities.py**: Contains extra utility functions. Contains bin-packing heuristic solver.
- **pure_MIP_tester.py**: Script for solving OR scheduling problem as a pure mixed integer programming problem.
- **network_tester.py**: Script for solving OR scheduling problem as a network flow problem.
- **loop_tester.py**: Script for solving OR scheduling problem with Benders' Cuts in a loop.
- **callback_tester.py**: Script for solving OR scheduling problem with Benders' Cuts in a callback.
- **models.py**: Contains classes for pure MIP, Network, Benders' loop and Benders' callback.

## Requirements 
To setup a conda environment with all necessary requirements to run the project ensure conda is installed and use the following command:
```
conda create --name <env_name> --file requirements.txt
```