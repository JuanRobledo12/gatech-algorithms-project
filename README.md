# Simulated Annealing Algorithm to Solve the Knapsack Problem

In this repository, I present my contribution to the final project for CSE6140 Computer Science and Engineering Algorithms course at Georgia Tech. The project consists of implementing a series of algorithms to cope with NP-Complete problems and learning how to evaluate their performance empirically. My contribution to this project was the simulated annealing algorithm, which is a stochastic local search algorithm that can be used to achieve good results when dealing with this type of problems. For more, please check the final report [here](Documents/final_report.pdf).

## Files

- `knapsack_sa.py`: This file contains the implementation of a custom simulated annealing algorithm that can be imported as a class in other scripts. My implementation allows the user to set random seeds for reproducibility as well as choosing from a variety of parameters to achieve better performances. Some of these parameters are the use of greedy initialization, restarts, automatic determination of the initial temperature, etc.

- `analysis_nb.ipynb`: This notebook has the analysis of the performance of the algorithm when using a specific set of parameters in different input files with distinct input sizes. This notebook can be used as a reference on how to use the algorithm properly.
