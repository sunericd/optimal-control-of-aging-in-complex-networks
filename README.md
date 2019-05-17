# Optimal control of aging in complex systems

This repository contains the code associated with the paper "Optimal control of aging in complex systems" by Eric D. Sun, Thomas C.T. Michaels and L. Mahadevan at the School of Engineering and Applied Sciences, Harvard University. The code is organized into several main Python scripts:
- "model.py" which includes relevant functions for building, aging, and repairing complex networks. The script also contains methods for validating visualizing, and approximating optimal repair protocols.
- "q_learning.py" which includes functions for applying a reinforcement learning (Q-learning) framework to determining optimal repair protocols for aging complex networks.
- "ode_solver.py" which includes functions that specify the nonlinear analytical model of aging in complex systems along with methods for solving the corresponding differential equations for optimal control.
- "linear_plots.py" which includes functions for plotting the optimal control solutions (linear)

The code is written in Python3 and requires the Numpy, Scipy, NetworkX, and Matplotlib packages among other common Python libraries. Please refer to the aforementioned scripts for a full list of the imported libraries.

In addition to the scripts, this repository contains several Jupyter notebooks which utilize the functions in the scripts to generate all of the key figures in the manuscript and the supplementary material. These code notebooks also contain several tests and clarifying examples for parties interested in using our functions to analyze aging in complex systems further. Additional examples are available in the "Misc" folder. We also include the Matlab script ("optimal_control.mat") that uses TOMLAB to solve the nonlinear vitality model for the optimal repair protocols in interdependent networks.

If you find the code helpful for your own work or project, please use the following citation:

#### Sun E.D., Michaels T.C.T., Mahadevan L. "Optimal control of aging in complex systems". 2019

A summary of some model parameters along with additional directions for running the code is provided in the supplementary material associated with the publication under the "Computational model" section.
