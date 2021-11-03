# Optimal control of aging in complex networks

## Citation:

**Sun, E. D., Michaels, T. C., & Mahadevan, L. (2020). Optimal control of aging in complex networks. Proceedings of the National Academy of Sciences, 117(34), 20404-20410. https://doi.org/10.1073/pnas.2006375117**

This repository contains the code associated with the paper "Optimal control of aging in complex systems" by Eric D. Sun, Thomas C.T. Michaels and L. Mahadevan at the School of Engineering and Applied Sciences, Harvard University. The code is organized into several main Python scripts:
- "model.py" which includes relevant functions for building, aging, and repairing complex networks. The script also contains methods for validating visualizing, and approximating optimal repair protocols.
- "q_learning.py" which includes functions for applying a reinforcement learning (Q-learning) framework to determining optimal repair protocols for aging complex networks.
- "ode_solver.py" which includes functions that specify the nonlinear analytical model of aging in complex systems along with methods for solving the corresponding differential equations for optimal control.
- "linear_plots.py" which includes functions for plotting the optimal control solutions (linear)

The code is written in Python3 and requires the Numpy, Scipy, NetworkX, and Matplotlib packages among other common Python libraries. Please refer to the aforementioned scripts for a full list of the imported libraries.

In addition to the scripts, this repository contains several Jupyter notebooks which utilize the functions in the scripts to generate all of the key figures in the manuscript and the supplementary material. These code notebooks also contain several tests and clarifying examples for parties interested in using our functions to analyze aging in complex systems further. Additional examples are available in the "Misc" folder. We also include the Matlab script ("optimal_control_aging.mat") that uses TOMLAB to solve the nonlinear vitality model for the optimal repair protocols in interdependent networks.
- "model_visualization.ipynb" corresponding to Figure 1BC, Figure S1
- "failure_times.ipynb" corresponding to Figure 1DEF
- "linear_optimal_control.ipynb" corresponding to Figure 2ABCD
- "nonlinear_optimal_control.ipynb" corresponding to Figure 2E
- "reinforcement_q_learning.ipynb" corresponding to Figure 3B
- "celegans_alphaktoglutarate.ipynb" corresponding to Figure 3C
- "network_topology.ipynb" corresponding to Figure S6
- "quadratic_repair_cost.ipynb" corresponding to Figure S8
- "cover_art_ideas.ipynb" corresponding to some journal cover art submissions

Some data that is used to generate the figures included in the paper are also present in this repository. They are in the correspondingly named folders:
- "Data" contains linear bang-bang simulated results and other miscellaneous data files
- "Nonlinear" contains nonlinear bang-bang simulated results for different networks
- "TOMLAB_data" contains optimal control solutions obtained via TOMLAB/propt on the nonlinear model
- "ReinforcementLearning" contains linear/nonlinear data on learned Q-matrices from Q-learning

A summary of some model parameters along with additional directions for running the code is provided in the supplementary material associated with the publication under the "Computational model" section.
