
# coding: utf-8

# # A Network Model for Aging
# ### Eric Sun
# ### Last updated 2/1/2019

# ### Outdated documentation

# simPopulation - iterates over 'pop_size' number of individuals and averages results
# 
# simIndividual - main function that runs the model for one individual
#     1. graph_type - 'scale_free', 'Grandom', 'ERRandom' (default) with identifier + '_s' for symmetric or + '_d' for directed
#     2. N - number of nodes
#     3. p - probability of attachment for random network :: degree/sum(degrees) probability for scale-free
#     4. (REMOVED) G - number of iterations to run, 'yolo' to run until all nodes failed
#     5. d - proportion of initial failed nodes (=1), for perfect initial system (d=0)
#     6. f - independent probability of failure
#     7. r - independent probability of repair (only for non-checking)
#     8. f_thresh - threshold fraction of live nodes below which system failure occurs
#     9. weight_type - 'uniform' (default) or 'degree' proportional weighting for contribution of node to vitality
#     
#     
#     threshold fraction of connections needed to force node failure, set to 0.0 for no threshold
# 
#     - Initializes adjacency matrix from parameters
#     - Runs iterations with checking strategy
#     - Reports results
# 
# Check(choice, kinetic_boolean, P_check, e)
#     1. Uniform - each node has check probability 'P_check', repair error 'e'
#     2. Biased - each node has check probability EXP[-(degree_k/tot_degree)/P_check], repair error 'e'
#     3. Kinetic - each node has check probability uniform or biased, repair error 'e'^N where N = number of proofs
# 
# Cost(costC, costR, costL):
#     1. 'longevity' - cost = cost = total(costR + costC) - longevity(costL*failure_time)
#     2. 'energy' - cost = total(costR + costC) - energy(costL*live_nodes)
#     
#     - costC - cost of checking; for each iteration, this is ~ N_checks times costC
#     - costR - cost of repair
#     - costL - cost of longevity/living (is subtractive; offsets costs)
# 
# Analyze - function that generates results and statistics from adjacency matrix
#     - For population of organisms:
#     1. Mortality Rate - mu(t) = -[s(t+1)-s(t)]/s(t), where s(t) is the fraction of live individuals at time t
#     
#     - For each individual (network):
#     2. Vitality - phi(t) = SUM(nodes)/N is the average "fitness"
#     3. Interdependence - lambda(phi(t)) = log(phi(t))/log(phi0(t)) where phi0(t) = exp[(-f+r)t] is the expectation
# 
# Report - generates figures, writes results file
#     - Generates figures for s / mortality rates / vitality / interdependence across time
#     - Saves figures as PNG file and results (mu, phi, lambda) as CSV file

# In[5]:

# Import necessary packages
#  %matplotlib notebook # run notebook inline
import numpy as np
import random
import math
import csv
import matplotlib.pyplot as plt
import networkx as nx
import os
import time
from tqdm import tqdm 
from mpl_toolkits.mplot3d import Axes3D


# In[6]:

def readInputs (input_file_path):
    '''
    Function for reading of an input file specifying the parameters of simPopulation()
    input_file_path (str) - path to the input text file
        First line of file are the tab-separated parameter names 
        Second line of the file are the tab-separated parameter values
    '''
    #print ('Reading input file at: ' + input_file_path)
    with open(input_file_path, 'rt') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        file_list = list(tsvin)
        
    # second row corresponds to input values
    spec_list = file_list[1]
    
    filename = str(spec_list[0])
    pop_size = int(spec_list[1])
    N = int(spec_list[2])
    p = float(spec_list[3])
    d = float(spec_list[4])
    f = float(spec_list[5])
    r = float(spec_list[6])
    f_thresh = float(spec_list[7])
    graph_type = str(spec_list[8])
    weight_type = str(spec_list[9])
    check_type = str(spec_list[10])
    kinetic = int(spec_list[11])
    P_check = float(spec_list[12])
    e = float(spec_list[13])
    cost_type = str(spec_list[14])
    costC = float(spec_list[15])
    costR = float(spec_list[16])
    costE = float(spec_list[17])
    costD = float(spec_list[18])
    costL = float(spec_list[19])
    P_repl = float(spec_list[20])
    costrepl = float(spec_list[21])
    max_repl = int(spec_list[22])
    repl_type = str(spec_list[23])
    node_type = str(spec_list[24])
    damage_type = str(spec_list[25])
    edge_type = str(spec_list[26])
    f_edge = float(spec_list[27])
    r_edge = float(spec_list[28])
    P_ablate = float(spec_list[29])
    costablate = float(spec_list[30])
    ablate_type = str(spec_list[31])
    repair_start = int(spec_list[32])
    repair_end = str(spec_list[33]) # str or int
    delay = int(spec_list[34])
    time_end = int(spec_list[35])
    dependency = float(spec_list[36])
    
    save = str(spec_list[37])
    plot = str(spec_list[38])
    write_inds = str(spec_list[39])
    
    std = 0.3
        
    # Printing parameter values
    for i, parameter in enumerate(file_list[0]):
        if i != 0 and i != len(spec_list)-1:
            print (str(parameter)+': '+str(spec_list[i]))
    
    # Cost_type
    if ' ,' in cost_type:
        cost_type = cost_type.split(' ,')
    elif ',' in cost_type:
        cost_type = cost_type.split(',')
    else:
        print ('Not using a cost_type...')
        cost_type = []
 
    simPopulation(filename, pop_size, N, p, d, f, r, f_thresh, graph_type, weight_type, check_type, kinetic, P_check, 
                  e, cost_type, costC, costR, costE, costD, costL, P_repl, costrepl, max_repl, repl_type,
                  node_type, damage_type, edge_type, f_edge, r_edge, std, P_ablate, costablate, ablate_type, 
                  repair_start, repair_end, delay, time_end, dependency, save, plot, write_inds)


# In[7]:

def optimizeBangBang_cluster (exp_name, pop_size, dependency, parameter_type, parameter_list, T1_list, T2_list, 
                              T_std, highres_step, T):
    '''
    Function to call to run a grid search for optimal start (T1) and stop (T2) times for a given parameter range on
    a SLURM-type computing cluster. Considers a repair-only model. USES A FIRST PASS, LOW-RES GRID SEARCH FOLLOWED BY
    A SECOND PASS, HIGH-RES GRID SEARCH ON A NEIGHBORHOOD AROUND THE FIRST PASS OPTIMUM.
    
    exp_name (str) - basename for the saved results file
    pop_size (int) - number of networks to use in the population
    dependency (float) - intederpendency, I
    parameter_type (str) - 'f', 'r', 'a' to specify the parameter that is variable
    parameter_list (list) - list or array of values that the parameter specified by parameter_type takes on
    T1_list (list) - list or array of T1 values to perform the first grid search pass
    T2_list (list) - list or array of T2 values to perform the first grid search pass
    T_std (int) - range (optimal T +- T_std) for the second high-res grid search pass
    highres_step (int) - step size to do the second grid search pass
    T (int) - final stop time for simulations, T1 < T, T2 < T
    '''
    
    if not os.path.exists('./BangBang'):
        os.makedirs('./BangBang')
    
    if not os.path.exists('./BangBang/T1_T2_mins/'):
        os.makedirs('./BangBang/T1_T2_mins/')
        
    if not os.path.exists('./BangBang/T1_T2_minsData/'):
        os.makedirs('./BangBang/T1_T2_minsData/')
        
    if not os.path.exists('./BangBang/ParamCurves/'):
        os.makedirs('./BangBang/ParamCurves/')
    
    if not os.path.exists('./BangBang/ParamCurvesData/'):
        os.makedirs('./BangBang/ParamCurvesData/')
    
    min_costs = []
    best_T1 = []
    best_T2 = []
    
    f = 0.025
    r = 0.01
    a = 10
    
    networks = [] # list of pop_size pre-made networks (averaged over for each param value)
    for i in range(pop_size):
        pA, pv = initIndividual (N=1000, graph_type='Grandom_s', p=0.1, d=0, edge_type='binary')
        networks.append([pA, pv])
    
    for p in tqdm(parameter_list):
        
        if parameter_type == 'f':
            f = p
        elif parameter_type == 'r':
            r = p
        elif parameter_type == 'a':
            a = p
        else:
            raise Exception("parameter_type is not valid!")
        
        save_tag = 'f'+str(f)+'_r'+str(r)+'_a'+str(a)+'_T'+str(T)+'_step'+str(highres_step)+'_d0_depoff_N1000'
        
        cost_results = []
        T1_results = []
        T2_results = []
        
        for t1 in T1_list:
            for t2 in T2_list:
                if t2 > t1:
                    T1_results.append(t1)
                    T2_results.append(t2)
                    cost = simPopulation(exp_name, pop_size=pop_size, f=f, r=r, graph_type='none',
                                         cost_type=['healthspan', a], repair_start=t1, repair_end=t2, 
                                         time_end=T, dependency=dependency, plot='no', preNet_list=networks)
                    cost_results.append(cost)
                    
        # Find best (T1, T2) to minimize cost
        min_cost, min_idx = min((val, idx) for (idx, val) in enumerate(cost_results))
        minT1 = T1_results[min_idx]
        minT2 = T2_results[min_idx]
        
        # HIGH-RES STEP
        
        cost_results = []
        T1_results = []
        T2_results = []
        
        #Loop with higher resolution around current min +- T_std
        for t1 in range(minT1-T_std, minT1+T_std+1, highres_step):
            if t1 > 0:
                for t2 in range(minT2-T_std, minT2+T_std+1, highres_step):
                    if t2 > t1:
                        T1_results.append(t1)
                        T2_results.append(t2)
                        cost = simPopulation(exp_name, pop_size=pop_size, f=f, r=r, graph_type='none',
                                             cost_type=['healthspan', a], repair_start=t1, repair_end=t2, 
                                             time_end=T, dependency=dependency, plot='no', preNet_list=networks)
                        cost_results.append(cost)
        
        # Find best (T1, T2) to minimize cost
        min_cost, min_idx = min((val, idx) for (idx, val) in enumerate(cost_results))
        minT1 = T1_results[min_idx]
        minT2 = T2_results[min_idx]
        
        min_costs.append(min_cost)
        best_T1.append(minT1)
        best_T2.append(minT2)
        
        # Saving results
        save_list = [[min_cost, minT1, minT2], cost_results, T1_results, T2_results]
        file = open('./BangBang/T1_T2_minsData/'+save_tag+'.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerows(save_list)
        
        # Plot results for f, r, a, d=0, depoff, N=1000
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        ax.plot_trisurf(T1_results, T2_results, cost_results)
#        ax.set_xlabel('T1')
#        ax.set_ylabel('T2')
#        ax.set_zlabel('Cost')
#        ax.text(minT1, minT2, min_cost, min_statement)
#        plt.savefig('./BangBang/T1_T2_mins/'+save_tag+'_SURF.png')
#        plt.close()

#        fig2 = plt.figure()
#        ax = fig2.add_subplot(111, projection='3d')
#        ax.scatter(T1_results, T2_results, cost_results)
#        ax.set_xlabel('T1')
#        ax.set_ylabel('T2')
#        ax.set_zlabel('Cost')
#        ax.text(minT1, minT2, min_cost, min_statement)
#        plt.savefig('./BangBang/T1_T2_mins/'+save_tag+'_SCATT.png')
#        plt.close()
        
        time.sleep(3)

    # Plot best T1 and T2 curves with x-axis as param
    #fig3 = plt.figure()
    #plt.plot(parameter_list, best_T1, 'g--', label='T1')
    #plt.plot(parameter_list, best_T2, 'r--', label='T2')
    #plt.title('Bang-Bang Optimal Repair ('+parameter_type+')')
    #plt.xlabel(parameter_type)
    #plt.ylabel('T')
    #plt.legend(loc='upper right')
    new_save_tag = 'vary'+parameter_type+'_'+save_tag
    #plt.savefig('./BangBang/ParamCurves/'+new_save_tag+'.png', dpi=800)
    #plt.show()
    
    # Save min_costs, bestT1, bestT2
    save_list = [min_costs, best_T1, best_T2]
    file = open('./BangBang/ParamCurvesData/'+new_save_tag+'.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerows(save_list)


# In[8]:

def optimizeBangBang (exp_name, pop_size, dependency, parameter_type, parameter_list, T1_list, T2_list, T):
    '''
    Function to call to run a grid search for optimal start (T1) and stop (T2) times for a given parameter range for a
    short duration. Considers a repair-only model. ONLY FIRST LOW-RES GRID SEARCH.
    
    exp_name (str) - basename for the saved results file
    pop_size (int) - number of networks to use in the population
    dependency (float) - intederpendency, I
    parameter_type (str) - 'f', 'r', 'a' to specify the parameter that is variable
    parameter_list (list) - list or array of values that the parameter specified by parameter_type takes on
    T1_list (list) - list or array of T1 values to perform the first grid search pass
    T2_list (list) - list or array of T2 values to perform the first grid search pass
    '''
    
    
    if not os.path.exists('./BangBang'):
        os.makedirs('./BangBang')
    
    if not os.path.exists('./BangBang/T1_T2_mins/'):
        os.makedirs('./BangBang/T1_T2_mins/')
        
    if not os.path.exists('./BangBang/T1_T2_minsData/'):
        os.makedirs('./BangBang/T1_T2_minsData/')
        
    if not os.path.exists('./BangBang/ParamCurves/'):
        os.makedirs('./BangBang/ParamCurves/')
    
    if not os.path.exists('./BangBang/ParamCurvesData/'):
        os.makedirs('./BangBang/ParamCurvesData/')
    
    min_costs = []
    best_T1 = []
    best_T2 = []
    
    f = 0.025
    r = 0.01
    a = 0.1
    
    for p in tqdm(parameter_list):
        
        if parameter_type == 'f':
            f = p
        elif parameter_type == 'r':
            r = p
        elif parameter_type == 'a':
            a = p
        else:
            raise Exception("parameter_type is not valid!")
        
        save_tag = 'f'+str(f)+'_r'+str(r)+'_a'+str(a)+'_T'+str(T)+'_d0_depoff_N1000'
        
        cost_results = []
        T1_results = []
        T2_results = []
        for t1 in T1_list:
            for t2 in T2_list:
                if t2 > t1:
                    T1_results.append(t1)
                    T2_results.append(t2)
                    cost = simPopulation(exp_name, pop_size=pop_size, f=f, r=r, graph_type='none',
                                         cost_type=['healthspan_norm', a], repair_start=t1, repair_end=t2, 
                                         time_end=T, dependency=dependency, plot='no')
                    cost_results.append(cost)
                    
        # Find best (T1, T2) to minimize cost
        min_cost, min_idx = min((val, idx) for (idx, val) in enumerate(cost_results))
        minT1 = T1_results[min_idx]
        minT2 = T2_results[min_idx]
        min_statement = "Minimum cost: " + str(min_cost) +' @ T1 = ' + str(minT1) + ', T2 = ' + str(minT2)
        
        # ADD another for loop with higher resolution around current min +- T_std
        
        min_costs.append(min_cost)
        best_T1.append(minT1)
        best_T2.append(minT2)
        
        # Saving results
        save_list = [[min_cost, minT1, minT2], cost_results, T1_results, T2_results]
        file = open('./BangBang/T1_T2_minsData/'+save_tag+'.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerows(save_list)
        
        # Plot results for f, r, a, d=0, depoff, N=1000
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(T1_results, T2_results, cost_results)
        ax.set_xlabel('T1')
        ax.set_ylabel('T2')
        ax.set_zlabel('Cost')
        ax.text(minT1, minT2, min_cost, min_statement)
        plt.savefig('./BangBang/T1_T2_mins/'+save_tag+'_SURF.png')
        plt.close()

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.scatter(T1_results, T2_results, cost_results)
        ax.set_xlabel('T1')
        ax.set_ylabel('T2')
        ax.set_zlabel('Cost')
        ax.text(minT1, minT2, min_cost, min_statement)
        plt.savefig('./BangBang/T1_T2_mins/'+save_tag+'_SCATT.png')
        plt.close()
        
        time.sleep(3)

    # Plot best T1 and T2 curves with x-axis as param
    fig3 = plt.figure()
    plt.plot(parameter_list, best_T1, 'g--', label='T1')
    plt.plot(parameter_list, best_T2, 'r--', label='T2')
    plt.title('Bang-Bang Optimal Repair ('+parameter_type+')')
    plt.xlabel(parameter_type)
    plt.ylabel('T')
    plt.legend(loc='upper right')
    new_save_tag = 'vary'+parameter_type+'_'+save_tag
    plt.savefig('./BangBang/ParamCurves/'+new_save_tag+'.png', dpi=800)
    plt.show()
    
    # Save min_costs, bestT1, bestT2
    save_list = [min_costs, best_T1, best_T2]
    file = open('./BangBang/ParamCurvesData/'+new_save_tag+'.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerows(save_list)


# In[9]:

def optimizeChecking_cluster (exp_name, pop_size, dependency, delay, parameter_type, parameter_list, 
                              cT1_list, cT2_list, T_std, highres_step, T):
    
    '''
    Function to call to run a grid search for optimal start (T1) and stop (T2) times for a given parameter range on
    a SLURM-type computing cluster. Considers a checking-then-repair model. USES A FIRST PASS, LOW-RES GRID SEARCH FOLLOWED BY
    A SECOND PASS, HIGH-RES GRID SEARCH ON A NEIGHBORHOOD AROUND THE FIRST PASS OPTIMUM.
    
    exp_name (str) - basename for the saved results file
    pop_size (int) - number of networks to use in the population
    dependency (float) - intederpendency, I
    parameter_type (str) - 'f', 'r', 'a' to specify the parameter that is variable
    parameter_list (list) - list or array of values that the parameter specified by parameter_type takes on
    cT1_list (list) - list or array of T1 values to perform the first grid search pass
    cT2_list (list) - list or array of T2 values to perform the first grid search pass
    T_std (int) - range (optimal T +- T_std) for the second high-res grid search pass
    highres_step (int) - step size to do the second grid search pass
    T (int) - final stop time for simulations, T1 < T, T2 < T
    '''
    
    # cT1 refers to start time of checking
    # T1 refers to start time of repair = check start + delay
    
    if not os.path.exists('./BangBang'):
        os.makedirs('./BangBang')
    
    if not os.path.exists('./BangBang/T1_T2_mins/'):
        os.makedirs('./BangBang/T1_T2_mins/')
        
    if not os.path.exists('./BangBang/T1_T2_minsData/'):
        os.makedirs('./BangBang/T1_T2_minsData/')
        
    if not os.path.exists('./BangBang/ParamCurves/'):
        os.makedirs('./BangBang/ParamCurves/')
    
    if not os.path.exists('./BangBang/ParamCurvesData/'):
        os.makedirs('./BangBang/ParamCurvesData/')
    
    min_costs = []
    best_T1 = []
    best_T2 = []
    
    # Get repair start and end from checking range
    T1_list = [ct1+delay for ct1 in cT1_list]
    T2_list = [ct2+delay for ct2 in cT2_list]
    
    f = 0.025
    r = 0.01
    a1 = 0.01
    a2 = 0.1
    
    networks = [] # list of pop_size pre-made networks (averaged over for each param value)
    for i in range(pop_size):
        pA, pv = initIndividual (N=1000, graph_type='Grandom_s', p=0.1, d=0, edge_type='binary')
        networks.append([pA, pv])
    
    for p in tqdm(parameter_list):
        
        if parameter_type == 'f':
            f = p
        elif parameter_type == 'r':
            r = p
        elif parameter_type == 'a1':
            a1 = p
        elif parameter_type == 'a1':
            a2 = p
        else:
            raise Exception("parameter_type is not valid!")
        
        save_tag = 'Check_f'+str(f)+'_r'+str(r)+'_a1'+str(a1)+'_a2'+str(a2)+'_T'+str(T)+'_step'+str(highres_step)+'_delay'+str(delay)+'_d0_depoff_N1000'
        
        cost_results = []
        T1_results = []
        T2_results = []
        for t1 in T1_list:
            for t2 in T2_list:
                if t2 > t1:
                    T1_results.append(t1)
                    T2_results.append(t2)
                    cost = simPopulation(exp_name, pop_size=pop_size, f=f, r=r, graph_type='none', check_type='none',
                                         P_check = 0.01, cost_type=['checking_delay', a1, a2], repair_start=t1, repair_end=t2, 
                                         time_end=T, dependency=dependency, plot='no', preNet_list=networks)
                    cost_results.append(cost)
                    
        # Find best (T1, T2) to minimize cost
        min_cost, min_idx = min((val, idx) for (idx, val) in enumerate(cost_results))
        minT1 = T1_results[min_idx]
        minT2 = T2_results[min_idx]
        
        # HIGH RES STEP
        cost_results = []
        T1_results = []
        T2_results = []
        
        for t1 in range(minT1-T_std, minT1+T_std+1, highres_step):
            if t1 > 0:
                for t2 in range(minT2-T_std, minT2+T_std+1, highres_step):
                    if (t2 > t1) and (t2 < T):
                        T1_results.append(t1)
                        T2_results.append(t2)
                        cost = simPopulation(exp_name, pop_size=pop_size, f=f, r=r, graph_type='none', check_type='none',
                                             P_check = 0.01, cost_type=['checking_delay', a1, a2], repair_start=t1, repair_end=t2, 
                                             time_end=T, dependency=dependency, plot='no')
                        cost_results.append(cost)
        
        min_cost, min_idx = min((val, idx) for (idx, val) in enumerate(cost_results))
        minT1 = T1_results[min_idx]
        minT2 = T2_results[min_idx]
        
        min_costs.append(min_cost)
        best_T1.append(minT1)
        best_T2.append(minT2)
        
        # Saving results
        save_list = [[min_cost, minT1, minT2], cost_results, T1_results, T2_results]
        file = open('./BangBang/T1_T2_minsData/'+save_tag+'.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerows(save_list)
        
#        cT1_results = [t1-delay for t1 in T1_results]
#        cT2_results = [t2-delay for t2 in T2_results]
        
        # Plot results for f, r, a, d=0, depoff, N=1000
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        ax.plot_trisurf(cT1_results, cT2_results, cost_results)
#        ax.set_xlabel('checking T1')
#        ax.set_ylabel('checking T2')
#        ax.set_zlabel('Cost')
#        ax.text(minT1-delay, minT2-delay, min_cost, min_statement)
#        plt.savefig('./BangBang/T1_T2_mins/'+save_tag+'_SURF.png')
#        plt.close()

#        fig2 = plt.figure()
#        ax = fig2.add_subplot(111, projection='3d')
#        ax.scatter(cT1_results, cT2_results, cost_results)
#        ax.set_xlabel('checking T1')
#        ax.set_ylabel('checking T2')
#        ax.set_zlabel('Cost')
#        ax.text(minT1-delay, minT2-delay, min_cost, min_statement)
#        plt.savefig('./BangBang/T1_T2_mins/'+save_tag+'_SCATT.png')
#        plt.close()
        
        time.sleep(3)
    
    #convert from repair to checking times
    best_cT1 = [t1-delay for t1 in best_T1]
    best_cT2 = [t2-delay for t2 in best_T2]
    
    # Plot best T1 and T2 curves with x-axis as param
    fig3 = plt.figure()
    plt.plot(parameter_list, best_cT1, 'g--', label='T1')
    plt.plot(parameter_list, best_cT2, 'r--', label='T2')
    plt.title('Bang-Bang Optimal Checking ('+parameter_type+')')
    plt.xlabel(parameter_type)
    plt.ylabel('T')
    plt.legend(loc='upper right')
    new_save_tag = 'vary'+parameter_type+'_'+save_tag
    plt.savefig('./BangBang/ParamCurves/'+new_save_tag+'.png', dpi=800)
    plt.show()
    
    # Save min_costs, bestT1, bestT2
    save_list = [min_costs, best_cT1, best_cT2]
    file = open('./BangBang/ParamCurvesData/'+new_save_tag+'.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerows(save_list)


# In[10]:

def optimizeChecking (exp_name, pop_size, dependency, delay, parameter_type, parameter_list, cT1_list, cT2_list, T):
    
    '''
    Function to call to run a grid search for optimal start (T1) and stop (T2) times for a given parameter range for a
    short duration. Considers a checking-then-repair model. ONLY FIRST LOW-RES GRID SEARCH.
    
    exp_name (str) - basename for the saved results file
    pop_size (int) - number of networks to use in the population
    dependency (float) - intederpendency, I
    parameter_type (str) - 'f', 'r', 'a' to specify the parameter that is variable
    parameter_list (list) - list or array of values that the parameter specified by parameter_type takes on
    cT1_list (list) - list or array of T1 values to perform the first grid search pass
    cT2_list (list) - list or array of T2 values to perform the first grid search pass
    '''
    
    # cT1 refers to start time of checking
    # T1 refers to start time of repair = check start + delay
    
    if not os.path.exists('./BangBang'):
        os.makedirs('./BangBang')
    
    if not os.path.exists('./BangBang/T1_T2_mins/'):
        os.makedirs('./BangBang/T1_T2_mins/')
        
    if not os.path.exists('./BangBang/T1_T2_minsData/'):
        os.makedirs('./BangBang/T1_T2_minsData/')
        
    if not os.path.exists('./BangBang/ParamCurves/'):
        os.makedirs('./BangBang/ParamCurves/')
    
    if not os.path.exists('./BangBang/ParamCurvesData/'):
        os.makedirs('./BangBang/ParamCurvesData/')
    
    min_costs = []
    best_T1 = []
    best_T2 = []
    
    # Get repair start and end from checking range
    T1_list = [ct1+delay for ct1 in cT1_list]
    T2_list = [ct2+delay for ct2 in cT2_list]
    
    f = 0.025
    r = 0.01
    a1 = 0.01
    a2 = 0.1
    
    for p in tqdm(parameter_list):
        
        if parameter_type == 'f':
            f = p
        elif parameter_type == 'r':
            r = p
        elif parameter_type == 'a1':
            a1 = p
        elif parameter_type == 'a1':
            a2 = p
        else:
            raise Exception("parameter_type is not valid!")
        
        save_tag = 'Check_f'+str(f)+'_r'+str(r)+'_a1'+str(a1)+'_a2'+str(a2)+'_T'+str(T)+'_delay'+str(delay)+'_d0_depoff_N1000'
        
        cost_results = []
        T1_results = []
        T2_results = []
        for t1 in T1_list:
            for t2 in T2_list:
                if t2 > t1:
                    T1_results.append(t1)
                    T2_results.append(t2)
                    cost = simPopulation(exp_name, pop_size=pop_size, f=f, r=r, graph_type='none', check_type='none',
                                         P_check = 0.01, cost_type=['checking_delay', a1, a2], repair_start=t1, repair_end=t2, 
                                         time_end=T, dependency=dependency, plot='no')
                    cost_results.append(cost)
                    
        # Find best (T1, T2) to minimize cost
        min_cost, min_idx = min((val, idx) for (idx, val) in enumerate(cost_results))
        minT1 = T1_results[min_idx]
        minT2 = T2_results[min_idx]
        min_statement = "Minimum cost: " + str(min_cost) +' @ T1 = ' + str(minT1-delay) + ', T2 = ' + str(minT2-delay)
        
        min_costs.append(min_cost)
        best_T1.append(minT1)
        best_T2.append(minT2)
        
        # Saving results
        save_list = [[min_cost, minT1, minT2], cost_results, T1_results, T2_results]
        file = open('./BangBang/T1_T2_minsData/'+save_tag+'.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerows(save_list)
        
        cT1_results = [t1-delay for t1 in T1_results]
        cT2_results = [t2-delay for t2 in T2_results]
        
        # Plot results for f, r, a, d=0, depoff, N=1000
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(cT1_results, cT2_results, cost_results)
        ax.set_xlabel('checking T1')
        ax.set_ylabel('checking T2')
        ax.set_zlabel('Cost')
        ax.text(minT1-delay, minT2-delay, min_cost, min_statement)
        plt.savefig('./BangBang/T1_T2_mins/'+save_tag+'_SURF.png')
        plt.close()

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.scatter(cT1_results, cT2_results, cost_results)
        ax.set_xlabel('checking T1')
        ax.set_ylabel('checking T2')
        ax.set_zlabel('Cost')
        ax.text(minT1-delay, minT2-delay, min_cost, min_statement)
        plt.savefig('./BangBang/T1_T2_mins/'+save_tag+'_SCATT.png')
        plt.close()
        
        time.sleep(3)
    
    # convert from repair to checking times
    best_cT1 = [t1-delay for t1 in best_T1]
    best_cT2 = [t2-delay for t2 in best_T2]
    
    # Plot best T1 and T2 curves with x-axis as param
    fig3 = plt.figure()
    plt.plot(parameter_list, best_cT1, 'g--', label='T1')
    plt.plot(parameter_list, best_cT2, 'r--', label='T2')
    plt.title('Bang-Bang Optimal Checking ('+parameter_type+')')
    plt.xlabel(parameter_type)
    plt.ylabel('T')
    plt.legend(loc='upper right')
    new_save_tag = 'vary'+parameter_type+'_'+save_tag
    plt.savefig('./BangBang/ParamCurves/'+new_save_tag+'.png', dpi=800)
    plt.show()
    
    # Save min_costs, bestT1, bestT2
    save_list = [min_costs, best_cT1, best_cT2]
    file = open('./BangBang/ParamCurvesData/'+new_save_tag+'.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerows(save_list)


# In[11]:

def examineMultipliers (name, mrange, m_idx, check_type, cost_type, graph_type='Grandom_s', num_networks=50, repair_end=100, dependency=0):
    '''
    Function for multiplying the cost of repair by a multiplier (m) across a list of m values. A scatter plot is 
    generated where the cost (obtained from the model) is plotted as a function of m
    
    Used primarily for the quadratic repair cost problem.
    '''
    
    costs = []
    m_list = np.arange(mrange[0],mrange[1],mrange[2])
    for m in m_list:
        check_type[m_idx] = m
        cost = simPopulation('numeric', pop_size=num_networks, N=1000, p=0.1, d=0, f=0.025, r=0, f_thresh=0.01,
                      graph_type=graph_type, weight_type='uniform', check_type=check_type, kinetic=1, P_check=1, e=0, cost_type=cost_type, 
                      costC=0.1, costR=1, costE=0.5, costD=0.5, costL=1, P_repl=0, costrepl=1, max_repl=1, repl_type='constant',
                      node_type='binary', damage_type='uniform', edge_type='binary', f_edge=0, r_edge=0, std=0.3, 
                      P_ablate=0,costablate=1,ablate_type='constant',repair_start=0,repair_end=repair_end,delay=0,time_end=repair_end,dependency=dependency,
                             save='no', plot='no')
        costs.append(cost)
    
    save_list = [m_list, costs]
    file = open(name+'.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerows(save_list)
    
    plt.figure()
    plt.scatter(m_list, costs)
    plt.xlabel('m')
    plt.ylabel('cost')
    plt.title(name)
    #plt.savefig(name+'.png', dpi=500)
    plt.show()


# In[12]:

def simPopulation (filename, pop_size=1, N=1000, p=0.1, d=0, f=0.025, r=0.01, f_thresh=0.1, 
                   graph_type='Grandom_s', weight_type='uniform', check_type='none', 
                   kinetic=1, P_check=0.01, e=0, cost_type=['basic'], costC=0, costR=0, costE=0, costD=0, costL=0, 
                   P_repl=0, costrepl=0, max_repl=1, repl_type='constant', node_type='binary', damage_type='uniform',
                   edge_type='binary', f_edge=0, r_edge=0, std=0.3, 
                   P_ablate=0, costablate=0, ablate_type='constant', repair_start=0, repair_end='none', delay=0,
                   time_end='none', dependency=0, save='no', plot='yes', write_inds='no', preNet_list=False,
                   equilibrate_failures=True):

    '''
    Main function for simulating a population of aging networks. Takes in all model parameters, runs simIndividual for 
    each of the members in the population, and reports population-level and averaged statistics.
    
    Many parameters are not present in the model highlighted in "Optimal control of aging in complex systems". 
    These are prefaced with "[OPT]".
    
    filename (str) - basename of saved results file
    pop_size (int) - number of networks in population
    N (int) - size of network (number of nodes)
    p (float) - probability of an edge between two nodes in Gilbert model (implementation varies slightly for ER/BA)
    d (float) - prenatal damage
    f (float) - failure rate (probability)
    r (float) - repair rate (probability)
    f_thresh (float) - threshold fraction of live nodes required to avoid network death
    graph_type (str) - 'Grandom_s' (symmetric Gilbert random), 'Grandom_d' (non-symmetric Gilbert random)
                       'ERrandom_s' (symmetric Erdos-Renyi random), 'ERrandom_d' (non-symmetric Erdos-Renyi random)
                       'scale_free_s' (symmetric Barabasi-Albert scale-free), 'scale_free_s' (non-symmetric BA)
    [OPT] weight_type (str) - 'uniform' (all nodes are equally weighted in vitality), 'biased' (nodes weighted by degree)
    check_type (str) - different checking strategies ('none', 'uniform', 'biased', 'age', 'block', 'time')
                       'none' = repair-only, 'uniform' = checking-then-repair
                       refer to Check_and_Repair() for other check_type's
    [OPT] kinetic (int) - number of additional proofreads/repair iterations at each time step
    P_check (float) - checking rate (probability)
    [OPT] e (float) - error of checking
    cost_type (str) - cost function used at each time step
    [OPT] costC (float) - check
    [OPT] costR (float) - repair
    [OPT] costE (float) - error
    [OPT] costD (float) - death
    [OPT] costL (float) - longevity
    [OPT] P_repl (float) - probability of a node replicatin at each time step
    [OPT] costrepl (float) - replication
    [OPT] maxrepl (int) - max number of replication events per node (i.e. Hayflick limit)
    [OPT] repl_type (str) - 'constant', 'expdec' (exponentially decaying)
    [OPT] node_type (str) - 'binary' (node take values of 1=alive, 0=dead), 'continuous' (nodes take value between 0-1)
    [OPT] damage_type (str) - 
    [OPT] edge_type (str) - 'binary' (edge take values of 1=present, 0=not), 'continuous' (edges take value between 0-1)
    [OPT] f_edge (float) - edge failure rate (probability)
    [OPT] r_edge (float) - edge repair rate (probability)
    [OPT] std (float) - standard deviation for 'continuous' initial node values, damage, and repair
    [OPT] P_ablate (float) - probability of ablation (removing a failed node)
    [OPT] costablate (float) - ablation
    [OPT] ablate_type (str) - 'constant', 'expdec' (exponentially decaying)
    repair_start (int) - time step at which to start repair (or check-then-repair) = T1
    repair_end (int) - time step at which to stop repair (or check-then-repair) = T2
    delay (int) - time delay (tau) where repair follows checking by [delay] time steps
    time_end (int or str) - time to end simulation (T) or 'none' if end determined only by f_thresh
    dependency (float) - I, interdependency value (fraction of live neighbors required to avoid auto. node failure)
    save (str) - 'yes' or 'no': save results file?
    plot (str) - 'yes' or 'no': show inline plots?
    write_inds (str) - 'yes' or 'no': save individual data?
    preNet_list (bool or list) - False = init networks later, [list] = use networks A, v in the list (for cluster use)
    '''
    
    # Records
    vitality = []
    interdependence = []
    failure_times = []
    costs = []
    
    # Runs a simulation for each individual in the population
    i = 0
    while i < pop_size:
        if preNet_list is False:
            vit, inter, ftime, cost = simIndividual(N,p,d,f,r,f_thresh,graph_type,weight_type,check_type,kinetic,P_check,e,
                                                    cost_type,costC,costR,costE,costD,costL,
                                                    P_repl,costrepl,max_repl,repl_type,
                                                    node_type, damage_type, edge_type, f_edge, r_edge, std,
                                                    P_ablate,costablate,ablate_type,repair_start,repair_end,delay,time_end,
                                                    dependency,preNet_list,equilibrate_failures)
        else:
            vit, inter, ftime, cost = simIndividual(N,p,d,f,r,f_thresh,graph_type,weight_type,check_type,kinetic,P_check,e,
                                                    cost_type,costC,costR,costE,costD,costL,
                                                    P_repl,costrepl,max_repl,repl_type,
                                                    node_type, damage_type, edge_type, f_edge, r_edge, std,
                                                    P_ablate,costablate,ablate_type,repair_start,repair_end,delay,time_end,
                                                    dependency,preNet_list[i],equilibrate_failures)
        vitality.append(vit)
        interdependence.append(inter)
        failure_times.append(ftime)
        costs.append(cost)
        i += 1

    # Experiment parameters string (tagged onto end of filename)
    try:
        end_tag = (node_type + '_' + graph_type + '_' + check_type + '_N' + str(N) + '_p' + str(p) 
                   + '_d' + str(d) + '_f' + str(f) + '_r' + str(r))
    except:
        end_tag = (node_type + '_' + graph_type + '_' + str(check_type[0]) + '_N' + str(N) + '_p' + str(p) 
                   + '_d' + str(d) + '_f' + str(f) + '_r' + str(r))
    
    # Generating figures, statistics (s and mortality), saving to CSV file
    totcost = Report(filename, end_tag, vitality, interdependence, failure_times, costs, cost_type, costL, 
                     save, plot, write_inds, f_thresh)
    
    return (totcost)
    #return(vitality[0][-1])


# In[13]:

def simIndividual (N,p,d,f,r,f_thresh,graph_type,weight_type,check_type,kinetic,P_check,e,
                   cost_type,costC,costR,costE,costD,costL,P_repl,costrepl,max_repl,repl_type,
                   node_type,damage_type,edge_type,f_edge,r_edge,std,P_ablate,costablate,ablate_type,
                   repair_start,repair_end,delay,time_end,dependency,preNet=False,equilibrate_failures=True):
    
    '''
    Simulates the aging of one individual network. The aging algorithm:
        1. Damages network according to f
        2. Checks and/or repairs according to P_check and r
        3. Fails nodes depending on interdependency I
        4. Measures vitality and cost
    
    Refer to simPopulation() for parameter definitions
    '''
    
    
    # Calculate check start and end (used for delayed checking costs)
    check_start = repair_start - delay
    if repair_end != 'none':
        check_end = repair_end - delay
    else:
        check_end = 'none'
    
    # Initialize individual adjacency matrix (A), state vector (v)
    if preNet is False: # make graph
        if node_type == 'binary':
            A, v = initIndividual (N, graph_type, p, d, edge_type)
        elif node_type == 'continuous':
            A, v = initIndividualCont (N, graph_type, p, d, edge_type, std)
        else:
            raise Exception ('node_type is not valid!')
    else: # use pre-made graphs
        A = np.copy(preNet[0])
        v = np.copy(preNet[1])
    
    # Initialize vector for number of nodes that the given is dependent on
    num_neigh = np.sum(A,axis=0)
    
    # Initialize replication counter vector
    repl_v = np.zeros(N)
    
    # Summary Lists
    vitality = []
    interdependence = []
    costs = []
                
    i = 0
    while i >= 0: # runs until all nodes broken
        
        # Gets weight and degree vectors
        degree_vec = getDegrees (A)
        weight_vec = getWeights (weight_type, A, v, degree_vec)
        
        # Analyze current state, compute summary statistics
        #try:
        #    vitality_i, interdependence_i = Analyze(v, f, r, i, weight_vec)
        #    vitality.append(vitality_i)
        #    interdependence.append(interdependence_i)
        #except:
        #    print ('Error calculating interdependence at timestep ' + str(i))
        #    i = i-1
        #    break
        
        vitality_i, interdependence_i = Analyze(v, f, r, i, weight_vec)
        vitality.append(vitality_i)
        interdependence.append(interdependence_i)
        
        # Calculate incremental cost at time step
        if i > 0:
            cost = cost_cr + cost_repl + cost_ablate
            cost = getCosts(cost, costE, costD, costL, cost_r, cost_c, vitality_i, cost_type)
            costs.append(cost)
        else:
            costs.append(0)
        
        # Break if vitality lower than threshold
        if np.sum(v)/len(v) <= f_thresh or np.sum(v)==0: # stops when vitality passes threshold
            break
            
        # Stops data collection if time_end reaches
        if time_end != 'none' and i >= time_end:
            break
        
        # More stringent break constrain for continuous node health
#        if (v>0.1).sum()/len(v) <= f_thresh: # health < 0.1 counts as failed
#            break
        
        # simulate stochastic damage
        A, v, f = Damage(A, v, f, damage_type, node_type, edge_type, f_edge, std, i)
        
        # Check and repair network
        if i >= repair_start:
            if repair_end != 'none':
                if i <= int(repair_end):
                    cost_cr, A, v, P_check, r = Check_and_Repair(A, v, r, check_type, kinetic, P_check, e, i, costC, costR, 
                                                              node_type, edge_type, r_edge, std)
                    cost_r = r
                else:
                    cost_cr = 0
                    cost_r = 0
            else:
                cost_cr, A, v, P_check, r = Check_and_Repair(A, v, r, check_type, kinetic, P_check, e, i, costC, costR, 
                                                              node_type, edge_type, r_edge, std)
                cost_r = r
        else:
            cost_cr = 0
            cost_r = 0
            
        # Checking delay costs
        if i >= check_start:
            if check_end != 'none':
                if i <= check_end:
                    cost_c = P_check
                else:
                    cost_c = 0
            else:
                cost_c = P_check
        else:
            cost_c = 0
            
        
        # Replicate
        if P_repl > 0:
            A, v, cost_repl, repl_v = Replicate (A, v, P_repl, costrepl, max_repl, repl_v, repl_type, i)
            costs.append(cost_repl)
        else:
            cost_repl = 0
        
        # Ablate
        if P_ablate > 0:
            A, v, cost_ablate = Ablate (A, v, P_ablate, costablate, ablate_type, i)
            costs.append(cost_ablate)
        else:
            cost_ablate = 0
        
        # dependency-related failure
        if dependency > 0:
            v = dependencyFail(A, v, num_neigh, dependency, equilibrate_failures)
            #v = dependencyFailOLD(A, v, dependency)
        
        i += 1
    
    # failure time
    failure_time = i
    
    # Final network size
    if P_repl > 0 or P_ablate > 0:
        print ('Final N = ' + str(len(v)))
    
    # returns summary
    return(vitality, interdependence, failure_time, costs)


# In[14]:

def initIndividual (N, graph_type, p, d, edge_type):
    '''
    Function for building binary network structures:
    Gilbert Random, Erdos-Renyi Random, Barabasi-Albert Scale-Free
        
    Refer to simPopulation() for parameter definitions.
    '''
    # Initialize adjacency matrix
    if (graph_type == "Grandom_s") or (graph_type == "Grandom_d"):
        A = makeGilbertRandom (N, graph_type, p)
        
    elif (graph_type == "ERrandom_s") or (graph_type == "ERrandom_d"):
        A = makeERRandom (N, graph_type, p)
        
    elif (graph_type == "scale_free_s") or (graph_type == "scale_free_d"):
        A = makeBAScaleFree (N, graph_type, p)
    
    elif graph_type == "none":
        A = np.ones((N,N))
    
    else:
        raise Exception ('Specified graph_type (' + graph_type + ') is not valid!')
    
    # Randomly weighted edges:
    if edge_type == 'binary':
        pass
    elif edge_type == 'random':
        A = randomWeights (A)
    else:
        raise Exception ("Specified edge_type is not valid!")
    
    # Initialize state vector (module/node vector)
    v = np.zeros(N)
    
    i = 0
    while i < N:
        rand_draw = random.uniform(0,1)
        if rand_draw < d:
            v[i] = 0 # failed
        else:
            v[i] = 1
        i += 1
    
    return (A, v)


# In[15]:

def initIndividualCont (N, graph_type, p, d, edge_type, std):
    '''
    Function for building continuous network structures:
        Gilbert Random, Erdos-Renyi Random, Barabasi-Albert Scale-Free
        
    Refer to simPopulation() for parameter definitions.
    '''
    # Initialize adjacency matrix
    if (graph_type == "Grandom_s") or (graph_type == "Grandom_d"):
        A = makeGilbertRandom (N, graph_type, p)
        
    elif (graph_type == "ERrandom_s") or (graph_type == "ERrandom_d"):
        A = makeERRandom (N, graph_type, p)
        
    elif (graph_type == "scale_free_s") or (graph_type == "scale_free_d"):
        A = makeBAScaleFree (N, graph_type, p)
    
    else:
        raise Exception ('Specified graph_type (' + graph_type + ') is not valid!')
        
    # Randomly weighted edges:
    if edge_type == 'binary':
        pass
    elif edge_type == 'random':
        A = randomWeights (A)
    else:
        raise Exception ("edge_type is not valid!")
    
    # Initialize state vector (module/node vector)
    v = np.zeros(N)
    
    i = 0
    while i < N:
        vit_draw = np.random.normal(1-d,std*(1-d)) # draw node vitality from Gaussian with mean=1-d and std=1/2 mean
        if vit_draw > 1: # greater than unity vitalities become one
            v[i] = 1.0
        elif vit_draw < 0: # negative vitalities become zero
            v[i] = 0.0
        else:
            v[i] = vit_draw
        i += 1
        
    return (A, v)


# In[16]:

def makeGilbertRandom (N, graph_type, p):
    '''
    Builds a Gilbert random network where edges are made with probaility p between any two nodes in the network
    '''
    # Empty adjacency matrix
    A = np.zeros((N, N))
    i = 0
    while i < N:
        j = 0
        while j < N:
            # Initialize A
            rand_draw = random.uniform(0,1)
            if graph_type == 'Grandom_s':
                if rand_draw < p:
                    A[i][j] = 1.0 # connections formed with prob p
                    A[j][i] = 1.0 # make symmetric
                else:
                    A[i][j] = 0.0
                    A[j][i] = 0.0
            elif graph_type == 'Grandom_d':
                if rand_draw < p:
                    A[i][j] = 1.0 # connections formed with prob p
                else:
                    A[i][j] = 0.0
            j += 1
        # eliminate self-references (no self-dependency)
        A[i][i] = 0.0
        i += 1
    
    return (A)


# In[17]:

def makeBAScaleFree (N, graph_type, p):
    '''
    Builds a Barabasi-Albert scale-free network using the NetworkX package
    Each new node establishes q=p*N number of new edges with probability that increases with the degree of exisiting
    nodes.
    '''
    q = round(p*N) # number of edges for each new node
    rand_seed = random.uniform(0,100)
    G=nx.barabasi_albert_graph(N, q)
    A = nx.to_numpy_matrix(G)
    # networkx defaults to no self references (diagonal = 0)
    return (A)


# In[18]:

def makeERRandom (N, graph_type, p): # Same as in Vural 2014
    '''
    Builds an Erdos-Renyi random network of form G(N,m) where m=q*N=N^2*p where the network structure is chosen with equal
    probability from all possible networks of size N with m edges.
    Uses NetworkX package
    '''
    m = round(p*N**2) # number of edges for each new node
    G=nx.gnm_random_graph(N,m)
    A = nx.to_numpy_matrix(G)
    # networkx defaults to no self references (diagonal = 0)
    return (A)


# In[19]:

def makeERRandomOLD (N, graph_type, p): # Same as in Vural 2014
    '''
    Builds an Erdos-Renyi random network with degree distribution Exp[-0.7*degree] (not dependent on p)
    0.7 is the value used in (Vural et al., 2014)
    '''
    # Empty adjacency matrix
    A = np.zeros((N, N))
    i = 0
    while i < N:
        j = 0
        while j < N:
            # Initialize A
            rand_draw = random.uniform(0,1)
            degree_j = np.sum(A[:,j])
            if graph_type == 'ERrandom_s':
                if rand_draw < math.exp(-0.7*degree_j):
                    A[i][j] = 1.0 # connections formed with prob p
                    A[j][i] = 1.0 # make symmetric
                else:
                    A[i][j] = 0.0
                    A[j][i] = 0.0
            elif graph_type == 'ERrandom_d':
                if rand_draw < math.exp(-0.7*degree_j):
                    A[i][j] = 1.0 # connections formed with prob p
                else:
                    A[i][j] = 0.0
            j += 1
        # eliminate self-references (no self-dependency)
        A[i][i] = 0.0
        i += 1    
    return (A)


# In[20]:

def randomWeights (A):
    '''
    Draws random weights to make a continuous network from a binary one
    '''
    # draws random number from normal distribution centered at 1 for edge weights
    new_A = A
    i = 0
    while i < A.shape[0]:
        j = 0
        while j < A.shape[1]:
            if A[i,j] == 1:
                rand_weight = np.random.normal(1,0.2)
                new_A[i,j] = rand_weight
            j += 1
        i += 1
    return (new_A)


# In[21]:

def getWeights (weight_type, A, v, degree_vec):
    '''
    Obtain the weight vector corresponding to the weighted contribution of a node to the network vitality
    '''
    if weight_type == 'uniform':
        weight_vec = np.ones(len(v))
    elif weight_type == 'biased':
        weight_vec = np.multiply(degree_vec, len(v)/np.sum(degree_vec)) # normalize so sum of all is len(v)
    else:
        raise Exception ('Specified weight_type is not valid!')
    return (weight_vec)


# In[22]:

def getDegrees (A):
    '''
    Get array of node degrees from the adjacency matrix A
    '''
    degree_vec = np.sum(A, axis=0)
    return (degree_vec)


# In[23]:

def Damage (A, v, f, damage_type, node_type, edge_type, f_edge, std, i):
    '''
    Damages nodes according to the damage_type and f. Optionally, damage edges according to edge_type and f_edge
    
    Refer to simPopulation() for parameter definitions.
    '''
    
    # Damage NODES
    if damage_type == 'uniform':
        v = damageUniform (A, v, f, node_type, std)
        
    elif damage_type == 'biased':
        v = damageBiased (A, v, f, node_type, std)
        
    elif damage_type == 'aggregate':
        v, f = damageAggregate (A, v, f, node_type, std, i)
        
    else:
        raise Exception ("damage_type is not valid!")
        
    # Damage EDGES
    if f_edge > 0:
        A = damageEdges(A, edge_type, f_edge, std)
        
    return (A, v, f)


# In[24]:

def damageUniform (A, v, f, node_type, std):
    '''
    Uniformly damage all nodes with probability f
    '''
    # stochastic damage and repair
    k = 0
    while k < len(v):
        
        if node_type == 'binary':
            rand_draw = random.uniform(0,1)
            if rand_draw < f and v[k] == 1:
                v[k] = 0 # breaks
                
        if node_type == 'continuous':
            damage_draw = np.random.normal(f,std*f)
            if damage_draw > 0:
                v[k] = v[k] - damage_draw # damages node
            else:
                continue
            if v[k] < 0: # sets minimum to zero
                v[k] = 0
        
        k+=1
    
    return (v)


# In[25]:

def damageBiased (A, v, f, node_type, std):
    '''
    Damage nodes according to their degree k with probability: 1-Exp[-k/SUM(k)/f]
    '''
    # stochastic damage and repair
    k = 0
    while k < len(v):
        
        degree_k = np.sum(A[:,k])
        tot_degree = np.sum(np.sum(A, axis=0))
        
        if node_type == 'binary':
            rand_draw = random.uniform(0,1)
            if rand_draw < 1-math.exp(-(degree_k/tot_degree)/f) and v[k] == 1:
                v[k] = 0 # breaks
                
        if node_type == 'continuous':
            mean_damage = 1-math.exp(-(degree_k/tot_degree)/f)
            damage_draw = np.random.normal(mean_damage,std*mean_damage)
            if damage_draw > 0:
                v[k] = v[k] - damage_draw # damages node
            else:
                continue
            if v[k] < 0: # sets minimum to zero
                v[k] = 0
        
        k+=1
    
    return (v)


# In[26]:

def damageAggregate (A, v, f, node_type, std, i):
    '''
    Damage uniformly with an increasing probability of failure with time
    '''
    
    new_f = f + 1/(2*(i+1)) * f
    
    v = damageUniform (A, v, f, node_type, std)
    
    return (v, new_f)


# In[27]:

def damageEdges(A, edge_type, f_edge, std):
    '''
    Damage edges with probability f_edge
    '''
    
    i = 0
    
    while i < A.shape[0]:
        j = 0
        while j < A.shape[1]:
            
            if edge_type == 'binary':
                rand_draw = random.uniform(0,1)
                if rand_draw < f_edge and A[i,j] == 1:
                    A[i,j] = 0 # breaks
            
            if edge_type == 'random':
                damage_draw = np.random.normal(f_edge, std*f_edge)
                if damage_draw > 0:
                    A[i,j] = A[i,j] - damage_draw
            j += 1
        i += 1
        
    return (A)


# In[28]:

def Check_and_Repair (A, v, r, check_type, kinetic, P_check, e, i, costC, costR, node_type, edge_type, r_edge, std):
    '''
    Performs checking and repair protocols according to check_type, P_check and r.
    
    Refer to simPopulation() for parameter definitions.
    '''
    # Repair NODES
    
    # Continuous
    if node_type == 'continuous':
        P_check = r # since no longer probabilities of checking!
    
    # Kinetic proofreading
    try:
        e = e**kinetic
        costR = costR*kinetic
    except:
        raise Exception("kinetic should be an integer!")
    
    # No checking (just repair rate)
    if check_type == 'none':
        v, cost = checkNone (A, v, r, costR, node_type, std)
    
    elif check_type == 'uniform':
        v, cost = checkUniform (A, v, costR, costC, P_check, e, node_type, std)

    elif check_type == 'biased':
        v, cost = checkBiased (A, v, costR, costC, P_check, e, node_type, std)
        
    elif check_type == 'age':
        v, cost = checkAge (A, v, costR, costC, P_check, e, i)
        
    elif check_type == 'block':
        v, cost = checkBlock (A, v, costR, costC, P_check, e, i)

    elif check_type == 'time':
        v, cost, P_check = checkTime (A, v, costR, costC, P_check, e, node_type, std, i)
    
    elif 'quadratic' in check_type:
        v, cost, r = repairQuadratic (A, v, r, costR, node_type, std, i, check_type)
    
    elif 'numeric' in check_type:
        v, cost, r = repairNumerical (A, v, r, costR, node_type, std, i, check_type)
        
    elif 'space' in check_type:
        v, cost, r = repairSpace (A, v, check_type, node_type, e, std)
    
    else:
        raise Exception ('Specified check_type is not valid!')
        
    # Repair EDGES
    if r_edge > 0:
        A = repairEdges (A, edge_type, r_edge, std)
    
    return (cost, A, v, P_check, r)


# In[29]:

def checkNone (A, v, r, costR, node_type, std):
    '''
    No checking is performed. (i.e. c=1, delay=0)
    Repair occurs for failed nodes at probability r.
    '''
    cost=0
    k=0
    
    while k < len(v):
        if node_type == 'binary':
            rand_draw = random.uniform(0,1)
            if rand_draw < r and v[k] == 0:
                v[k] = 1 # repairs
                cost += costR
                
        elif node_type == 'continuous':
            repair_draw = np.random.normal(r,std*r)
            if repair_draw > 0:
                v[k] = v[k] + repair_draw # repairs node
            else:
                continue
            if v[k] > 1: # sets maximum to one
                v[k] = 1
            
        k+=1
    
    return (v, cost)


# In[30]:

def checkUniform (A, v, costR, costC, P_check, e, node_type, std):
    '''
    [DEPRECATED]
    Uniform checking
    '''
    cost = 0
    k = 0
    while k < len(v):
        if node_type == 'binary':
            rand_draw = random.uniform(0,1)
            if rand_draw < P_check: # checks
                cost += costC # adds checking cost
                if v[k] == 0:
                    rand_draw = random.uniform(0,1)
                    cost += costR # adds repair cost
                    if rand_draw > e: # error rate
                        v[k] = 1 # repaired
                    else:
                        v[k] = 0 # repair error
                        
        elif node_type == 'continuous':
            cost += costC
            if v[k] < 1:
                repair_draw = np.random.normal(P_check,std*P_check)
                cost += costR
                rand_draw = random.uniform(0,1)
                if rand_draw > e and repair_draw > 0:
                    v[k] = v[k] + repair_draw
                else:
                    continue # repair fail
                if v[k] > 1:
                    v[k] = 1 # set max to 1
                    
        k += 1
    
    return (v, cost)


# In[31]:

def checkBiased (A, v, costR, costC, P_check, e, node_type, std):
    '''
    [DEPRECATED]
    Check biased towards high-degree nodes: 1-math.exp(-(degree_k/tot_degree)/P_check)
    '''
    
    cost = 0
    k = 0
    
    while k < len(v):
        if node_type == 'binary':
            degree_k = np.sum(A[:,k])
            tot_degree = np.sum(np.sum(A, axis=0))
            rand_draw = random.uniform(0,1)
            if rand_draw < 1-math.exp(-(degree_k/tot_degree)/P_check): # check probability proportional to degree
                cost += costC # adds checking cost
                if v[k] == 0:
                    rand_draw = random.uniform(0,1)
                    cost += costR # adds repair cost
                    if rand_draw > e: # error rate
                        v[k] = 1 # repaired
                    else:
                        v[k] = 0 # repair error
                        
        # NEED TO EDIT (DEFUNCT): mean of means for gamma should be "r" or "P_check=r"
        elif node_type == 'continuous':
            degree_k = np.sum(A[:,k])
            tot_degree = np.sum(np.sum(A, axis=0))
            cost += costC
            if v[k] < 1:
                new_P_check = 1-math.exp(-(degree_k/tot_degree)/P_check)
                repair_draw = np.random.normal(new_P_check,std*new_P_check)
                cost += costR
                rand_draw = random.uniform(0,1)
                if rand_draw > e and repair_draw > 0:
                    v[k] = v[k] + repair_draw
                else:
                    continue # repair fail
                if v[k] > 1:
                    v[k] = 1 # set max to 1
        
        k += 1
        
    return (v, cost)


# In[32]:

def repairSpace (A, v, check_type, node_type, e, std):
    '''
    No checking.
    Repair as a function of connectivity (degree=k): 1-math.exp(-beta*(degree_k/tot_degree))
    beta is the parameter that defines the family of repair functions.
    '''
    
    cost = 0
    k = 0
    
    beta = check_type[1]
    degrees = []
    
    while k < len(v):
        if node_type == 'binary':
            degree_k = np.sum(A[:,k])
            degrees.append(degree_k)
            tot_degree = np.sum(np.sum(A, axis=0))
            rand_draw = random.uniform(0,1)
            if rand_draw < 1-math.exp(-beta*(degree_k/tot_degree)): # check probability proportional to degree
                if v[k] == 0:
                    rand_draw = random.uniform(0,1)
                    if rand_draw > e: # error rate
                        v[k] = 1 # repaired
                    else:
                        v[k] = 0 # repair error
                        
        # NEED TO EDIT (DEFUNCT): mean of means for gamma should be "r" or "P_check=r"
        elif node_type == 'continuous':
            degree_k = np.sum(A[:,k])
            degrees.append(degree_k)
            tot_degree = np.sum(np.sum(A, axis=0))
            if v[k] < 1:
                new_P_check = 1-math.exp(-beta*(degree_k/tot_degree))
                repair_draw = np.random.normal(new_P_check,std*new_P_check)
                rand_draw = random.uniform(0,1)
                if rand_draw > e and repair_draw > 0:
                    v[k] = v[k] + repair_draw
                else:
                    continue # repair fail
                if v[k] > 1:
                    v[k] = 1 # set max to 1
        
        k += 1
        
    # approximately linear for small k/SUM(k)
    kmax = max(degrees) / np.sum(np.sum(A, axis=0))
    r = check_type[1] * kmax/2
        
    return (v, cost, r)


# In[33]:

def checkAge (A, v, costR, costC, P_check, e, i):
    '''
    [DEPRECATED]
    Checking following an age replacement policy where the age at replacement is a function of the degree:
        round(1/(1-math.exp(-(degree_k/tot_degree)/P_check)))
    '''
    cost = 0
    k = 0
    while k < len(v):
        degree_k = np.sum(A[:,k])
        tot_degree = np.sum(np.sum(A, axis=0))
        T = round(1/(1-math.exp(-(degree_k/tot_degree)/P_check)))
        if (i+1) % T == 0:
            #print (str(T) + ' divides ' + str(i+1))
            cost += costC # adds checking cost
            if v[k] == 0:
                rand_draw = random.uniform(0,1)
                cost += costR # adds repair cost
                if rand_draw > e: # error rate
                    v[k] = 1 # repaired
                else:
                    continue # repair error
        k += 1
        
    return (v, cost)


# In[34]:

def checkBlock (A, v, costR, costC, P_check, e, i):
    '''
    [DEPRECATED]
    Checking following a block replacement policy where all units are replaced at time period: 1/P_check
    '''
    cost = 0
    k = 0
    T = round(1/P_check)
    if (i+1) % T == 0:
        #print (str(T) + ' divides ' + str(i+1))
        while k < len(v):
            cost += costC # adds checking cost
            if v[k] == 0:
                rand_draw = random.uniform(0,1)
                cost += costR # adds repair cost
                if rand_draw > e: # error rate
                    v[k] = 1 # repaired
                else:
                    continue # repair error
            k += 1
    
    return (v, cost)


# In[35]:

def checkTime (A, v, costR, costC, P_check, e, node_type, std, i):
    '''
    [DEPRECATED]
    Checking probability that decays with time
    '''
    
#    new_r = r + 1/(2*i) * r
    new_P_check = P_check + 1/(2*(i+1)) * P_check
    
    v, cost = checkUniform(A, v, costR, costC, new_P_check, e, node_type, std)
    
    return (v, cost, new_P_check)


# In[36]:

def repairNumerical (A, v, r, costR, node_type, std, i, check_type):
    '''
    Reads in a file of repair rates in time (filename specified by second element of check_type).
    Updates the global repair rate r to the file repair rate specified at the i-th step.
    '''
    i = i-1
    cost=0
    k=0
    
    while k < len(v):
        if node_type == 'binary':
            rand_draw = random.uniform(0,1)
            if rand_draw < r and v[k] == 0:
                v[k] = 1 # repairs
                cost += costR
                
        elif node_type == 'continuous':
            repair_draw = np.random.normal(r,std*r)
            if repair_draw > 0:
                v[k] = v[k] + repair_draw # repairs node
            else:
                continue
            if v[k] > 1: # sets maximum to one
                v[k] = 1
            
        k+=1
    
    ### UPDATE r
    filename = check_type[1]
    with open(filename, 'r') as repair_file:
        reader = csv.reader(repair_file, delimiter=',')
        repair_list = list(reader)
    multiplier = check_type[2]
    new_r = multiplier*float(repair_list[i][0])
    
    return (v, cost, new_r)


# In[37]:

def repairQuadratic (A, v, r, costR, node_type, std, i, check_type):
    '''
    [DEPRECATED]
    Quadratic repair function
    '''
    cost=0
    k=0
    
    while k < len(v):
        if node_type == 'binary':
            rand_draw = random.uniform(0,1)
            if rand_draw < r and v[k] == 0:
                v[k] = 1 # repairs
                cost += costR
                
        elif node_type == 'continuous':
            repair_draw = np.random.normal(r,std*r)
            if repair_draw > 0:
                v[k] = v[k] + repair_draw # repairs node
            else:
                continue
            if v[k] > 1: # sets maximum to one
                v[k] = 1
            
        k+=1
    
    # UPDATE r
    # vertex of symmetric parabola
    vertex_t = check_type[1]
    vertex_r = check_type[2]
    
    r_zero1 = 0
    r_zero2 = 2*vertex_t
    
    # y = -[a(x-h)^2 + k] with vertex (h, k)
    # (0, 0) --> a*h^2 + k = 0 --> a = -k/h^2
    a = -vertex_r/(vertex_t**2)
    new_r = (a*(i-vertex_t)**2+vertex_r)
    
#    if i % 10 == 0:
#        print (str(i)+', '+str(new_r)+', '+str(r))
    
    return (v, cost, new_r)


# In[38]:

def repairEdges(A, edge_type, r_edge, std):
    '''
    [DEPRECATED]
    Repair at probability r_edge for edges
    '''
    
    i = 0
    while i < A.shape[0]:
        j = 0
        while j < A.shape[1]:
            
            if edge_type == 'binary':
                rand_draw = random.uniform(0,1)
                if rand_draw < r_edge and A[i,j] == 0:
                    A[i,j] = 1 # repairs
            
            if edge_type == 'random':
                repair_draw = np.random.normal(r_edge, std*r_edge)
                if repair_draw > 0:
                    A[i,j] = A[i,j] + repair_draw
            j += 1
        i += 1
        
    return (A)


# In[39]:

def Replicate (A, v, P_repl, costrepl, max_repl, repl_v, repl_type, i):
    '''
    [DEPRECATED]
    Live nodes replicate at probability "P_repl" at each time step.
    Associated cost for each replication event is "costrepl".
    Replicated nodes are identical to parent node (same edges and cont. vitality if applicable).
    "repl_v" keeps track of the number of replication events per node.
    "max_repl" specifies the upper limit on number of replication events per node.
    "repl_type" = constant (uniform across time and node) or expdec (decays with time)
    '''
    new_v = v
    new_A = A
    cost = 0
        
    if repl_type == "constant":
        P_repl = P_repl
    elif repl_type == "expdec":
        P_repl = P_repl * math.exp(-i)
    
    k = 0

    while k < len(v):
        if repl_v[k] < max_repl:
            if v[k] == 1:
                rand_draw = random.uniform(0,1)
                if rand_draw < P_repl: # NEED TO CHANGE FOR CONTINUOUS
                    # Replicates and update A if functional and with certain probability
                    cost += costrepl
                    new_node_in = new_A[k,:]
                    new_node_out = np.append(new_A[:,k], 0) # add 0 (no self connection)
                    new_A = np.vstack((new_A, new_node_in))
                    new_A = np.column_stack((new_A, new_node_out))

                    # update v, repl_v
                    new_v = np.append(new_v, 1) # new node is functional
                    repl_v[k] += 1 # increment replicated node
                    repl_v = np.append(repl_v, repl_v[k]) # replicated node as same replication # as parent
        k+=1
        
    return (new_A, new_v, cost, repl_v)


# In[40]:

#A = np.zeros((4,4))
#v = np.array([1,1,1,1])
#repl_v = ([5,6,7,8])
#new_A, new_v, cost, repl_v = Replicate (A, v, 0.8, 1, 10, repl_v, repl_type, 1)
#print (new_A)
#print(new_v)
#print(cost)
#print(repl_v)


# In[41]:

def Ablate (A, v, P_ablate, costablate, ablate_type, i):
    
    '''
    Failed nodes are removed (from A and v) with probability "P_ablate".
    Associated cost is "costablate".
    "ablate_type" = constant (uniform across time and node) or expdec (decays with time)
    '''
    
    new_v = v
    new_A = A
    cost = 0
        
    if ablate_type == "constant":
        P_ablate = P_ablate
    elif ablate_type == "expdec":
        P_ablate = P_ablate * math.exp(-i)
    
    k = 0
    
    ablate_idxs = []
    
    while k < len(v):
        if v[k]==0: # NEED TO CHANGE FOR CONTINUOUS
            rand_draw = random.uniform(0,1)
            if rand_draw < P_ablate:
                cost += costablate
                ablate_idxs.append(k)
        k+=1
        
    # remove node from A
    new_A = np.delete(new_A, ablate_idxs, 0)
    new_A = np.delete(new_A, ablate_idxs, 1)
    # remove node from v
    new_v = np.delete(new_v, ablate_idxs)
    
    return (new_A, new_v, cost)


# In[51]:

def dependencyFail (A, v, num_neigh, dependency, equilibrate_failures):
    '''
    Fail live nodes if the fraction of live neighbors is below the "dependency" value (i.e. I).
    equilibrate_failures [Boolean] = whether to equilibrate failures or not
    '''
    # Break if majority of connections are broken (i.e. num_good < 0.5*num_total)
    v_new = np.copy(v)
    if equilibrate_failures is False:
        num_good = np.matmul(A.T, v) # vector of number of good neighbors
        for k in range(len(v)):
            try:
                if num_good[k] < dependency*num_neigh[k]:
                    v_new[k] = 0 # breaks
            except: # for BA and ER graphs
                if num_good[0,k] < dependency*num_neigh[0,k]:
                    v_new[k] = 0 # breaks
    else: # equilibrate
        keep_going = True
        while keep_going is True:
            num_good = np.matmul(A.T, v_new) # vector of number of good neighbors
            num_breaks = 0
            for k in range(len(v)):
                try:
                    if num_good[k] < dependency*num_neigh[k]:
                        if v_new[k] == 1:
                            num_breaks += 1
                        v_new[k] = 0 # breaks
                except: # for BA and ER graphs
                    if num_good[0,k] < dependency*num_neigh[0,k]:
                        if v_new[k] == 1:
                            num_breaks += 1
                        v_new[k] = 0 # breaks
            if num_breaks == 0:
                keep_going = False
                    
    return (v_new)


# In[43]:

def dependencyFailOLD (A, v, dependency):
    '''
    [DEPRECATED]
    Old function for interdependecny-based failure. Slower.
    '''
    # Break if majority of connections are broken (i.e. num_good < 0.5*num_total)
    v_new = v
    k = 0
    while k < len(v):
        edge_idxs = np.nonzero(A[k,:])[0] # If A is not symmetric, rows correspond to input and columns to outputs
        try:
            num_good = np.sum(v[edge_idxs]) 
        except:
            print (A)
            print (edge_idxs)
            print (len(edge_idxs[1]))
            print (len(v))

        if num_good < dependency*np.sum(A[k,:]):
            v_new[k] = 0 # breaks
            
        k += 1
        
    return (v_new)


# In[44]:

def getCosts (cost, costE, costD, costL, costr, costc, vitality_i, cost_type):
    '''
    Calculate costs from accured costs or from preset functions.
    Functions used in "Optimal Control of Aging in Complex Systems"
        'healthspan': cost = alpha * r - phi 
        'healthspan_quadratic': cost = alpha * r^2 - phi
        'checking_delay': cost = alpha_1 * c + alpha_2 * r - phi
    '''
    # Failure Costs
    total_cost = 0
    
    if 'basic' in cost_type: # Basic Barlow type cost
        total_cost += cost
        
    if 'energy' in cost_type: # Resource production offsets cost (proportional to # live nodes)
        total_cost += -(costE*np.sum(v))
    
    if 'duration' in cost_type: # Low durability system has self-induced harm/cost (proportional to # dead nodes)
        total_cost += (costD*(len(v)-np.sum(v)))
        
    if 'healthspan' in cost_type:
        a = cost_type[1]
        total_cost = a * costr - vitality_i
        
    if 'healthspan_norm' in cost_type: # a*repair - vitality for bang-bang repair
        if costr > 0:
            a = cost_type[1]/costr # cost_type[1] = 0.1 works!
        else:
            a = 1
        total_cost = a * costr - vitality_i
        
    if 'healthspan_quadratic' in cost_type:
        # first arg is alpha (BIG), second arg is vertex t, third arg is vertex r(t)
        if costr > 0:
            a = cost_type[1] # cost_type[1] = 8000 or 10000 works!
        else:
            a = 1
        total_cost = a * costr**2 - vitality_i
        
#    if 'healthspan_ratio' in cost_type: # repair/vitality cost for bang-bang repair
#        if vitality_i == 0:
#            total_cost = costr/0.01
#        else:
#            total_cost = costr/vitality_i
            
    if 'checking_delay' in cost_type: # a1*checking + a2*repair + vitality for bang-bang checking
        if costc > 0:
            a1 = cost_type[1]/costc
        else:
            a1 = 1
        if costr > 0:
            a2 = cost_type[2]/costr
        else:
            a2 = 1
        total_cost = a1*costc + a2*costr - vitality_i
    
    return (total_cost)


# In[45]:

def Analyze(v, f, r, i, weight_vec):
    '''
    Calculate network-level statistics: interdependence and vitality of network
    '''
    
    # Calculated weighted node vector
    weighted_v = np.multiply(v, weight_vec)
    
    # Calculate vitality
    vitality = np.sum(weighted_v)/len(weighted_v) # fraction of live nodes (weighted)
    
    # Calculate interdependence
    if f == r:
        interdependence = 0
    elif vitality == 0:
        interdependence = math.log(0.0001)/math.log(math.exp((-f+r)*(i+1)))
    else:
        interdependence = math.log(vitality)/math.log(math.exp((-f+r)*(i+1)))
    
    return (vitality, interdependence)


# In[46]:

def Report(filename, end_tag, vitality, interdependence, failure_times, costs, cost_type, costL, 
           save, plot, write_inds, f_thresh):
    '''
    Calculate population-level statistics:
        mean interdependence, mean vitality, mean failure time, mean costs, mortality
        (see Methods or markdown for specific equations corresponding to these calculations)
    Optionally make plots of these values as functions of time.
    '''
    
    ######
    vitality_m = []
    interdependence_m = []
    s = []
    mortality = []
    costs_m = []
    failure_times_m = np.mean(failure_times)
    ######
    
    #oldest_idx = failure_times.index(max(failure_times)) # finds idx of longest-lived individual
    
    t = 0
    while t < max(failure_times)+1:
        vit = []
        inter = []
        cost = []
        num_alive = 0
        
        ind = 0
        while ind < len(vitality):
            if len(vitality[ind]) > t: # if ind alive at this time
                num_alive += 1
                vit.append(vitality[ind][t])
                inter.append(interdependence[ind][t])
                cost.append(costs[ind][t])
            else: 
                # append dead individual values (to avoid skewed statistics from long-lived)
#                vit.append(vitality[ind][-1])
#                inter.append(interdependence[ind][-1])
                # OR: append (0)
                vit.append(0)
                inter.append(0)
                
                cost.append(0)
            ind += 1
            
        vitality_m.append(np.mean(vit))
        interdependence_m.append(np.mean(inter))
        costs_m.append(np.mean(cost))
        
        # Calculate mortality and s
        s_t = num_alive/len(vitality)
        s.append(s_t) # fraction of individuals alive at time t
        
        if t > 0:
            #mort = -(s_t-s[t-1])/s_t # from Vural
            mort = (s[t-1]-s_t)/s[t-1] # real def
            mortality.append(mort)
        
        t += 1
        
    mortality.append(0)
    
    # Calculate costs
    tot_cost = sum(costs_m)
#    if cost_type == 'longevity_old':
#        tot_cost += -(costL*failure_times_m) # should give the same as subtracting each ind separately
    if 'longevity' in cost_type:
        tot_cost += costL*(1/failure_times_m) # keeps cost always positive, long life minimizes costL (large)    
    
    
    # SAVE DATA
    
    if save == 'yes':
        print ('Saving results...')
        if not os.path.exists('./Data'):
            print ("Making ./Data/ directory")
            os.makedirs('./Data')
        if not os.path.exists('./Figures'):
            print ("Making ./Figures/ directory")
            os.makedirs('./Figures')
            
        # INDIVIDUAL
        i = 0
        while i < len(vitality):
            data_list = []
            data_list.append(vitality[i])
            data_list.append(costs[i]) 
            data_list.append(interdependence[i])
            #data_list.append([failure_times[i]])

            if write_inds == 'yes':
                file = open('Data/'+filename+end_tag+'_'+str(i)+'.csv', 'w', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerows(data_list)
                
            i+=1

        # MEANS
        data_list = []
        data_list.append(vitality_m)
        data_list.append(costs_m) 
        data_list.append(interdependence_m)
        data_list.append(s) 
        data_list.append(mortality)
        #data_list.append([failure_times_m])
        #data_list.append([tot_cost])

        file = open('Data/'+filename+end_tag+'_MEAN.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerows(data_list)
    
    # Making and Saving Figures as PNGs
    time = np.arange(1,len(vitality_m)+1)
    
    # Vitality
    
    # COMMENT OUT (just for extending vitality_m range):
    #while len(vitality_m) < 30:
    #    vitality_m.append(0)
    #time = np.arange(1,30+1)
    ####
    if save == 'yes' or plot == 'yes':
        plt.figure(figsize=(6,4.2))
        for vit in vitality:
            #plt.plot(np.arange(1,len(vit)+1,1), vit, color='#000080', alpha=0.1)
            plt.plot(np.arange(0,len(vit),1), vit, color='#000080', alpha=0.1) 
        plt.plot([t-1 for t in time], vitality_m, color='#000080', alpha=0.7, linewidth=3.0, label='Simulated')
        ### Theory (linear) plot
        f=0.025
        r=0
        vitality_theory = [(np.exp((-f-r)*(t-1))*(f+np.exp((f+r)*(t-1))*r))/(f+r) for t in time]
        plt.plot([t-1 for t in time], vitality_theory, color='m', linestyle='--', alpha=0.5, linewidth=2.0, label='Linear Theory')
        ###
        #plt.plot(time, np.multiply(np.ones(len(time)), f_thresh), color='#FFC0CB', linestyle='--', alpha=1.0, linewidth=1.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        #plt.title("Network Vitality", fontsize=22)
        plt.xlabel("Time, $t$", fontsize=16)
        plt.ylabel("Vitality, $\phi$", fontsize=16)
        #plt.annotate('Mean Failure Time: ' + str(round(failure_times_m)-1), 
        #             xy=(0.45, 0.88), xycoords='axes fraction', fontsize=12)
        plt.ylim([-0.05,1.05])
        #plt.xlim([0,30])
        plt.xlim([-0.05, max(time)+0.05])
        plt.legend(loc='lower left')
        plt.tight_layout()
    if save == 'yes':
        plt.savefig('Figures/'+filename+'_vitality', dpi=800)
    if plot == 'yes':
        plt.show()
    
    #time = np.arange(1,len(interdependence_m)+1) # COMMENT OUT
    # Interdependence
    if save == 'yes' or plot == 'yes':
        plt.figure()
        plt.plot(time, interdependence_m, 'b')
        plt.title("Interdependence")
        plt.xlabel("Time (t)")
        plt.ylabel("Interdependence")
    if save == 'yes':
        plt.savefig('Figures/'+filename+'_interdependence', dpi=500)
    if plot == 'yes':
        plt.show()
    
    # s (fraction alive)
    if save == 'yes' or plot == 'yes':
        plt.figure()
        plt.plot(time, s, 'g')
        plt.title("Fraction Individuals Alive")
        plt.xlabel("Time t")
        plt.ylabel("s")
    if save == 'yes':
        plt.savefig('Figures/'+filename+'_s', dpi=500)
    if plot == 'yes':
        plt.show()
    
    # Mortality
    if save == 'yes' or plot == 'yes':
        plt.figure()
        plt.plot(time, mortality, 'c')
        plt.title("Mortality")
        plt.xlabel("Time (t)")
        plt.ylabel("Mortality Rate")
    if save == 'yes':
        plt.savefig('Figures/'+filename+'_mortality', dpi=500)
    if plot == 'yes':
        plt.show()
    
    # Costs
    if save == 'yes' or plot == 'yes':
        plt.figure()
        plt.plot(time, costs_m, 'k')
        plt.title("Cost")
        plt.xlabel("Time (t)")
        plt.ylabel("Check Cost")
        plt.annotate('Total Cost: ' + str(round(sum(costs_m), 2)), xy=(0.05, 0.9), xycoords='axes fraction')
    if save == 'yes':
        plt.savefig('Figures/'+filename+'_cost', dpi=500)
    if plot == 'yes':
        plt.show()
        
    return (tot_cost)