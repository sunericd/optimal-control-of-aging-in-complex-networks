
# coding: utf-8

# # A Network Model for Ageing
# ### Eric Sun
# ### Last updated 6/7/2018

# ### Documentation moved to Latex document

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

# In[1]:

# Import necessary packages
# import networkx as nx   <-- may be useful when things get complicated (edges contain info, etc)
#  %matplotlib notebook
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


# In[2]:

def readInputs (input_file_path):
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


# In[3]:

def optimizeBangBang_cluster (exp_name, pop_size, dependency, parameter_type, parameter_list, T1_list, T2_list, 
                              T_std, highres_step, T):
    
    
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
                                         time_end=T, dependency=dependency, plot='no')
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
                                             time_end=T, dependency=dependency, plot='no')
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


# In[4]:

def optimizeBangBang (exp_name, pop_size, dependency, parameter_type, parameter_list, T1_list, T2_list, T):
    
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


# In[5]:

def optimizeChecking_cluster (exp_name, pop_size, dependency, delay, parameter_type, parameter_list, 
                              cT1_list, cT2_list, T_std, highres_step, T):
    
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
                                         time_end=T, dependency=dependency, plot='no')
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


# In[6]:

def optimizeChecking (exp_name, pop_size, dependency, delay, parameter_type, parameter_list, cT1_list, cT2_list, T):
    
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


# In[7]:

def examineMultipliers (name, mrange, m_idx, check_type, cost_type, graph_type='Grandom_s', num_networks=50, repair_end=100, dependency=0):
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


# In[8]:

def simPopulation (filename, pop_size=1, N=1000, p=0.1, d=0, f=0.0025, r=0.001, f_thresh=0.1, 
                   graph_type='Grandom_s', weight_type='uniform', check_type='none', 
                   kinetic=1, P_check=0.001, e=0.1, cost_type=['basic'], costC=0, costR=0, costE=0, costD=0, costL=0, 
                   P_repl=0, costrepl=0, max_repl=1, repl_type='constant', node_type='binary', damage_type='uniform',
                   edge_type='binary', f_edge=0, r_edge=0, std=0.3, 
                   P_ablate=0, costablate=0, ablate_type='constant', repair_start=0, repair_end='none', delay=0,
                   time_end='none', dependency=0.5, save='no', plot='yes', write_inds='no'):

    
    # Records
    vitality = []
    interdependence = []
    failure_times = []
    costs = []
    
    # Runs a simulation for each individual in the population
    i = 0
    while i < pop_size:
        vit, inter, ftime, cost = simIndividual(N,p,d,f,r,f_thresh,graph_type,weight_type,check_type,kinetic,P_check,e,
                                                cost_type,costC,costR,costE,costD,costL,
                                                P_repl,costrepl,max_repl,repl_type,
                                                node_type, damage_type, edge_type, f_edge, r_edge, std,
                                                P_ablate,costablate,ablate_type,repair_start,repair_end,delay,time_end,
                                                dependency)
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
    
    return (vitality[0][-1])


# In[9]:

def simIndividual (N,p,d,f,r,f_thresh,graph_type,weight_type,check_type,kinetic,P_check,e,
                   cost_type,costC,costR,costE,costD,costL,P_repl,costrepl,max_repl,repl_type,
                   node_type,damage_type,edge_type,f_edge,r_edge,std,P_ablate,costablate,ablate_type,
                   repair_start,repair_end,delay,time_end,dependency):
    
    # Calculate check start and end (used for delayed checking costs)
    check_start = repair_start - delay
    if repair_end != 'none':
        check_end = repair_end - delay
    else:
        check_end = 'none'
    
    # Initialize individual adjacency matrix (A), state vector (v)
    if node_type == 'binary':
        A, v = initIndividual (N, graph_type, p, d, edge_type)
    elif node_type == 'continuous':
        A, v = initIndividualCont (N, graph_type, p, d, edge_type, std)
    else:
        raise Exception ('node_type is not valid!')
    
    # Initialize replication # vector
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
            v = dependencyFail(A, v, dependency)
        
        i += 1
    
    # failure time
    failure_time = i
    
    # Final network size
    if P_repl > 0 or P_ablate > 0:
        print ('Final N = ' + str(len(v)))
    
    # returns summary
    return(vitality, interdependence, failure_time, costs)


# In[10]:

def initIndividual (N, graph_type, p, d, edge_type):
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


# In[11]:

def initIndividualCont (N, graph_type, p, d, edge_type, std):
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


# In[12]:

def makeGilbertRandom (N, graph_type, p):
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


# In[13]:

def makeBAScaleFree (N, graph_type, p):
    m = round(p*N) # number of edges for each new node
    rand_seed = random.uniform(0,100)
    G=nx.barabasi_albert_graph(N, m)
    A = nx.to_numpy_matrix(G)
    # networkx defaults to no self references (diagonal = 0)
    return (A)


# In[14]:

def makeERRandom (N, graph_type, p): # Same as in Vural 2014
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


# In[15]:

def randomWeights (A):
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


# In[16]:

def getWeights (weight_type, A, v, degree_vec):
    if weight_type == 'uniform':
        weight_vec = np.ones(len(v))
    elif weight_type == 'biased':
        weight_vec = np.multiply(degree_vec, len(v)/np.sum(degree_vec)) # normalize so sum of all is len(v)
    else:
        raise Exception ('Specified weight_type is not valid!')
    return (weight_vec)


# In[17]:

def getDegrees (A):
    degree_vec = np.sum(A, axis=0)
    return (degree_vec)


# In[18]:

def Damage (A, v, f, damage_type, node_type, edge_type, f_edge, std, i):
    
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


# In[19]:

def damageUniform (A, v, f, node_type, std):
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


# In[20]:

def damageBiased (A, v, f, node_type, std):
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


# In[21]:

def damageAggregate (A, v, f, node_type, std, i):
    
    new_f = f + 1/(2*(i+1)) * f
    
    v = damageUniform (A, v, f, node_type, std)
    
    return (v, new_f)


# In[22]:

def damageEdges(A, edge_type, f_edge, std):
    
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


# In[23]:

def Check_and_Repair (A, v, r, check_type, kinetic, P_check, e, i, costC, costR, node_type, edge_type, r_edge, std):
    
    # Repair NODES
    
    # Continuous
    if node_type == 'continuous':
        P_check = r # since no longer probabilities of checking!
    
    # Kinetic proofreading
    try:
        e = e**kinetic
        costR = costR*kinetic
    except:
        raise Exception("kinetic should be a number!")
    
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


# In[24]:

def checkNone (A, v, r, costR, node_type, std):
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


# In[25]:

def checkUniform (A, v, costR, costC, P_check, e, node_type, std):
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


# In[26]:

def checkBiased (A, v, costR, costC, P_check, e, node_type, std):
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


# In[27]:

def repairSpace (A, v, check_type, node_type, e, std):
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


# In[28]:

def checkAge (A, v, costR, costC, P_check, e, i):
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


# In[29]:

def checkBlock (A, v, costR, costC, P_check, e, i):
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


# In[30]:

def checkTime (A, v, costR, costC, P_check, e, node_type, std, i):
    
#    new_r = r + 1/(2*i) * r
    new_P_check = P_check + 1/(2*(i+1)) * P_check
    
    v, cost = checkUniform(A, v, costR, costC, new_P_check, e, node_type, std)
    
    return (v, cost, new_P_check)


# In[31]:

def repairNumerical (A, v, r, costR, node_type, std, i, check_type):
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


# In[32]:

def repairQuadratic (A, v, r, costR, node_type, std, i, check_type):
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


# In[33]:

def repairEdges(A, edge_type, r_edge, std):
    
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


# In[34]:

def Replicate (A, v, P_repl, costrepl, max_repl, repl_v, repl_type, i):
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


# In[35]:

#A = np.zeros((4,4))
#v = np.array([1,1,1,1])
#repl_v = ([5,6,7,8])
#new_A, new_v, cost, repl_v = Replicate (A, v, 0.8, 1, 10, repl_v, repl_type, 1)
#print (new_A)
#print(new_v)
#print(cost)
#print(repl_v)


# In[36]:

def Ablate (A, v, P_ablate, costablate, ablate_type, i):
    
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


# In[37]:

#A = np.zeros((4,4))
#v = np.array([0,0,0,0])
#A, v, cost = Ablate (A, v, 0.5, 1, 'constant', 1)
#print(A)
#print(v)
#print(cost)


# In[38]:

def dependencyFail (A, v, dependency):
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


# In[39]:

def getCosts (cost, costE, costD, costL, costr, costc, vitality_i, cost_type):
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


# In[40]:

def Analyze(v, f, r, i, weight_vec):
    
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


# In[42]:

def Report(filename, end_tag, vitality, interdependence, failure_times, costs, cost_type, costL, 
           save, plot, write_inds, f_thresh):
    
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
            data_list.append([failure_times[i]])

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
        data_list.append([failure_times_m])
        data_list.append([tot_cost])

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
        plt.figure(figsize=(6,3))
        for vit in vitality:
            plt.plot(np.arange(1,len(vit)+1,1), vit, 'g', alpha=0.1)
        plt.plot(time, vitality_m, 'g--', linewidth=3.0)
        plt.plot(time, np.multiply(np.ones(len(time)), f_thresh), 'r--', alpha=0.5, linewidth=1.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        #plt.title("Network Vitality", fontsize=22)
        plt.xlabel("Time, $t$", fontsize=18)
        plt.ylabel("Vitality, $\phi$", fontsize=18)
        plt.annotate('Mean Failure Time: ' + str(round(failure_times_m)), 
                     xy=(0.45, 0.88), xycoords='axes fraction', fontsize=12)
        plt.ylim([-0.05,1.05])
        plt.xlim([0,30])
        #plt.xlim([-0.05, max(time)+0.05])
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


