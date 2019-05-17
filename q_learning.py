#####################
# Copyright Eric Sun 2019
#####################
# Python script containing all main functions for applying reinforcement learning to the control of aging
# complex networks. The script contains learning functions in the state-action space of (vitality, repair)
# and also for the (unpublished) reinforcement learning of (time, repair) protocols.
#####################

from model import *
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm


def QLearning_t (actions, pop_size=5, num_episodes=1, T=100, f=0.025, alpha=10, p_explore=0.05, tstep=1):
    '''
    [DEPRECATED]
    Old version:
    Q-learning function:
        States = Time step, t
        Actions = Repair rate, r
    Parameters:
        actions [list] = list of repair rates
        pop_size [int] = number of networks to use at each iteration (averaged over)
        T [int] = timeframe to simulate for
        f [float] = 0.025 (failure rate)
        alpha [float] = 10. (cost of repair)
        p_explore [float] = prob of exploration
        tstep [int] = 1 (timestep)
    '''
    # Define states and action space
    states = list(range(0,T+1,tstep))
    Q_matrix = np.zeros([len(states), len(actions)])
    discount = 0.95
    learning_rate = 0.1
    
    # Q-learning
    i = 0
    while i < num_episodes:
        # Global variables
        vitality = 1.0
        state = states[0]
        done = False
        while done == False:
            state_idx = getStateIdx(state, states) # get state index
            action_idx = q_learning_action(state_idx, Q_matrix, p_explore) # get best action for state
            action = actions[action_idx]
            new_state, reward, done, vitality = aging_env_t (state, action, T, vitality, alpha, pop_size, tstep) # test environment
            new_state_idx = getStateIdx(new_state, states) # get new state index
            # update state and Q matrix
            state_idx, Q_matrix = q_learning_updating(state_idx, action_idx, reward, new_state_idx, Q_matrix, learning_rate, discount)
            state = states[state_idx]
        i += 1
    
    return (Q_matrix)


def QLearning_t_fast (actions, pop_size=5, num_episodes=1, T=100, N=100, f=0.025, alpha=10, p_explore=0.05, tstep=1,
                     learning_rate=0.1, discount=0.95, reward='normal', decay = True):
    '''
    [DEPRECATED]
    Most updated version for States = time Q-learning
    Q-learning function:
        States = Time step, t
        Actions = Repair rate, r
    Parameters:
        actions [list] = list of repair rates
        pop_size [int] = number of networks to use at each iteration (averaged over)
        T [int] = timeframe to simulate for
        f [float] = 0.025 (failure rate)
        alpha [float] = 10. (cost of repair)
        p_explore [float] = prob of exploration
        tstep [int] = 1 (timestep)
        learning_rate [float] = learning rate for Q-learning (alpha)
        discount [float] = discount term for Q-learning (gamma)
        reward [string] = type of reward function ('normal'=healthspan)
        decay [boolean] = decay in learning_rate and p_explore ?
    '''
    # Define states and action space
    states = list(range(0,T+1,tstep))
    Q_matrix = np.zeros([len(states), len(actions)])
    
    # Set decay steps
    if decay is True:
        step_lr = learning_rate/num_episodes
        step_exp = p_explore/num_episodes
    
    # Q-learning
    for i in tqdm(range(num_episodes)):
        # Global variables
        vitality = 1.0
        state = states[0]
        done = False
        # Build network (A, v)
        A, v = initIndividual (N=N, graph_type='Grandom_s', p=0.1, d=0, edge_type='binary')
        while done == False:
            state_idx = getStateIdx(state, states) # get state index
            action_idx = q_learning_action(state_idx, Q_matrix, p_explore) # get best action for state
            action = actions[action_idx]
            new_state, reward, done, A, v, vitality = aging_env_t_fast (state, action, A, v, f, vitality, T, alpha, pop_size, reward, discount) # test environment
            new_state_idx = getStateIdx(new_state, states) # get new state index
            # update state and Q matrix
            state_idx, Q_matrix = q_learning_updating(state_idx, action_idx, reward, new_state_idx, Q_matrix, learning_rate, discount)
            state = states[state_idx]
        if decay is True:
            learning_rate += -step_lr
            p_explore += -step_exp
        time.sleep(3) # progress increment    
    
    return (Q_matrix)


def QLearning_v (actions, pop_size=5, num_episodes=1, T=100, N=1000, f=0.025, alpha=10, p_explore=0.05):
    '''
    [DEPRECATED]
    '''
    # Define states and action space
    vstep = 1/N # discrete vitality steps
    states = list(np.arange(0,1+vstep,vstep))
    sigfig = round(math.log10(N))
    states = [round(state, sigfig) for state in states]
    Q_matrix = np.zeros([len(states), len(actions)])
    discount = 0.95
    learning_rate = 0.1
    
    # Q-learning
    i = 0
    while i < num_episodes:
        # Global variables
        t = 0
        rand_idx = random.randint(0, len(states)-1) # randomly initialize vitality
        state = states[rand_idx]
        done = False
        while done == False:
            state_idx = getStateIdx(state, states) # get state index
            action_idx = q_learning_action(state_idx, Q_matrix, p_explore) # get best action for state
            action = actions[action_idx]
            new_state, reward, done, t = aging_env_v (state, action, N, T, t, alpha, pop_size) # test environment
            new_state = round(new_state, sigfig)
            new_state_idx = getStateIdx(new_state, states) # get new state index
            # update state and Q matrix
            state_idx, Q_matrix = q_learning_updating(state_idx, action_idx, reward, new_state_idx, Q_matrix, learning_rate, discount)
            state = states[state_idx]
        i += 1
    
    return (Q_matrix)


def QLearning_v_fast_cluster (actions, pop_size=1, num_episodes=100, T=100, N=100, f=0.025, alpha=10, p_explore=0.1,
                     learning_rate=0.1, discount=0.975, reward='normal', decay=[0,0], I=0, num_nets=50):
    '''
    CURRENNT Q-LEARNING MODEL
    Q-learning function:
        States = Vitality, phi
        Actions = Repair rate, r
    Parameters:
        actions [list] = list of repair rates
        pop_size [int] = number of networks to use at each iteration (averaged over) [SET TO 1]
        T [int] = timeframe to simulate for
        f [float] = 0.025 (failure rate)
        alpha [float] = 10. (cost of repair)
        p_explore [float] = prob of exploration
        tstep [int] = 1 (timestep)
        learning_rate [float] = learning rate for Q-learning (alpha)
        discount [float] = discount term for Q-learning (gamma)
        reward [string] = type of reward function ('normal'=healthspan)
        decay [list of float] = decay rate for p_explore and learning_rate respectively
        I [float] = interdependency failure criterion (I=0 -> linear, independent network)
        num_nets [int] = number of networks to generate and cycle through during training episodes
    '''
    # Define states and action space
    vstep = 1/N # discrete vitality steps
    states = list(np.arange(0,1+vstep,vstep))
    sigfig = round(math.log10(N))
    states = [round(state, sigfig) for state in states]
    Q_matrix = np.zeros([len(states), len(actions)])
        
    # Initialize networks (num_nets)
    networks = []
    for g in range(num_nets):
        pA, pv = initIndividual (N=N, graph_type='Grandom_s', p=0.1, d=0, edge_type='binary')
        networks.append([pA, pv])
    
    # Q-learning
    for i in tqdm(range(num_episodes)):
        # Global variables
        rand_idx = random.randint(0, len(states)-1) # randomly initialize vitality
        state = states[-1]
        done = False
        t = 0
        # Retrieve network with circular indexing
        [A_orig, v_orig] = networks[i % len(networks)]
        A = np.copy(A_orig)
        v = np.copy(v_orig)
        while done == False:
            state_idx = getStateIdx(state, states) # get state index
            action_idx = q_learning_action(state_idx, Q_matrix, p_explore) # get best action for state
            action = actions[action_idx]
            new_state, reward, done, A, v, t = aging_env_v_fast (state, action, A, v, f, t, T, alpha, pop_size, reward, discount, I) # test environment
            new_state = round(new_state, sigfig)
            new_state_idx = getStateIdx(new_state, states) # get new state index
            # update state and Q matrix
            state_idx, Q_matrix = q_learning_updating(state_idx, action_idx, reward, new_state_idx, Q_matrix, learning_rate, discount)
            state = states[state_idx]
            # decay learning rate and exploration
        p_explore = exp_decay(decay[0], i)
        learning_rate = exp_decay(decay[1], i)
        time.sleep(3) # progress increment
    
    return (Q_matrix)


def exp_decay (lambda_val, t):
    '''
    Exponential decay at rate lambda_val at time t
    '''
    return (math.exp(-lambda_val*t))


def QLearning_v_fast_cluster_unchanged (actions, pop_size=1, num_episodes=100, T=100, N=100, f=0.025, alpha=10, p_explore=0.1,
                     learning_rate=0.1, discount=0.975, reward='normal', decay=True, I=0, num_nets=50):
    '''
    OLD Q-LEARNING MODEL
    Q-learning function:
        States = Vitality, phi
        Actions = Repair rate, r
    Parameters:
        actions [list] = list of repair rates
        pop_size [int] = number of networks to use at each iteration (averaged over) [SET TO 1]
        T [int] = timeframe to simulate for
        f [float] = 0.025 (failure rate)
        alpha [float] = 10. (cost of repair)
        p_explore [float] = prob of exploration
        tstep [int] = 1 (timestep)
        learning_rate [float] = learning rate for Q-learning (alpha)
        discount [float] = discount term for Q-learning (gamma)
        reward [string] = type of reward function ('normal'=healthspan)
        decay [list of float] = decay rate for p_explore and learning_rate respectively
        I [float] = interdependency failure criterion (I=0 -> linear, independent network)
        num_nets [int] = number of networks to generate and cycle through during training episodes
    '''
    # Define states and action space
    vstep = 1/N # discrete vitality steps
    states = list(np.arange(0,1+vstep,vstep))
    sigfig = round(math.log10(N))
    states = [round(state, sigfig) for state in states]
    Q_matrix = np.zeros([len(states), len(actions)])
        
    # Initialize networks (num_nets)
    networks = []
    for g in range(num_nets):
        pA, pv = initIndividual (N=N, graph_type='Grandom_s', p=0.1, d=0, edge_type='binary')
        networks.append([pA, pv])
    
    # Get decay step
    if decay is True:
        step_exp = p_explore/num_episodes
    
    # Q-learning
    for i in tqdm(range(num_episodes)):
        # Global variables
        state = states[-1] # 1/16/2019: changed from [0]
        done = False
        t = 0
        # Retrieve network with circular indexing
        [A_orig, v_orig] = networks[i % len(networks)]
        A = np.copy(A_orig)
        v = np.copy(v_orig)
        while done == False:
            state_idx = getStateIdx(state, states) # get state index
            action_idx = q_learning_action(state_idx, Q_matrix, p_explore) # get best action for state
            action = actions[action_idx]
            new_state, reward, done, A, v, t = aging_env_v_fast (state, action, A, v, f, t, T, alpha, pop_size, reward, discount, I) # test environment
            new_state = round(new_state, sigfig)
            new_state_idx = getStateIdx(new_state, states) # get new state index
            # update state and Q matrix
            state_idx, Q_matrix = q_learning_updating(state_idx, action_idx, reward, new_state_idx, Q_matrix, learning_rate, discount)
            state = states[state_idx]
            # decay learning rate and exploration
        if decay is True:
            p_explore += -step_exp
        time.sleep(3) # progress increment
    
    return (Q_matrix)


def QLearning_v_fast (actions, pop_size=1, num_episodes=100, T=100, N=100, f=0.025, alpha=10, p_explore=0.1,
                     learning_rate=0.1, discount=0.95, reward='normal', decay = True, I = 0):
    '''
    OLD Q-LEARNING MODEL (for local machine use)
    Q-learning function:
        States = Vitality, phi
        Actions = Repair rate, r
    Parameters:
        actions [list] = list of repair rates
        pop_size [int] = number of networks to use at each iteration (averaged over) [SET TO 1]
        T [int] = timeframe to simulate for
        f [float] = 0.025 (failure rate)
        alpha [float] = 10. (cost of repair)
        p_explore [float] = prob of exploration
        tstep [int] = 1 (timestep)
        learning_rate [float] = learning rate for Q-learning (alpha)
        discount [float] = discount term for Q-learning (gamma)
        reward [string] = type of reward function ('normal'=healthspan)
        decay [list of float] = decay rate for p_explore and learning_rate respectively
        I [float] = interdependency failure criterion (I=0 -> linear, independent network)
        num_nets [int] = number of networks to generate and cycle through during training episodes
    '''
    # Define states and action space
    vstep = 1/N # discrete vitality steps
    states = list(np.arange(0,1+vstep,vstep))
    sigfig = round(math.log10(N))
    states = [round(state, sigfig) for state in states]
    Q_matrix = np.zeros([len(states), len(actions)])
    
    # Set decay steps
    if decay is True:
        step_exp = p_explore/num_episodes
    
    # Q-learning
    for i in tqdm(range(num_episodes)):
        # Global variables
        rand_idx = random.randint(0, len(states)-1) # randomly initialize vitality
        state = states[0]
        done = False
        t = 0
        # Build network (A, v)
        A, v = initIndividual (N=N, graph_type='Grandom_s', p=0.1, d=0, edge_type='binary')
        while done == False:
            state_idx = getStateIdx(state, states) # get state index
            action_idx = q_learning_action(state_idx, Q_matrix, p_explore) # get best action for state
            action = actions[action_idx]
            new_state, reward, done, A, v, t = aging_env_v_fast (state, action, A, v, f, t, T, alpha, pop_size, reward, discount, I) # test environment
            new_state = round(new_state, sigfig)
            new_state_idx = getStateIdx(new_state, states) # get new state index
            # update state and Q matrix
            state_idx, Q_matrix = q_learning_updating(state_idx, action_idx, reward, new_state_idx, Q_matrix, learning_rate, discount)
            state = states[state_idx]
            # decay learning rate and exploration
        if decay is True:
            #learning_rate += -step_lr
            p_explore += -step_exp
        time.sleep(3) # progress increment
    
    return (Q_matrix)



def aging_env_t (state, action, T, vitality, alpha, pop_size, tstep):
    '''
    [DEPRECATED]
    Age the network according to network aging model for the time-state Q-learning
    '''
    # get new vitality
    new_vitality = simPopulation('env', pop_size=pop_size, N=100, p=0.1, d=1-vitality, f=0.025, r=action, f_thresh=0,
    graph_type='Grandom_s', weight_type='uniform', check_type='none', kinetic=1, P_check=1, e=0, cost_type=['healthspan', 10], 
    costC=0.1, costR=1, costE=0.5, costD=0.5, costL=1, P_repl=0, costrepl=1, max_repl=1, repl_type='constant',
    node_type='binary', damage_type='uniform', edge_type='binary', f_edge=0, r_edge=0, std=0.3, 
    P_ablate=0,costablate=1,ablate_type='constant',repair_start=0,repair_end=1,delay=0,time_end=tstep,dependency=0,save='no',plot='no')
    
    # Calculate reward = -cost
    
    # SUGGESTIONS:
    # maybe a minus time or something like that would be good here
    # **coarser t intervals?
    # ***use simIndivudal components to make faster
    # **quadratic cost
    # ***how to relate reward to cost integrand?
    # ***run longer for convergence
    reward = new_vitality - alpha*action
    
    # Increment state
    new_state = state + tstep
    
    # Determines if episode is over
    if new_state == T:
        done = True
    else:
        done = False
    
    return (new_state, reward, done, new_vitality)


def aging_env_t_fast (state, action, A, v, f, vitality, T, alpha, pop_size, reward, discount):
    '''
    [DEPRECATED]
    Faster model
    Age the network according to network aging model for the time-state Q-learning
    '''
    ##### Default variables #####
    damage_type = 'uniform'
    node_type = 'binary'
    edge_type = 'binary'
    f_edge = 0
    r_edge = 0
    std = 0
    i = 0
    P_check = 0
    check_type = 'none'
    weight_type = 'uniform'
    kinetic = 1
    e = 0
    costC = 0
    costR = 0
    dependency = 0 # toggle for nonlinear model
    #############################
    
    # Seet repair as action
    r = action
    
    # Damage network
    A, v, f = Damage(A, v, f, damage_type, node_type, edge_type, f_edge, std, i)
    
    # Repair network
    cost_cr, A, v, P_check, r = Check_and_Repair(A, v, r, check_type, kinetic, P_check, e, i, costC, costR, 
                                                              node_type, edge_type, r_edge, std)
    # Interdependency failure
    if dependency > 0:
        v = dependencyFail(A, v, dependency)
    
    # Get vitality
    degree_vec = getDegrees (A)
    weight_vec = getWeights (weight_type, A, v, degree_vec)
    vitality, interdependence_i = Analyze(v, f, r, i, weight_vec)
        
    # Increment time
    state += 1
    new_state = state
    
    # Calculate reward 
    if reward == 'quadratic':
        reward = vitality - alpha*action**2
    elif reward == 'half_vit': # phi@0.5 --> zero cost
        reward = vitality - alpha*action
        reward += -0.4
    elif reward == 'half_crit_vit': #phi@0.75 --> zero cost
        reward = vitality - alpha*action
        reward += -0.65
    elif reward == 'exp_decay':
        reward = (vitality - alpha*action) * math.exp(-discount*(new_state-1))
    else: # basic cost
        reward = vitality - alpha*action
    
    # Determines if episode is over
    if state == T:
        done = True
    else:
        done = False
    
    return (new_state, reward, done, A, v, vitality)


def aging_env_v (state, action, N, T, t, alpha, pop_size):
    '''
    [DEPRECATED]
    Old function for aging network for vitality-state Q-learning
    '''
    # get new vitality
    new_state = simPopulation('env', pop_size=pop_size, N=N, p=0.1, d=1-state, f=0.025, r=action, f_thresh=0,
    graph_type='Grandom_s', weight_type='uniform', check_type='none', kinetic=1, P_check=1, e=0, cost_type=['healthspan', 10], 
    costC=0.1, costR=1, costE=0.5, costD=0.5, costL=1, P_repl=0, costrepl=1, max_repl=1, repl_type='constant',
    node_type='binary', damage_type='uniform', edge_type='binary', f_edge=0, r_edge=0, std=0.3, 
    P_ablate=0,costablate=1,ablate_type='constant',repair_start=0,repair_end=1,delay=0,time_end=1,dependency=0,save='no',plot='no')
    
    reward = new_state - alpha*action
    
    # Increment state
    new_t = t + 1
    
    # Determines if episode is over
    if new_t == T:
        done = True
    else:
        done = False
    
    return (new_state, reward, done, new_t)


def aging_env_v_fast (state, action, A, v, f, t, T, alpha, pop_size, reward, discount, I = 0):
    '''
    Main/current function for implementing an aging environment for the learning agent
    '''
    ##### Default variables #####
    damage_type = 'uniform'
    node_type = 'binary'
    edge_type = 'binary'
    f_edge = 0
    r_edge = 0
    std = 0
    i = 0
    P_check = 0
    check_type = 'none'
    weight_type = 'uniform'
    kinetic = 1
    e = 0
    costC = 0
    costR = 0
    dependency = I # toggle for nonlinear model
    #############################
    
    # Seet repair as action
    r = action
    
    # Damage network
    A, v, f = Damage(A, v, f, damage_type, node_type, edge_type, f_edge, std, i)
    
    # Repair network
    cost_cr, A, v, P_check, r = Check_and_Repair(A, v, r, check_type, kinetic, P_check, e, i, costC, costR, 
                                                              node_type, edge_type, r_edge, std)
    # Interdependency failure
    if dependency > 0:
        v = dependencyFail(A, v, dependency)
    
    # Get vitality
    degree_vec = getDegrees (A)
    weight_vec = getWeights (weight_type, A, v, degree_vec)
    vitality, interdependence_i = Analyze(v, f, r, i, weight_vec)
    
    # Set new state to new vitality
    new_state = vitality
 
    reward = new_state - alpha*action
    
    # Increment time
    t += 1
    
    # Determines if episode is over
    if t == T:
        done = True
    else:
        done = False
    
    return (new_state, reward, done, A, v, t)


# In[14]:

def q_learning_action(s, Q_matrix, p_explore):
    rand_draw = random.uniform(0,1)
    if rand_draw < p_explore:
        action_idx = random.randint(0, Q_matrix.shape[1]-1)
    else:
        action_idx = np.argmax(Q_matrix[s, :])
    return (action_idx)

def q_learning_updating(s, a, reward, s2, Q_matrix, learning_rate, discount):
    Q_matrix[s, a] = (1 - learning_rate)*Q_matrix[s, a] + learning_rate*(reward + discount*np.amax(Q_matrix[s2, :]))
    s = s2
    return (s, Q_matrix)

def getActionIdx (action, actions):
    idx = actions.index(action)
    return (idx)

def getStateIdx (state, states):
    idx = states.index(state)
    return (idx)


### Visualization of optimal Q matrix


def visualizeQ_t (Q_matrix, actions, filename, N=1000, T=100, f=0.025):
    
    num_states = Q_matrix.shape[0]
    step_size = T/num_states
    state_vec = np.arange(0,T,step_size)
    state_vec = state_vec.tolist()
    
    action_vec = []
    vitalities_vec = []
    
    # fetch optimal actions (repair)
    for state in state_vec:
        state_idx = getStateIdx(state, state_vec)
        best_action_idx = np.argmax(Q_matrix[state_idx, :])
        best_action = actions[best_action_idx]
        action_vec.append(best_action)
        
    # Calculate vitality curve
    vitalities = []
    vitality = 1
    for t in state_vec:
        state_idx = getStateIdx(t, state_vec)
        best_action_idx = np.argmax(Q_matrix[state_idx, :])
        best_action = actions[best_action_idx]
        vitality += -(f*vitality)
        vitality += best_action*(1-vitality)
        vitalities.append(vitality) # for plotting
        if vitality < 0:
            vitality = 0
        
    # Save Q_matrix
    np.savetxt('T_'+filename+'.txt', Q_matrix)
    
    # PLOTS
    plt.figure(figsize=(10,2.5))
    plt.subplot(1,2,1)
    plt.title('Vitality', fontsize=14)
    plt.xlabel('Time, $t$', fontsize=12)
    plt.ylabel('Vitality, $\phi$', fontsize=12)
    plt.plot(state_vec, vitalities, 'g', linewidth=2.5)
    
    plt.subplot(1,2,2)
    plt.title('Optimal repair', fontsize=14)
    plt.xlabel('Time, $t$', fontsize=12)
    plt.ylabel('Repair rate, $r$', fontsize=12)
    plt.ylim([0-max(action_vec)*0.05, max(action_vec)*1.05])
    plt.plot(state_vec, action_vec, 'g', linewidth=2.5)
    plt.savefig('./Figures/reinforcement/T_'+filename+'.png', dpi=500)
    plt.tight_layout()
    plt.show()


def visualizeQ_v (Q_matrix, actions, filename, N=1000, T=100, f=0.025):
    
    num_states = Q_matrix.shape[0]
    step_size = 1
    t_vec = np.arange(0,T,step_size)
    t_vec = t_vec.tolist()
    
    # Define state vec
    vstep = 1/N
    state_vec = list(np.arange(0,1+vstep,vstep))
    sigfig = round(math.log10(N))
    state_vec = [round(state, sigfig) for state in state_vec]
    
    action_vec = []
    
    # fetch optimal actions (repair)
    vitalities = []
    vitality = 1
    vitality_rounded = 1
    for t in t_vec:
        state_idx = getStateIdx(vitality_rounded, state_vec)
        best_action_idx = np.argmax(Q_matrix[state_idx, :])
        best_action = actions[best_action_idx]
        action_vec.append(best_action)
        vitality += -(f*vitality)
        vitality += best_action*(1-vitality)
        vitality_rounded = round(vitality, sigfig)
        vitalities.append(vitality) # for plotting
        if vitality < 0:
            vitality = 0
    
    # Save Q_matrix
    np.savetxt(filename+'.txt', Q_matrix)


def viz_from_file (filename, actions, N=100, T=100, f=0.025):
    # read from file
    Q_matrix = np.genfromtxt('results_files/'+filename+'.txt')
    print ('Q_matrix shape = (' + str(Q_matrix.shape[0])+ ', ' + str(Q_matrix.shape[1])+')')
    visualizeQ_v (Q_matrix, actions, filename, N, T, f)


import numpy as np
import math
import matplotlib.pyplot as plt
import os

def plotResults (directory, param, p_range, repair_list=[0,0.01], N=100, T=100, plot=False, fine_res=False,
                binning=False):
    '''
    directory = path to directory containing ordered files
    param = 'alpha', 'failure', 'repair', 'gamma'
    p_range = list or array of ordered parameter values
    '''
    # returned values
    cT1 = []
    aT1 = []
    
    # order in files
    os.chdir(directory)
    files = filter(os.path.isfile, os.listdir(directory)) # filter nonfiles
    files = [os.path.join(directory, f) for f in files] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x)) # sort by date (parameter ordering)
    
    # Default param values
    f = 0.025
    r = 0.01
    alpha = 10
    gamma = 0.95
    d = 0
    
    # Analyze results
    for i in range(len(files)):
        # Get right param assignment
        p = p_range[i]
        if param == 'alpha':
            alpha = p
        elif param == 'failure':
            f = p
        elif param == 'repair':
            r = p
            repair_list = [0, r]
        elif param == 'gamma':
            gamma = p
        else:
            return Exception("'param' must be 'alpha', 'failure', 'repair', or 'gamma'!")
        
        # analyze
        compT1, anlyT1 = validate_analytics(files[i], [f,r,alpha,gamma,d,T], repair_list, N)
        cT1.append(compT1)
        aT1.append(anlyT1)
        
    if fine_res is True:
        aT1 = []
        a_range = np.linspace(min(p_range), max(p_range), 1000)
        for a in a_range:
            if param == 'alpha':
                alpha = a
            elif param == 'failure':
                f = a
            elif param == 'repair':
                r = a
            elif param == 'gamma':
                gamma = a
            aT1.append(analytical_T1([f,r,alpha,gamma,d,T]))
    
    # Plot
    if plot != False:
        plt.figure(figsize=(6,3))
        # Plot learned results
        if binning is False:
            plt.scatter(p_range[:len(files)], cT1, c='k', marker='o', label='Q-learning')
        else:
            param_vals = p_range[:len(files)]
            binned_vals = []
            binned_params = []
            for j in range(0,len(cT1),binning): # binning is an int specifying number of points to average over
                binned_vals.append(np.mean(np.array(cT1[j:j+binning])))
                binned_params.append(np.mean(np.array(param_vals[j:j+binning])))
            plt.scatter(binned_params, binned_vals, c='k', marker='o', label='Q-learning')

                
        # Plot analytics
        if fine_res is True:
            plt.plot(a_range, aT1, 'k', linewidth=3, alpha=0.6, label='Analytic')
        else:
            plt.plot(p_range[:len(files)], aT1, 'k', linewidth=3, alpha=0.6, label='Analytic')
        plt.ylabel('Switching Time, $T_1$', fontsize=14)
        if param == 'alpha':
            plt.xlabel('Repair Cost, '+r'$\alpha$', fontsize=14)
            plt.xlim(0,None)
        elif param == 'failure':
            plt.xlabel('Failure rate, $f$', fontsize=14)
            plt.xlim(0,None)
        elif param == 'repair':
            plt.xlabel('Repair rate, $r$', fontsize=14)
            plt.xlim(0,None)
        elif param == 'gamma':
            plt.xlabel('Reward Discount, '+r'$\gamma$', fontsize=14)
        plt.ylim(0,T)
        plt.tight_layout()
        plt.savefig('../../'+plot+'.png', dpi=800)
        plt.show()
        
    return (cT1, aT1)


def validate_analytics(filepath, analytic_params, repair_list=[0,0.01], N=100):
    '''
    Function for visualizing/validating computational Q-learning results to inf. hor. analytical optimal controls
    
    datafile = name of Q-learning results text file
    repair_list = list of repair rates
    N = number of nodes
    T = simulated time
    analytic_params = list of parameters of form [f, r, alpha, gamma, d]
    '''
    f = analytic_params[0]
    T = analytic_params[5]
    Q_matrix = np.genfromtxt(filepath) # read file
    repair_vec, vitality_vec = get_protocol(Q_matrix, repair_list, N, T, f) # get repair protocol and vitalities
    time_vec = np.arange(0,len(repair_vec)) # get time vec
    comp_T1 = get_T1(time_vec, repair_vec, T) # get T1 from Q-learning
    anly_T1 = analytical_T1(analytic_params) # get T1 from analytics
        
    return (comp_T1, anly_T1)

    
def get_protocol (Q_matrix, actions, N, T, f):
    '''
    Retrieve optimal repair protocol by deterministic simulation of a network with failure rate f and using the
    learned Q matrix for optimal repairs from states of vitality.
    '''
    num_states = Q_matrix.shape[0]
    step_size = 1
    t_vec = np.arange(0,T,step_size)
    t_vec = t_vec.tolist()
    
    # Define state vec
    vstep = 1/N
    state_vec = list(np.arange(0,1+vstep,vstep))
    sigfig = round(math.log10(N))
    state_vec = [round(state, sigfig) for state in state_vec]
    
    action_vec = []
    
    # fetch optimal actions (repair)
    vitalities = []
    vitality = 1
    vitality_rounded = 1
    for t in t_vec:
        state_idx = getStateIdx(vitality_rounded, state_vec)
        best_action_idx = np.argmax(Q_matrix[state_idx, :])
        best_action = actions[best_action_idx]
        action_vec.append(best_action)
        vitality += -(f*vitality)
        vitality += best_action*(1-vitality)
        vitality_rounded = round(vitality, sigfig)
        vitalities.append(vitality) # for plotting
        if vitality < 0:
            vitality = 0
    
    return (action_vec, vitalities)
 
def get_T1 (time_vec, repair_vec, T, p=0.5):
    '''
    Get T1 choice that optimizes classification of r=0 and r=r points
    '''
    if np.count_nonzero(repair_vec) > 0.1*T: # check to make sure not flat
        error_count = []
        for i in range(len(time_vec)): # count classification errors
            err_lower = np.count_nonzero(repair_vec[:i])
            err_upper = len(repair_vec[i:]) - np.count_nonzero(repair_vec[i:])
            error_count.append(err_lower+err_upper)
        err, min_err_idx = min((val, idx) for (idx, val) in enumerate(error_count)) # get T1 with min error
        T1 = time_vec[min_err_idx]
    else:
        T1 = float('NaN') # out of frame
    
    return (T1)
    
    
def get_T1_sliding_frame (time_vec, repair_vec, T, p=0.5):
    '''
    [DEPRECATED]
    T1 will be approximated by the first time point (t) 
    after which there is at least p proportion of nonzero repair in next fifth
    '''
    for i in range(len(time_vec)):
        if i < round((1-0.2)*T):
            if repair_vec[i] > 0 and np.count_nonzero(repair_vec[i:i+round(0.2*T)]) > round(T*0.2*p):
            #if np.sum(np.subtract(np.array(time_vec[i:i+round(p*T)]), np.multiply(np.ones(round(p*T)),repair_vec[i]))) == 0:
                T1 = i
                break
        else:
            T1 = float('NaN') # out of frame
    
    return (T1)


def analytical_T1 (params_list):
    '''
    Analytical approximation for the infinite horizon optimal repair switching time, T1
    '''
    
    f = params_list[0]
    r = params_list[1]
    alpha = params_list[2]
    gamma = params_list[3]
    d = params_list[4]
    T = params_list[5]
    
    # convert gamma
    gamma = -math.log(gamma)
    
    # calculate T1
    try:
        T1 = 1/(f) * math.log((1-d)/(1-alpha*(f+r+gamma)))
    except:
        T1 = float('NaN') # out of frame
    
    return (T1)


def numerical_T1 (params_list):
    '''
    Numerical approximation for the infinite horizon optimal repair switching time, T1
    '''
    from scipy.optimize import fsolve
    
    f = params_list[0]
    r = params_list[1]
    alpha = params_list[2]
    gamma = params_list[3]
    d = params_list[4]
    T = params_list[5]
    
    # large approx for T and T2
    
    # convert gamma
    gamma = -math.log(gamma)
    
    # calculate T1 (and T2)
    init_guess=20
    T1, T2 = fsolve(numerical_function, [init_guess, T-init_guess], args=(f,r,gamma,alpha,d,T))
    if T1 > 100:
        T1 = float('nan')
    
    return (T1)


def numerical_function (t, f, r, g, alpha, d, T):
    '''
    System of equations to solve for T1 (and T2)
    '''
    [t1, t2] = t
    #t2 = T # T2 --> inf approx.
    f1 = ((1-d)*np.exp(-f*t1)-1) * ((np.exp((f+g)*(t2-T))-1)/(f+g)*np.exp((r+f+g)*(t1-t2))+(np.exp((r+f+g)*(t1-t2))-1)/(f+r+g)) - alpha
    f2 = ((1-d)*np.exp(-f*t1)*np.exp(-(r+f)*(t2-t1))+(r*(1-np.exp(-(r+f)*(t2-t1))))/(r+f)-1)*(np.exp((f+g)*(t2-T))-1)/(f+g) - alpha
    return ([f1,f2])


def numerical_function_bang (t, f, r, g, alpha, d, T):
    # system of equations to solve for T1 and T2
    [t1, t2] = t
    f1 = (np.exp(-f*t1)-1) * ((r+f)*np.exp(f*(t2-T))*np.exp((r+f)*(t1-t2)) - r*np.exp((r+f)*(t1-t2))-f) - alpha*f*(r+f)
    f2 = ((r+f)*np.exp(-f*t1)*np.exp(-(r+f)*(t2-t1))-r*np.exp(-(r+f)*(t2-t1))-f) * (np.exp(f*(t2-T))-1) - alpha*f*(r+f)
    return ([f1, f2])


#### Function for Bang/Switch Matching
def plotMatching (filename, savename, f, r, alpha, gamma, d, T=100, N=100):
    '''
    Make plots of repair as a function of time where the switching time is determined analytically and the
    optimal repair protocol is plotted as a scatter (overlayed)
    '''
    
    # Plot analytic switch
    aT1 = analytical_T1([f, r, alpha, gamma, d, T]) # get analytical switch time
    t_range = np.linspace(0,T,1000)
    r_range = [0 if t<aT1 else r for t in t_range]
    plt.figure(figsize=(6,3))
    plt.plot(t_range, r_range, 'b', linewidth=2.5, alpha=0.3, label='Analytic $r(t)$')
    
    # Plot learned repair
    Q_matrix = np.genfromtxt(filename) # read file
    repair_vec, vitality_vec = get_protocol(Q_matrix, [0, r], N, T, f) # get repair protocol and vitalities
    time_vec = np.arange(0,len(repair_vec)) # get time vec
    plt.scatter(np.array(time_vec), np.array(repair_vec), color='b', s=7, label='Learned $r(t)$')
    
    print (get_T1(time_vec, repair_vec, T))
    
    #plt.legend(loc='center right')
    plt.xlabel('Time, $t$', fontsize=14)
    plt.ylabel('Repair, $r(t)$', fontsize=14)
    plt.xlim(0,T)
    plt.tight_layout()
    plt.savefig('/Users/edsun/Desktop/RESEARCH/Mathematical_Aging/ReinforcementLearning/Matching_'+savename+'.png', dpi=700)
    plt.show()

# functions for averaging T1 from multiple runs
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def extract(raw_string, start_marker, end_marker):
    start = raw_string.index(start_marker) + len(start_marker)
    end = raw_string.index(end_marker, start)
    return (raw_string[start:end])

def plotResultsAVG (directory, param, p_range, repair_list=[0,0.01], N=100, T=100, soln_type='analytic', plot=False, fine_res=False,
                binning=False):
    '''
    directory = path to directory containing subdirectories with ordered files
    param = 'alpha', 'failure', 'repair', 'gamma'
    p_range = list or array of ordered parameter values
    repair_list = list of actions (floats)
    N = network size
    T = time for simulation
    soln_type = 'numeric' (numerical approx. to exact soln) or 'analytic' (approx. soln)
    plot = Boolean
    fine_res = Boolean (use True): finer reseoltuon for soln_type plot
    binning = Boolean: bin parameter ranges together and get average switch time?
    '''
    # returned values
    cT1 = []
    aT1 = []
    cT1_std = []
    
    # Get all subdirectories
    subdirs = get_immediate_subdirectories(directory)
    
    # order in files (for first subdir)
    first_dir_path = directory+'/'+subdirs[0]+'/results/'
    files = [f for f in os.listdir(first_dir_path) if os.path.isfile(os.path.join(first_dir_path, f))]
    files = [os.path.join(first_dir_path, f) for f in files] # add path to each file
    
    # Default param values
    f = 0.025
    r = 0.01
    alpha = 10
    gamma = 0.975
    d = 0
    
    # Analyze results
    comp_p_range = []
    
    for file in files:
        if param == 'failure':
            marker1 = '_f'
            marker2 = '_r'
        elif param == 'repair':
            marker1 = '_r'
            marker2 = '_a'
        elif param == 'alpha':
            marker1 = '_a'
            marker2 = '_g'
        elif param == 'gamma':
            marker1 = '_g'
            marker2 = '_N'
        
        # take out parameter value from filename
        p = float(extract(os.path.basename(file),marker1,marker2)) 
            
        # Get right param assignment
        #p = p_range[i]
        if param == 'alpha':
            alpha = p
        elif param == 'failure':
            f = p
        elif param == 'repair':
            r = p
            repair_list = [0, r]
        elif param == 'gamma':
            gamma = p
            # Skip gamma = 1
            if p > 0.998:
                continue
        else:
            return Exception("'param' must be 'alpha', 'failure', 'repair', or 'gamma'!")
        
        # append computational param values
        comp_p_range.append(p)
        
        # analyze (get all and average)
        c_list = []
        for sub in subdirs:
            filepath = directory+'/'+sub+'/results/'+os.path.basename(file)
            try:
                compT1, anlyT1 = validate_analytics(filepath, [f,r,alpha,gamma,d,T], repair_list, N)
                c_list.append(compT1)
            except: continue
        cT1.append(sum(c_list)/len(c_list))
        cT1_std.append(np.std(np.array(c_list)))
        aT1.append(anlyT1)
        
    if fine_res is True:
        aT1 = []
        a_range = np.linspace(min(p_range), max(p_range), 1000)
        for a in a_range:
            if param == 'alpha':
                alpha = a
            elif param == 'failure':
                f = a
            elif param == 'repair':
                r = a
            elif param == 'gamma':
                gamma = a
                
            # Get analytical or numerical solution
            if soln_type == 'analytic':
                aT1.append(analytical_T1([f,r,alpha,gamma,d,T]))
            elif soln_type == 'numeric':
                aT1.append(numerical_T1([f,r,alpha,gamma,d,T]))
            else:
                raise Exception ("soln_type not recognized!")
    
    # Plot
    if plot != False:
        plt.figure(figsize=(6,4))
        # Plot learned results
        if binning is False:
            plt.errorbar(comp_p_range, cT1, yerr=cT1_std, color='0.5', linestyle="None")
            plt.scatter(comp_p_range, cT1, c='0.5', marker='o', s=25, edgecolors='none', label='Q-learning')
        else:
            param_vals = comp_p_range
            binned_vals = []
            binned_params = []
            for j in range(0,len(cT1),binning): # binning is an int specifying number of points to average over
                binned_vals.append(np.mean(np.array(cT1[j:j+binning])))
                binned_params.append(np.mean(np.array(param_vals[j:j+binning])))
            plt.scatter(binned_params, binned_vals, c='k', marker='o', label='Q-learning')

                
        # Plot analytics
        if fine_res is True:
            # Smoothen numerics with filter
            if soln_type == 'numeric':
                from scipy.ndimage.filters import median_filter
                aT1 = median_filter(aT1,size=100)
            plt.plot(a_range, aT1, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
        else:
            plt.plot(p_range, aT1, 'm', linestyle='--', linewidth=3, alpha=0.6, label='Theoretical')
        plt.ylabel('Switching Time, $T_1$', fontsize=14)
        if param == 'alpha':
            plt.xlabel('Repair Cost, '+r'$\alpha$', fontsize=14)
            plt.xlim(0,15)
        elif param == 'failure':
            plt.xlabel('Failure rate, $f$', fontsize=14)
            plt.xlim(0.005,0.051)
        elif param == 'repair':
            plt.xlabel('Repair rate, $r$', fontsize=14)
            plt.xlim(0,None)
        elif param == 'gamma':
            plt.xlabel('Reward Discount, '+r'$\gamma$', fontsize=14)
            plt.xlim(0.945,0.998)
        plt.ylim(0,T)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('../../'+plot+'.png', dpi=800)
        plt.show()
        
    return (cT1, aT1)


def plotMatchingAVG (path, filename, savename, f, r, alpha, gamma, d, T=100, N=100, method='mode'):
    '''
    path = path to dir of subdirs
    filename = specific txt file form to visualize
    method = 'mean' --> average the repairs (not binary), 'mode' --> binary, most frequent
    '''
    from scipy import stats
    # Plot analytic switch
    aT1 = analytical_T1([f, r, alpha, gamma, d, T]) # get analytical switch time
    t_range = np.linspace(0,T,1000)
    r_range = [0 if t<aT1 else r for t in t_range]
    plt.figure(figsize=(6,4))
    plt.plot(t_range, r_range, '#000080', linewidth=2.5, alpha=0.3, label='Theoretical $r(t)$')
    
    # Plot learned repair
    subdirs = get_immediate_subdirectories(path)  # get subdirs in path
    repair_vecs = []
    for sub in subdirs: # loop subdirs
        filepath = path+'/'+os.path.basename(sub)+'/results/'+os.path.basename(filename)
        Q_matrix = np.genfromtxt(filepath) # read file
        repair_vec, vitality_vec = get_protocol(Q_matrix, [0, r], N, T, f) # get repair protocol and vitalities
        repair_vecs.append(repair_vec)
    
    # Average with mean or mode method
    repair_mat = np.vstack(repair_vecs)
    if method == 'mean':
        repair_vec = np.mean(repair_mat,axis=0)
    elif method == 'mode':
        repair_vec = []
        for n in range(repair_mat.shape[1]):
            repair_vec.append(stats.mode(repair_mat[:,n])[0])
    else:
        raise Exception ('method not recognized!')
    time_vec = np.arange(0,len(repair_vec)) # get time vec
    plt.scatter(time_vec.tolist(), repair_vec, color='#000080', s=7, alpha=0.7, label='Learned $r(t)$')
    
    plt.xlabel('Time, $t$', fontsize=14)
    plt.ylabel('Repair, $r(t)$', fontsize=14)
    plt.xlim(0,T)
    plt.ylim(-0.001, None)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('/Users/edsun/Desktop/RESEARCH/Mathematical_Aging/ReinforcementLearning/Matching_'+savename+'.png', dpi=700)
    plt.show()
    
    
def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


# Nonlinear RL cluster visualization
def plotNonlinear (I_list, directories_list, param, p_range, repair_list=[0,0.01], N=100, T=100, soln_type='analytic', plot=False, fine_res=False,
                binning=False):
    '''
    I_list = list of interdependency (I) values between 0 and 1
    directories_list = list of paths to directory containing subdirectories with ordered files
    param = 'alpha', 'failure', 'repair', 'gamma'
    p_range = list or array of ordered parameter values
    repair_list = list of actions (floats)
    N = network size
    T = time for simulation
    soln_type = 'numeric' (numerical approx. to exact soln) or 'analytic' (approx. soln)
    plot = Boolean
    fine_res = Boolean (use True): finer reseoltuon for soln_type plot
    binning = Boolean: bin parameter ranges together and get average switch time?
    '''
    colors = ['#000080', '#FFC0CB', '#FFA500', 'm', 'k', 'g', 'r', 'c', 'y']
    all_cT1 = []
    all_aT1 = []
    comp_p_ranges = []
    for directory in directories_list:
        # returned values
        cT1 = []
        aT1 = []
        cT1_std = []

        # Get all subdirectories
        subdirs = get_immediate_subdirectories(directory)

        # order in files (for first subdir)
        first_dir_path = directory+'/'+subdirs[0]+'/results/'
        files = [f for f in os.listdir(first_dir_path) if os.path.isfile(os.path.join(first_dir_path, f))]
        files = [os.path.join(first_dir_path, f) for f in files] # add path to each file

        # Default param values
        f = 0.025
        r = 0.01
        alpha = 10
        #gamma = 0.95
        gamma = 0.975
        d = 0

        # Analyze results
        comp_p_range = []

        for file in files:
            if param == 'failure':
                marker1 = '_f'
                marker2 = '_r'
            elif param == 'repair':
                marker1 = '_r'
                marker2 = '_a'
            elif param == 'alpha':
                marker1 = '_a'
                marker2 = '_g'
            elif param == 'gamma':
                marker1 = '_g'
                marker2 = '_N'

            # take out parameter value from filename
            p = float(extract(os.path.basename(file),marker1,marker2)) 

            # Get right param assignment
            if param == 'alpha':
                alpha = p
            elif param == 'failure':
                f = p
            elif param == 'repair':
                r = p
                repair_list = [0, r]
            elif param == 'gamma':
                gamma = p
                if p > 0.998:
                    continue
            else:
                return Exception("'param' must be 'alpha', 'failure', 'repair', or 'gamma'!")

            # append computational param values
            comp_p_range.append(p)

            # analyze (get all and average)
            c_list = []
            for sub in subdirs:
                filepath = directory+'/'+sub+'/results/'+os.path.basename(file)
                try:
                    compT1, anlyT1 = validate_analytics(filepath, [f,r,alpha,gamma,d,T], repair_list, N)
                    c_list.append(compT1)
                except: continue
            cT1.append(sum(c_list)/len(c_list))
            cT1_std.append(np.std(np.array(c_list)))
            aT1.append(anlyT1)
        
        comp_p_ranges.append(comp_p_range)
        
        if fine_res is True:
            aT1 = []
            a_range = np.linspace(min(p_range), max(p_range), 1000)
            for a in a_range:
                if param == 'alpha':
                    alpha = a
                elif param == 'failure':
                    f = a
                elif param == 'repair':
                    r = a
                elif param == 'gamma':
                    gamma = a

                # Get analytical or numerical solution
                if soln_type == 'analytic':
                    aT1.append(analytical_T1([f,r,alpha,gamma,d,T]))
                elif soln_type == 'numeric':
                    aT1.append(numerical_T1([f,r,alpha,gamma,d,T]))
                else:
                    raise Exception ("soln_type not recognized!")
            
        all_cT1.append(cT1)
        all_aT1.append(aT1)
            
    for idx, I in enumerate(I_list):
        # Plot
        if plot != False:
            # Plot learned results
            if binning is False:
                plt.scatter(comp_p_range, all_cT1[idx], c=colors[idx%len(colors)], marker='o', s=25, edgecolors='none')
                dub_list = [list(x) for x in zip(*sorted(zip(comp_p_range, all_cT1[idx]), key=lambda pair: pair[0]))]
                x_list = dub_list[0]
                y_list = dub_list[1]
                plt.plot(x_list, y_list, c=colors[idx], label='$I=$'+str(I))
            else:
                param_vals = comp_p_range
                binned_vals = []
                binned_params = []
                for j in range(0,len(cT1),binning): # binning is an int specifying number of points to average over
                    binned_vals.append(np.mean(np.array(all_cT1[idx][j:j+binning])))
                    binned_params.append(np.mean(np.array(param_vals[j:j+binning])))
                plt.scatter(binned_params, binned_vals, marker='o', label='$I=$'+str(I))

            plt.ylabel('Switching Time, $T_1$', fontsize=14)
            if param == 'alpha':
                plt.xlabel('Repair Cost, '+r'$\alpha$', fontsize=14)
                plt.xlim(0,15)
            elif param == 'failure':
                plt.xlabel('Failure rate, $f$', fontsize=14)
                plt.xlim(0.005,0.051)
            elif param == 'repair':
                plt.xlabel('Repair rate, $r$', fontsize=14)
                plt.xlim(0,None)
            elif param == 'gamma':
                plt.xlabel('Reward Discount, '+r'$\gamma$', fontsize=14)
                plt.xlim(0.945,0.998)
            plt.ylim(0,T)
            
    # Plot rescaled and rescaling coefficients [ONLY VALID FOR ALPHA]
    best_coeffs = []
    numerical_soln = all_aT1[0] # at I=0
    plt.plot(a_range, numerical_soln, label='numeric')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('../../'+plot+'_rl.png', dpi=800)
    plt.show()
    
    
    
    for idx, I in enumerate(I_list):
        parameter_list = comp_p_ranges[idx]
        distances = []
        coeffs_list = np.linspace(0.5,4,100)
        parameter_list = comp_p_ranges[idx]
        for coeff in coeffs_list:
            new_param_list = parameter_list/coeff
            numer_points = []
            computational_cleaned = []
            for jdx, prm in enumerate(new_param_list):
                if math.isnan(all_cT1[idx][jdx]):
                    continue
                else:
                    closest = min(a_range, key=lambda x:abs(x-prm))
                    closest_idx = np.where(a_range==closest)
                    numer_points.append(numerical_soln[closest_idx[0][0]])
                    computational_cleaned.append(all_cT1[idx][jdx])
                    
            distances.append(np.linalg.norm(np.array(numer_points)-np.array(computational_cleaned)))

        best_idx = distances.index(min(distances))
        best_coeff = coeffs_list[best_idx]
        best_coeffs.append(best_coeff)

    plt.figure(figsize=(6,2.5))
    plt.plot(I_list, [1/b for b in best_coeffs],color='k', alpha=0.5, linewidth=1.5)
    plt.scatter(I_list, [1/b for b in best_coeffs],color='k',alpha=0.8, s=30)
    plt.xlabel('Interdependency, $I$', fontsize=14)
    plt.ylabel('Scaling coefficient, $C(I)$', fontsize=12)
    plt.xlim(0,0.2)
    if max(I_list)<0.2:
        plt.xlim(0,max(I_list))
    plt.tight_layout()
    plt.savefig('C_I_rl.png',dpi=700)
    plt.show()
    
    colors = ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r']
    for idx, I in enumerate(I_list):
        plt.scatter(comp_p_ranges[idx]/best_coeffs[idx], all_cT1[idx], color=colors[idx], label='$I=$'+str(I))
        plt.legend()
    plt.savefig('collapsed_rl.png', dpi=800)
    plt.show()