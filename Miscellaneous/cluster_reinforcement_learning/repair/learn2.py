from model import *
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm
from q_learning import *

f = 0.025
r = 0.01
alpha = 10
gamma = 0.975

for r in np.linspace(0.026,0.05,10):
    actions = [0, r]
    exp_name = 'results/'+'_f'+str(f)+'_r'+str(r)+'_a'+str(alpha)+'_g'+str(gamma)+'_N100_T100_15000ep_lr01_expD'
    Q = QLearning_v_fast_cluster (actions, pop_size=1, num_episodes=15000, T=100, N=100, f=f, alpha=alpha, p_explore=1, 
                      learning_rate=1, discount=gamma, decay=[0.0005,0.0005], I=0, num_nets=50)
    visualizeQ_v(Q, actions, exp_name, N=100, T=100)



