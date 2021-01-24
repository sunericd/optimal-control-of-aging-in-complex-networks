#####################
# Copyright Eric Sun 2019
#####################
# Python script containing all functions for solving the differential equations for optimal control
# and for the dynamics of the model (i.e. vitality). Also contains Python implementation of the 
# analytical model. Also contains functions for quadratic repair.
#####################

import numpy as np
from scipy.integrate import solve_bvp, odeint
import matplotlib.pyplot as plt

# Quadratic repair
	
def diff_y (t, y, alpha, f, g):
    lambd, phi = y
    dydt = [1 + (f+g)*lambd + lambd**2*(phi-1)/(2*alpha), -lambd * (1-phi)**2 / (2*alpha) - (f+g)*phi]
    return (dydt)


def diff_y_tlast (y, t, alpha, f, g):
    lambd, phi = y
    dydt = [1 + (f+g)*lambd + lambd**2*(phi-1)/(2*alpha), -lambd * (1-phi)**2 / (2*alpha) - (f+g)*phi]
    return (dydt)

#def bc(y0, y1, alpha, f, g):
#    return [y0[1]-1, y1[0]]


def iterateAlpha (alpha_list, f, g, t_range, color_list=['r', 'g', 'b', 'm', 'k', 'c'], save='no'):
    
    t = np.linspace(t_range[0], t_range[1], t_range[2])
    ystart = np.zeros((2, t.size))
    
    r_t_list = []
    
    
    
    for i, alpha in enumerate(alpha_list):
        result = solve_bvp(lambda t, y: diff_y(t, y, alpha=alpha, f=f, g=g),
                           lambda y0, y1: bc(y0, y1, alpha=alpha, f=f, g=g),
                           t, ystart)

        lambd_t = np.array(result.y[0])
        phi_t = np.array(result.y[1])
        r_t = np.divide(np.multiply(lambd_t, np.subtract(phi_t, 1)),(2*alpha))
        r_t_list.append(r_t)
        plt.plot(t, r_t, label='alpha = ' + str(alpha), color=color_list[i])
        
        if save != 'no':
            np.savetxt('Data/Quadratic/'+save+'_'+str(alpha)+'.csv', r_t, delimiter=",")
        
    #plt.legend(loc='best')
    plt.xlabel('Time, $t$', fontsize=18)
    plt.ylabel('Repair rate, $r(t)$', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.xlim(0,101)
    if save != 'no':
        plt.savefig('Figures/Quadratic/'+save+'.png', dpi=800)
    plt.show()
    
def iterateGamma (g_list, f, alpha, t_range, color_list=['r', 'g', 'b', 'm', 'k', 'c'], save='no'):
    
    t = np.linspace(t_range[0], t_range[1], t_range[2])
    ystart = np.zeros((2, t.size))
    
    r_t_list = []
    
    
    
    for i, g in enumerate(g_list):
        result = solve_bvp(lambda t, y: diff_y(t, y, alpha=alpha, f=f, g=g),
                           lambda y0, y1: bc(y0, y1, alpha=alpha, f=f, g=g),
                           t, ystart)

        lambd_t = np.array(result.y[0])
        phi_t = np.array(result.y[1])
        r_t = np.divide(np.multiply(lambd_t, np.subtract(phi_t, 1)),(2*alpha))
        r_t_list.append(r_t)
        plt.plot(t, r_t, label='gamma = ' + str(g), color=color_list[i])
        
        if save != 'no':
            np.savetxt('Data/Quadratic/'+save+'_'+str(g)+'_gamma.csv', r_t, delimiter=",")
        
    #plt.legend(loc='best')
    plt.xlabel('Time, $t$', fontsize=18)
    plt.ylabel('Repair rate, $r(t)$', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.xlim(0,101)
    plt.legend(loc='best')
    if save != 'no':
        plt.savefig('Figures/Quadratic/'+save+'_gamma.png', dpi=800)
    plt.show()



# # Nonlinear Model with $h(I,\Phi)$ and $m(I,\Phi)$

from scipy.integrate import solve_bvp, odeint
from scipy.special import binom
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad, elementwise_grad
import math
from model import *

def dphi_dt (t, phi, I, f, r, alpha, n):
    '''
    dphi/dt differential equation with full nonlinear terms (i.e. h(I,phi) and m(I,phi))
    Can be solved using standard scipy.integrate.odeint to obtain: phi(t) and lambda(t)
        which in turn can be used to obtain the optimal repair protocol: r(t)
    Parameters include I (interdependency), f (failure rate), r (repair rate), 
        alpha (cost of repair), n=N*p (mean # of neighbors)
    '''
    # Extract lambda and phi values
    # Get f_eff, h, df/dphi, dh/dphi
    feff = get_feff(I,phi,f,n)
    h = get_h(I,phi,n)
    # Get r0
    ro = np.ones(len(phi))*r
    output = -feff*phi+ro*h*(1-phi)
    return (output)

#def get_m(I,phi,n):
#    m = binom(n,I*n) * phi**(I*n) * (1-phi)**(n-(I*n))
#    return (m)

#def get_h(I,phi,n):
#    h_minus = np.zeros(len(phi))
#    for i in range(round(I*n)):
#        #print (binom(n,i) * phi**i * (1-phi)**(n-i))
#        h_minus += binom(n,i) * phi**i * (1-phi)**(n-i)
#    h = 1-h_minus
#    return (h)

#def get_feff(I,phi,f,n):
#    m = get_m(I,phi,n)#/phi
#    #feff = f/(1-(I*n)*m*(1-f)) # first-order
#    k = I*n
#    feff = 2*f/(1+(f-1)*k*m+(1+(f-1)*k*m*(2+2*f-2*f*k+(f-1)*k*m))**(1/2))# second-order
#    return (feff)




from scipy.integrate import solve_bvp, odeint
from scipy.special import binom
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad, elementwise_grad
import math
from model import *


def lambda_phi_system_odes (t, X, I, f, r, alpha, n):
    '''
    Coupled system of differential equations corresponding to dlambda/dt and dphi/dt
    Can be solved using standard scipy.integrate.solve_bvp to obtain: phi(t) and lambda(t)
        which in turn can be used to obtain the optimal repair protocol: r(t)
    Parameters include I (interdependency), f (failure rate), r (repair rate), 
        alpha (cost of repair), n=N*p (mean # of neighbors)
    '''
    # Extract lambda and phi values
    lamb, phi = X
    # Get f_eff, h, df/dphi, dh/dphi
    feff = get_feff(I,phi,f,n)
    h = get_h(I,phi,n)
    dfdphi = get_dfdphi(I,phi,f,n)
    dhdphi = get_dhdphi(I,phi,n)
    # Get r0 (linear)
    ro = np.zeros(len(phi))
    
    # clean up phi after collapse
    #for i, p in enumerate(phi):
    #    if p < 0:
    #        phi[i:] = np.zeros(len(phi[i:]))
    
    for idx in range(len(phi)):
        #print (lam*h[idx]*(1-phi[idx]))
        if lamb[idx]*h[idx]*(phi[idx]-1) >= alpha:
            ro[idx] = r
        else:
            ro[idx] = 0
    '''
    # Get ro (higher-order)
    beta = 1.2
    ro = ((phi-1)*lamb*h/(beta*alpha))**(1/(beta-1))
    '''
    
    output = [1+(feff+ro*h)*lamb+lamb*np.array(dfdphi)*phi-ro*lamb*np.array(dhdphi)*(1-phi),
                     -feff*phi+ro*h*(1-phi)]
    return (output)

def bc(x0, x1, I, f, r, alpha, n):
    return [x0[1]-1, x1[0]]

'''
def dlambda_dt(lamb,t,I,alpha,f,r):
    feff = get_feff(I,phi,f)
    h = get_h(I,phi)
    dfdphi = get_dfdphi(I,phi,f)
    dhdphi = get_dhdphi(I,phi)
    return (1+(feff+ro*h)*lamb+lamb*dfdphi*phi-ro*lamb*dhdphi*(1-phi))

def dphi_dt(phi,t,I,alpha,f,r):
    feff = get_feff(I,phi,f)
    h = get_h(I,phi)
    return(-feff*phi+ro*h*(1-phi))
    
def get_r0(t,I,alpha,f,r):
    lamb = dlambda_dt # solve for lambda(t)
    phi = dphi_dt # solve for phi
    h = get_h(I,phi)
    if lamb*h*(1-phi) > alpha:
        ro = r
    else:
        ro = 0
    return (ro)
'''

def get_feff(I,phi,f,n):
    m = get_m(I,phi,n)#/phi
    feff = f/(1-(I*n)*m*(1-f)) ## ORIGINAL: 11/16/2019
    #feff = f+(I*n)*(1-f)*f*m
    return (feff)

get_dfdphi = elementwise_grad(get_feff,1)

def get_m(I,phi,n):
    m = binom(n,I*n) * phi**(I*n) * (1-phi)**(n-(I*n))
    return (m)

def get_h(I,phi,n):
    h_minus = np.zeros(len(phi))
    for i in range(round(I*n)):
        h_minus += binom(n,i) * phi**i * (1-phi)**(n-i)
    h = 1-h_minus
    return (h)



def iterateI (I_list, t_range, f, r, alpha, n, color_list=['r','g','b','m','k','c','y'], save='no', plot='yes'):
    '''
    Solver for the coupled system of ODEs of the full nonlinear model to obtain the optimal repair protocol
    ''' 
    t = np.linspace(t_range[0], t_range[1], t_range[2])
    #line_down = np.ones(t.size) - t/np.max(t)
    ystart = np.vstack((np.zeros(t.size), np.ones(t.size)))#np.ones((2, t.size))*0.5
    #xstart[0] = line_down
    #xstart[1] = line_down
    
    r_t_lists = []
    
    for i, I in enumerate(I_list):
        result = solve_bvp(lambda t, y: lambda_phi_system_odes(t, y, I=I, f=f, r=r, alpha=alpha, n=n),
                           lambda y0, y1: bc(y0, y1, I=I, f=f, r=r, alpha=alpha, n=n),
                           t, ystart, tol=1e4)
        lambd_t = np.array(result.y[0])
        phi_t = np.array(result.y[1])        
        h_t = get_h(I,phi_t,n)
        # Linear
        r_t_list = []
        for idx, lambd in enumerate(lambd_t):
            if -lambd_t[idx]*h_t[idx]*(1-phi_t[idx]) >= alpha:
                ro = r
            else:
                ro = 0
            r_t_list.append(ro)
        '''
        # Nonlinear
        beta = 1.2
        r_t_list = ((phi_t-1)*lambd_t*h_t/(beta*alpha))**(1/(beta-1))
        '''
        if plot == 'yes':
            plt.plot(result.x, r_t_list, label='I = ' + str(I), color=color_list[i])
            #plt.plot(result.x, lambd_t)
            #plt.plot(result.x, phi_t)
        
        if save != 'no':
            np.savetxt('Data/NonlinearModel/'+save+'_I'+str(I)+'.csv', r_t_list, delimiter=",")
            
        # Make previous soln the guess for the next one
        current_time = result.x[0]
        redundant_indices = []
        for idx, time_val in enumerate(result.x):
            if time_val > current_time:
                current_time = time_val
            else:
                redundant_indices.append(idx)
        
        ystart = np.vstack((np.delete(lambd_t,redundant_indices),np.delete(phi_t,redundant_indices)))
        t = np.delete(result.x, redundant_indices)
        r_t_lists.append(np.delete(np.array(r_t_list),redundant_indices))
        
    if plot == 'yes':
        plt.legend(loc='best')
        plt.xlabel('Time, $t$', fontsize=18)
        plt.ylabel('Repair rate, $r(t)$', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(-0.0005,r+0.0005)
        #plt.tight_layout()
        if save != 'no':
            plt.savefig('Figures/NonlinearModel/'+save+'.png', dpi=800)
        plt.show()
    return (r_t_lists)