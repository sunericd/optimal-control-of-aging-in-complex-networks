#####################
# Copyright Eric Sun 2019
#####################
# Python script containing all functions for solving the linear optimal control problem and generating
# corresponding plots. Also contains some miscellaneous functions for further plotting of results.
#####################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import math
from scipy.optimize import fsolve


def plotHD (savename_list, list_of_results_files, plot_details_list, dpi=500):

    for n, filename in enumerate(list_of_results_files):
        
        if 'T1_T2' in plot_details_list[n]:
            
            input_file_path = './Data/' + filename + '.csv'
            with open(input_file_path, 'rt') as tsvin:
                tsvin = csv.reader(tsvin, delimiter=',')
                row_list = list(tsvin)
                T1_list = [float(i) for i in row_list[1]]
                T2_list = [float(i) for i in row_list[2]]
        
            # TO FIX BUG
            for i, t1 in enumerate(T1_list):
                if t1 < 0:
                    T1_list[i] = 0
                if T2_list[i] > 100:
                    T2_list[i] = 100

            parameter_list = np.arange(plot_details_list[n][1], plot_details_list[n][2], plot_details_list[n][3])
            parameter_type = plot_details_list[n][4]

            plt.figure()
            plt.plot(parameter_list, T1_list, 'g--', label='T1')
            plt.plot(parameter_list, T2_list, 'r--', label='T2')
            plt.title('Bang-Bang Optimal Repair ('+parameter_type+')')
            plt.xlabel(parameter_type)
            plt.ylabel('T')
            plt.legend(loc='upper right', fontsize=12)
            plt.savefig('Figures/'+savename_list[n]+'.png', dpi=dpi)
            plt.show()


def plotHD_wCurve (savename_list, list_of_results_files, plot_details_list, curve_type='analytic', dpi=500, flip=False,
                  fill=False):
    '''
    High-def plots of computational results with curves:
        savename_list = list of paths/filenames to save figures to
        list_of_results_files = paths to results files (from runSim)
        plot_details_list = ['T1_T2', int(min of param range), int(max of param range), int(delta param), param_type]
            param_type = 'r', 'f', r'$\alpha$', etc (x-axis label)
        curve_type = 'analytic' (just analytical approx.), 'numeric' (numerical soln), 'both' (both plotted)
        dpi = dots per inch for figures
    '''
    # default param values
    f = 0.025
    r = 0.01
    alpha = 10
    d = 0
    T = 100
    
    for n, filename in enumerate(list_of_results_files):
        
        if 'T1_T2' in plot_details_list[n]:
            
            input_file_path = './Data/' + filename + '.csv'
            with open(input_file_path, 'rt') as tsvin:
                tsvin = csv.reader(tsvin, delimiter=',')
                row_list = list(tsvin)
                T1_list = [float(i) for i in row_list[1]]
                T2_list = [float(i) for i in row_list[2]]
        
            # TO FIX BUG
            for i, t1 in enumerate(T1_list):
                if t1 < 0:
                    T1_list[i] = 0
                if T2_list[i] > 100:
                    T2_list[i] = 100

            parameter_list = np.arange(plot_details_list[n][1], plot_details_list[n][2], plot_details_list[n][3])
            parameter_type = plot_details_list[n][4]
            
            if flip is False:
                plt.figure(figsize=(6,3))
            else:
                plt.figure(figsize=(6,4))
            
            # Plot computational
            if flip is False:
                plt.scatter(parameter_list, T1_list, color='k', alpha=0.2, s=20)
                plt.scatter(parameter_list, T2_list, color='k', alpha=0.4, s=20)
            # Plot analytical
            if curve_type is 'analytic':
                a_parameter_list, a_T1_list, a_T2_list = getAnalytic (parameter_list, parameter_type, f, r, alpha, d, T)
                plt.plot(a_parameter_list, a_T1_list, 'g', alpha=0.5, linewidth=2, label='$T_1$')
                plt.plot(a_parameter_list, a_T2_list, 'r', alpha=0.5, linewidth=2, label='$T_2$')
            elif curve_type is 'numeric':
                n_parameter_list, n_T1_list, n_T2_list = getNumeric (parameter_list, parameter_type, f, r, alpha, d, T)
                if flip is True:
                    plt.plot(n_T1_list, n_parameter_list, color='#000080', alpha=0.5, linewidth=2, label='$T_1$')
                    plt.plot(n_T2_list, n_parameter_list, color='#000080', alpha=1.0, linewidth=2, label='$T_2$')
                if fill is True:
                    if parameter_type == '$f$': # to not shade for numerical issues at start
                        plt.fill_betweenx(n_parameter_list[60:], n_T1_list[60:], n_T2_list[60:], color='#000080', alpha=0.05)
                    else:
                        plt.fill_betweenx(n_parameter_list, n_T1_list, n_T2_list, color='#000080', alpha=0.05)
                else:
                    plt.plot(n_parameter_list, n_T1_list, 'm', alpha=0.5, linewidth=2, label='$T_1$')
                    plt.plot(n_parameter_list, n_T2_list, 'm', alpha=1.0, linewidth=2, label='$T_2$')
            else: # BOTH
                a_parameter_list, a_T1_list, a_T2_list = getAnalytic (parameter_list, parameter_type, f, r, alpha, d, T)
                plt.plot(a_parameter_list, a_T1_list, 'g--', alpha=0.5, linewidth=2)
                plt.plot(a_parameter_list, a_T2_list, 'r--', alpha=0.5, linewidth=2)
                n_parameter_list, n_T1_list, n_T2_list = getNumeric (parameter_list, parameter_type, f, r, alpha, d, T)
                plt.plot(n_parameter_list, n_T1_list, 'g', alpha=0.5, linewidth=2, label='$T_1$')
                plt.plot(n_parameter_list, n_T2_list, 'r', alpha=0.5, linewidth=2, label='$T_2$')
                
            if flip is True:
                plt.xlim(-1,101)
                plt.ylim(0,max(parameter_list))
                if parameter_type == '$r$':
                    plt.ylabel('Repair rate, ' + parameter_type, fontsize=14)
                elif parameter_type == '$f$':
                    plt.ylabel('Failure rate, ' + parameter_type, fontsize=14)
                else:
                    plt.ylabel('Cost of repair, ' + parameter_type, fontsize=14)
                plt.xlabel('Time, $t$', fontsize=14)
            else:
                plt.ylim(-1,101)
                plt.xlim(0,max(parameter_list))
                if parameter_type == '$r$':
                    plt.xlabel('Repair rate, ' + parameter_type, fontsize=14)
                elif parameter_type == '$f$':
                    plt.xlabel('Failure rate, ' + parameter_type, fontsize=14)
                else:
                    plt.xlabel('Cost of repair, ' + parameter_type, fontsize=14)
                plt.ylabel('Time, $t$', fontsize=14)
                
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tick_params(axis='both', which='minor', labelsize=12)
            
            if parameter_type == r'$\alpha$':
                from matplotlib.ticker import StrMethodFormatter
                plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 2 decimal places
            
            plt.legend(loc='upper right', fontsize=12)
            plt.tight_layout()
            plt.savefig('Figures/'+savename_list[n]+'.png', dpi=dpi)
            plt.show()
           
        
def getAnalytic (parameter_list, parameter_type, f, r, alpha, d, T):
    '''
    Gets analytical solutions for N points specified in the parameter_list range
    '''
    N = 1000 # points to use
    new_param_list = np.linspace(min(parameter_list), max(parameter_list), N)
    T1_list = []
    T2_list = []
        
    for p in new_param_list:
        if parameter_type == '$r$':
            r = p
        elif parameter_type == '$f$':
            f = p
        elif parameter_type == r'$\alpha$':
            alpha = p
        try:
            T1 = 1/f * math.log(1/(1-alpha*(f+r)))
            T2 = T - T1
        except:
            T1 = float('NaN')
            T2 = float('NaN')
        T1_list.append(T1)
        T2_list.append(T2)
        
    return (new_param_list, T1_list, T2_list)


def getNumeric (parameter_list, parameter_type, f, r, alpha, d, T):
    '''
    Gets analytical solutions for N points specified in the parameter_list range
    '''
    N = 1000 # points to use
    new_param_list = np.linspace(min(parameter_list), max(parameter_list), N)
    T1_list = []
    T2_list = []
        
    for p in new_param_list:
        if parameter_type == '$r$':
            r = p
        elif parameter_type == '$f$':
            f = p
        elif parameter_type == r'$\alpha$':
            alpha = p
        T1, T2 = fsolve(numerical_function, [20,80], args=(f,r,alpha,T))
        T1_list.append(T1)
        T2_list.append(T2)
        
    return (new_param_list, T1_list, T2_list)

def numerical_function (t, f, r, alpha, T):
    # system of equations to solve for T1 and T2
    [t1, t2] = t
    f1 = (np.exp(-f*t1)-1) * ((r+f)*np.exp(f*(t2-T))*np.exp((r+f)*(t1-t2)) - r*np.exp((r+f)*(t1-t2))-f) - alpha*f*(r+f)
    f2 = ((r+f)*np.exp(-f*t1)*np.exp(-(r+f)*(t2-t1))-r*np.exp(-(r+f)*(t2-t1))-f) * (np.exp(f*(t2-T))-1) - alpha*f*(r+f)
    return ([f1, f2])



def plotNonlin (savename, list_of_results_files, list_labels, plot_details_list, range_2, dpi=500, nofit=True):
    
    from numpy.polynomial import polynomial as P
    
    colors = ['k', 'b', 'c', 'g', 'r', 'm']
    corr_vals = ['$I=0$', '$I=0.05$', '$I=0.1$', '$I=0.2$', '$I=0.3$', '$I=0.4$']
    
    parameter_list = np.arange(plot_details_list[1], plot_details_list[2], plot_details_list[3])
    parameter_type = plot_details_list[4]
    
    for n, filename in enumerate(list_of_results_files):
                    
        input_file_path = './Nonlinear/' + filename + '.csv'
        with open(input_file_path, 'rt') as tsvin:
            tsvin = csv.reader(tsvin, delimiter=',')
            row_list = list(tsvin)
            T1_list = [float(i) for i in row_list[1]]
            T2_list = [float(i) for i in row_list[2]]

        # TO FIX BUG
        for i, t1 in enumerate(T1_list):
            if t1 < 0:
                T1_list[i] = 0
            if T2_list[i] > 100:
                T2_list[i] = 100
                
        if len(T1_list) != len(parameter_list): # try second range
            parameter_list = np.arange(range_2[0], range_2[1], range_2[2])

        coefs, stats = P.polyfit(parameter_list,np.array(T1_list),4,full=True)
        fitted_T1 = [coefs[0]+coefs[1]*x+coefs[2]*x**2+coefs[3]*x**3+coefs[4]*x**4 for x in parameter_list]
        coefs, stats = P.polyfit(parameter_list,np.array(T2_list),4,full=True)
        fitted_T2 = [coefs[0]+coefs[1]*x+coefs[2]*x**2+coefs[3]*x**3+coefs[4]*x**4 for x in parameter_list]

        if nofit is True:
            m = corr_vals.index(list_labels[n])
            plt.scatter(parameter_list, T1_list, color=colors[m], alpha=0.5, s=10, label=list_labels[n] + ' $T_1$')
            plt.scatter(parameter_list, T2_list, color=colors[m], s=10, label=list_labels[n] + ' $T_2$')
            plt.plot(parameter_list, T1_list, colors[m], alpha=0.5, linewidth=0.5)
            plt.plot(parameter_list, T2_list, colors[m], linewidth=0.5)
        else:
            plt.scatter(parameter_list, T1_list, color=colors[n], alpha=0.5, s=10, label=list_labels[n] + ' $T_1$')
            plt.scatter(parameter_list, T2_list, color=colors[n], s=10, label=list_labels[n] + ' $T_2$')

            plt.plot(parameter_list, fitted_T1, color=colors[n], alpha=0.5)
            plt.plot(parameter_list, fitted_T2, color=colors[n])
            
    plt.xlabel(parameter_type, fontsize=14)
    plt.ylabel('Time, $t$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(min(parameter_list), max(parameter_list))
    plt.ylim(-1,101)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig('Figures/'+savename+'.png', dpi=dpi, bbox_inches='tight')
    plt.show()



def plotTogether(list_of_results_files, legend_list, scale_type=['none'], 
                 color_list=['k', 'c', 'orange', 'olive', 'purple', 'brown'], i_thick=-1, cost_curve='no',
                 save='no'):
    
    vitality_list = []
    costs_list = [] 
    interdependence_list = []
    s_list = [] 
    mortality_list = []
    failuretime_list = []
    totcost_list = []
    
    for filename in list_of_results_files:
        input_file_path = './Data/' + filename + '.csv'
        with open(input_file_path, 'rt') as tsvin:
            tsvin = csv.reader(tsvin, delimiter=',')
            row_list = list(tsvin)
            vitality_list.append([float(i) for i in row_list[0]])
            costs_list.append([float(i) for i in row_list[1]])
            interdependence_list.append([float(i) for i in row_list[2]])
            if '_MEAN' in filename:
                s_list.append([float(i) for i in row_list[3]])
                mortality_list.append([float(i) for i in row_list[4]])
                failuretime_list.append([float(i) for i in row_list[5]][0])
                totcost_list.append([float(i) for i in row_list[6]][0])
            else:
                failuretime_list.append([float(i) for i in row_list[3]][0])
    
    if vitality_list:
        plt.figure()
        for i, vitality in enumerate(vitality_list):
            if i != i_thick:
                time, xtitle = getTime(vitality, scale_type, i)
                plt.plot(time, vitality, label=legend_list[i], color=color_list[i])
            else:
                time, xtitle = getTime(vitality, scale_type, i)
                plt.plot(time, vitality, label=legend_list[i], color=color_list[i], linewidth=2.5)
        plt.ylabel('Vitality, $\phi$', fontsize=18)
        plt.xlabel('Time, $t$', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='upper right', fontsize=12)
        if save != 'no':
            plt.savefig('./Figures/Together_'+save+'.png', dpi=800)
        plt.show()
    
    if costs_list:
        plt.figure()
        plt.title('Cost')
        for i, costs in enumerate(costs_list):
            time, xtitle = getTime(costs, scale_type, i)
            plt.plot(time, costs, label=legend_list[i])
        plt.ylabel('costs')
        plt.xlabel(xtitle)
        plt.legend(loc='upper left')
        plt.show()
        
    if interdependence_list:
        plt.figure()
        plt.title('Interdependence')
        for i, interdependence in enumerate(interdependence_list):
            time, xtitle = getTime(interdependence, scale_type, i)
            plt.plot(time, interdependence, label=legend_list[i])
        plt.ylabel('interdependence')
        plt.xlabel(xtitle)
        plt.legend(loc='upper center')
        plt.show()
        
    if s_list:
        plt.figure()
        plt.title('Survivorship')
        for i, s in enumerate(s_list):
            time, xtitle = getTime(s, scale_type, i)
            plt.plot(time, s, label=legend_list[i])
        plt.ylabel('Fraction Alive (s)')
        plt.xlabel(xtitle)
        plt.legend(loc='upper right')
        plt.show()
        
    if mortality_list:
        plt.figure()
        plt.title('Mortality')
        for i, mortality in enumerate(mortality_list):
            time, xtitle = getTime(mortality, scale_type, i)
            plt.plot(time, mortality, label=legend_list[i])
        plt.ylabel('mortality')
        plt.xlabel(xtitle)
        plt.legend(loc='upper right')
        plt.show()
            
    if failuretime_list:
        plt.figure()
        plt.title('Failure Times')
        plt.ylabel('failure time')
        plt.xlabel('model')
        plt.bar(range(len(failuretime_list)), failuretime_list, color=[0.5,0,0], align='center')
        plt.xticks(range(len(failuretime_list)), legend_list, size='small')
        plt.show()
        
    if totcost_list:
        if cost_curve == 'no':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.title('Final Cost')
            plt.ylabel('cost')
            plt.xlabel('repair start')
            plt.scatter(range(len(totcost_list)), totcost_list, marker='o')
            for i,j in zip(range(len(totcost_list)), totcost_list):
                ax.annotate(str(round(j,2)),xy=(i+0.05,j))
            plt.xticks(range(len(totcost_list)), legend_list, size='small')
            plt.tight_layout()
            if save != 'no':
                plt.savefig('./Figures/Together_cost_'+save+'.png', dpi=800)
            plt.show()
        elif isinstance(cost_curve, list): # cost_curve is a x_range
            from scipy.interpolate import spline
            T = np.array(cost_curve)
            xnew = np.linspace(T.min(),T.max(),300) #300 represents number of points to make between T.min and T.max
            smooth_costs = spline(T,totcost_list,xnew)
            plt.plot(xnew,smooth_costs)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel('Cost, $c$', fontsize=18)
            plt.xlabel('Symmetric repair duration, $[t]$', fontsize=18)
            plt.tight_layout()
            if save != 'no':
                plt.savefig('./Figures/Together_cost_'+save+'.png', dpi=800)
            plt.show()
        else:
            from scipy.interpolate import spline
            T = np.arange(0, len(totcost_list), 1)
            xnew = np.linspace(T.min(),T.max(),300) #300 represents number of points to make between T.min and T.max
            smooth_costs = spline(T,totcost_list,xnew)
            plt.plot(xnew,smooth_costs)
            plt.xticks(range(len(totcost_list)), legend_list)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel('Cost, $c$', fontsize=18)
            plt.xlabel('Symmetric repair duration, $[t]$', fontsize=18)
            plt.tight_layout()
            if save != 'no':
                plt.savefig('./Figures/Together_cost_'+save+'.png', dpi=800)
            plt.show()



def plotTogether3D(list_of_results_files, xtitle, x, ytitle, y):
    
    vitality_list = []
    costs_list = [] 
    interdependence_list = []
    s_list = [] 
    mortality_list = []
    failuretime_list = []
    totcost_list = []
    
    for filename in list_of_results_files:
        input_file_path = './Data/' + filename + '.csv'
        with open(input_file_path, 'rt') as tsvin:
            tsvin = csv.reader(tsvin, delimiter=',')
            row_list = list(tsvin)
            vitality_list.append([float(i) for i in row_list[0]])
            costs_list.append([float(i) for i in row_list[1]])
            interdependence_list.append([float(i) for i in row_list[2]])
            if '_MEAN' in filename:
                s_list.append([float(i) for i in row_list[3]])
                mortality_list.append([float(i) for i in row_list[4]])
                failuretime_list.append([float(i) for i in row_list[5]][0])
                totcost_list.append([float(i) for i in row_list[6]][0])
            else:
                failuretime_list.append([float(i) for i in row_list[3]][0])
        
    if totcost_list:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, y)
        #ax.plot_surface(X, Y, totcost_list)
        ax.plot_trisurf(x, y, totcost_list)
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.set_zlabel('Cost')
        plt.show()
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.scatter(x, y, totcost_list)
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.set_zlabel('Cost')
        plt.show()
    
        min_val, min_idx = min((val, idx) for (idx, val) in enumerate(totcost_list))
        print ("Minimum cost: " + str(min_val) +' @ ' + xtitle + ' = ' + str(x[min_idx]) + ', ' + ytitle + ' = ' + str(y[min_idx]))



def getTime(values, scale_type, i):
    
    if 'none' in scale_type:
        time = np.arange(1,len(values)+1)
        xtitle = "Time"
    
    elif 'f+r_rescaled' in scale_type:
        f = float(scale_type[1+2*i])
        r = float(scale_type[2+2*i])
        time = np.arange(1,len(values)+1)
        time = np.multiply(time, f+r)
        xtitle = "(f+r) * Time"
        
    return (time, xtitle)


def plotParams (filename_list, legend_list, title='', xtitle='', ytitle='', color_list=['r', 'g', 'b', 'm', 'k', 'c'],
                i_thick=-1, save='no', xlim='none', ylim='none'):
    
    for i, filename in enumerate(filename_list):
        input_file_path = './Data/' + filename + '.csv'
        with open(input_file_path, 'rt') as tsvin:
            tsvin = csv.reader(tsvin, delimiter=',')
            row_list = list(tsvin)
            x_list = [float(i) for i in row_list[0]]
            y_list = [float(i) for i in row_list[1]]
        if i != i_thick:
            plt.plot(x_list, y_list, label=legend_list[i], color=color_list[i])
        else: # plot line thicker
            plt.plot(x_list, y_list, label=legend_list[i], color=color_list[i], linewidth=3.0)
    
    plt.xlabel(xtitle, fontsize=18)
    plt.ylabel(ytitle, fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12)
    if xlim != 'none':
        plt.xlim((xlim[0], xlim[1]))
    if ylim != 'none':
        plt.ylim((ylim[0], ylim[1]))
    if save != 'no':
        plt.savefig('./Figures/'+save+'.png', dpi=800)
    plt.show()