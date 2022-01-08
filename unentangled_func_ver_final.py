#!/usr/bin/env python
# coding: utf-8

# In[ ]:

""" Final version of the unentangled system before entanglements were introduced. 
    Results were used as a control wrt the entangled system results for comparison. """


""" INPUT PARAMETERS """

# Import Packages
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import scipy.stats
import cProfile
import math

# functions for fitting
def func_linear(x,m,c):
    return m*x + c
    


# In[ ]:


""" RAW DATA FUNCTION """

def make_moves(nbeads):
    """
    Returns x,y,z position values for a generated chain
    
    Inputs: nbeads (int):       Desired number of beads on chain
            boundary_r (float): Limits of boundary (Cube) 
    """
    boundary_r = 10
    new_origin = np.random.uniform(-boundary_r, boundary_r, (3,))
    r_vector = np.zeros((nbeads,3), dtype = float)
    r_vector[0] = new_origin 
    
    
    for i in range(1,nbeads):
        while True: # until True
            cubepoint = np.random.uniform(-1,1, (3,)) # generate a movement
            if np.dot(cubepoint,cubepoint) <= 1: # magnitude less than 1
                 break
        new_array = np.add(new_origin, cubepoint) # apply movement
        r_vector[i] = new_array # record position
        new_origin = new_array # new position is origin
        
    return r_vector


# In[ ]:


""" OPERATOR FUNCTION """ 


def chain_length(bead1, bead2, pop_array):
    """
    Returns total chain length between two beads
    Input: bead1 (int):          first bead
           bead2 (int):          second bead
           pop_array (float arr):array of position values
         
    """
    length_sum = 0
    cond = bead2 - bead1
    for i in range(cond):
        if bead1+i > cond:
            break
        else:
            length_sum += np.linalg.norm(pop_array[bead1+i,:] - pop_array[bead1-1+i,:])
    return length_sum


# In[ ]:


""" OPERATOR FUNCTION """

def mean_progress(array):
    """
    Input: An array of select bead movements for a chosen bead on a time evolved chain 
    Ouput: A (max_time-1) length array of mean squared displacement values
    
    """
    prog_list = []
    
    for t in range(1, len(array)): # iterate over all possible time intervals up to m_time-1
        sum_count = 0
        combo_count = 0
        for a in range(len(array)-t):    # iterate over all possible a values 0,1,2,3..()
            combo_count += 1
            sum_count += array[a+t] - array[a]
        
        prog_list.append(sum_count/combo_count) # return the mean squared displacement for every time unit

    return np.array(prog_list)


# In[ ]:


""" OPERATOR FUNCTION """

def mean_squared_progress(array):
    """
    Input: An array of select bead movements for a chosen bead on a time evolved chain 
    
    Ouput: A (len(array)-1) length array of mean squared displacement values
    
    """
    prog_list = []
    
    for t in range(1, len(array)): # iterate over all possible time intervals up to m_time-1
        sum_count = 0
        combo_count = 0
        for a in range(len(array)-t):    # iterate over all possible a values 0,1,2,3..()
            combo_count += 1
            displacement = array[a+t] - array[a]
            sum_count += np.dot(displacement, displacement)
        
        prog_list.append(sum_count/combo_count) # return the mean squared displacement for every time unit

    return np.array(prog_list)


# In[ ]:


""" RAW DATA FUNCTION """  

def make_chains(num_of_polymers, nbeads):
    """
    Makes a 3D matrix containing 'num_of_chains' number of chains
    
    Input: num_of_polymers (int): Number of chains to be in system
           nbeads (int)         : Beads on each generated chain
    
    """
    
    chain_array = np.zeros((num_of_polymers, nbeads, 3), dtype = float)
    
    for i in range(num_of_polymers): # num_of_chains is the 3D array depth 
        x = make_moves(nbeads)
        chain_array[i,:,:] = make_moves(nbeads)
        
    return chain_array


# In[ ]:


""" BATCH PROCESSING FUNCTION & GRAPHING AND STATISTICAL ANALYSIS FUNCTION """

def gradient_func(num_of_batches, batch_size, nbeads):
    """
    Returns linear fitting parameters with error and plots
    a graph with error bars.
    
    Inputs: num_of_batches (int): number of batches
            batch_size:           runs ber batch
            nbeads:               number of chain beads
    """
    gradient_array = np.zeros((num_of_batches,2), dtype = float)
    intermediate_array = np.ones((nbeads-1, num_of_batches), dtype = float)
    
    for i in range(num_of_batches):
        batch_array = np.zeros((nbeads-1), dtype = float) 
        for j in range(batch_size): # make batch 
            batch_array += mean_squared_progress(make_moves(nbeads)) # repeat MSP  
        intermediate_array[:,i] = batch_array * 1/(batch_size)
        mini_result_array = intermediate_array[:,i] # normalise
        varebs, errorr = scipy.optimize.curve_fit(func_linear, np.arange(1, nbeads, 1), mini_result_array)
        gradient_array[i,:] = varebs # 'm' and 'c'
    
    # Prepare plots and parameters
    progress_matrix = np.mean(intermediate_array, axis=1)
    mean_of_means = np.mean(gradient_array, axis=0)
    se_array = scipy.stats.sem(gradient_array)
    sigma_matrix = ((np.std(intermediate_array, axis=1)).reshape(nbeads-1, 1))/((num_of_batches)**0.5)
    
    plt.style.use('ggplot')
    plt.errorbar(np.arange(1,nbeads,1), progress_matrix, yerr=sigma_matrix.flatten(), fmt='.k', ecolor = 'red')
    plt.title("Mean squared displacement against step number")
    plt.xlabel('n')
    plt.ylabel('$<|r(i+n) - r(i)|^2>$')
    return mean_of_means[0], se_array[0], mean_of_means[1], se_array[1] # m, merr, c, cerr
        

print(gradient_func(10,100,100))
 


# In[ ]:


""" OPERATOR FUNCTION """

def time_evolution(pop_array):
    """
    Time evolves a chain bead wise
    
    Input: pop_array (float array): Position values of chain to be evolved
    
    """
    input_array = pop_array
    dt = len(pop_array)
    max_time = dt**2
    for i in range(max_time):
        
        for j in range(dt): #dt =nbeads
            kth = random.randint(0,dt-1) # pick a random bead 
            
            while True: # make a vector that is less than 1 magnitude
                    cubepoint = np.random.uniform(-1,1, (3,))
                    if np.dot(cubepoint,cubepoint) <= 1:
                        break
                        
            new_position = input_array[kth, :] + cubepoint # prospective movement
            
            if (kth == 0 and np.dot(input_array[kth+1,:] - new_position, input_array[kth+1,:] - new_position) <= 1): # check if a starting chain end bead
                input_array[kth, :] = new_position # accept position
                
            elif (kth == dt-1 and np.dot(input_array[kth-1,:] - new_position, input_array[kth-1,:] - new_position) <= 1): # check if an final chain end bead
                input_array[kth, :] = new_position # accept position
                
            elif (np.dot(input_array[kth-1,:] - new_position, input_array[kth-1,:] - new_position) <= 1 and np.dot(input_array[kth+1,:] - new_position, input_array[kth+1,:] - new_position) <= 1): # check if chosen bead is an end bead of the chain
                input_array[kth, :] = new_position # accept position
            
            else:
                pass # reject the movement
            
                
    return input_array


def test():
    print(time_evolution(make_moves(100)))
    print("\n")
cProfile.run('test()', sort = "cumtime")


# In[ ]:


""" OPERATOR FUNCTION """
# Returns a population array of the position of a chosen bead over time, every time unit

def select_time_evolution(pop_array, chosen_bead):
    """
    Time evolves a chain bead wise recording movements of a specific bead at each time unit
    
    Input: pop_array (float array): Position values of chain to be evolved
           chosen_bead (int) : bead to track
    """
    input_array = pop_array
    dt = len(pop_array)
    max_time = 2000
    evolved_array = np.zeros((max_time, 3), dtype = float)
    for i in range(max_time):
        evolved_array[i,:] = input_array[chosen_bead-1, :]
        for j in range(dt):
            kth = random.randint(0,dt-1) # pick a random bead 
            
            while True: # make a vector that is less than 1 magnitude
                    cubepoint = np.random.uniform(-1,1, (3,))
                    if np.dot(cubepoint,cubepoint) <= 1:
                        break
                        
            new_position = input_array[kth, :] + cubepoint # prospective movement
            
            if (kth == 0 and np.dot(input_array[kth+1,:] - new_position, input_array[kth+1,:] - new_position) <= 1): # check if a starting chain end bead
                input_array[kth, :] = new_position # accept position
                
            elif (kth == dt-1 and np.dot(input_array[kth-1,:] - new_position, input_array[kth-1,:] - new_position) <= 1): # check if an final chain end bead
                input_array[kth, :] = new_position # accept position
                
            elif (np.dot(input_array[kth-1,:] - new_position, input_array[kth-1,:] - new_position) <= 1 and np.dot(input_array[kth+1,:] - new_position, input_array[kth+1,:] - new_position) <= 1): # check if chosen bead is an end bead of the chain
                input_array[kth, :] = new_position # accept position
            
            else:
                pass # reject the movement
            

    return evolved_array


# In[ ]:


""" BATCH PROCESSING FUNCTION """

def TE_gradient_func(num_of_batches, batch_size, nbeads):
    """
    Returns averaged slope and error of slope for num_of_sims repetitions of mean squared progress with up to n steps
    
    Inputs: num_of_batches (int): number of batches
            batch_size (int):     runs ber batch
            nbeads (int):         number of chain beads
    """
    gradient_array = np.zeros((num_of_batches,2), dtype = float)
    conds = np.array([0.60,0])
    xrange = np.arange(1, nbeads, 1)
    
    for i in range(num_of_batches):
        print("// Batch Number: " + str(i))
        batch_array = np.zeros((nbeads-1), dtype = float) # initialise empty array
        for j in range(batch_size): # make batch by repeating chain function
            print("Simulation Number: " + str(j))
            batch_array += mean_squared_progress(time_evolution(make_moves(nbeads)))   
        mini_result_array = (1/batch_size) * batch_array
        varebs, errorr = scipy.optimize.curve_fit(func_linear, xrange, mini_result_array, conds)
        gradient_array[i,:] = np.array([varebs[0], varebs[1]])
        
    mean_of_means = (1/num_of_batches) * gradient_array.sum(axis=0)
    se_array = scipy.stats.sem(gradient_array)
    my_tuple = mean_of_means[0], se_array[0], mean_of_means[1], se_array[1]
    return my_tuple
  

def test():
    print(TE_gradient_func(5, 20, 10))
cProfile.run('test()', sort = "cumtime")


# In[ ]:


""" BATCH PROCESSING FUNCTION & GRAPHING AND STATISTICAL ANALYSIS FUNCTION """

def select_TE_gradient_func(num_of_batches, batch_size, chosen_bead, nbeads):
    """
    Plots a logarithmic graph of mean squared progress of a select time evolution against time
    averaged over many polymer chains of length 'nbeads' for a chosen bead on the chain.
    Returns linear fitted parameters for each regime.
    
    !WARNING: THE MAX TIME MUST BE SUFFICIENTLY LARGE FOR 3RD REGIME TO APPEAR!
    
    Inputs: num_of_batches (int): number of batches
            batch_size (int): number of simulations per batch
            chosen_bead (int): selected bead to track
            nbeads (int):      number of chain beads
    """
    spec_max_time = (2000)-1 # Check that this is the same value in select_time_evolution
    t1_gradient_array = np.zeros((num_of_batches,2), dtype = float)
    tphys_gradient_array = np.zeros((num_of_batches,2), dtype = float)   
    trepeat_gradient_array = np.zeros((num_of_batches,2), dtype = float) 
    intermediate_array = np.ones((spec_max_time, num_of_batches), dtype = float)
    
    for i in range(num_of_batches):
        batch_array = np.zeros((spec_max_time), dtype = float) 
        for j in range(batch_size): # make batch 
            batch_array += mean_squared_progress(select_time_evolution(make_moves(nbeads), chosen_bead)) # repeat MSP  
        intermediate_array[:,i] = batch_array * 1/(batch_size) 
        mini_result_array = intermediate_array[:,i]
        
        # Define msp<0.6 regime
        t1_reg = mini_result_array[(mini_result_array<=0.6)]
        t1_reg_xvals = np.arange(1, t1_reg.size+1, 1)
        varebs1, errorr1 = scipy.optimize.curve_fit(func_linear, np.log(t1_reg_xvals), np.log(t1_reg))
        t1_gradient_array[i,:] = varebs1 # 'm' and 'c'
        
        # Define 0.6<msp<=0.6*N regime
        tphys_reg = mini_result_array[(mini_result_array >0.6) & (mini_result_array<=0.6*nbeads)]
        tphys_reg_xvals = np.arange(t1_reg.size+1, t1_reg.size+tphys_reg.size+1, 1)
        varebs2, errorr2 = scipy.optimize.curve_fit(func_linear, np.log(tphys_reg_xvals), np.log(tphys_reg))
        tphys_gradient_array[i,:] = varebs2 # 'm' and 'c'
        
        # Define msp>0.6*N regime
        trepeat_reg = mini_result_array[(mini_result_array>0.6*nbeads)]
        trepeat_reg_xvals = np.arange(t1_reg.size+tphys_reg.size+1, mini_result_array.size+1, 1)
        varebs3, errorr3 = scipy.optimize.curve_fit(func_linear, np.log(trepeat_reg_xvals), np.log(trepeat_reg))
        trepeat_gradient_array[i,:] = varebs3 # 'm' and 'c'
        
    # Prepare total msp matrix with error for both regimes   
    progress_matrix = np.mean(intermediate_array, axis=1)
    sigma_matrix = ((np.std(intermediate_array, axis=1)).reshape(spec_max_time, 1))/((num_of_batches)**0.5)
    mean_of_means = np.mean(t1_gradient_array, axis=0)
    se_array = scipy.stats.sem(t1_gradient_array)
    
    # Prepare msp<0.6 regime to plot
    plot1 = plt.figure(1)
    mean_of_means1 = np.mean(t1_gradient_array, axis=0)
    se_array1 = scipy.stats.sem(t1_gradient_array)
    my_tuple1 = mean_of_means1, se_array1
    maj_t1_reg = progress_matrix[(progress_matrix <= 0.6)]
    maj_t1_reg_xvals = np.arange(1, maj_t1_reg.size+1, 1)
    maj_t1_error = sigma_matrix[0: maj_t1_reg.size]/ (progress_matrix[0: maj_t1_reg.size])[:,None] # ln_err(x) = err(x)/x
    plt.errorbar(np.log(maj_t1_reg_xvals), np.log(maj_t1_reg), yerr=maj_t1_error.flatten(), fmt='.k', ecolor = 'red')
    plt.title("Mean squared displacement against time for regime 1")
    plt.xlabel('Log(t)')
    plt.ylabel('Log($<| r(t+a) - r(a) |^2>$)')
    plt.show()
    
    # Prepare msp>0.6 and <=0.6*nbeads regime to plot
    plot2 = plt.figure(2)
    mean_of_meansphys = np.mean(tphys_gradient_array, axis=0)
    se_arrayphys = scipy.stats.sem(tphys_gradient_array)
    my_tuplephys = mean_of_meansphys, se_arrayphys
    maj_tphys_reg = progress_matrix[(progress_matrix >0.6) & (progress_matrix<=0.6*nbeads)]
    maj_tphys_reg_xvals = np.arange(maj_t1_reg.size+1, maj_t1_reg.size+maj_tphys_reg.size+1, 1)
    maj_tphys_error = sigma_matrix[maj_t1_reg.size+1: maj_t1_reg.size+maj_tphys_reg.size+1]/ (progress_matrix[maj_t1_reg.size+1: maj_t1_reg.size+maj_tphys_reg.size+1])[:,None]
    plt.errorbar(np.log(maj_tphys_reg_xvals), np.log(maj_tphys_reg),yerr= maj_tphys_error.flatten(), fmt='.k', ecolor = 'red')
    plt.title("Mean squared displacement against time for regime 2")
    plt.xlabel('Log(t)')
    plt.ylabel('Log($<| r(t+a) - r(a) |^2>$)')
    plt.show()
    
    # Prepare msp>0.6*nbeads regime to plot
    plot2 = plt.figure(3)
    mean_of_meansrepeat = np.mean(trepeat_gradient_array, axis=0)
    se_arrayrepeat = scipy.stats.sem(trepeat_gradient_array)
    my_tuplerepeat = mean_of_meansrepeat, se_arrayrepeat
    maj_trepeat_reg = progress_matrix[(progress_matrix>0.6*nbeads)]
    maj_trepeat_reg_xvals = np.arange(maj_t1_reg.size+maj_tphys_reg.size+1, progress_matrix.size+1, 1)
    maj_trepeat_error = sigma_matrix[maj_t1_reg.size+maj_tphys_reg.size: progress_matrix.size+1]/ (progress_matrix[maj_t1_reg.size+maj_tphys_reg.size: progress_matrix.size+1])[:,None]
    plt.errorbar(np.log(maj_trepeat_reg_xvals), np.log(maj_trepeat_reg),yerr= maj_trepeat_error.flatten(), fmt='.k', ecolor = 'red')
    plt.title("Mean squared displacement against time for regime 3")
    plt.xlabel('Log(t)')
    plt.ylabel('Log($<| r(t+a) - r(a) |^2>$)')
    plt.show()
    
    return my_tuple1, my_tuplephys, my_tuplerepeat
    
select_TE_gradient_func(5, 20,5,10)


# In[ ]:


""" BATCH PROCESSING FUNCTION & GRAPHING AND STATISTICAL ANALYSIS FUNCTION """

def select_TE_gradient_func_variable_beads(num_of_batches, batch_size, min_beads, max_beads):
    """
    Plots a graph of mean squared progress of a select time evolution of the central bead
    batch averaged for a chain of variable length. Outputs these evolutions overlapped on
    a graph. 
    
    Inputs: num_of_batches (int): number of batches
            batch_size (int): number of simulations per batch
            min_beads (int): minimum chain length
            max_beads (int): maximum chain length
    
    """
    for k in range(min_beads, max_beads+1):
        print("//// CHAIN NUMBER \\\\: " + str(k))
        spec_max_time = (2000)-1 # Check that this is the same value in select_time_evolution   
        intermediate_array = np.ones((spec_max_time, num_of_batches), dtype = float)
    
        for i in range(num_of_batches):
            print(" - Batch - : " + str(i))
            batch_array = np.zeros((spec_max_time), dtype = float) 
            for j in range(batch_size): # make batch 
                print("Simulation " + str(j))
                cent_bead = math.ceil(k/2) # Choose the central bead (approximately)
                batch_array += mean_squared_progress(select_time_evolution(make_moves(k), cent_bead)) # repeat MSP  
            intermediate_array[:,i] = batch_array * 1/(batch_size) 
        
          
        progress_matrix = np.mean(intermediate_array, axis=1)
        xvalus = np.arange(1, progress_matrix.shape[0]+1, 1)
        
        plt.figure(0)
        plt.plot(xvalus , progress_matrix)
        plt.title("Mean squared displacement against time", fontdict = {'fontsize' : 11})
        plt.xlabel('$t$')
        plt.ylabel('$<| r(t+a) - r(a) |^2>$')
        
        plt.figure(1)
        plt.plot(np.log(xvalus) , np.log(progress_matrix))
        plt.title("Logarithmic plot of mean squared displacement against time", fontdict = {'fontsize' : 11})
        plt.xlabel('$ln(t)$')
        plt.ylabel('$ln(<| r(t+a) - r(a) |^2>)$')
        
        plt.figure(2)
        plt.plot(np.log(xvalus/k**2) , np.log(progress_matrix/k))
        plt.title("Scaled logarithmic plot of mean squared displacement against time", fontdict = {'fontsize' : 11})
        plt.xlabel('$ln(t/N^2)$')
        plt.ylabel('$ln(<| r(t+a) - r(a) |^2>/N)$')
    
    
    plt.show()
    
print(select_TE_gradient_func_variable_beads(5, 20, 20, 25))


# In[ ]:


""" RAW DATA / OPERATOR FUNCTION """

def system_time_evolution(system_array):
    """
    Time evolves a system of chains and returns the position of each bead
    in the system as "snapshots" in an output 4D array.
    
    Input: pop_array (float array): Position values of chain to be evolved
    
    """
    input_array = system_array
    asize = input_array.shape
    dt = asize[0]*asize[1] # number of chains * number of beads
    max_time = dt**2
    snap_array = np.zeros((max_time, asize[0], asize[1], asize[2]))
    for i in range(max_time):
        snap_array[i] = input_array
        for j in range(dt): 
            zth = random.randint(0, asize[0]-1) # pick a random chain index
            kth = random.randint(0,asize[1]-1) # pick a random bead index
            while True: # make a vector that is less than 1 magnitude
                    cubepoint = np.random.uniform(-1,1, (3,))
                    if np.dot(cubepoint,cubepoint) <= 1:
                        break
                        
            new_position = input_array[zth,kth, :] + cubepoint # prospective movement
            
            if (kth == 0 and np.dot(input_array[zth,kth+1,:] - new_position, input_array[zth,kth+1,:] - new_position) <= 1): # check if a starting chain end bead
                input_array[zth,kth, :] = new_position # accept position
                
            elif (kth == asize[1]-1 and np.dot(input_array[zth,kth-1,:] - new_position, input_array[zth,kth-1,:] - new_position) <= 1): # check if an final chain end bead
                input_array[zth,kth, :] = new_position # accept position
                
            elif (np.dot(input_array[zth,kth-1,:] - new_position, input_array[zth,kth-1,:] - new_position) <= 1 and np.dot(input_array[zth,kth+1,:] - new_position, input_array[zth,kth+1,:] - new_position) <= 1): # check if chosen bead is an end bead of the chain
                input_array[zth,kth, :] = new_position # accept position
            
            else:
                pass # reject the movement
            
                
    return snap_array


# In[ ]:


""" OPERATOR FUNCTION """

def sys_mean_squared_progress(snapshots_array):
    """
    Calculates the mean squared progress given a 4D snapshot array
    and averaging over the middle 1/3 bead paths for all chains
    
    Inputs: snapshot_array (array): Snapshots of 3D polymer systems
    
    """
    num_of_beads = snapshots_array.shape[2]
    ub_lim = math.floor(2/3 * num_of_beads)
    lb_lim = math.ceil(1/3 * num_of_beads)
    middle_third_array = snapshots_array[:,:,lb_lim:ub_lim+1,:]
    ssize = middle_third_array.shape

    MSD = 0
    for i in range(ssize[1]):
        print("// Chain Number: " + str(i))
        for j in range(ssize[2]):
            print("    Bead Number: " + str(j))
            single_bead_evolution = middle_third_array[:,i,j,:] 
            MSD += mean_squared_progress(single_bead_evolution)

    MSD /= (ssize[1] * ssize[2])

    return MSD 

np.save('4Dtestarray.npy', sys_mean_squared_progress(system_time_evolution(make_chains(3,13))))


# In[ ]:


""" GRAPHING AND STATISTICAL ANALYSIS ROUTINE """
"""
    Plots the time regimes for the system time evolution
    Inputs: make_chains() 
            system_time_evolution()
    
"""
extract_arr = np.load('4Dtestarray.npy')
plot_arr = sys_mean_squared_progress(extract_arr)
new_plot_arr = plot_arr[plot_arr<=0.6]
xvalues = np.log(np.arange(1, new_plot_arr.shape[0]+1, 1))
plt.scatter(xvalues, np.log(new_plot_arr), marker="X")    
plt.title("Mean squared displacement against time for regime 1 \n", fontdict = {'fontsize' : 11})
plt.xlabel('Log(t)')
plt.ylabel('Log($<| r(t+a) - r(a) |^2>$)')

varebs, errorr = scipy.optimize.curve_fit(func_linear, xvalues, np.log(new_plot_arr) )

plt.show()
print("  Gradient:  Intercept:  ")
print(varebs)


# In[ ]:


""" GRAPHING AND STATISTICAL ANALYSIS ROUTINE """
"""
    Shows 3D graph system of all chains in space 
    
"""
print_array = make_chains(20,300)
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(6):
    plot_array = print_array[i,:,:]
    print_arrayx = plot_array[:,0].flatten()
    print_arrayy = plot_array[:,1].flatten()
    print_arrayz = plot_array[:,2].flatten()
    ax.plot(print_arrayx, print_arrayy, print_arrayz)

plt.style.use('ggplot')


plt.show()

