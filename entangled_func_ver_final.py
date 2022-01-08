#!/usr/bin/env python
# coding: utf-8

# In[ ]:

""" This is the final version that was used to produce the diffusion curves. Note 
    that certain functions such as those calculating chain length or mean squared
    displacement are omitted. Created by Akhilesh Mocherla.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import scipy.stats
import cProfile
import math
from itertools import islice
import collections
from numba import jit
import time
start_time = time.time()

""" UTILITY FUNCTION """
# functions for fitting
def func_linear(x,m,c):
    return m*x + c

""" UTILITY FUNCTION """
# Skip indices function
def skip_indices(iterator, r):
    """ Skips r indices of the iterator. E.g if called on the 5th iteration with r=3,
        will skip to the 8th index, skipping 5,6,7th indices.
        
        INPUT: iterator (list) - range of numbers to be iterated over 
                                 (need to use 'iter()' with 'range()')
               r (int)         - number of indices to skip
     """
    if r is None:
        # put iterator in a length 0 double end queue
        collections.deque(iterator, maxlen=0)
    else:
        # skip to the r indices ahead
        next(islice(iterator, r-1, r-1), None)
        
        
""" RAW DATA FUNCTION """
def make_moves(nbeads):
    """
    Returns x,y,z position values for a generated chain
    
    Inputs: nbeads (int):       Desired number of beads on chain
            boundary_r (float): Limits of boundary (Cube) 
    """
    boundary_r = 3
    new_origin = np.random.uniform(-boundary_r, boundary_r, (3,))
    r_vector = np.zeros((nbeads,3))
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



def make_chains(num_of_polymers, nbeads):
    """
    Makes a 3D matrix containing 'num_of_chains' number of chains
    
    Input: num_of_polymers (int): Number of chains to be in system
           nbeads (int)         : Beads on each generated chain
    
    """
    
    chain_array = np.zeros((num_of_polymers, nbeads, 3))
    
    for i in range(num_of_polymers): # num_of_chains is the 3D array depth 
        chain_array[i,:,:] = make_moves(nbeads)
        
    return chain_array


""" RAW DATA FUNCTION """
@jit(nopython=True)
def system_time_evolution(system_array):
    """
    Time evolves a system of chains and returns the position of each bead
    in the system as "snapshots" in an output 4D array.
    
    Input: pop_array (float array): Position values of chain to be evolved
    
    """ 
    entanglements_on = False
    tol = -15
    L = 3 # NEEDS TO MATCH BOUNDARY R
    periodic_divisor = np.array([L,L,L])
    offset = periodic_divisor/2
    input_array = system_array
    asize = input_array.shape
    dt = asize[0]*asize[1] # number of chains * number of beads
    max_time = 15*dt**2
    snap_array = np.zeros((max_time, asize[0], asize[1], asize[2]))
    for i in range(max_time):
        snap_array[i] = input_array
        for j in range(dt): 
    
                
            done = False # refresh entanglement check to be verified 
            zth = random.randint(0, asize[0]-1) # pick a random chain index
            kth = random.randint(0,asize[1]-1) # pick a random bead index
            
            while True: # make a vector that is less than 1 magnitude
                    cubepoint = np.random.uniform(-1,1, (3,))
                    if np.dot(cubepoint,cubepoint) <= 1:
                        break
                        
            new_position = input_array[zth,kth, :] + cubepoint # prospective movement
            
            # First end Bead
            if (kth == 0 and np.dot(input_array[zth,kth+1,:] - new_position, input_array[zth,kth+1,:] - new_position) <= 1): # check if a starting chain end bead
                if entanglements_on == False:
                    input_array[zth,kth, :] = new_position # accept position
                    
                
                first_centroid = (input_array[zth,kth,:] + input_array[zth,kth+1,:] + new_position)/3 # centroid
                
                B = input_array[zth,kth,:]
                B_cent_mag = np.sqrt(np.dot(B- first_centroid,B- first_centroid))
                
                Bdash = new_position 
                Bdash_cent_mag = np.sqrt(np.dot(Bdash- first_centroid,Bdash- first_centroid))
                
                C =  input_array[zth,kth+1,:] 
                C_cent_mag =  np.sqrt(np.dot(C- first_centroid,C- first_centroid))
                
                V_mag = np.amax(np.array([B_cent_mag,Bdash_cent_mag, C_cent_mag]))
                
                for chain in range(asize[0]):
                    if (done == True):
                        break # entangle
                    for bead in range(asize[1]-1): # exclude the end one, bead corresponds to bond ahead of it
                        if (chain == zth and (bead == kth or bead == kth+1)): # skip bonds in/adjacent the tetrahedron
                            pass
                        else:
                            bond_midpoint =  (input_array[chain, bead, :] + input_array[chain, bead+1,:])/2
            
                            bond  = input_array[chain, bead, :] - input_array[chain, bead+1,:]  - offset
                            bond_mag = np.sqrt(np.dot(bond,bond))
                            separ = np.mod((bond_midpoint - first_centroid + offset), periodic_divisor) - offset
                            
                            cond1 =  np.sqrt(np.dot(separ,separ))
                            cond2 = V_mag + bond_mag/2
                            
                            correction = separ - bond_midpoint + first_centroid # save some modulo arithmetic
                        
                            if (cond1 <= cond2):
                                # edges to bead P
                                PB = B - input_array[chain, bead, :] - correction # B-P
                                PBdash = Bdash - input_array[chain, bead, :] - correction # B'-P 
                                PC = C - input_array[chain, bead, :] - correction # C-P
                                
                                # edges to bead Q
                                QB = B - input_array[chain, bead+1, :] - correction # B-Q
                                QBdash = Bdash - input_array[chain, bead+1, :] - correction # B'-Q
                                QC = C - input_array[chain, bead+1, :] - correction # C-Q
                                
                                # Shared tetraheadron volume
                                vPQBBdash = abs(np.linalg.det(np.column_stack((bond, PB, PBdash))))
                                                               
                                # triangle BB'C
                                vPBBdashC = abs(np.linalg.det(np.column_stack((PB, PBdash, PC))))
                                vQBBdashC = abs(np.linalg.det(np.column_stack((QB, QBdash, QC))))
                                vPQBC = abs(np.linalg.det(np.column_stack((bond, PB, PC))))
                                vPQBdashC = abs(np.linalg.det(np.column_stack((bond, PBdash, PC))))
                                
                                # check triangle
                                inequal2 = abs(vPBBdashC+vPBBdashC-vPQBC-vPQBdashC-vPQBBdash) # thread BB'C
                                                
                                if (inequal2<tol):
                                    done = True # entangle
                                    break  
                                else:
                                    pass # skip to checking the next bond
                                
                                
                            else:
                                pass # accept position
                
                if done == False: # if no block
                    input_array[zth,kth, :] = new_position # accept position
            
                
            # Last end bead     
            elif (kth == asize[1]-1 and np.dot(input_array[zth,kth-1,:] - new_position, input_array[zth,kth-1,:] - new_position) <= 1): # check if an final chain end bead
                if entanglements_on == False:
                    input_array[zth,kth, :] = new_position # accept position
                
                last_centroid = (input_array[zth,kth-1,:] + input_array[zth,kth,:] + new_position)/3 # centroid
                
                A = input_array[zth,kth-1,:] 
                A_cent_mag = np.sqrt(np.dot(A- last_centroid,A- last_centroid))
                
                B = input_array[zth,kth,:]
                B_cent_mag = np.sqrt(np.dot(B- last_centroid,B- last_centroid))
                
                Bdash = new_position 
                Bdash_cent_mag = np.sqrt(np.dot(Bdash- last_centroid,Bdash- last_centroid))
                
                V_mag = np.amax(np.array([A_cent_mag,B_cent_mag,Bdash_cent_mag]))
                
                for chain in range(asize[0]):
                    if (done == True):
                        break # entangle 
                    for bead in range(asize[1]-1): # exclude the end one, bead corresponds to bond ahead of it # exclude the end one, bead corresponds to bond ahead of it
                        if (chain == zth and (bead == kth-1 or bead == kth-2)): # skip bonds in/adjacent the tetrahedron
                            pass #
                        else:
                            bond_midpoint =  (input_array[chain, bead, :] + input_array[chain, bead+1,:])/2
                            
                            bond  = input_array[chain, bead, :] - input_array[chain, bead+1,:] 
                            bond_mag = np.sqrt(np.dot(bond,bond))
                        
                            separ = np.mod((bond_midpoint - last_centroid + offset), periodic_divisor) - offset
                            cond1 =  np.sqrt(np.dot(separ,separ))
                            cond2 = V_mag + bond_mag/2
                            
                            correction = separ - bond_midpoint + last_centroid # save some modulo arithmetic
                        
                            if (cond1 <= cond2):
                                
                                
                                # edges to bead P
                                PA = A - input_array[chain, bead, :] - correction # A-P
                                PB = B - input_array[chain, bead, :] - correction # B-P
                                PBdash = Bdash - input_array[chain, bead, :] - correction # B'-P 
                                
                                # edges to bead Q
                                QA = A - input_array[chain, bead+1, :] - correction # A-Q
                                QB = B - input_array[chain, bead+1, :] - correction # B-Q
                                QBdash = Bdash - input_array[chain, bead+1, :] - correction # B'-Q
                                
                                # Shared tetraheadron volume
                                vPQBBdash = abs(np.linalg.det(np.column_stack((bond, PB, PBdash))))
                                                
                                # triangle ABB'
                                vPABBdash = abs(np.linalg.det(np.column_stack((PA, PB, PBdash))))
                                vQABBdash = abs(np.linalg.det(np.column_stack((QA, QB, QBdash))))
                                vPQAB = abs(np.linalg.det(np.column_stack((bond, PA, PB))))
                                vPQABdash = abs(np.linalg.det(np.column_stack((bond, PA, PBdash))))
                                
                                # check triangle
                                inequal1  = abs(vPABBdash+vQABBdash-vPQAB-vPQABdash-vPQBBdash) # thread ABB'               
                                if (inequal1<tol):
                                    done = True # entangle
                                    break  
                                else:
                                    pass # skip to checking the next bond
                                
                                
                            else:
                                pass # skip checking this bond and move onto the next one
                    
                if done == False: # if no block
                    input_array[zth,kth, :] = new_position # accept position 
                    
                    
            # All other beads    
            elif (np.dot(input_array[zth,kth-1,:] - new_position, input_array[zth,kth-1,:] - new_position) <= 1 and np.dot(input_array[zth,kth+1,:] - new_position, input_array[zth,kth+1,:] - new_position) <= 1): # check if chosen bead is an end bead of the chain
                if entanglements_on == False:
                    input_array[zth,kth, :] = new_position # accept position
                    
                mid_centroid = (input_array[zth,kth-1,:] + input_array[zth,kth,:] + input_array[zth,kth+1,:] + new_position)/4 # centroid
                
                
                A = input_array[zth,kth-1,:] 
                A_cent_mag = np.sqrt(np.dot(A- mid_centroid,A- mid_centroid))
                
                B = input_array[zth,kth,:]
                B_cent_mag = np.sqrt(np.dot(B- mid_centroid,B- mid_centroid))
                
                Bdash = new_position 
                Bdash_cent_mag = np.sqrt(np.dot(Bdash- mid_centroid,Bdash- mid_centroid))
                                   
                C =  input_array[zth,kth+1,:] 
                C_cent_mag =  np.sqrt(np.dot(C- mid_centroid,C- mid_centroid))
                
                V_mag = np.amax(np.array([A_cent_mag,B_cent_mag,Bdash_cent_mag,C_cent_mag]))
                
                for chain in range(asize[0]):
                    if (done == True):
                        break # entangle 
                    for bead in range(asize[1]-1): # exclude the end one, bead corresponds to bond ahead of it
                        if (chain == zth and (bead == kth or bead == kth-1)): # skip bonds in the tetrahedron
                            pass 

                        else:
                            
                            bond_midpoint =  (input_array[chain, bead, :] + input_array[chain, bead+1,:])/2 #(P+Q)/2

                            bond  = input_array[chain, bead+1, :] - input_array[chain, bead,:] # Q-P
                            bond_mag = np.sqrt(np.dot(bond,bond))
                            
                            separ = np.mod((bond_midpoint - mid_centroid + offset), periodic_divisor) - offset
                            cond1 =  np.sqrt(np.dot(separ, separ))

                            cond2 = V_mag + bond_mag/2
                            
                            correction = separ - bond_midpoint + mid_centroid # save some modulo arithmetic
                        
                            if (cond1 <= cond2):
                                
                                # edges to bead P
                                PA = A - input_array[chain, bead, :] - correction # A-P
                                PB = B - input_array[chain, bead, :] - correction # B-P
                                PBdash = Bdash - input_array[chain, bead, :] - correction # B'-P 
                                PC = C - input_array[chain, bead, :] - correction # C-P
                                
                                # edges to bead Q
                                QA = A - input_array[chain, bead+1, :] - correction # A-Q
                                QB = B - input_array[chain, bead+1, :] - correction # B-Q
                                QBdash = Bdash - input_array[chain, bead+1, :] - correction # B'-Q
                                QC = C - input_array[chain, bead+1, :] - correction # C-Q
                                
                                # Shared tetraheadron volume
                                vPQBBdash = abs(np.linalg.det(np.column_stack((bond, PB, PBdash))))
                                                
                                # Face ABB'
                                vPABBdash = abs(np.linalg.det(np.column_stack((PA, PB, PBdash))))
                                vQABBdash = abs(np.linalg.det(np.column_stack((QA, QB, QBdash))))
                                vPQAB = abs(np.linalg.det(np.column_stack((bond, PA, PB))))
                                vPQABdash = abs(np.linalg.det(np.column_stack((bond, PA, PBdash))))
                                                
                                # Face BB'C
                                vPBBdashC = abs(np.linalg.det(np.column_stack((PB, PBdash, PC))))
                                vQBBdashC = abs(np.linalg.det(np.column_stack((QB, QBdash, QC))))
                                vPQBC = abs(np.linalg.det(np.column_stack((bond, PB, PC))))
                                vPQBdashC = abs(np.linalg.det(np.column_stack((bond, PBdash, PC))))
                                
                                # check both triangles
                                inequal1  = abs(vPABBdash+vQABBdash-vPQAB-vPQABdash-vPQBBdash) # thread ABB'
                                inequal2 = abs(vPBBdashC+vQBBdashC-vPQBC-vPQBdashC-vPQBBdash) # thread BB'C
                                if (inequal1<tol and bead != kth-2) or (inequal2<tol and bead != kth+1):
                                    done = True # entangle
                                    break  
                                else:
                                    pass # skip to checking the next bond
                                                
                                
                            else:
                                pass # skip to checking the next bond 
                
                
                if done == False:
                    input_array[zth,kth, :] = new_position # accept position
                
            else:
                pass # reject the movement
            
                
    return snap_array

# Repeat same simulation parameters , 5-50 is generally instructive.
np.save('1.npy', system_time_evolution(make_chains(5,50)))
np.save('2.npy', system_time_evolution(make_chains(5,50)))
np.save('3.npy', system_time_evolution(make_chains(5,50)))

print("--- %s seconds ---" % (time.time() - start_time))

# In[ ]:


""" OPERATOR FUNCTION """
def mean_squared_progress(pop_array):
    """
    Returns an array which contains all <|r(i+n) - r(i)|^2> 
    (averaged over number of iterations) for a given n
    
    Inputs: n (int): step_number
            pop_array: m x 3 matrix containing xyz positions on each row. 
            
    Outputs: len(array)-1 x 1 matrix containing msd values for all step numbers n
    """
    #initialise list to contain each 
    progress_array = []
    
    for size in range(1,len(pop_array)):
        selected_beads = [pop_array[i] for i in np.arange(0,len(pop_array),size)]
        distances = [(np.dot(selected_beads[i] - selected_beads[i-1],selected_beads[i] - selected_beads[i-1])) for i in range(1,len(selected_beads))]

        mean = np.mean(distances)
        progress_array.append(mean)
    return np.array(progress_array)



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

    MSD = np.zeros((snapshots_array.shape[0]-1, ssize[1]*ssize[2]))
    for i in range(ssize[1]):
        print("// Chain Number: " + str(i))
        for j in range(ssize[2]):
            print("    Bead Number: " + str(j))
            single_bead_evolution = middle_third_array[:,i,j,:] 
            MSD[:,i*ssize[2]+j] = mean_squared_progress(single_bead_evolution)
        
    
    print("Done!")

    return MSD


""" OPERATOR FUNCTION """
def get_gradients(stack_array):
    size = stack_array.shape
    grad1_ray = np.zeros((size[1],2)) # column 1 is m, column 2 is c
    grad2_ray = np.zeros((size[1],2))
    grad3_ray = np.zeros((size[1],2))
    resulta = np.zeros((3,4))
    
    for i in range(size[1]):
        original_arr = stack_array[:,i]
        
        # fit to the first regime
        t1_reg = np.log(original_arr[(original_arr<=0.6)])
        t1_reg_xvals = np.log(np.arange(1, t1_reg.size+1, 1))
        varebs1, errorr1 = scipy.optimize.curve_fit(func_linear, t1_reg_xvals, t1_reg)
        grad1_ray[i] = varebs1
        
        # fit to the second regime
        tphys_reg = np.log(original_arr[(original_arr >0.6) & (original_arr<=0.6*10)])
        tphys_reg_xvals = np.log(np.arange(t1_reg.size+1, t1_reg.size+tphys_reg.size+1, 1))
        varebs2, errorr2 = scipy.optimize.curve_fit(func_linear, tphys_reg_xvals, tphys_reg)
        grad2_ray[i] = varebs2
        
        # fit to the third regime
        trepeat_reg = np.log(original_arr[(original_arr>0.6*10)])
        if len(trepeat_reg)==0:
            grad3_ray[i,:] = True
            continue
        trepeat_reg_xvals = np.log(np.arange(t1_reg.size+tphys_reg.size+1, original_arr.size+1, 1))
        varebs3, errorr3 = scipy.optimize.curve_fit(func_linear, trepeat_reg_xvals, trepeat_reg)
        grad3_ray[i] = varebs3
        
        
    # fix the grad 3 array for False entries
    rlist = []
    for j in range(len(grad3_ray)):
        if grad3_ray[j,1] and grad3_ray[j,0] == True:
            rlist.append(j)
            
    grad3_ray = np.delete(grad3_ray,rlist,0)
        
    # prepare entries    
    err1 = scipy.stats.sem(grad1_ray, axis=0) #m_err, c_err
    err2 = scipy.stats.sem(grad2_ray, axis=0)
    err3 = scipy.stats.sem(grad3_ray, axis=0)
    
    mean1 = np.mean(grad1_ray, axis=0) # m, c
    mean2 = np.mean(grad2_ray, axis=0)
    mean3 = np.mean(grad3_ray, axis=0)
    
    # prepare result array
    resulta[0,0] = mean1[0]
    resulta[0,1] = err1[0]
    resulta[0,2] = mean1[1]
    resulta[0,3] = err1[1]
    
    resulta[1,0] = mean2[0]
    resulta[1,1] = err2[0]
    resulta[1,2] = mean2[1]
    resulta[1,3] = err2[1]
    
    resulta[2,0] = mean3[0]
    resulta[2,1] = err3[0]
    resulta[2,2] = mean3[1]
    resulta[2,3] = err3[1]
    
    
    return resulta # m, m_err, c, c_err


convert_array1 = np.load('1.npy')
convert_array2 = np.load('2.npy')
convert_array3 = np.load('5_50thirdunent.npy')

np.save('MSD1.npy', sys_mean_squared_progress(convert_array1))
np.save('MSD2.npy', sys_mean_squared_progress(convert_array2))
np.save('MSD3.npy', sys_mean_squared_progress(convert_array3))


print("--- %s seconds ---" % (time.time() - start_time))

# In[6]:


a = np.load('X.npy') # unentangled
b = np.load('Y.npy') # entangled

original_arr = np.mean(a, axis=1)
plot_arr = np.mean(b, axis=1)     
plt.style.use('fast')

# raw error arrays for unentangled and entangled
raw_a_err = scipy.stats.sem(a, axis=1)
raw_b_err = scipy.stats.sem(b, axis=1)

# amended errors for log graphs using ln_err(x) = err(x)/x

a_err = raw_a_err/original_arr 
b_err = raw_b_err/plot_arr

f0 = plt.figure(0)
plt.errorbar(np.log(np.arange(1,original_arr.size+1, 1)),np.log(original_arr), yerr=a_err)
plt.errorbar(np.log(np.arange(1,plot_arr.size+1, 1)),np.log(plot_arr), yerr=b_err)
plt.title("Mean squared displacement against time \n", fontdict = {'fontsize' : 14})
plt.xlabel('Log(t)')
plt.ylabel('Log($<| r(t+a) - r(a) |^2>$)')


# regime 1
f1 = plt.figure(1)
t1_reg = original_arr[(original_arr<=0.6)]
t1_reg_xvals = np.log(np.arange(1, t1_reg.size+1, 1))
t1_err = a_err[np.arange(0, t1_reg.size, 1)]
plot_arr1 = plot_arr[(plot_arr <= 0.6)]
plot_err1 = b_err[np.arange(0, plot_arr1.size, 1)]
xvalues1 = np.log(np.arange(1, np.log(plot_arr1).shape[0]+1, 1))
plt.title("Mean squared displacement against time for regime 1 repeat \n", fontdict = {'fontsize' : 14})
plt.xlabel('Log(t)')
plt.ylabel('Log($<| r(t+a) - r(a) |^2>$)')
plt.errorbar(t1_reg_xvals, np.log(t1_reg), yerr=t1_err)
plt.errorbar(xvalues1, np.log(plot_arr1), yerr =plot_err1 )



# regime 2
f2 = plt.figure(2)
tphys_reg = original_arr[(original_arr >0.6) & (original_arr<=0.6*10)]
tphys_reg_xvals = np.log(np.arange(t1_reg.size+1, t1_reg.size+tphys_reg.size+1, 1))
tphys_err = a_err[np.arange(t1_reg.size, t1_reg.size+tphys_reg.size, 1)]
plot_arr2 = plot_arr[(plot_arr > 0.6) & (plot_arr <= 0.6*10)]
plot_err2 = b_err[np.arange(plot_arr1.size, plot_arr1.size+plot_arr2.size, 1)]
xvalues2 = np.log(np.arange(np.log(plot_arr1).shape[0]+1, np.log(plot_arr1).shape[0]+np.log(plot_arr2).shape[0]+1, 1))
plt.title("Mean squared displacement against time for regime 2 repeat \n", fontdict = {'fontsize' : 14})
plt.xlabel('Log(t)')
plt.ylabel('Log($<| r(t+a) - r(a) |^2>$)')
plt.errorbar(tphys_reg_xvals, np.log(tphys_reg), yerr=tphys_err)
plt.errorbar(xvalues2, np.log(plot_arr2), yerr=plot_err2)

# regime 3

f3 = plt.figure(3)
trepeat_reg = original_arr[(original_arr>0.6*10)]
trepeat_reg_xvals = np.log(np.arange(t1_reg.size+tphys_reg.size+1, original_arr.size+1, 1))
trepeat_err = a_err[np.arange(t1_reg.size+tphys_reg.size, original_arr.size, 1)]
plot_arr3 = plot_arr[(plot_arr > 0.6*10)]
plot_err3 = b_err[np.arange(plot_arr1.size+plot_arr2.size, plot_arr.size, 1)]
xvalues3 = np.log(np.arange(np.log(plot_arr1).shape[0]+np.log(plot_arr2).shape[0]+1, np.log(plot_arr1).shape[0]+np.log(plot_arr2).shape[0]+np.log(plot_arr3).shape[0]+1, 1))
plt.title("Mean squared displacement against time for regime 3 repeat \n", fontdict = {'fontsize' : 14})
plt.xlabel('Log(t)')
plt.ylabel('Log($<| r(t+a) - r(a) |^2>$)')
plt.errorbar(trepeat_reg_xvals, np.log(trepeat_reg), yerr=trepeat_err)
plt.errorbar(xvalues3, np.log(plot_arr3), plot_err3)



plt.show()

print("For unentangled")
print(get_gradients(a))
print("For entangled")
print(get_gradients(b))


# In[ ]:




