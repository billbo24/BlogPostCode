#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:35:08 2023

@author: williamfloyd
Alright going to take on a very ambitious project here.  Want to see if I can
use cell phone towers to find a person in the wilderness.  

The problem: You are lost in the wilderness and no one knows where you are.  Miraculously
though you have cell phone service, and more importantly, the cell phone towers in this area
exhibit a weird property: The likelihood that your call is routed through a tower is 
inverserly proportional to the distance you are from the tower.  What I want to know:
    1) Can we pin point your location based on a few phone calls? 
    2) How many towers do we need for a good guess
    3) How many calls do we need for a good guess.  
    


Since the scope of this is pretty big, here are a few things I need to sort out.  Will
check them off up here as I get them done.  Need to be very clean and organized with 
this project

1) 2-d plotting.  I'm sure this can be done in seaborn or matplotlib.  I think both a heatmap
    and 3-d rendering could look nice.  Might experiment with both
2) A clear and clean way to store the coordinates of the cell phone towers
3) Perform the bayes shit in 2 dimensions.  I'm pretty sure it won't be much different than the 
    1-d case i.e. f(x,y | e) = f(e | x,y)*f(x,y) / int(all xy) f(e|x,y)*f(x,y)
    little sloppy with notation here but you get the idea
    
    
This shows a good way to iterate over a matrix Z.  This is assuming Z was made
using f(X,Y) for netmesh's X and Y
#By convention, net mesh makes each row a fixed y value and column a fixed x value
for row in range(len(Z)):
    for col in range(len(Z[0])):
        print(f"x={X[row,col]},y={Y[row,col]},z={Z[row,col]}")


NOTE 
in the general case, we calculate the probability like so
1) total the distance to every tower
2) Add it up
3) calculate (total distance/distance_i) for each tower.  Call it share_i
4) sum those numbers up.  Call it TOTAL
5) prob_i = share_i / TOTAL
    
    ex. distances 1,3,5
    2) 9
    3) 9,3,1.8
    4) 13.8
    5) 0.65, 0.22,0.13

In the two tower case, it's a bit easier.  If the distances are 1, k, then 
the probs are k/k+1 and 1/k+1
"""

import matplotlib.pyplot as plt
import numpy as np
import random


from matplotlib import cm
from matplotlib.ticker import LinearLocator


'''
For right now this is how we'll kick everything off.  Z is effectively our function 
at this point, and any changes to it we can make elsewhere
'''
def combine_towers(*towers):
    ans = []
    for i in towers:
        ans.append(i)
    return ans

def initializer():
    #I can't tell if this is a good idea, but I like making one of these 
    #'initializer functions' to make all my shit
    #This stuff is always gonna happen right away
    x_min,x_max = 0,1
    y_min,y_max = 0,1
    area = (x_max-x_min)*(y_max-y_min)
    n_x,n_y = 40,40
    
    X = np.linspace(x_min, x_max, n_x)
    Y = np.linspace(y_min, y_max, n_y)
    X, Y = np.meshgrid(X, Y)
    
    f = lambda x,y: (1/area)
    pdf = np.vectorize(f)
    
    Z = pdf(X,Y)
    
    tower1 = [0.1,0.1]
    tower2 = [0.5,0.1]
    tower3 = [0.8,0.8]
    tower4 = [0.2,0.75]
    tower5 = [0.5,0.5]
    towers = combine_towers(tower1,tower2,tower3,tower4,tower5)
    
    
    
    return X,Y,Z,towers
    


#Alright for the purposes of this problem we are going to build a 3-d plotting function
#to make our lives easier.  Going to take arguments X,Y,Z which are all matrices
def three_d_plotter(X,Y,Z):
    #Now remember, for this problem our function is not going to be 
    #a nice tidy f(x,y).  It's going to be an explicit map of 10000 points
    #to their z coordinate
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    z_max = np.max(Z) + 1
    z_min = np.min(Z) - 1
    
    # Customize the z axis.
    ax.set_zlim(z_min, z_max)
    ax.zaxis.set_major_locator(LinearLocator(5))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    

    plt.show()

def get_distance(x,y,point_x,point_y):
    #point is an x,y pair
    return ((point_x-x)**2 + (point_y-y)**2)**0.5


def compute_shares(tower_i_dist,total_dist):
    #both arguments are matrices
    #They should have the same dimensions
    num_rows = len(total_dist)
    num_cols = len(total_dist[0])
    
    #create an empty matrix to store the answers
    ans = np.zeros(shape=(num_rows,num_cols))
    
    #calculate share for each x,y point
    for row in range(num_rows):
        for column in range(num_cols):
            ans[row,column] = total_dist[row,column] / tower_i_dist[row,column]
            
    return ans

def get_probs(X,Y,towers):
    #We need a few empty matrices of this size so we'll make it now
    num_rows = len(X)
    num_cols = len(X[0])     
    
    skeleton = np.zeros((num_rows,num_cols))

    #need to know how many towers here
    num_tow = len(towers)
    
    #create vectorized function
    temp = np.vectorize(get_distance)
    
    dist_matrices = {}
    
    for i in range(num_tow):
        key_string = f"tow_{i+1}"
        dist_matrices[key_string] = temp(X,Y,towers[i][0],towers[i][1])
 
    #Now we need to convert these distances to probabilities
    #Create a blank canvas
    combo_dist = skeleton.copy()
    
    
    for key_name in dist_matrices.keys():
        #add the distances to the other tower
        combo_dist += dist_matrices[key_name]
    
    #Alright now we need to take all of the individual distance matrices
    #and determine the probabilities as outlined above.  
    #We've totaled the distances, so now we need to compute the SHARES
    #For each tower.  This is the total distance at point x,y divided by
    #the distance to tower i.  
    shares = {}
    
    for key_name in dist_matrices.keys():
        shares[key_name] = compute_shares(dist_matrices[key_name], combo_dist)
    
    #Now we need a total share matrix
    total_share = skeleton.copy()
    for key_name in shares.keys():
        total_share += shares[key_name]
    
    #And finally compute probabilities
    tower_probs = {}
    
    #This will allow us to broadcast division to the matrices
    basic_divide = lambda x,y: x/y
    for key_name in shares.keys():
        tower_probs[key_name] = basic_divide(shares[key_name],total_share)
    

    
    return tower_probs
    


#Alright, now that we know the probabilities of going through each tower, 
#all that's left to do is go through our "evidence" and perform the bayes step

def bayes_step(X,Y,Z,evidence,probs):
    #probs is a constant, so no need to recompute it
    #Really all we need to do here is compute the likelihood of 
    #Observing the piece of evidence.  This is the integral of our pdf
    #multiplied by the likelihood of that tower at that particular point
    #We know the numebr of X and Y steps so we can use a basic riemann 
    #sum to compute the integral
    #Remember that rows are y coordinates and columns are x's.  ALso because
    #we have, say k points, that means there are only k-1 gaps
    nx = len(X[0])
    ny = len(X)
    
    x_step = 1/(nx-1)
    y_step = 1/(ny-1)
    
    square_area = x_step*y_step
    ANS = Z.copy()
    
    prob_e = 0
    for x in range(nx):
        for y in range(ny):
            #crude integral.  Could update the Z[x,y] to take an avg over the interval
            prob_e += Z[x,y]*probs[evidence][x,y]*square_area
    
    #Not giving thigns that sum exactly to 1 but that's okay for now
    #print(f"Current likelihood of {evidence} is {prob_e}")
    
    #finally we do this step
    for x in range(nx):
        for y in range(ny):
            #crude integral.  Could update the Z[x,y] to take an avg over the interval
            ANS[x,y] = (Z[x,y]*probs[evidence][x,y])/prob_e
    
    #gonna cheat a bit here and renormalize
    renorm = 0
    for x in range(nx):
        for y in range(ny):
            #crude integral.  Could update the Z[x,y] to take an avg over the interval
            renorm += ANS[x,y]*square_area
    
    
    ANS =(1/renorm)*ANS
    
    return ANS


def view_towers(towers,my_loc):
    #This is kind of a way to visualize the towers' locations
    #plt.pcolormesh(X,Y,tower_probs['tow_2'])
    #for i in range(len(towers)): #just to label the towers
        #tower_num = i+1
        #plt.text(towers[i][0], towers[i][1],f"{tower_num}",c='white')
    X_points = []
    Y_points = []
    for t in towers:
        X_points.append(t[0])
        Y_points.append(t[1])
    
    fig,ax = plt.subplots()
    X_points.append(my_loc[0])
    Y_points.append(my_loc[1])
    ax.scatter(X_points,Y_points)
    
    for i in range(len(towers)):
        label = f"tower {i+1}"
        x = X_points[i]
        y = Y_points[i]
        ax.annotate(label,(x,y))
    
    ax.annotate("My Location",(X_points[-1],Y_points[-1]))
    plt.show()


def return_data(x,probs):
    for i in range(len(probs)):
        if x<=probs[i]:
            return i #this should always terminate
    return 100

def get_call_data(location,towers,num_points):
    #Now we're going to randomly generate our call info
    distances = []
    for t in towers:
        distances.append(get_distance(location[0], location[1], t[0], t[1]))
    
    print(distances)
    
    #Using methodology outlines above
    total_distance = sum(distances)
    shares = [total_distance/i for i in distances]
    total_shares = sum(shares)
    
    #Really not that bad
    probs = [i/total_shares for i in shares]
    
    #Now we need a cumulative vector
    generator = [probs[0]]
    for i in range(1,len(probs)):
        generator.append(probs[i]+generator[i-1])
    
    data_points = []
    #Now just to generate our data
    for i in range(num_points):
        dice_roll = random.random()
        tow_index = return_data(dice_roll, generator) #see above for definition
        data_points.append(f"tow_{tow_index+1}")
    
    #print(probs,generator)
    return data_points




X,Y,Z,towers = initializer()
my_location = [0.5,0.8]
probs = get_probs(X,Y,towers)
view_towers(towers,my_location)




evidence = get_call_data(my_location,towers,50)
#print(evidence)
counter = 0
for e in evidence:
    counter += 1
    Z = bayes_step(X,Y,Z,e,probs)
    
    if counter % 10 == 0:
        plt.pcolormesh(X,Y,Z)
        for i in range(len(towers)): #just to label the towers
            tower_num = i+1
            plt.text(towers[i][0], towers[i][1],f"{tower_num}",c='white')
        
        
        plt.annotate("My Location", xy=(my_location[0],my_location[1]), c='white',
                     arrowprops=dict(facecolor='white',shrink=0.5))
        plt.title(f"after {counter} observations, {e} was last")
        plt.show()

#three_d_plotter(X,Y,Z)
plt.pcolormesh(X,Y,Z)
for i in range(len(towers)): #just to label the towers
    tower_num = i+1
    plt.text(towers[i][0], towers[i][1],f"{tower_num}",c='white')

plt.annotate("My Location", xy=(my_location[0],my_location[1]), c='white',
             arrowprops=dict(facecolor='white',shrink=0.5))

plt.title(f"after {counter} observations, {e} was last")
plt.show()