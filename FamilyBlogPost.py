#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:59:11 2023

@author: williamfloyd
Potential blog post idea: want to see how long it takes (generations) for the
last name nguyen to take over vietnam.  

Rules: 
    1) Strictly Patronymic naming convention
    2) random number of children
    3) randomly split boy girl
    
Alright initial results are surprising! It looks like a highly variable process that can produce all sorts
of outcomes.      

Tried 100 trials with 50% and 43 had Nguyen dominate.  I suspect it should be 50% in the long run 
but who has time for that
"""

import random
import pandas as pd
import matplotlib.pyplot as plt


#Now it will all be pretty simple.  Basically assume we get the maximum number of 
#reproducing couples each generation, make them have kids, then throw out the old gen

def pdf(): #this is where we'll put the pdf.  I honestly can't 
    #remember what you call the pdf if it's discrete...maybe pmf?
    return [0.15,0.2,0.2,0.2,0.25]


def get_cdf():
    my_pdf = pdf()
    ans = [my_pdf[0]] #start with beginning number
    running_total = my_pdf[0]
    len_pdf = len(my_pdf)
    for i in range(1,len_pdf):
        running_total += my_pdf[i]
        ans.append(running_total)
    #print(ans)
    return ans


def num_kids():
    cdf = get_cdf()
    random_num = random.random()
    keep_track = 0 #this keeps track of where we are in the cdf
    for i in cdf:
        if random_num < i:
            return keep_track
        keep_track += 1
    #in case there's some weird thing with 1 not being less than 1
    return keep_track


 

def make_kids(nguyen_bool):
    #First we need to determine how many kids there will be
    #Note that this need not be uniformly distributed, and I don't
    #know how that may change things.  
    #We should make this a different function
    kid_num = num_kids()
    

    #Now we need to assign gender
    the_kids = [random.randint(0, 1) for i in range(kid_num)]
    boys = sum(the_kids)
    
    girls = kid_num - boys
    
    #Let's say 1 is a boy and 0 is a girl
    return boys,girls
    

def make_new_gen(men,women):
    #this is going to take the last generation and use it to make the
    #generation
    
    #get the total number of men and women
    #well the number of women is just a number
    
    #men we've just got to add the two pops
    total_men = men['nguyen'] + men['other']
    
    #now if the number of women exceeds the number of men, there's essentially
    #no issue. We can dive right into the pairing off subroutine
    #if there are more men we have to choose who gets left out
    
    #make the male population
    nguyen = [1 for i in range(men['nguyen'])]
    other = [0 for i in range(men['other'])]
    male_pop = nguyen+other
    random.shuffle(male_pop)
    
    
    if total_men > women: #this is when men exceed women
        male_pop = male_pop[:women]
        
    
    if women > total_men:
        women = total_men
    
    new_nguyen_men = 0
    new_other_men = 0
    new_women = 0
    
    for i in range(women):
        new_boys,new_girls = make_kids(1)
        if male_pop[i] == 1: #this is a nguyen
            new_nguyen_men += new_boys
        else:
            new_other_men += new_boys
        new_women += new_girls
        
        
    men['nguyen'] = new_nguyen_men
    men['other'] = new_other_men
    
    return men,new_women
 
def get_nguyen_prop(men): #only need to look at the men to determine the number with the 
    #last name Nguyen
    return men['nguyen'] / (men['nguyen']+men['other'])

def simulate_generations(num_gens,nguyen_prop):
    
    
    #Creating our first population
    first_pop_size = 1000 #100 pairs of people
    women = 1000
    #men = [[int(first_pop_size*nguyen_prop),1],[int(first_pop_size*(1-nguyen_prop)),0]]
    men = {}
    men['nguyen'] = int(first_pop_size*nguyen_prop)
    men['other'] = int(first_pop_size*(1-nguyen_prop))
    
    #Populations get big fast lol.  We're going to put in a cap 
    #to keep it manageable.  Will renormalize each gen if needed
    max_pop = 1000
    
    props = [nguyen_prop]
    
    for gen in range(num_gens):
        #create the new generation
        new_men,new_women = make_new_gen(men,women)
    
        #see what we've got
        #print(new_men,new_women)
        
        new_prop = get_nguyen_prop(new_men)
        print(f"{gen+1}th generation is {new_prop:.1%} Nguyen")
        print(f"men was {men} women was {women}")
        
        props.append(new_prop)
        
        if new_prop > 0.99:
            print(f'100% of the pop has the last name after {gen} generations')
            
            return 1
        
        
        if new_prop < 0.01:
            print(f"population died out after {gen} generations :(")
            return 0
        
        #out with the old, in with the new
        men,women = new_men,new_women
        #print(f"number of women is {women}")
        
        if women > max_pop: #there should be the same number of men and women
            scaling_factor = max_pop / women
            print(f"before {women}")
            women = int(women*scaling_factor)
            print(f"scaled by {scaling_factor} we have {women} women")
            men['nguyen'] = int(men['nguyen']*scaling_factor)
            men['other'] = int(men['other']*scaling_factor)
    plt.plot(props)
    



def simulate_many_times(num_iters):
    #I want to see what proportion of simulations result in Nguyen dominance
    #compared to Nguyen annihilation
    running_total = 0
    for i in range(1,num_iters+1):
        running_total += simulate_generations(10000,0.6)
    
    print(f"number of successful wins was {running_total}")
    print(f"rate of successful wins was {running_total/num_iters}")
    return running_total

simulate_many_times(2)
