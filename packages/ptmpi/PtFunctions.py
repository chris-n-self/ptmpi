# 23/02/16
# Chris Self

import numpy as np

# This function decides whether two chains with temps T1=1/beta1 and T2=1/beta2
# should swap temperatures given that their current vsectors have free energies
# F1 and F2
# if running in parallel need to pass this function the log-partition functions
# rather than the sector obj
def decide_pt_switch( F_11, F_22, F_12, F_21 ):
    prob_switch = min( 1, np.exp( (F_12 + F_21 - F_11 - F_22) ) )
    return ( np.random.random()<prob_switch )

# function that doubles the density in a beta set within a given range of beta
def double_beta_density_between_a_b( betas, beta_a, beta_b ):
    beta_a = 1.0*beta_a
    beta_b = 1.0*beta_b

    # ensure b_a < b_b
    if (beta_b<beta_a):
        beta_a,beta_b = beta_b,beta_a

    # find the betas in the set that are closest to b_a and b_b
    ordered_betas = np.sort(betas)
    index_beta_a,index_beta_b = 0,0

    # detect error that beta_a < min(betas)
    if (beta_a<ordered_betas[0]):
        print('WARNING, tried to double beta density between '+str(beta_a)+' and '+str(beta_b)+' but this failed because the beta set begins at '+str(ordered_betas[0])+'. Please ensure the window is inside the initial range of betas.')
        return betas

    # iterate through the ordered set of temperatures until we find the closest beta to
    # b_a and b_b
    while( (index_beta_a+1!=len(ordered_betas)) and abs(ordered_betas[index_beta_a]-beta_a)>abs(ordered_betas[index_beta_a+1]-beta_a) ):
        index_beta_a += 1
    index_beta_b = index_beta_a
    while( (index_beta_b+1!=len(ordered_betas)) and abs(ordered_betas[index_beta_b]-beta_b)>abs(ordered_betas[index_beta_b+1]-beta_b) ):
        index_beta_b += 1

    # detect error that index_beta_a = index_beta_b
    if (index_beta_a == index_beta_b):
        print('WARNING, tried to double beta density between '+str(beta_a)+' and '+str(beta_b)+' but this failed because there is only one beta in this range. Please choose a larger window of betas.')
        return betas

    # iterate over the range adding in new temperatures
    for bb in range(index_beta_a,index_beta_b):
        betas = np.append( betas, ordered_betas[bb] + (ordered_betas[bb+1]-ordered_betas[bb])/2.0 )

    return betas

