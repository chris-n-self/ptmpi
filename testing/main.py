#!/bin/env python
#
# 31/01/16
# Chris Self

import sys
from mpi4py import MPI
# load run functions
import Parallel

# if the function is the main function ...
if __name__ == '__main__':

    # if running with MPI initialise the messaging passing evironment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # get code versions
    jsci_version = jsci.get_version_string()
    kithcmb_version = kithcmb.get_version_string()

    # set run parameters
    L = 4
    J = 1
    kappa = 0.1
    length_of_program = 1000

    output_name = 'test_process'+str(process_rank_)+'_L'+str(L)+'_betaset140216'

    # the list of quantities we will track
    # avg_quantities = [ 'energy','links','vortices','specific_heat','gap','participation_ratio','nu','thermal_correlation_matrix','thermal_current_matrix','pt_swap_success' ]
    avg_quantities = [ 'energy','links','vortices','specific_heat','gap','participation_ratio','pt_swap_success' ]
    
    # each process has a copy of the same temperature set
    beta_set = np.linspace(-2,2,8)
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 2.0,0 )
    # beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 2.0,0.8 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 2.0,1.8 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 2.0,1.9 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 1.2,0.8 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 1.2,1.05 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 0.6,0.3 )
