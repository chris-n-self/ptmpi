#!/bin/env python
#
# 31/01/16
# Chris Self

import sys
import time
import numpy as np
from mpi4py import MPI
# import ptmpi packages
import ptmpi
from ptmpi import PtMPI
from ptmpi import PtFunctions
# import Kitaev Honeycomb package
import kithcmb
from kithcmb import ThermalVortexSectors as vs
# import jsci, CT's enhanced json stream write package
import jsci
from jsci import WriteStream as jsciwrite
from jsci import Coding as jscicoding

# if the function is the main function ...
if __name__ == '__main__':

    # if running with MPI initialise the messaging passing evironment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # get code versions
    jsci_version = jsci.get_version_string()
    kithcmb_version = kithcmb.get_version_string()
    ptmpi_version = ptmpi.get_version_string()

    # set run parameters
    L = 4
    J = 1
    kappa = 0.1
    length_of_program = 5000
    output_name = 'test_process'+str(rank)+'_L'+str(L)+'_betaset140216'

    # the list of quantities we will track
    # avg_quantities = [ 'energy','links','vortices','specific_heat','gap','participation_ratio','nu','thermal_correlation_matrix','thermal_current_matrix','pt_swap_success' ]
    avg_quantities = [ 'energy','links','vortices','specific_heat','gap','participation_ratio','pt_swap_success' ]
    
    # each process has a copy of the same temperature set
    beta_set = np.linspace(-2,2,8)
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 2.0,0 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 2.0,1.8 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 2.0,1.9 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 1.2,0.8 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 1.2,1.05 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 0.6,0.3 )
    # order the temperatures it and raise them to two powers of 10
    betas = [ -1.*10**bb for bb in beta_set]
    betas = np.sort(betas)

    # throw an error if the number of pt chains is different from the number of
    # processes
    number_pt_chains = len(beta_set)
    if (comm.Get_size() != number_pt_chains):
        print('ERROR, the number of processes is different from the number of temperatures')
        print('ABORTING SIMULATION')
        sys.exit(0)

    # initialise ptmpi controller object
    mpi_pi_handler = PtMPI.PtMPI( comm,rank,length_of_program )

    # initialise this process's copy of the system
    print (str(rank)+': initialising copy of the system')
    kh_sys = vs.ThermalVortexSectors(L,J,kappa,betas[ mpi_pi_handler.get_current_temp_index() ])
    # begin in a random vortex sector rather than the ground state
    kh_sys.set_sign_disorder_random_vortex_configuration()

    # initialise round counter, parallel tempering swaps are the unit of our monte-carlo time
    # and each unit of time corresponds to a Metropolis sweep i.e. O(L^2) metropolis steps
    mc_time_step = 0
    print('BEGINNING SIMULATION in process '+str(rank))

    # time to start outputting data
    thermalisation_time = int(np.floor(length_of_program/5.0))

    # open output stream and add header information to the top of the file
    with open( output_name+'.json', 'w' ) as file:
        out = jsciwrite.FileWriteStream(file, indent=2)
        with out.wrap_object():
            out.write_pair('code_versions', { 'ptmpi':ptmpi_version, 'kithcmb':kithcmb_version, 'jsci':jsci_version } )
            out.write_pair('readme', "test")
            out.write_pair('specification', {'length_of_program':length_of_program,'L':L,'J':J,'kappa':kappa,'pt_chains':number_pt_chains})
            out.write_pair('quantities', avg_quantities )
            out.write_pair('betas', betas, jscicoding.NumericEncoder )
            out.write_pair('pt_swaps_normalisation', sum(mpi_pi_handler.pt_subsets) )
            out.write_key('time_series')
            
            # store start time
            start_time = time.time()
            start_block_time = time.time()

            with out.wrap_array():
                while ( mc_time_step < length_of_program ):
                    with out.wrap_object():
                        # store the beta_index the process was at
                        out.write_pair('beta_index',mpi_pi_handler.get_current_temp_index())

                        # print progress to stdout every X bins
                        progress_time_unit = int(np.floor(length_of_program/100.))
                        make_noise = ((rank==0) and (mc_time_step%progress_time_unit==0) and (mc_time_step>0))
                        if make_noise:
                            end_block_time = time.time()
                            time_per_block = (end_block_time-start_block_time)*1.
                            projected_total_runtime = (length_of_program / (1.*progress_time_unit)) * time_per_block
                            print('===========================')
                            print('process 0 at mc_time_step '+str(mc_time_step))
                            print('last block of '+str(progress_time_unit)+' bins took: '+str(time_per_block)+' seconds')
                            print('estimated total run-time:'+str(projected_total_runtime)+' seconds')
                            print('of which '+str(projected_total_runtime - (end_block_time - start_time))+' seconds remain')
                            start_block_time = time.time()

                        # do a sweep of updates in each chain
                        kh_sys.metropolis_sweep( betas[ mpi_pi_handler.get_current_temp_index() ] )
                        # print(str(rank)+': finished MC sweep')
                        
                        # output all the data from this time step to the stream
                        if make_noise:
                            output_start_time = time.time()
                            print('outputting data...')

                        for quant in avg_quantities:
                            try:
                                out.write_pair( quant, kh_sys.get_quantity(quant, betas[ mpi_pi_handler.get_current_temp_index() ]), jscicoding.NumericEncoder )
                            except vs.UnknownQuantity:

                                # want to do a type check here for if the system has the following methods
                                if quant is 'nu':
                                    # the real space chern number
                                    if (mc_time_step>thermalisation_time):
                                        out.write_pair( quant, kh_sys.get_real_space_chern_number(betas[ mpi_pi_handler.get_current_temp_index() ]), jscicoding.NumericEncoder )
                                elif quant is 'thermal_correlation_matrix':
                                    # the thermal correlation matrix
                                    # if kh_sys.correl_matrix is None:
                                        # kh_sys.compute_correlation_matrix(betas[ mpi_pi_handler.get_current_temp_index() ])
                                    # out.write_pair( quant, kh_sys.correl_matrix, jscicoding.NumericEncoder )
                                    if (mc_time_step>thermalisation_time):
                                        if kh_sys.correl_matrix is None:
                                            kh_sys.get_correlation_matrix(betas[ mpi_pi_handler.get_current_temp_index() ])
                                        thermal_correlation_matricies[ mpi_pi_handler.get_current_temp_index() ] += kh_sys.correl_matrix
                                elif quant is 'thermal_current_matrix':
                                    # the thermal current matrix
                                    # out.write_pair( quant, kh_sys.compute_thermal_current_matrix(betas[ mpi_pi_handler.get_current_temp_index() ]), jscicoding.NumericEncoder )
                                    if (mc_time_step>thermalisation_time):
                                        thermal_current_matricies[ mpi_pi_handler.get_current_temp_index() ] += kh_sys.get_thermal_current_matrix(betas[ mpi_pi_handler.get_current_temp_index() ])
                                elif quant is 'pt_swap_success':
                                    pass
                                else:
                                    print('WARNING avg_quantities contains an unknown quantity '+str(quant)+', skipping it')
                                    pass

                        if make_noise:
                            print('took '+str(time.time()-output_start_time)+' seconds.')

                        # parallel tempering swap step
                        curr_beta_index = mpi_pi_handler.get_current_temp_index()
                        alt_beta_index = mpi_pi_handler.get_alternative_temp_index()
                        success_flag = mpi_pi_handler.pt_step( kh_sys.get_log_partition_function( betas[curr_beta_index] ), kh_sys.get_log_partition_function( betas[alt_beta_index] ) )
                        if not success_flag is None:
                            out.write_pair( 'pt_swap_success', success_flag )

                    mc_time_step += 1

            # record the run time of the stage
            end_time = time.time()
            run_time = (end_time-start_time)
            
            # add final data to the output, the runtime
            out.write_pair( 'process_details', {'process_rank':rank,'run_time':run_time} )
                  