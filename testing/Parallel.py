# 23/02/16
# Chris Self

import sys
import json
import time
import numpy as np
from mpi4py import MPI
# import Kitaev Honeycomb package
import kithcmb
from kithcmb import ThermalVortexSectors as vs
# import jsci, CT's enhanced json stream write package
import jsci
from jsci import WriteStream as jsciwrite
from jsci import Coding as jscicoding
import PtFunctions

def runParallelMachinesStreamOutput( sys_size_, Tset_offset_, version_string_, readme_, comm_, process_rank_, kithcmb_version_=None, jsci_version_=None ):
    """
    the way the parallelisation works is that each process is a single copy of the system.
    They each have a process id (their rank) and a temperature id. When the simulation
    starts these are the same. At a PT exchange step, sets of two processes next to each other
    in TEMPERATURE exchange MPI messages allowing them to work out whether they will swap
    temperature or not. If they do the temperature id of the processes change. So that we can
    keep track of which process is next to which in temperature id every process has a pointer
    to the process id of the systems either side of it in temperature, these start out as just
    +/- 1 the rank. When two systems exchange they also send each other their old pointers so
    that each process can adjust its own.
    
    If for example we had initially had, (process id):(temp id)
    
    1:1 <-> 2:2 <-> 3:3 <-> 4:4 <-> 5:5   &   2,3 exchange temperature, this goes to:
    
    1:1 <-> 3:2 <-> 2:3 <-> 4:4 <-> 5:5   but we want to keep the processes (copies of sys) in
                                          place so what we actually do is:
    1:1 <---------> 3:2 <-
                          |
         -----------------
        |
         -> 2:3 <---------> 4:4 <-> 5:5   the processes stay in place and all we do is
                                          adjust pointers
    
    One tricky detail is that besides the communication between the swap pair that is already
    taking place we must notify the nearest neighbours to the pair (in this case 1 & 4) that
    their pointers need to be updated. This is the synchronisation step. It only need to be
    done when the swap subset changes.
    """
    # on the cluster I had problems with getting the versions at runtime so I now pass them as arguments, if
    # they have not been passed then we get them here
    if jsci_version_ is None:
        jsci_version_ = jsci.get_version_string()
    if kithcmb_version_ is None:
        kithcmb_version_ = kithcmb.get_version_string()

    # set run parameters
    L = int(sys_size_)
    J = 1
    kappa = 0.1
    TSet_offset = float(Tset_offset_)
    length_of_program = 50000

    output_name = 'cluster_torus_negativetemp_process'+str(process_rank_)+'_L'+str(L)+'_betaset140216'

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
    
    # (optionally) offset the temperature set
    for bi in range(len(beta_set)):
        beta_set[bi] = beta_set[bi] + TSet_offset
    
    # order the temperatures it and raise them to two powers of 10
    betas = [ -1.*10**bb for bb in beta_set]
    betas = np.sort(betas)
    number_pt_chains = len(beta_set)
    
    # throw an error if the number of pt chains is different from the number of
    # processes
    if (comm_.Get_size() != number_pt_chains):
        print('ERROR, the number of processes is different from the number of temperatures')
        print('ABORTING SIMULATION')
        sys.exit(0)

    # the initial beta index of each process is just equal to its process rank
    beta_index = process_rank_

    # process 0 pre-generates the list of which random subset the pt exchanges occur
    # within during each round, these are then broadcasted to all the other processes.
    if (process_rank_==0):
        pt_subsets = np.random.randint(2,size=length_of_program)
        # print(pt_subsets)
    else:
        pt_subsets = np.empty(length_of_program, dtype=np.int64)
    comm_.Bcast(pt_subsets, root=0)

    # each process has two pointers pointing to the processes running  the
    # temperature directly above and below this process. This is used to
    # identify proper pairs in the PT rounds
    # We use -1 to indicate that the process is at an end of the chain
    up_pointer = int(process_rank_+1)
    if (up_pointer==number_pt_chains):
        up_pointer = -1
    down_pointer = int(process_rank_-1)

    #    if (beta_index%2==pt_subsets[0]):
    #        print(str(process_rank_)+': points '+str(down_pointer)+'<- [->'+str(up_pointer)+']')
    #    else:
    #        print(str(process_rank_)+': points ['+str(down_pointer)+'<-] ->'+str(up_pointer))

    # initialise this process's copy of the system
    print (str(process_rank_)+': initialising copy of the system')
    kh_sys = vs.ThermalVortexSectors(L,J,kappa,betas[beta_index])
    # begin in a random vortex sector rather than the ground state
    kh_sys.set_sign_disorder_random_vortex_configuration()

    # initialise round counter, parallel tempering swaps are the unit of our monte-carlo time
    # and each unit of time corresponds to a Metropolis sweep i.e. O(L^2) metropolis steps
    mc_time_step = 0
    print('BEGINNING SIMULATION in process '+str(process_rank_))

    # initialise sync step pointers
    if (beta_index%2==pt_subsets[mc_time_step]):
        # points down to the top of the pair below
        sync_step_pointer = down_pointer
        sync_pointer_direction = 0
    else:
        # points up to the bottom of the pair above
        sync_step_pointer = up_pointer
        sync_pointer_direction = 1

    # due to practical limitations with processing the data its not possible to stream all the matricies
    # (correlation matrix) and (current matrix) into output file so we obtain the averages of them at each
    # temperature as we go and output them at the end. Each process has an array of each type of matricies
    # that is (number_pt_chains) deep and includes its contributions to the averages at that temperature.
    # We also count the number of times each process was at each temperature in another array to be able
    # to correctly weight the average later. 
    # Unfortunately since we are now collecting averages rather than the time series this require us to
    # separate the algorithm into thermalisation and sampling phases explicitly rather than selecting a 
    # cut-off in post-processing by observing the time series.
    thermalisation_time = int(np.floor(length_of_program/5.0))
    thermal_correlation_matricies = [ np.zeros((2*L**2,2*L**2), dtype=np.complex128) for i in range(number_pt_chains) ]
    thermal_current_matricies = [ np.zeros((2*L**2,2*L**2), dtype=np.complex128) for i in range(number_pt_chains) ]
    process_at_temp_counts = [ 0 for i in range(number_pt_chains) ]

    # open output stream and add header information to the top of the file
    with open( output_name+'.json', 'w' ) as file:
        out = jsciwrite.FileWriteStream(file, indent=2)
        with out.wrap_object():
            out.write_pair('code_versions', { 'pt':version_string_, 'kithcmb':kithcmb_version_, 'jsci':jsci_version_ } )
            out.write_pair('readme', readme_)
            out.write_pair('specification', {'length_of_program':length_of_program,'L':L,'J':J,'kappa':kappa,'pt_chains':number_pt_chains})
            out.write_pair('quantities', avg_quantities )
            out.write_pair('betas', betas, jscicoding.NumericEncoder )
            out.write_pair('pt_swaps_normalisation', sum(pt_subsets) )
            out.write_key('time_series')
            
            # store start time
            start_time = time.time()
            start_block_time = time.time()

            with out.wrap_array():
                while True:
                
                    # into the time-series we output a dictionary for this process's data from the MC time step
                    # out.write_key(str(mc_time_step))

                    with out.wrap_object():
                        # store the beta_index the process was at
                        out.write_pair('beta_index',beta_index)

                        # print progress to stdout every X bins
                        progress_time_unit = int(np.floor(length_of_program/100.))
                        make_noise = ((process_rank_==0) and (mc_time_step%progress_time_unit==0) and (mc_time_step>0))
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
                        kh_sys.metropolis_sweep( betas[beta_index] )
                        # print(str(process_rank_)+': finished MC sweep')
                        
                        # output all the data from this time step to the stream
                        if make_noise:
                            output_start_time = time.time()
                            print('outputting data...')

                        # (in sampling phase) count the number of times each process is at given temp
                        if (mc_time_step>thermalisation_time):
                            process_at_temp_counts[beta_index] += 1

                        for quant in avg_quantities:
                            try:
                                out.write_pair( quant, kh_sys.get_quantity(quant, betas[beta_index]), jscicoding.NumericEncoder )
                            except vs.UnknownQuantity:

                                # want to do a type check here for if the system has the following methods
                                if quant is 'nu':
                                    # the real space chern number
                                    if (mc_time_step>thermalisation_time):
                                        out.write_pair( quant, kh_sys.get_real_space_chern_number(betas[beta_index]), jscicoding.NumericEncoder )
                                elif quant is 'thermal_correlation_matrix':
                                    # the thermal correlation matrix
                                    # if kh_sys.correl_matrix is None:
                                        # kh_sys.compute_correlation_matrix(betas[beta_index])
                                    # out.write_pair( quant, kh_sys.correl_matrix, jscicoding.NumericEncoder )
                                    if (mc_time_step>thermalisation_time):
                                        if kh_sys.correl_matrix is None:
                                            kh_sys.get_correlation_matrix(betas[beta_index])
                                        thermal_correlation_matricies[beta_index] += kh_sys.correl_matrix
                                elif quant is 'thermal_current_matrix':
                                    # the thermal current matrix
                                    # out.write_pair( quant, kh_sys.compute_thermal_current_matrix(betas[beta_index]), jscicoding.NumericEncoder )
                                    if (mc_time_step>thermalisation_time):
                                        thermal_current_matricies[beta_index] += kh_sys.get_thermal_current_matrix(betas[beta_index])
                                elif quant is 'pt_swap_success':
                                    pass
                                else:
                                    print('WARNING avg_quantities contains an unknown quantity '+str(quant)+', skipping it')
                                    pass

                        if make_noise:
                            print('took '+str(time.time()-output_start_time)+' seconds.')

                        # parallel tempering swap step
                        # ============================
                        # the process that handles the decision making is always the one at lower temperature.
                        # The outline of the messaging scheme is:
                        #   T+1 -> T: its log-part func evaluated at T+1 and T (F22 and F21) + its pointers
                        #       T makes swap decision
                        #           T -> T+1: swap decison + its pointers
                        # if the swap was accepted each process uses the pointers it recieved from the other
                        # to update its neighbours on the temperature ladder
                        #
                        # I use the MPI tags purely to indicate (here in the code) which are matching messages
                            
                        # the value in the pt_subsets array indicates whether the even or odd subsets
                        # control the swap, i.e. if the exchanges are between even<->(even+1) or
                        # odd<->(odd+1).
                        #
                        # The systems at the end of the temperature chain sit out if they have no swap
                        # partner, indicated by the relevant pointer being (-1)
                        
                        # SYNC STEP
                        # =========
                        # detect a change in the pt subset
                        if (mc_time_step>0 and pt_subsets[mc_time_step]!=pt_subsets[mc_time_step-1]):
                            # print(str(process_rank_)+': beginning PT sync round')
                        
                            # handle the processes that began the last pt subset at the bottom of the pair, i.e. that are syncing
                            # with the pair below
                            if sync_pointer_direction==0:
                        
                                # identify if during the last pt subset this process instead ended up on the
                                # top of the pair
                                process_has_swapped_during_last_subset = (beta_index%2!=pt_subsets[mc_time_step-1])
                        
                                # if the sync step pointer is -1 this indicates the process is at the lowest temperature and has no
                                # pair to sync with below, so nothing needs to be done
                                if (sync_step_pointer!=-1):
                            
                                    # wait for a signal from the process at the end of the sync pointer
                                    # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for info from '+str(sync_step_pointer))
                                    incoming_data = np.empty(1, dtype=np.int64)
                                    comm_.Recv( incoming_data, source=sync_step_pointer, tag=14 )
                            
                                    # *** -1 flags that the that process that sent the signal is still at the top of the pair below ***
                                    # any other value tells us the process rank of the process that is now instead at the top
                                    new_top_process_subset_below = incoming_data[0]
                            
                                    # send a message back encoding whether or not this process is still at the bottom
                                    # of the pair, if it is not send the process rank of the process that now is
                                    # print(str(process_rank_)+': at temp '+str(beta_index)+' sending info to '+str(sync_step_pointer))
                                    if process_has_swapped_during_last_subset:
                                        outgoing_data = np.array([down_pointer])
                                    else:
                                        outgoing_data = np.array([-1])
                                    comm_.Send( outgoing_data, dest=sync_step_pointer, tag=16 )
                            
                                else:
                                    new_top_process_subset_below=-1
                        
                                # if this process ended up on the top of the pair we need to sync within the pair
                                if process_has_swapped_during_last_subset:
                        
                                    # wait for a message from the process partner telling this process what
                                    # its up-pointer should be pointing to
                                    # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for info from '+str(down_pointer))
                                    incoming_data = np.empty(1, dtype=np.int64)
                                    comm_.Recv( incoming_data, source=down_pointer, tag=18 )
                                    if incoming_data[0]!=-1:
                                        up_pointer = incoming_data[0]
                        
                                    # send a message to the processes partner in the pair containing what its
                                    # down-pointer should be pointing to
                                    # print(str(process_rank_)+': at temp '+str(beta_index)+' sending info to '+str(down_pointer))
                                    outgoing_data = np.array([new_top_process_subset_below])
                                    comm_.Send( outgoing_data, dest=down_pointer, tag=20 )
                        
                                # else this process just has to update its down pointer
                                else:
                                    if new_top_process_subset_below!=-1:
                                        down_pointer = new_top_process_subset_below
                        
                            # handle the processes that began the last pt subset at the top of the pair, i.e. that are syncing
                            # with the pair above
                            else:
                        
                                # identify if during the last pt subset this process instead ended up on the
                                # bottom of the pair
                                process_has_swapped_during_last_subset = (beta_index%2==pt_subsets[mc_time_step-1])
                        
                                # if the sync step pointer is -1 this indicates the process is at the highest temperature and has no
                                # pair to sync with above, so nothing needs to be done
                                if (sync_step_pointer!=-1):
                        
                                    # send a message to the process at the other end of the sync pointer telling it
                                    # whether this process is still on the top of the pair or not
                                    # print(str(process_rank_)+': at temp '+str(beta_index)+' sending info to '+str(sync_step_pointer))
                                    if process_has_swapped_during_last_subset:
                                        outgoing_data = np.array([up_pointer])
                                    else:
                                        outgoing_data = np.array([-1])
                                    comm_.Send( outgoing_data, dest=sync_step_pointer, tag=14 )
                            
                                    # wait for a message back telling us whether the process we sent the message to
                                    # is still at the bottom of the pair or not
                                    # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for info from '+str(sync_step_pointer))
                                    incoming_data = np.empty(1, dtype=np.int64)
                                    comm_.Recv( incoming_data, source=sync_step_pointer, tag=16 )
                            
                                    # *** -1 flags that the that process that sent the signal is still at the bottom of the pair above ***
                                    # any other value tells us the process rank of the process that is now instead at the bottom
                                    new_bottom_process_subset_above = incoming_data[0]
                            
                                else:
                                    new_bottom_process_subset_above = -1
                        
                                # if this process ended up on the bottom of the pair we need to sync within the pair
                                if process_has_swapped_during_last_subset:
                        
                                    # send a message to the processes partner in the pair containing what its
                                    # up-pointer should be pointing to
                                    # print(str(process_rank_)+': at temp '+str(beta_index)+' sending info to '+str(down_pointer))
                                    outgoing_data = np.array([new_bottom_process_subset_above])
                                    comm_.Send( outgoing_data, dest=up_pointer, tag=18 )
                        
                                    # wait for a message from the process partner telling this process what
                                    # its down-pointer should be pointing to
                                    # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for info from '+str(down_pointer))
                                    incoming_data = np.empty(1, dtype=np.int64)
                                    comm_.Recv( incoming_data, source=up_pointer, tag=20 )
                                    if incoming_data[0]!=-1:
                                        down_pointer = incoming_data[0]
                                
                                # else this process just has to update its up-pointer
                                else:
                                    if new_bottom_process_subset_above!=-1:
                                        up_pointer = new_bottom_process_subset_above
                        
                            # set sync pointers for next sync round
                            if (beta_index%2==pt_subsets[mc_time_step]):
                                # print(str(process_rank_)+': points '+str(down_pointer)+'<- [->'+str(up_pointer)+']')
                                
                                # points down to the top of the pair below
                                sync_step_pointer = down_pointer
                                sync_pointer_direction = 0
                            else:
                                # print(str(process_rank_)+': points ['+str(down_pointer)+'<-] ->'+str(up_pointer))

                                # points up to the bottom of the pair above
                                sync_step_pointer = up_pointer
                                sync_pointer_direction = 1
                        
                        # PT SWAPS
                        # ========
                        # print(str(process_rank_)+': beginning PT swap round')
                        if ((beta_index%2==pt_subsets[mc_time_step]) and (up_pointer!=-1)):
                        
                            # controller chain at T recieves data from T+1 and makes a decision
                            incoming_data = np.empty(4, dtype=np.float64)
                            # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for info from '+str(up_pointer)+' at temp '+str(beta_index+1))
                            comm_.Recv( incoming_data, source=up_pointer, tag=10 )
                            # print(str(process_rank_)+': at temp '+str(beta_index)+' recieved info from '+str(up_pointer))

                            # unpack incoming data
                            F_22 = incoming_data[0]
                            F_21 = incoming_data[1]
                            TplusOne_up_pointer = int(incoming_data[2])
                            TplusOne_down_pointer = int(incoming_data[3])

                            # get this processes log-partition functions for pt comparisons
                            F_11 = kh_sys.get_log_partition_function( betas[beta_index] )
                            F_12 = kh_sys.get_log_partition_function( betas[beta_index+1] )

                            # decide whether to make pt switch
                            pt_switch_decision = int(PtFunctions.decide_pt_switch_parallel( F_11, F_22, F_12, F_21 ))

                            # send decision to paired process
                            sending_data = np.array([ pt_switch_decision, up_pointer, down_pointer ])
                            # print(str(process_rank_)+': at temp '+str(beta_index)+' sending decision to '+str(up_pointer)+' at temp '+str(beta_index+1))
                            comm_.Send( sending_data, dest=up_pointer, tag=12 )

                            # if swap was accepted handle change
                            if ( pt_switch_decision==1):
                                # print(str(process_rank_)+': changing temp from beta_index='+str(beta_index)+' to beta_index='+str(beta_index+1))
                                
                                # record accepted swap
                                out.write_pair( 'pt_swap_success', 1 )
                                
                                # increase temperature index from T->T+1
                                beta_index += 1
                                
                                # update pointers
                                down_pointer = up_pointer
                                up_pointer = TplusOne_up_pointer
                                # print(str(process_rank_)+': points ['+str(down_pointer)+'<-] ->'+str(up_pointer))

                            else:

                                # record rejected swap
                                out.write_pair( 'pt_swap_success', 0 )

                        elif ((beta_index%2!=pt_subsets[mc_time_step]) and (down_pointer!=-1)):
                            # non-controller chain at T+1 sends data to T and waits for decision
                            
                            # pack data for sending to T-1, this is [F_22, F_21, up_pointer, down_pointer]
                            sending_data = np.array([ kh_sys.get_log_partition_function( betas[beta_index] ), kh_sys.get_log_partition_function( betas[beta_index-1] ), np.float64(up_pointer), np.float64(down_pointer) ])
                            # print(str(process_rank_)+': at temp '+str(beta_index)+' sending info to '+str(down_pointer)+' at temp '+str(beta_index-1))
                            comm_.Send( sending_data, dest=down_pointer, tag=10 )

                            # wait for decision data
                            decision_data = np.empty(3, dtype=np.int64)
                            # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for decision from '+str(down_pointer)+' at temp '+str(beta_index-1))
                            comm_.Recv( decision_data, source=down_pointer, tag=12 )
                            # print(str(process_rank_)+': at temp '+str(beta_index)+' recieved decision from '+str(down_pointer))
                                
                            # unpack decision data
                            pt_switch_decision = decision_data[0]
                            TminusOne_up_pointer = decision_data[1]
                            TminusOne_down_pointer = decision_data[2]
                                
                            # if swap was accepted handle change
                            if ( pt_switch_decision==1 ):
                                # print(str(process_rank_)+': changing temp from beta_index='+str(beta_index)+' to beta_index='+str(beta_index-1))
                                
                                # decrease temperature index from T+1->T
                                beta_index -= 1
                                
                                # update pointers
                                up_pointer = down_pointer
                                down_pointer = TminusOne_down_pointer
                                # print(str(process_rank_)+': points '+str(down_pointer)+'<- [->'+str(up_pointer)+']')

                        # test for convergence, if the condition is not met go onto the next bin
                        # if has_converged( stage, deltas[stage], averages[stage] ):
                        if (mc_time_step>(length_of_program-2)):
                            break
                        else:
                            mc_time_step += 1

            # record the run time of the stage
            end_time = time.time()
            run_time = (end_time-start_time)

            # output the average thermal correlations
            out.write_key( 'mean_thermal_correlation' )
            with out.wrap_array():
                for correl in thermal_correlation_matricies:
                    out.write_value( correl, cls=jscicoding.NumericEncoder )

            # output the average thermal currents
            out.write_key( 'mean_thermal_currents' )
            with out.wrap_array():
                for curr in thermal_current_matricies:
                    out.write_value( curr, cls=jscicoding.NumericEncoder )

            # output the process temperature counts
            out.write_pair( 'process_at_temp_counts', process_at_temp_counts )
            
            # add final data to the output, the runtime
            out.write_pair( 'process_details', {'process_rank':process_rank_,'run_time':run_time} )

def runParallelMachines( comm_, process_rank_ ):
    """
    the way the parallelisation works is that each process is a single copy of the system.
    They each have a process id (their rank) and a temperature id. When the simulation
    starts these are the same. At a PT exchange step, sets of two processes next to each other
    in TEMPERATURE exchange MPI messages allowing them to work out whether they will swap
    temperature or not. If they do the temperature id of the processes change. So that we can
    keep track of which process is next to which in temperature id every process has a pointer
    to the process id of the systems either side of it in temperature, these start out as just
    +/- 1 the rank. When two systems exchange they also send each other their old pointers so
    that each process can adjust its own.
    
    If for example we had initially had, (process id):(temp id)
    
    1:1 <-> 2:2 <-> 3:3 <-> 4:4 <-> 5:5   &   2,3 exchange temperature, this goes to:
    
    1:1 <-> 3:2 <-> 2:3 <-> 4:4 <-> 5:5   but we want to keep the processes (copies of sys) in
                                          place so what we actually do is:
    1:1 <---------> 3:2 <-
                          |
         -----------------
        |
         -> 2:3 <---------> 4:4 <-> 5:5   the processes stay in place and all we do is
                                          adjust pointers
    
    One tricky detail is that besides the communication between the swap pair that is already
    taking place we must notify the nearest neighbours to the pair (in this case 1 & 4) that
    their pointers need to be updated. This is the synchronisation step. It only need to be
    done when the swap subset changes.
    """
    # set run parameters
    L = 10
    J = 1
    kappa = 0.1
    output_name = 'cluster_cylinder_offset+0.1_process'+str(process_rank_)+'_L'+str(L)+'_betaset140216'
    length_of_program = 50000
    
    # the list of quantities we will track
    avg_quantities = [ 'energy','links','vortices','specific_heat','gap','participation_ratio','nu' ]
    
    # each process has a copy of the same temperature set
    beta_set = np.linspace(-2,2,8)
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 2.0,0 )
    # beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 2.0,0.8 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 2.0,1.8 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 2.0,1.9 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 1.2,0.8 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 1.2,1.05 )
    beta_set = PtFunctions.double_beta_density_between_a_b( beta_set, 0.6,0.3 )
    
    # (optionally) offset the temperature set
    offset = 0.1
    for bi in range(len(beta_set)):
        beta_set[bi] = beta_set[bi] + offset
    
    # order the temperatures it and raise them to two powers of 10
    betas = [10**bb for bb in beta_set]
    betas = np.sort(betas)
    number_pt_chains = len(beta_set)
    
    # throw an error if the number of pt chains is different from the number of
    # processes
    if (comm_.Get_size() != number_pt_chains):
        print('ERROR, the number of processes is different from the number of temperatures')
        print('ABORTING SIMULATION')
        sys.exit(0)
        
    # the initial beta index of each process is just equal to its process rank
    beta_index = process_rank_

    # process 0 pre-generates the list of which random subset the pt exchanges occur
    # within during each round, these are then broadcasted to all the other processes.
    if (process_rank_==0):
        pt_subsets = np.random.randint(2,size=length_of_program)
        # print(pt_subsets)
    else:
        pt_subsets = np.empty(length_of_program, dtype=np.int64)
    comm_.Bcast(pt_subsets, root=0)

    # each process has two pointers pointing to the processes running  the
    # temperature directly above and below this process. This is used to
    # identify proper pairs in the PT rounds
    # We use -1 to indicate that the process is at an end of the chain
    up_pointer = int(process_rank_+1)
    if (up_pointer==number_pt_chains):
        up_pointer = -1
    down_pointer = int(process_rank_-1)

    #    if (beta_index%2==pt_subsets[0]):
    #        print(str(process_rank_)+': points '+str(down_pointer)+'<- [->'+str(up_pointer)+']')
    #    else:
    #        print(str(process_rank_)+': points ['+str(down_pointer)+'<-] ->'+str(up_pointer))

    # initialise this process's copy of the system
    print (str(process_rank_)+': initialising copy of the system')
    kh_sys = vs.ThermalVortexSectorRealSpaceCorrelations(L,J,kappa)
    kh_sys.log_partition_func = kh_sys.get_log_partition_function( betas[beta_index] )

    # parallel tempering rounds are taken as the base unit of simulation time.
    # mt_sweeps_per_ptround sets the number of MC sweeps per parallel tempering
    # round; if it is set to 1 there is no averaging and the data output will
    # purely be the time series of the values after each sweep operation, if
    # it is >1 the data output is the course-grained time series of the average values
    # of the quantity for that number of sweeps between each pt
    mt_sweeps_per_ptround = 1

    # initialise round counter
    mc_time_step = 0
    print('BEGINNING SIMULATION in process '+str(process_rank_))

    # initialise sync step pointers
    if (beta_index%2==pt_subsets[mc_time_step]):
        # points down to the top of the pair below
        sync_step_pointer = down_pointer
        sync_pointer_direction = 0
    else:
        # points up to the bottom of the pair above
        sync_step_pointer = up_pointer
        sync_pointer_direction = 1

    # initialise the data containers for this process
    time_series = dict( zip(avg_quantities,[{0: {beta_index : 0.0} } for l in range(len(avg_quantities))]) )
    pt_swaps = dict(zip( range(number_pt_chains),[0 for i in range(number_pt_chains)] ))
    
    # store start time
    start_time = time.time()
    start_block_time = time.time()

    while True:
        announcement_time_unit = 500
        if ((process_rank_==0) and (mc_time_step%announcement_time_unit==0) and (mc_time_step>0)):
            end_block_time = time.time()
            time_per_bin = (end_block_time-start_time)/(mc_time_step+1.0)
            print('process 0 at mc_time_step '+str(mc_time_step))
            print('last block of '+str(announcement_time_unit)+' bins took: '+str(end_block_time - start_block_time)+' seconds')
            print('projected total run-time:'+str(length_of_program * time_per_bin)+' seconds')
            print('of which '+str(length_of_program * time_per_bin - (end_block_time - start_time))+' seconds remains')
            start_block_time = time.time()

        # do a sweep of updates in each chain
        # if mt_sweeps_per_ptround>1 the stored value is averaged
        for mtstep in range(mt_sweeps_per_ptround):
            kh_sys.metropolis_sweep( betas[beta_index] )
            # print(str(process_rank_)+': finished MC sweep')
            
            # update all the time series
            for quant in avg_quantities:
                try:
                    val = kh_sys.get_quantity( quant, betas[beta_index] )
                    time_series[quant][mc_time_step][beta_index] += val / (1.0*mt_sweeps_per_ptround)
                except vs.UnknownQuantity:
                    # the unknown quantity is the real space chern number
                    nu = kh_sys.get_real_space_chern_number( betas[beta_index] )
                    nu /= (1.0*mt_sweeps_per_ptround)
                    time_series['nu'][mc_time_step][beta_index] = [ float(np.real(nu)),float(np.imag(nu)) ]

        # parallel tempering swap step
        # ============================
        # the process that handles the decision making is always the one at lower temperature.
        # The outline of the messaging scheme is:
        #   T+1 -> T: its log-part func evaluated at T+1 and T (F22 and F21) + its pointers
        #       T makes swap decision
        #           T -> T+1: swap decison + its pointers
        # if the swap was accepted each process uses the pointers it recieved from the other
        # to update its neighbours on the temperature ladder
        #
        # I use the MPI tags purely to indicate (here in the code) which are matching messages
            
        # the value in the pt_subsets array indicates whether the even or odd subsets
        # control the swap, i.e. if the exchanges are between even<->(even+1) or
        # odd<->(odd+1).
        #
        # The systems at the end of the temperature chain sit out if they have no swap
        # partner, indicated by the relevant pointer being (-1)
        
        # SYNC STEP
        # =========
        # detect a change in the pt subset
        if (mc_time_step>0 and pt_subsets[mc_time_step]!=pt_subsets[mc_time_step-1]):
            # print(str(process_rank_)+': beginning PT sync round')
        
            # handle the processes that began the last pt subset at the bottom of the pair, i.e. that are syncing
            # with the pair below
            if sync_pointer_direction==0:
        
                # identify if during the last pt subset this process instead ended up on the
                # top of the pair
                process_has_swapped_during_last_subset = (beta_index%2!=pt_subsets[mc_time_step-1])
        
                # if the sync step pointer is -1 this indicates the process is at the lowest temperature and has no
                # pair to sync with below, so nothing needs to be done
                if (sync_step_pointer!=-1):
            
                    # wait for a signal from the process at the end of the sync pointer
                    # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for info from '+str(sync_step_pointer))
                    incoming_data = np.empty(1, dtype=np.int64)
                    comm_.Recv( incoming_data, source=sync_step_pointer, tag=14 )
            
                    # *** -1 flags that the that process that sent the signal is still at the top of the pair below ***
                    # any other value tells us the process rank of the process that is now instead at the top
                    new_top_process_subset_below = incoming_data[0]
            
                    # send a message back encoding whether or not this process is still at the bottom
                    # of the pair, if it is not send the process rank of the process that now is
                    # print(str(process_rank_)+': at temp '+str(beta_index)+' sending info to '+str(sync_step_pointer))
                    if process_has_swapped_during_last_subset:
                        outgoing_data = np.array([down_pointer])
                    else:
                        outgoing_data = np.array([-1])
                    comm_.Send( outgoing_data, dest=sync_step_pointer, tag=16 )
            
                else:
                    new_top_process_subset_below=-1
        
                # if this process ended up on the top of the pair we need to sync within the pair
                if process_has_swapped_during_last_subset:
        
                    # wait for a message from the process partner telling this process what
                    # its up-pointer should be pointing to
                    # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for info from '+str(down_pointer))
                    incoming_data = np.empty(1, dtype=np.int64)
                    comm_.Recv( incoming_data, source=down_pointer, tag=18 )
                    if incoming_data[0]!=-1:
                        up_pointer = incoming_data[0]
        
                    # send a message to the processes partner in the pair containing what its
                    # down-pointer should be pointing to
                    # print(str(process_rank_)+': at temp '+str(beta_index)+' sending info to '+str(down_pointer))
                    outgoing_data = np.array([new_top_process_subset_below])
                    comm_.Send( outgoing_data, dest=down_pointer, tag=20 )
        
                # else this process just has to update its down pointer
                else:
                    if new_top_process_subset_below!=-1:
                        down_pointer = new_top_process_subset_below
        
            # handle the processes that began the last pt subset at the top of the pair, i.e. that are syncing
            # with the pair above
            else:
        
                # identify if during the last pt subset this process instead ended up on the
                # bottom of the pair
                process_has_swapped_during_last_subset = (beta_index%2==pt_subsets[mc_time_step-1])
        
                # if the sync step pointer is -1 this indicates the process is at the highest temperature and has no
                # pair to sync with above, so nothing needs to be done
                if (sync_step_pointer!=-1):
        
                    # send a message to the process at the other end of the sync pointer telling it
                    # whether this process is still on the top of the pair or not
                    # print(str(process_rank_)+': at temp '+str(beta_index)+' sending info to '+str(sync_step_pointer))
                    if process_has_swapped_during_last_subset:
                        outgoing_data = np.array([up_pointer])
                    else:
                        outgoing_data = np.array([-1])
                    comm_.Send( outgoing_data, dest=sync_step_pointer, tag=14 )
            
                    # wait for a message back telling us whether the process we sent the message to
                    # is still at the bottom of the pair or not
                    # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for info from '+str(sync_step_pointer))
                    incoming_data = np.empty(1, dtype=np.int64)
                    comm_.Recv( incoming_data, source=sync_step_pointer, tag=16 )
            
                    # *** -1 flags that the that process that sent the signal is still at the bottom of the pair above ***
                    # any other value tells us the process rank of the process that is now instead at the bottom
                    new_bottom_process_subset_above = incoming_data[0]
            
                else:
                    new_bottom_process_subset_above = -1
        
                # if this process ended up on the bottom of the pair we need to sync within the pair
                if process_has_swapped_during_last_subset:
        
                    # send a message to the processes partner in the pair containing what its
                    # up-pointer should be pointing to
                    # print(str(process_rank_)+': at temp '+str(beta_index)+' sending info to '+str(down_pointer))
                    outgoing_data = np.array([new_bottom_process_subset_above])
                    comm_.Send( outgoing_data, dest=up_pointer, tag=18 )
        
                    # wait for a message from the process partner telling this process what
                    # its down-pointer should be pointing to
                    # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for info from '+str(down_pointer))
                    incoming_data = np.empty(1, dtype=np.int64)
                    comm_.Recv( incoming_data, source=up_pointer, tag=20 )
                    if incoming_data[0]!=-1:
                        down_pointer = incoming_data[0]
                
                # else this process just has to update its up-pointer
                else:
                    if new_bottom_process_subset_above!=-1:
                        up_pointer = new_bottom_process_subset_above
        
            # set sync pointers for next sync round
            if (beta_index%2==pt_subsets[mc_time_step]):
                # print(str(process_rank_)+': points '+str(down_pointer)+'<- [->'+str(up_pointer)+']')
                
                # points down to the top of the pair below
                sync_step_pointer = down_pointer
                sync_pointer_direction = 0
            else:
                # print(str(process_rank_)+': points ['+str(down_pointer)+'<-] ->'+str(up_pointer))

                # points up to the bottom of the pair above
                sync_step_pointer = up_pointer
                sync_pointer_direction = 1
        
        # PT SWAPS
        # ========
        # print(str(process_rank_)+': beginning PT swap round')
        if ((beta_index%2==pt_subsets[mc_time_step]) and (up_pointer!=-1)):
        
            # controller chain at T recieves data from T+1 and makes a decision
            incoming_data = np.empty(4, dtype=np.float64)
            # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for info from '+str(up_pointer)+' at temp '+str(beta_index+1))
            comm_.Recv( incoming_data, source=up_pointer, tag=10 )
            # print(str(process_rank_)+': at temp '+str(beta_index)+' recieved info from '+str(up_pointer))

            # unpack incoming data
            F_22 = incoming_data[0]
            F_21 = incoming_data[1]
            TplusOne_up_pointer = int(incoming_data[2])
            TplusOne_down_pointer = int(incoming_data[3])

            # get this processes log-partition functions for pt comparisons
            F_11 = kh_sys.get_log_partition_function( betas[beta_index] )
            F_12 = kh_sys.get_log_partition_function( betas[beta_index+1] )

            # decide whether to make pt switch
            pt_switch_decision = int(PtFunctions.decide_pt_switch_parallel( F_11, F_22, F_12, F_21 ))

            # send decision to paired process
            sending_data = np.array([ pt_switch_decision, up_pointer, down_pointer ])
            # print(str(process_rank_)+': at temp '+str(beta_index)+' sending decision to '+str(up_pointer)+' at temp '+str(beta_index+1))
            comm_.Send( sending_data, dest=up_pointer, tag=12 )

            # if swap was accepted handle change
            if ( pt_switch_decision==1):
                # print(str(process_rank_)+': changing temp from beta_index='+str(beta_index)+' to beta_index='+str(beta_index+1))
                
                # record accepted swap
                pt_swaps[beta_index] += 1
                
                # increase temperature index from T->T+1
                beta_index += 1
                
                # update pointers
                down_pointer = up_pointer
                up_pointer = TplusOne_up_pointer
                # print(str(process_rank_)+': points ['+str(down_pointer)+'<-] ->'+str(up_pointer))

        elif ((beta_index%2!=pt_subsets[mc_time_step]) and (down_pointer!=-1)):
            # non-controller chain at T+1 sends data to T and waits for decision
            
            # pack data for sending to T-1, this is [F_22, F_21, up_pointer, down_pointer]
            sending_data = np.array([ kh_sys.get_log_partition_function( betas[beta_index] ), kh_sys.get_log_partition_function( betas[beta_index-1] ), np.float64(up_pointer), np.float64(down_pointer) ])
            # print(str(process_rank_)+': at temp '+str(beta_index)+' sending info to '+str(down_pointer)+' at temp '+str(beta_index-1))
            comm_.Send( sending_data, dest=down_pointer, tag=10 )

            # wait for decision data
            decision_data = np.empty(3, dtype=np.int64)
            # print(str(process_rank_)+': at temp '+str(beta_index)+' waiting for decision from '+str(down_pointer)+' at temp '+str(beta_index-1))
            comm_.Recv( decision_data, source=down_pointer, tag=12 )
            # print(str(process_rank_)+': at temp '+str(beta_index)+' recieved decision from '+str(down_pointer))
                
            # unpack decision data
            pt_switch_decision = decision_data[0]
            TminusOne_up_pointer = decision_data[1]
            TminusOne_down_pointer = decision_data[2]
                
            # if swap was accepted handle change
            if ( pt_switch_decision==1 ):
                # print(str(process_rank_)+': changing temp from beta_index='+str(beta_index)+' to beta_index='+str(beta_index-1))
                
                # decrease temperature index from T+1->T
                beta_index -= 1
                
                # update pointers
                up_pointer = down_pointer
                down_pointer = TminusOne_down_pointer
                # print(str(process_rank_)+': points '+str(down_pointer)+'<- [->'+str(up_pointer)+']')

        # test for convergence, if the condition is not met go onto the next bin
        # if has_converged( stage, deltas[stage], averages[stage] ):
        if (mc_time_step>(length_of_program-2)):
            break
        else:
            mc_time_step += 1
            # print(str(process_rank_)+': at bin index '+str(mc_time_step))
            for quant in avg_quantities:
                time_series[quant][mc_time_step] = {beta_index : 0.0}

    # record the run time of the stage
    end_time = time.time()
    run_time = (end_time-start_time)
    
    # output data
    # We output all the data as JSON databases. This allows us to carry out whatever kind of analysis we want to do after with
    # minimal complexity. We outout variances, averages, and pt_swaps all in the same file, along with configuration data about
    # the run and any extra notes
    json_data = dict()
    json_data['process_details'] = {'process_rank':process_rank_,'run_time':run_time}
    json_data['pt_swaps'] = pt_swaps
    json_data['pt_swaps_normalisation'] = sum(pt_subsets)
    json_data['time_series'] = time_series
    json_data['quantities'] = avg_quantities
    json_data['betas'] = [float(b) for b in betas]
    json_data['specification'] = {'length_of_program':length_of_program,'L':L,'J':J,'kappa':kappa,'mt_sweeps_per_ptround':mt_sweeps_per_ptround,'pt_chains':number_pt_chains}
    json_data['readme'] = 'First test of the parallel code'
    
    with open( output_name+'.json', 'w') as fp:
        json.dump(json_data, fp)
        fp.close()
