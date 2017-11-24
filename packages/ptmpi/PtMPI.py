# 24/04/2016
# Chris Self

import PtFunctions
import numpy as np
from mpi4py import MPI

class PtMPI:
    """
    PARALLEL TEMPERING - MESSAGE PASSING INTERFACE class

    Arguments:
    ==========
    mpi_comm_ : MPI communication environment
    mpi_rank_ : process rank
    length_of_program_ : number of swap rounds that will be run

    Outline of program:
    ===================
    Each process is a single copy of the system. They each have a process id (their rank) and 
    a temperature id. When the simulation starts these are the same. At a pt exchange step, 
    sets of two processes next to each other in temperature exchange mpi messages allowing them
    to work out whether they will swap temperature or not. 

    When we make a SWAP we do not move the copies of the systems, instead we exchange the 
    temperature indexes. Pointers to other processes keep track of the processes that neighbour 
    this one in temperatures. At each swap the pointers are exchanged. If for example we had 
    initially had, (process id):(temp id)
    
    1:1 <-> 2:2 <-> 3:3 <-> 4:4 <-> 5:5   &  2,3 exchange temperature, this goes to:
    
    1:1 <-> 3:2 <-> 2:3 <-> 4:4 <-> 5:5   but we want to keep the processes (copies of sys)
                                          in place so what we actually do is:
    1:1 <---------> 3:2 <-
                          |               (to make sure that 1 also points to 3, rather than
         -----------------                continuing to point to 2, we also need to SYNC)
        |
         -> 2:3 <---------> 4:4 <-> 5:5   the processes stay in place and all we do is
                                          adjust pointers

    Each round of swaps we select a subset of pairs (1,2),(3,4),etc. or (2,3),(4,5),etc. These
    are chosen by a list of random numbers 0 or 1 generated at the beginning of the program. The
    process with rank 0 generates this list and broadcasts it to other processes. During the swap
    step only the paired processes communicate.

    When we switch between the subsets an additional SYNC step is needed. If in the next step
    in the example above the subset changes from (2,3),(4,5),etc. -> (1,2),(3,4),etc. then the
    up-pointer from 1 has to be synced with the pair (2,3). 

    Stored properties:
    ==================
    self.disable_swaps : set swap prob -> 0 to reject all swaps
    -----
    self.mpi_comm_world : MPI communication environment
    self.mpi_process_rank : process rank
    self.beta_index : index in the temperature set of this process's current temp
    -----
    (SWAP STEP VARIABLES)
    self.pt_subsets : list of binary values indicating how processes are paired for swaps
    self.mpi_process_up_pointer : process rank of the system at the temperature above
    self.mpi_process_down_pointer : process rank of the system with the temperature below
    self.prev_pt_subset : 
    -----
    (SYNC STEP VARIABLES)
    self.mpi_sync_step_pointer : pointer to the process this will communicate with in the next sync step
    self.mpi_sync_pointer_direction : information about whether this pointer points up or down

    """
    def __init__( self, mpi_comm_,mpi_rank_, length_of_program_, disable_swaps=False ):
        """
        """
        self.disable_swaps = disable_swaps

        # store the self.mpi_comm_world and the process rank
        self.mpi_comm_world = mpi_comm_
        self.mpi_process_rank = mpi_rank_

        # initialise all vars
        self.reset(length_of_program_)

    def reset( self,length_of_program_ ):
        """
        set all variables to initial state
        """
        # store the current temperature of this process as its position in the list of temperatures
        self.beta_index = self.mpi_process_rank

        """
        BROADCAST PT-SUBSETS LIST
        process 0 pre-generates the list of which random subset the pt exchanges occur
        within during each round, these are then broadcasted to all the other processes.
        """
        if ( self.mpi_process_rank == 0 ):
            self.pt_subsets = np.random.randint(2,size=length_of_program_)
            # turn into bool array
            #self.pt_subsets = np.logical_and(self.pt_subsets,self.pt_subsets) 
        else:
            self.pt_subsets = np.empty(length_of_program_,dtype=int)
            #self.pt_subsets = np.empty(length_of_program_,dtype=np.bool)
        self.mpi_comm_world.Bcast([self.pt_subsets,MPI.INT], root=0)
        #self.mpi_comm_world.Bcast([self.pt_subsets,MPI.BOOL], root=0)
        self.prev_pt_subset = -1
        # convert pt_subsets to regular array
        self.pt_subsets = [ rr for rr in self.pt_subsets ]
        #self.pt_subsets = [ int(rr) for rr in self.pt_subsets ]

        """
        INITIALISE POINTERS TO NEIGHBOURING PROCESSES
        each process has two pointers pointing to the processes running the
        temperature directly above and below this process. This is used to
        identify proper pairs in the PT rounds
        We use -1 to indicate that the process is at an end of the chain
        """
        self.mpi_process_up_pointer = int(self.mpi_process_rank)+1
        if ( self.mpi_process_up_pointer == int(self.mpi_comm_world.Get_size()) ):
            self.mpi_process_up_pointer = -1
        self.mpi_process_down_pointer = int(self.mpi_process_rank)-1

        # print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' initial self.mpi_process_up_pointer ', self.mpi_process_up_pointer
        # print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' initial self.mpi_process_down_pointer ', self.mpi_process_down_pointer

        """
        INITIALISE SYNC STEP POINTERS
        """
        if (self.beta_index%2 == self.pt_subsets[-1]):
            # points down to the top of the pair below
            self.mpi_sync_step_pointer = self.mpi_process_down_pointer
            self.mpi_sync_pointer_direction = 0
        else:
            # points up to the bottom of the pair above
            self.mpi_sync_step_pointer = self.mpi_process_up_pointer
            self.mpi_sync_pointer_direction = 1

    def get_current_temp_index( self ):
        """ """
        return self.beta_index

    def get_alternative_temp_index( self ):
        """ """
        if (self.beta_index%2 == self.pt_subsets[-1]) and (not (self.mpi_process_up_pointer == -1)):
            return self.beta_index + 1
        elif (not ( self.beta_index%2 == self.pt_subsets[-1] )) and (not (self.mpi_process_down_pointer == -1)):
            return self.beta_index - 1
        else:
            return 0

    def pt_step( self, energy_, curr_temp_, alt_temp_ ):
        """ """
        # sync prcesses if needed
        self.pt_sync()

        # carry out swaps and return success flag
        return self.pt_swap( energy_, curr_temp_, alt_temp_ )

    def pt_sync( self ):
        """
        """
        #print self.mpi_process_rank,' : at sync'
        #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' self.prev_pt_subset ', self.prev_pt_subset
        #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' self.pt_subsets[-1] ', self.pt_subsets[-1]
        if (not ( self.prev_pt_subset == self.pt_subsets[-1] )) and (not ( self.prev_pt_subset == -1 )):
            #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' syncing'
            #print(str(self.mpi_process_rank)+': points '+str(self.mpi_process_down_pointer)+'<- [->'+str(self.mpi_process_up_pointer)+']')

            # handle the processes that began the last pt subset at the bottom of the pair, i.e. that are syncing
            # with the pair below
            #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' self.mpi_sync_pointer_direction ', self.mpi_sync_pointer_direction
            if self.mpi_sync_pointer_direction == 0:
        
                # identify if during the last pt subset this process instead ended up on the
                # top of the pair
                _process_has_swapped_during_last_subset = not ( self.beta_index%2 == self.prev_pt_subset )
        
                # if the sync step pointer is -1 this indicates the process is at the lowest temperature and has no
                # pair to sync with below, so nothing needs to be done
                #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' self.mpi_sync_step_pointer ', self.mpi_sync_step_pointer
                #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' ( self.mpi_sync_step_pointer == -1 ) ', ( self.mpi_sync_step_pointer == -1 )
                if not ( self.mpi_sync_step_pointer == -1 ):
            
                    # wait for a signal from the process at the end of the sync pointer
                    #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' waiting for info from '+str(self.mpi_sync_step_pointer))
                    incoming_data = np.empty(1, dtype=int)
                    self.mpi_comm_world.Recv( [incoming_data,MPI.INT], source=self.mpi_sync_step_pointer, tag=14 )
            
                    # *** -1 flags that the that process that sent the signal is still at the top of the pair below ***
                    # any other value tells us the process rank of the process that is now instead at the top
                    _new_top_process_subset_below = incoming_data[0]
            
                    # send a message back encoding whether or not this process is still at the bottom
                    # of the pair, if it is not send the process rank of the process that now is
                    #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' sending info to '+str(self.mpi_sync_step_pointer))
                    if _process_has_swapped_during_last_subset:
                        outgoing_data = np.array([ self.mpi_process_down_pointer ])
                    else:
                        outgoing_data = np.array([-1])
                    self.mpi_comm_world.Send( [outgoing_data,MPI.INT], dest=self.mpi_sync_step_pointer, tag=16 )
            
                else:
                    _new_top_process_subset_below = -1
        
                # if this process ended up on the top of the pair we need to sync within the pair
                if _process_has_swapped_during_last_subset:
        
                    # wait for a message from the process partner telling this process what
                    # its up-pointer should be pointing to
                    #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' waiting for info from partner '+str(self.mpi_process_down_pointer))
                    incoming_data = np.empty(1, dtype=int)
                    self.mpi_comm_world.Recv( [incoming_data,MPI.INT], source=self.mpi_process_down_pointer, tag=18 )
                    if incoming_data[0]!=-1:
                        self.mpi_process_up_pointer = incoming_data[0]
        
                    # send a message to the processes partner in the pair containing what its
                    # down-pointer should be pointing to
                    #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' sending info to partner '+str(self.mpi_process_down_pointer))
                    outgoing_data = np.array([_new_top_process_subset_below])
                    self.mpi_comm_world.Send( [outgoing_data,MPI.INT], dest=self.mpi_process_down_pointer, tag=20 )
        
                # else this process just has to update its down pointer
                else:
                    if _new_top_process_subset_below!=-1:
                        self.mpi_process_down_pointer = _new_top_process_subset_below
        
            # handle the processes that began the last pt subset at the top of the pair, i.e. that are syncing
            # with the pair above
            else:
        
                # identify if during the last pt subset this process instead ended up on the
                # bottom of the pair
                _process_has_swapped_during_last_subset = ( self.beta_index%2 == self.prev_pt_subset )
        
                # if the sync step pointer is -1 this indicates the process is at the highest temperature and has no
                # pair to sync with above, so nothing needs to be done
                if not ( self.mpi_sync_step_pointer == -1 ):
        
                    # send a message to the process at the other end of the sync pointer telling it
                    # whether this process is still on the top of the pair or not
                    #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' sending info to '+str(self.mpi_sync_step_pointer))
                    if _process_has_swapped_during_last_subset:
                        outgoing_data = np.array([ self.mpi_process_up_pointer ])
                    else:
                        outgoing_data = np.array([-1])
                    self.mpi_comm_world.Send( [outgoing_data,MPI.INT], dest=self.mpi_sync_step_pointer, tag=14 )
            
                    # wait for a message back telling us whether the process we sent the message to
                    # is still at the bottom of the pair or not
                    #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' waiting for info from '+str(self.mpi_sync_step_pointer))
                    incoming_data = np.empty(1, dtype=int)
                    self.mpi_comm_world.Recv( [incoming_data,MPI.INT], source=self.mpi_sync_step_pointer, tag=16 )
            
                    # *** -1 flags that the that process that sent the signal is still at the bottom of the pair above ***
                    # any other value tells us the process rank of the process that is now instead at the bottom
                    _new_bottom_process_subset_above = incoming_data[0]
            
                else:
                    _new_bottom_process_subset_above = -1
        
                # if this process ended up on the bottom of the pair we need to sync within the pair
                if _process_has_swapped_during_last_subset:
        
                    # send a message to the processes partner in the pair containing what its
                    # up-pointer should be pointing to
                    #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' sending info to partner '+str(self.mpi_process_down_pointer))
                    outgoing_data = np.array([_new_bottom_process_subset_above])
                    self.mpi_comm_world.Send( [outgoing_data,MPI.INT], dest=self.mpi_process_up_pointer, tag=18 )
        
                    # wait for a message from the process partner telling this process what
                    # its down-pointer should be pointing to
                    #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' waiting for info from partner '+str(self.mpi_process_down_pointer))
                    incoming_data = np.empty(1, dtype=int)
                    self.mpi_comm_world.Recv( [incoming_data,MPI.INT], source=self.mpi_process_up_pointer, tag=20 )
                    if not ( incoming_data[0] == -1 ):
                        self.mpi_process_down_pointer = incoming_data[0]
                
                # else this process just has to update its up-pointer
                else:
                    if _new_bottom_process_subset_above!=-1:
                        self.mpi_process_up_pointer = _new_bottom_process_subset_above
        
            # set sync pointers for next sync round
            if ( self.beta_index%2 == self.pt_subsets[-1] ):
                #print(str(self.mpi_process_rank)+': points '+str(self.mpi_process_down_pointer)+'<- [->'+str(self.mpi_process_up_pointer)+']')
                
                # points down to the top of the pair below
                self.mpi_sync_step_pointer = self.mpi_process_down_pointer
                self.mpi_sync_pointer_direction = 0
            else:
                #print(str(self.mpi_process_rank)+': points ['+str(self.mpi_process_down_pointer)+'<-] ->'+str(self.mpi_process_up_pointer))

                # points up to the bottom of the pair above
                self.mpi_sync_step_pointer = self.mpi_process_up_pointer
                self.mpi_sync_pointer_direction = 1

    def pt_swap( self, energy_, curr_temp_, alt_temp_ ):
        """
        """
        #print self.mpi_process_rank,' : at swap'
        _curr_pt_subset = self.pt_subsets.pop()
        self.prev_pt_subset = _curr_pt_subset

        #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' self.beta_index%2 ', self.beta_index%2
        #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' _curr_pt_subset ', _curr_pt_subset
        #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' self.mpi_process_up_pointer ', self.mpi_process_up_pointer
        #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' self.mpi_process_down_pointer ', self.mpi_process_down_pointer
        if (self.beta_index%2==_curr_pt_subset) and (not (self.mpi_process_up_pointer == -1)):
            #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' swapping'
                        
            # controller chain at T recieves data from T+1 and makes a decision
            incoming_data = np.empty(4, dtype=float)
            #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' waiting for info from '+str(self.mpi_process_up_pointer)+' at temp '+str(self.beta_index+1))
            try:
                self.mpi_comm_world.Recv( [incoming_data,MPI.FLOAT], source=self.mpi_process_up_pointer, tag=10 )
            except OverflowError:
                print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' incoming_data ', incoming_data
                print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' self.mpi_process_up_pointer ', self.mpi_process_up_pointer
            #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' recieved info from '+str(self.mpi_process_up_pointer))

            # unpack incoming data
            _F_22 = incoming_data[0]
            _F_21 = incoming_data[1]
            _TplusOne_up_pointer = int(incoming_data[2])
            _TplusOne_down_pointer = int(incoming_data[3])

            # get this processes log-partition functions for pt comparisons
            _F_11 = energy_*curr_temp_
            _F_12 = energy_*alt_temp_

            # decide whether to make pt switch
            if self.disable_swaps:
                _pt_switch_decision = 0
            else:
                _pt_switch_decision = int( PtFunctions.decide_pt_switch( _F_11, _F_22, _F_12, _F_21 ) )

            # send decision to paired process
            sending_data = np.array([ _pt_switch_decision, self.mpi_process_up_pointer, self.mpi_process_down_pointer ])
            #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' sending decision to '+str(self.mpi_process_up_pointer)+' at temp '+str(self.beta_index+1))
            self.mpi_comm_world.Send( [sending_data,MPI.INT], dest=self.mpi_process_up_pointer, tag=12 )

            # if swap was accepted handle change
            if ( _pt_switch_decision == 1 ):
                #print(str(self.mpi_process_rank)+': changing temp from self.beta_index='+str(self.beta_index)+' to self.beta_index='+str(self.beta_index+1))
                
                # increase temperature index from T->T+1
                self.beta_index += 1
                
                # update pointers
                self.mpi_process_down_pointer = self.mpi_process_up_pointer
                self.mpi_process_up_pointer = _TplusOne_up_pointer
                #print(str(self.mpi_process_rank)+': points ['+str(self.mpi_process_down_pointer)+'<-] ->'+str(self.mpi_process_up_pointer))

            return _pt_switch_decision

        elif (not ( self.beta_index%2 == _curr_pt_subset )) and (not ( self.mpi_process_down_pointer == -1 )):
            #print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' swapping'
            # non-controller chain at T+1 sends data to T and waits for decision
            
            # pack data for sending to T-1, this is [F_22, F_21, self.mpi_process_up_pointer, self.mpi_process_down_pointer]
            _F_22 = energy_*curr_temp_
            _F_21 = energy_*alt_temp_
            sending_data = np.array([ _F_22, _F_21, np.float64(self.mpi_process_up_pointer), np.float64(self.mpi_process_down_pointer) ])
            #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' sending info to '+str(self.mpi_process_down_pointer)+' at temp '+str(self.beta_index-1))
            try:
                self.mpi_comm_world.Send( [sending_data,MPI.FLOAT], dest=self.mpi_process_down_pointer, tag=10 )
            except OverflowError:
                print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' sending_data ', sending_data
                print str(self.mpi_process_rank),': at temp ',str(self.beta_index), ' self.mpi_process_down_pointer ', self.mpi_process_down_pointer


            # wait for decision data
            decision_data = np.empty(3, dtype=int)
            #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' waiting for decision from '+str(self.mpi_process_down_pointer)+' at temp '+str(self.beta_index-1))
            self.mpi_comm_world.Recv( [decision_data,MPI.INT], source=self.mpi_process_down_pointer, tag=12 )
            #print(str(self.mpi_process_rank)+': at temp '+str(self.beta_index)+' recieved decision from '+str(self.mpi_process_down_pointer))
                
            # unpack decision data
            _pt_switch_decision = decision_data[0]
            _TminusOne_up_pointer = decision_data[1]
            _TminusOne_down_pointer = decision_data[2]
                
            # if swap was accepted handle change
            if ( _pt_switch_decision==1 ):
                #print(str(self.mpi_process_rank)+': changing temp from self.beta_index='+str(self.beta_index)+' to self.beta_index='+str(self.beta_index-1))
                
                # decrease temperature index from T+1->T
                self.beta_index -= 1
                
                # update pointers
                self.mpi_process_up_pointer = self.mpi_process_down_pointer
                self.mpi_process_down_pointer = _TminusOne_down_pointer
                #print(str(self.mpi_process_rank)+': points '+str(self.mpi_process_down_pointer)+'<- [->'+str(self.mpi_process_up_pointer)+']')

        return None
