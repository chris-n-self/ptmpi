# ptmpi
A python class that carries out an MPI implementation of parallel tempering. Communication costs is kept minimal by exchanging temperatures rather than the state of the modelled system

## Usage

[Parallel tempering](https://en.wikipedia.org/wiki/Parallel_tempering) is a monte-carlo method used to obtain equilibrium statistics for a physical system over a range of temperatures. When the energy landscape of the system is complex it can hugely speed up the convergence of ensemble averages, especially at low temperatures. It works by simulating *N* copies of the system (replicas) evolving independently at different temperatures [*T1*, *T2*, *T3*, ... ]. Periodically replicas at different temperatures are exchanged with some probability.

The main class (`ptmpi.swaphandler`) supports a fully parallelised implementation of parallel tempering using mpi4py (message passing interface for python). Each replica runs as a separate parallel process and they communicate via an mpi4py object. To minimise message passing the replicas stay in place and only the temperatures are exchanged between the processes at swaps. It is this exchange of temperatures that ptmpi handles. Another class (`ptmpi.filehandler`) provides a context manager that makes it easy for the processes to share a set of output files. This is useful since we will typically want the MCMC timeseries generated to be at fixed temperature.

The class is independent of the system being simulated or any details of the simulation. During regular MCMC update steps each replica queries ptmpi to obtain its current temperature index. At parallel tempering swap steps the replicas pass the information needed to decide whether a swap will occur to ptmpi.

### example code

A fully worked example, using the [two-dimensional Ising model](https://en.wikipedia.org/wiki/Ising_model), is given in the examples folder. It can be run using the shell script `run_2d-ising_example.sh` and the main python script is `2d-ising.py`. An annotated version of `2d-ising.py` is also provided, highlighting the important parts of the code.

## Setup

This package requires the python package `mpi4py` and an MPI library such as `MPICH`. These can both be installed through anaconda. If you are using pip instead then `MPICH` must be installed separately.