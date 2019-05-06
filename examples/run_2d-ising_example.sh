
# cleanup from any previous runs
rm -r *.txt
if [ -d output ]; then rm -r output; fi
mkdir output

# run mpiexec with 20 processes, executing python script '2d-ising.py'
mpiexec -n 20 python 2d-ising.py
