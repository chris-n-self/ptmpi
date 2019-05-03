
# cleanup from any previous runs
rm -r *.txt
cd output 
rm -r *.json
cd ..

# run mpiexec with 20 processes, executing python script '2d-ising.py'
mpiexec -n 20 python 2d-ising.py
