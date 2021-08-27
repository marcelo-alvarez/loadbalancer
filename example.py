from mpi4py import MPI
import numpy as np
import time
import numpy.random
from loadbalancer import loadbalancer

# this is the number of jobs the loadbalancer will run
Njobs = 200
group_size = 2

# set up a dummy work function that sleeps for jobtime*job_index seconds
# generally this function takes two arguments, an MPI communicator and
# the index of the job to do, where job_index = 0 ... Njobs-1
jobtime  = 0.001;
workfunc = lambda comm,job_index: time.sleep(((job_index+1)*jobtime))

# initialize the load balancer, passing it the work function
lb = loadbalancer(workfunc,comm=MPI.COMM_WORLD,Njobs=Njobs,group_size=group_size)

# get wall-clock
t0 = time.time()

# run the load balancer
lb.run();

dt=time.time()-t0

# report on wall-clock time vs expected optimal time
if(MPI.COMM_WORLD.rank==0):
    Ngroups = (MPI.COMM_WORLD.size-1) // group_size
    optimal_time = jobtime * Njobs * (1 + Njobs) / 2 / Ngroups
    print(Njobs,'jobs in ',Ngroups,'groups took',dt,'seconds')
    print('optimal time is',optimal_time)
