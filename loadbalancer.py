from mpi4py import MPI
import numpy as np
import time
import numpy.random

class loadbalancer:

    def __init__(self, workfunc, **kwargs):

        # user provided function that will do the work
        self._workfunc = workfunc

        self.comm       = kwargs.get('comm',None)
        self.Njobs      = kwargs.get('Njobs',2)
        self.group_size = kwargs.get('group_size',1)

        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # numpy array for sending and receiving job indices
        self.job_buff = np.zeros(1,dtype=np.int32)

        if self.group_size > self.size - 1:
            raise Exception("can't have group_size larger than world size - 1")

        self.Ngroups  = (self.size-1) // self.group_size
        self.group    = (self.rank-1) // self.group_size

        # don't do the split for rank=0; it only uses the global communicator
        if self.rank > self.group_size * self.Ngroups: return

        # split all ranks > 0 into new communicators of size group_size
        if self.rank == 0: self.group = -1
        self.groupcomm = self.comm.Split(color=self.group)
        self.grouprank = self.groupcomm.Get_rank()

        if self.group_size != self.groupcomm.Get_size() and self.rank != 0:
            print(self.rank,self.group_size,self.groupcomm.Get_size(),flush=True)
            raise Exception("can't have group_size != group_size")

    def _assign_job(self,worker,job):

        # assign job to all processes in group worker
        # and return list of handles corresponding to
        # confirmation of completion of current job
        reqs = []
        for grouprank in range(self.group_size):
            destrank = worker * self.group_size + grouprank + 1
            self.job_buff[0] = job
            self.comm.Send(self.job_buff,dest=destrank)
            reqs.append(self.comm.Irecv(self.job_buff,source=destrank))
        return reqs

    def _dismiss_worker(self,worker):

        # send job = -1 messages to all processes in group worker signaling
        # there is no more work to be done
        for grouprank in range(self.group_size):
            destrank = worker * self.group_size + grouprank + 1
            self.job_buff[0] = -1
            self.comm.Send(self.job_buff,dest=destrank)

    def _checkreqlist(self,reqs):

        # check if all processes with handles in reqs have reported back
        for req in reqs:
            if not req.Test():
                return False
        return True

    def _schedule(self):

        # bookkeeping
        waitlist        = [] # message handles for pending worker groups 
        worker_groups   = [] # worker assigned

        # start by assigning Ngroups jobs to each of the Ngroup groups
        nextjob=0
        for job in range(self.Ngroups):
            worker = nextjob
            reqs=self._assign_job(worker,nextjob)
            waitlist.append(reqs)
            worker_groups.append(worker)
            nextjob += 1

        # the scheduler waits for jobs to be completed;
        # when one is complete it assigns the next job
        # until there are none left
        Ncomplete = 0
        while(Ncomplete < self.Njobs):
            # iterate over list of currently pending group of processes
            for i in range(len(waitlist)):
                # check for completion of all processes in this worker group
                if self._checkreqlist(waitlist[i]):
                    # all ranks group doing job corresponding to place i in waitlist
                    # have returned; identify this worker group and remove it from the
                    # waitlist and worker list
                    worker = worker_groups[i]
                    Ncomplete += 1
                    waitlist.pop(i)
                    worker_groups.pop(i)
                    if Ncomplete < self.Njobs:
                        # more jobs to do; assign processes in group worker
                        # the job with index nextjob, increment, and beak to
                        # start the loop over the waitlist over again
                        reqs=self._assign_job(worker,nextjob)
                        waitlist.append(reqs)
                        worker_groups.append(worker)
                        nextjob += 1
                        break
                    else:
                        # no more jobs to do; dismiss all processes in all groups,
                        # causing all workers to return
                        for group in range(self.Ngroups): self._dismiss_worker(group)
                        return

        return

    def _work(self):

        # listen for job assignments from the scheduler
        while True:
            self.comm.Recv(self.job_buff,source=0); job = self.job_buff[0]
            if job < 0:
                # job < 0 means no more jobs to do; return
                return
            else:
                # received a message from scheduler with job >= 0
                # call work function using group communicator and
                # send non-blocking message on completion
                self._workfunc(self.groupcomm,job)
                self.comm.Isend(self.job_buff,dest=0)

    def run(self):

        # main function of class, run scheduler on rank = 0
        # and worker on all other ranks
        if self.rank==0:
            self._schedule()
        else:
            self._work()
