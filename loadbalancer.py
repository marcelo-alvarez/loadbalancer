from mpi4py import MPI
import numpy as np
import time
import numpy.random

class loadbalancer:

    def __init__(self, workfunc, **kwargs):

        self._workfunc = workfunc

        self.comm       = kwargs.get('comm',None)
        self.Njobs      = kwargs.get('Njobs',2)
        self.group_size = kwargs.get('group_size',1)

        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.active = True

        if self.group_size > self.size - 1:
            raise Exception("can't have group_size larger than world size - 1")

        self.Ngroups  = (self.size-1) // self.group_size
        self.group    = (self.rank-1) // self.group_size
        if self.rank == 0: self.group = -1

        if self.rank > self.group_size * self.Ngroups:
            self.active = False
            return

        self.groupcomm = self.comm.Split(color=self.group)
        self.grouprank = self.groupcomm.Get_rank()

        if self.group_size != self.groupcomm.Get_size() and self.rank != 0:
            print(self.rank,self.group_size,self.groupcomm.Get_size(),flush=True)
            raise Exception("can't have group_size != group_size")

        self.Nworkers = self.group_size - 1
        self.job_buff = np.zeros(1,dtype=np.int32)

    def _assign_job(self,worker,job):

        reqs = []
        for grouprank in range(self.group_size):
            destrank = worker * self.group_size + grouprank + 1
            self.job_buff[0] = job
            self.comm.Send(self.job_buff,dest=destrank)
            reqs.append(self.comm.Irecv(self.job_buff,source=destrank))
        return reqs

    def _dismiss_worker(self,worker):

        for grouprank in range(self.group_size):
            destrank = worker * self.group_size + grouprank + 1
            self.job_buff[0] = -1
            self.comm.Send(self.job_buff,dest=destrank)

    def _checkreqlist(self,reqs):

        for req in reqs:
            if not req.Test():
                return False
        return True

    def _schedule(self):

        # bookkeeping
        waitlist        = [] # message state of assignments
        assignment_list = [] # jobs assigned
        worker_groups   = [] # worker assigned

        # start by assigning Ngroups jobs to the rest of the groups
        nextjob=0
        for job in range(self.Ngroups):
            worker = nextjob
            reqs=self._assign_job(worker,nextjob)
            waitlist.append(reqs)
            assignment_list.append(job)
            worker_groups.append(worker)
            nextjob += 1

        # the scheduler waits for jobs to be completed
        # when one is complete it assigns the next job
        # until there are none left
        Ncomplete = 0
        while(Ncomplete < self.Njobs):
            for i in range(len(waitlist)):
                if self._checkreqlist(waitlist[i]):
                    worker = worker_groups[i]
                    job    = assignment_list[i]
                    Ncomplete += 1
                    waitlist.pop(i)
                    worker_groups.pop(i)
                    assignment_list.pop(i)
                    if Ncomplete < self.Njobs:
                        reqs=self._assign_job(worker,nextjob)
                        waitlist.append(reqs)
                        assignment_list.append(nextjob)
                        worker_groups.append(worker)
                        nextjob += 1
                    else:
                        for group in range(self.Ngroups): self._dismiss_worker(group)
                        return
                        break
        return

    def _work(self):

        while True:
            self.comm.Recv(self.job_buff,source=0); job = self.job_buff[0]
            if job < 0:
                return
            else:
                self._workfunc(self.groupcomm,job)
                self.comm.Isend(self.job_buff,dest=0)

    def run(self):
        if self.rank==0:
            self._schedule()
        else:
            self._work()
