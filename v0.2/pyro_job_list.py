import Pyro4
import numpy as np
import uuid

class Job(object):
    def __init__(self,command=None,data=None):
        self.id = str(uuid.uuid4())
        self.command = command
        self.data = data
        self.status = "pending"
    def __str__(self):
        return str(self.command) + "(" + str(self.data) + "): " + str(self.status)

class PyroJobList(object):
    def __init__(self):
        self.keep_alive = True
        self.jobs = [Job('wait',n) for n in (np.random.randint(10,size=5)+10).tolist()]
        self.print_jobs()
    def print_jobs(self):
        print "listing ",len(self.jobs)," jobs:"
        for j in self.jobs:
            print "\t",str(j)
    def get_jobs(self,status="pending"):
        if status is None:
            return [j.id for j in self.jobs]
        return [j.id for j in self.jobs if j.status == status]
    def get_status_stats(self):
        statuses = [j.status.split(" ")[0] for j in self.jobs]
        return {s: len([stat for stat in statuses if stat == s]) for s in set(statuses)}
    def status(self,job,status):
        print "Job status: ",status
        for j in self.jobs:
            if j.id == job:
                j.status = status
                #if status == 'done':
                #    self.jobs.remove(j)
    def get_job_description(self,job):
        for j in self.jobs:
            if j.id == job:
                return { 'command': j.command, 'data': j.data }
        return None
    def keep_running(self):
        return len(self.get_jobs("done")) + len(self.get_jobs("failed")) < len(self.get_jobs(None)) or self.keep_alive

thing=PyroJobList()
daemon=Pyro4.Daemon()
nameserver = Pyro4.locateNS()
uri=daemon.register(thing)
nameserver.register("ILISTER#"+str(uri), uri, metadata={"silver_joblist"})
del nameserver
print(uri)
daemon.requestLoop(loopCondition=thing.keep_running)