import Pyro4
import time
from threading  import Thread, Event

def thread_f(assigned_job,assigned_job_list_uri,*args,**kwargs):
    with Pyro4.Proxy(assigned_job_list_uri ) as job_list:
        job_list.status(assigned_job,'running')
        job_description = job_list.get_job_description(assigned_job)
        if job_description is None:
            print "Thread: job_description is bad"
            return
        command, data = job_description["command"], job_description["data"]
    print "Thread: running job..."
    try:
        if command == "wait":
            for i in range(data):
                print ".",
                percent_done = round(100*float(i)/float(data))
                Pyro4.Proxy(assigned_job_list_uri ).status(assigned_job,'running '+str(percent_done)+'%')
                time.sleep(1.0)
        print " done."
        Pyro4.Proxy(assigned_job_list_uri ).status(assigned_job,'done')
    except:
        Pyro4.Proxy(assigned_job_list_uri ).status(assigned_job,'failed')
    print "Thread: done."

class PyroJobWorker(object):
    def __init__(self):
        self.jobs_done =[]
        self.thread = None
        self.assigned_job = None
    def idle(self):
        if self.thread is not None:
            if self.thread.is_alive():
                return False
            self.thread = None
        return True
    def free(self):
        return self.idle() and self.assigned_job == None
    def assign(self,job,job_list_uri):
        if not self.free():
            return False
        self.assigned_job = job
        self.assigned_job_list_uri = job_list_uri
        print "got assigned job!",job
        return True
    def run(self):
        if not self.idle() or self.assigned_job is None:
            return False
        self.jobs_done.append(self.assigned_job)
        print "Starting thread..."
        self.thread = Thread(target = thread_f, args=(self.assigned_job,self.assigned_job_list_uri,))
        self.thread.daemon = True # thread dies with the program
        self.thread.start()
        self.assigned_job = None

thing=PyroJobWorker()
daemon=Pyro4.Daemon()
nameserver = Pyro4.locateNS()
uri=daemon.register(thing)
nameserver.register("IWORKER#"+str(uri), uri, metadata={"silver_jobworker"})
del nameserver
print(uri)
daemon.requestLoop()