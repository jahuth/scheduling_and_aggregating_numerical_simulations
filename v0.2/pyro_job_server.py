import Pyro4
import time
import datetime
from collections import Counter

global_log = []

while True:
    nameserver = Pyro4.locateNS()
    job_lists = nameserver.list(metadata_all={"silver_joblist"}).values()
    job_workers = nameserver.list(metadata_all={"silver_jobworker"}).values()
    del nameserver
    all_jobs = []
    all_job_stats = Counter({})
    for jl in job_lists:
        try:
            with Pyro4.Proxy(jl) as job_list_proxy:
                list_of_jobs = job_list_proxy.get_jobs("pending")
                all_jobs.extend([(j,jl) for j in list_of_jobs])
                all_job_stats = all_job_stats +  Counter(job_list_proxy.get_status_stats())
        except:
            pass
    for k in all_job_stats.keys():
        print k, ':', all_job_stats[k]
    available_job_workers = []
    for w in job_workers:
        try:
            with Pyro4.Proxy(w) as worker_proxy:
                if worker_proxy.free():
                    available_job_workers.append(w)
        except:
            pass
    if len(all_jobs) == 0:
        time.sleep(2.0)
    elif len(available_job_workers) == 0:
        time.sleep(2.0)
    else:
        print 'Jobs queued:',len(job_lists)
        print 'Workers idle:',len(available_job_workers)
        while len(all_jobs) > 0 and len(available_job_workers) > 0:
            next_job,job_list = all_jobs.pop()
            next_worker = available_job_workers.pop()
            print "Assigning job:"
            print "\t",next_job,job_list
            print "\t",next_worker
            print ["Assigning job:",(next_job,job_list),"to worker",next_worker,datetime.datetime.now()]
            global_log.append(["Assigning job:",(next_job,job_list),"to worker",next_worker,datetime.datetime.now()])
            with Pyro4.Proxy(next_worker) as worker:
                if worker.assign(next_job,job_list):
                    Pyro4.Proxy(job_list).status(next_job,'assigned')
                    async_worker = Pyro4.async(worker)
                    async_worker.run()
                else:
                    all_jobs.append((next_job,job_list))
        time.sleep(0.1)
        print 'Jobs:',len(all_jobs)
        print 'Workers:',len(available_job_workers)