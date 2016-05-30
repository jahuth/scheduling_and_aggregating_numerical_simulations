"""

This module creates schedulable tasks that are systematically completed.


"""

import json
import numpy
import time
import resource
import weakref
import tarfile
import tempfile
import re
import time
import sys, StringIO
import traceback

module_verbose = True
"""
def memory_usage_resource():
    import resource
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem
"""
class MemoryManager(object):
    def __init__(self):
        #self.data_objects = []
        self.session_refs = []
    def get_footprint(self):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    def count_sessions(self):
        objs = 0
        for s in self.session_refs:
            session = s()
            if session is not None:
                objs += 1
        return objs
    def count_existing(self):
        objs = 0
        for s in self.session_refs:
            session = s()
            if session is not None:
                objs += sum(session.root_task.get_tasks(lambda x: not x.data.empty ))
        return objs
    def count_in_memory(self):
        objs = 0
        for s in self.session_refs:
            session = s()
            if session is not None:
                objs += sum(session.root_task.get_tasks(lambda x: x.data.loaded_objects() ))
        return objs
    def clear_oldest_until_below(self,threshold=1500.0):
        while self.get_footprint() > threshold: 
            if self.count_in_memory() <= 1:
                break
            print("attempting to remove data...")
            self.clear_oldest()
            #for session in self.sessions:
            #   datas = sorted(session.root_task.get_tasks(lambda x: (x.data.last_use,x.data), lambda x: x.data.in_memory),key=lambda x: x[0])
            #   if len(datas) > 0:
            #       datas[0][1].save(purge_memory=True)
    def clear_oldest(self):
        datas = []
        for s in self.session_refs:
            session = s()
            if session is not None:
                datas.extend(session.root_task.get_tasks(lambda x: (x.data.last_use,x.data), lambda x: x.data.loaded_objects() > 0))
        datas = sorted(datas,key=lambda x: x[0])
        if len(datas) > 0:
            datas[0][1].save(purge_memory=True)
    def clear_memory(self):
        for s in self.session_refs:
            session = s()
            if session is not None:
                datas = session.root_task.get_tasks(lambda x: x.data, lambda x: x.data.loaded_objects() > 0)
                for data in datas:
                    data.save(purge_memory=True)

meman = MemoryManager()

class Experiment(object):
    def __init__(self,source):
        self.source = source
        self.sessions = []
        source_str = open(source, 'r').readlines()
        if not source_str[0].startswith('#!silver_gui'):
            raise Exception('Not an experiment file!')
    def create_session(self,filename=None):
        if filename is None:
            filename = "Session"
        return Session(self,filename)
    def load_sessions(self):
        try:
            with open(self.source+'_session.json', 'r') as fp:
                self.sessions = json.load(fp)
        except:
            self.sessions = []
        return self.sessions
    def add_session(self,session):
        self.load_sessions()
        self.sessions.append(session.filename)
        with open(self.source+'_session.json', 'wb') as fp:
            json.dump(list(set(self.sessions)), fp, cls=NumpyAwareJSONEncoder)
    def get_sandbox(self):
        import imp
        import os
        from os.path import split
        #os.chdir(split(self.source)[0])
        sand = imp.load_source('tmp.experiment', self.source)
        return sand

class Session(object):
    def __init__(self,experiment,filename,readonly=False):
        self.experiment = experiment
        while filename.endswith('.json'):
            filename = filename[:-5]
        self.filename = filename
        self.root_task = None
        self.commands = {}
        self.readonly = readonly
        meman.session_refs.append(weakref.ref(self))
    def to_json(self):
        return {
            'experiment': self.experiment.source,
            'filename':self.filename,
            'tasks': self.root_task.to_json() if self.root_task is not None else None}
    def create_from_json(self, j):
        self.experiment = Experiment(j['experiment'])
        self.filename = j['filename']
        self.root_task = Task(j['tasks'])
    def load(self):
        self.readonly = True
        with open(self.filename+'.json', 'r') as fp:
            j = json.load(fp)
            self.create_from_json(j)
    def check_json(self):
        with open(self.filename+'.json', 'r') as fp:
            j = json.load(fp)
    def save(self):
        if self.experiment is not None:
            self.experiment.add_session(self)
        with open(self.filename+'.json', 'wb') as fp:
            json.dump(self.to_json(), fp, cls=NumpyAwareJSONEncoder)

    def walk_task_tree(self, func):
        d = func(self.root_task)
        if d is None:
            return None
        d["subtasks"] = []
        for s in t.subtasks:
            a = walk_task_tree(s, func)
            if a is not None:
                d["subtasks"].append(a)
        return d
    def guess_time(self, task):
        tag_times = self.root_task.list_tag_times()
        times = []
        for tag in task.tags:
            if tag in tag_times:
                times.extend(tag_times[tag])
        if len(times) > 0:
            return numpy.median(times)
        return 0
    def guess_total_time(self):
        tag_times = self.root_task.list_tag_times()
        tasks = self.root_task.get_tasks()
        total_time = 0
        for task in tasks:
            times = []
            if task.cmd.is_complete:
                continue
            for tag in task.tags:
                if tag in tag_times:
                    times.extend(tag_times[tag])
            if len(times) > 0:
                total_time += numpy.median(times)
        return total_time

class WritableLog():
    def __init__(self,fun):
        self.fun = fun
    def write(self,*args,**kwargs):
        self.fun(*args,**kwargs)
    def flush(self,*args,**kwargs):
        pass

class Command(object):
    def __init__(self,cmd=None,interpreter="python",ret=None,running=False,is_complete=False,starttime=None,endtime=None,completion_time=None,stdout=None,stderr=None):
        self.cmd = cmd
        self.interpreter = interpreter
        self.is_complete = is_complete
        self.ret = ret
        self.running = running
        if self.cmd is None:
            self.is_complete = True
        self.starttime = starttime
        self.endtime = endtime
        self.completion_time = completion_time
        self.stdout = stdout
        self.stderr = stderr
    def reset(self):
        self.is_complete = False
        self.ret = None
        self.running = False
        if self.cmd is None:
            self.is_complete = True
        self.starttime = None
        self.endtime = None
        self.completion_time = None
        self.stdout = None
        self.stderr = None
    def to_json(self):
        d = {
        'cmd': self.cmd,
        'interpreter': self.interpreter,
        'ret':self.ret,
        'running':self.running,
        'is_complete':self.is_complete,
        'starttime': self.starttime,
        'endtime': self.endtime,
        'completion_time': self.completion_time,
        'stdout':self.stdout,
        'stderr': self.stderr
        }
        return dict((k, v) for k, v in d.iteritems() if v is not None)
    def run(self,variables={},verbose=False, log_func=None):
        if log_func is None:
            def log_func(s='',c=''):
                print s
        if self.running:
            return "The Command is already running"
        self.starttime = time.time()
        self.running = True
        if self.interpreter == 'python':
            if verbose:
                log_func("running python code...")
                log_func('| |  '+'\n| |  '.join(self.cmd.split('\n')))
            locals().update(variables)
            locals().update({'log':log_func})
            stdout = sys.stdout
            sys.stdout = WritableLog(log_func)
            try:
                exec self.cmd
            except Exception as e:
                sys.stdout = stdout
                log_func(str(e))
                log_func(traceback.format_exc())
                self.exception = e
                self.traceback = traceback.format_exc()
                self.ret = False
                self.endtime = time.time()
                self.completion_time = self.endtime - self.starttime
                if verbose:
                    log_func("time:",self.endtime - self.starttime)
                raise
                return False
            else:
                sys.stdout = stdout
                self.ret = True
                self.is_complete = True
                self.endtime = time.time()
                self.completion_time = self.endtime - self.starttime
                if verbose:
                    log_func("time:",self.endtime - self.starttime)
                    log_func("--- OK ---")
                return True
        elif self.interpreter == 'shell':
            import subprocess
            if verbose:
                log_func("running shell command...")
                log_func(' #  '+str(self.cmd))
            self.stdout = ""
            self.stderr = ""
            stdout = sys.stdout
            sys.stdout = WritableLog(log_func)
            p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            sys.stdout = stdout
            while p.returncode is None:
                (o,e) = p.communicate()
                log_func(o,'stdout')
                log_func(e,'stderr')
                self.stdout += o
                self.stderr += e
            self.is_complete = True
            self.ret = p.returncode
            self.endtime = time.time()
            self.completion_time = self.endtime - self.starttime
            if verbose:
                log_func("time:",self.endtime - self.starttime)
            if verbose:
                if self.ret == 0:
                    log_func("--- OK ---")
                else:
                    log_func(self.stdout)
                    log_func(self.stderr)
                    log_func("--- Return: "+str(self.ret) + " ---")
            return True
    def failed(self):
        if self.is_complete is False:
            return False
        if self.interpreter == 'python':
            return not self.ret
        elif self.interpreter == 'shell':
            if self.ret is not 0:
                return True
            return False

import pickle
import os

def data_to_html(data):
    import numpy
    if type(data) == numpy.ndarray:
        if len(data.shape) == 1:
            if data.shape[0] > 20:
                return "<div class='list'><div class='list_info'>np.array<sub>"+str(data.shape[0])+"</sub></div> " + ' '.join([str(d) for d in data[:20]]) + " ...</div> "
            return "<div class='list'><div class='list_info'>np.array<sub>"+str(data.shape[0])+"</sub></div> " + ' '.join([str(d) for d in data]) + "</div> "
        elif len(data.shape) == 0:
            return str(data)
        elif len(data.shape) == 2:
            data_ = data
            axis = ['x','y']
            if data.shape[1] < data.shape[0]:
                data_ = data.transpose()
                axis = ['y','x']
            return "<div class='list'><div class='list_info'>2d matrix<sub>"+str(data.shape[0])+"x"+str(data.shape[1])+"</sub> (shown as "+",".join(axis)+")</div><table class='matrix'>"+("".join(["<tr>"+("".join(["<td>"+str(r2)+"</td>" for r2 in r1]))+"</tr>" for r1 in data_[:100,:100]]))+"</table></div>"
        elif len(data.shape) == 3:
            data_ = data
            axis = ['x','y','z']
            if data_.shape[1] < data_.shape[0]:
                data_ = numpy.swapaxes(data,0,1)
                axis = [axis[1],axis[0],axis[2]]
            if data_.shape[2] < data_.shape[1]:
                data_ = numpy.swapaxes(data_,1,2)
                axis = [axis[0],axis[2],axis[1]]
            if data_.shape[1] < data_.shape[0]:
                data_ = numpy.swapaxes(data_,0,1)
                axis = [axis[1],axis[0],axis[2]]
            return "<div class='list'><div class='list_info'>3d matrix<sub>"+str(data.shape[0])+"x"+str(data.shape[1])+"x"+str(data.shape[2])+"</sub> ("+",".join(axis)+")</div><div class='multi_matrix'>" + ("".join(["<p>"+axis[2]+": "+str(z)+"</p><table class='matrix'>"+("".join(["<tr>"+("".join(["<td>"+str(r2)+"</td>" for r2 in r1]))+"</tr>" for r1 in r]))+"</table>" for z,r in enumerate(data_[:100,:100,:100])]))+"</div></div>"
        else:
            return "<div class='list'><div class='list_info'>Multidimensional numpy array</div> " +  str(data) + "</div> "
    if type(data) == list or type(data) == tuple:
        def smallify(d):
            if len(d) < 15:
                return "<div class='list'><div class='list_info'>list of "+str(len(d))+" elements</div>"+ (" ".join([data_to_html(dd) for dd in d]))+ "</div>"
            return "<div class='list'><div class='list_info'>list of "+str(len(d))+" elements</div>"+ (" ".join([data_to_html(dd) for dd in d[:15]] + ['...']) ) + "</div>"
        return str(smallify(data))
    if type(data) == dict:
        s = []
        for k in data.keys():
            s.append("<div class='key'>"+str(k)+"</div><div class='value'>"+data_to_html(data[k])+"</div>")
        return "<div class='dict'>{<div class='dict_entry'>"+"<div class='comma'>,</div></div><div class='dict_entry'>".join(s)+"</div>}</div>"
    return str(data)

def data_to_metadata(data):
    meta = {'html': data_to_html(data),'type':str(type(data))}
    try:
        meta['class'] = str(data.__class__.__name__)
    except:
        pass
    if type(data) == list or type(data) == tuple or type(data) == dict:
        meta['len'] = len(data)
    if type(data) == dict:
        meta['keys'] = data.keys()
    if type(data) == numpy.ndarray:
        meta['dtype'] = data.dtype
        meta['shape'] = data.shape
        meta['nbytes'] = data.nbytes
        try:
            meta['min'] = data.min()
            meta['max'] = data.max()
            meta['mean'] = data.mean()
            meta['var'] = data.var()
        except:
            pass
    return meta

class AbstractTaskDataContainer(object):
    def __init__(self, name='', do_purge_after_exit=True):
        self.last_use = None
        self.filename = name
        self.empty = True
        self.do_purge_after_exit = do_purge_after_exit
        self.opened_as_context_manager = False
    def loaded_objects(self):
        return 0
    def get(self):
        pass
    def set(self,key,data):
        pass
    def append(self,key,data):
        pass
    def exists(self):
        if not os.path.exists(self.filename):
            return False
        return True
    def size(self):
        return os.path.getsize(self.filename)/(1024.0*1024.0)
    def load(self):
        pass
    def save(self,purge_memory=False):
        pass
    def __enter__(self):
        self.last_use = time.time()
        self.opened_as_context_manager = True
        if module_verbose:
            print 'open',time.time()
            self.opened_at = time.time()
        return self
    def __exit__(self, type, value, tb):
        self.last_use = time.time()
        self.opened_as_context_manager = False
        if module_verbose:
            print 'close',time.time(), time.time()-self.opened_at 
        self.save(purge_memory=self.do_purge_after_exit)
    def savefig(self,key="", fig=None, close=True):
        self.last_use = time.time()
        import base64, urllib
        import StringIO
        imgdata = StringIO.StringIO()
        if fig is None:
            from matplotlib.pylab import gcf
            fig = gcf()
        fig.savefig(imgdata, format='png')
        imgdata.seek(0) 
        image = base64.encodestring(imgdata.buf) 
        self.append(key,"<img src='data:image/png;base64," + urllib.quote(image) + "'>")
    def display(self,key):
        from IPython.core.display import HTML, display
        def h(l):
            if type(l) is list:
                for ll in l:
                    display(HTML(ll))
            else:
                display(HTML(l))
        h(self.get(key))


class TaskDataContainerTar(AbstractTaskDataContainer):
    def __init__(self, data=None, name='', do_purge_after_exit=True,save_meta_data=True,save_on_every_change=True,load=False,new=False):
        AbstractTaskDataContainer.__init__(self,name,do_purge_after_exit)
        if self.filename.endswith('.tar'):
            self.tarfilename = self.filename
        else:
            self.tarfilename = self.filename + '.tar'
        self._keys = []
        self._members = {}
        self._data = {}
        self._in_memory = {}
        self._meta_keys = []
        self._meta_members = {}
        self._meta_data = {}
        self._meta_in_memory = {}
        self._data_not_synced = []
        self.save_meta_data = save_meta_data
        self.save_on_every_change = save_on_every_change
        self.contains_duplicates = False
        if data is not None:
            self._data = data
            self._data_not_synced.extend(self._data.keys())
            self._in_memory.update(dict([(k,True) for k in self._data.keys()]))
            self.synced_with_file = False
            self.empty = False
            if not os.path.exists(self.tarfilename):
                self.synced_with_file = True
        else:
            self._in_memory = {}
            if os.path.exists(self.tarfilename):
                self.synced_with_file = False
                self.empty = False
            else:
                self.synced_with_file = True
                self.empty = True
        self._members_loaded_from_file = False
        if new and os.path.exists(self.tarfilename):
                os.remove(self.tarfilename)
        if load:
            self.load()
    def loaded_objects(self):
        c = 0
        for v in self._in_memory.values():
            if v:
                c += 1
        return c
    def keys(self,no_old_versions=False):
        if no_old_versions:
            return [k for k in self.keys() if not re.search(r'\.\d+$', k)]
        return list(set(self._data.keys() + self._members.keys()))
    def load(self,force_reload=False):
        if not os.path.exists(self.tarfilename):
            self.data = None
            return
        if not self._members_loaded_from_file or force_reload:
            with tarfile.TarFile(self.tarfilename,'r') as f:
                try:
                    if module_verbose:
                        print("loading ",self.tarfilename, " | size: ", os.path.getsize(self.tarfilename)/(1024*1024)," MB")
                    self._keys = []
                    self.update_members(f)
                    self.empty = len(self.keys()) > 0
                    self.synced_with_file = True
                    self._in_memory.update(dict((k,False) for k in self._keys))
                    self.last_use = time.time()
                    self._members_loaded_from_file = True
                except:
                    return
    def update_members(self,tar_file_handle):
        self._members = {}
        self._meta_members = {}
        for n in tar_file_handle.getnames():
            if not n.endswith('.meta') and (n not in self._in_memory or not self._in_memory[n]):
                self._keys.append(n)
                self._members[n] = tar_file_handle.getmember(n)
            if n.endswith('.meta') and (n[:-5] not in self._in_memory or not self._in_memory[n[:-5]]):
                self._meta_keys.append(n[:-5])
                self._meta_members[n[:-5]] = tar_file_handle.getmember(n)
    def re_save(self):
        if os.path.exists(self.tarfilename):
            with tarfile.TarFile(self.tarfilename+'_','a') as new_tar_data:
                with tarfile.TarFile(self.tarfilename,'r') as old_tar_data:
                    for name in list(set(old_tar_data.getnames())):
                        new_tar_data.addfile(old_tar_data.getmember(name),old_tar_data.extractfile(old_tar_data.getmember(name)))
            os.rename(self.tarfilename+'_',self.tarfilename)
            self.contains_duplicates = False
    def _save_single_key(self,key,data,file_handle=None):
        if file_handle is not None:
            (fd, tmpfilename) = tempfile.mkstemp()
            with open(tmpfilename,'w') as tmpfile:
                pickle.dump(data, tmpfile)
            file_handle.add(tmpfilename, key)
            os.close(fd)
            os.remove(tmpfilename)
        else:
            with tarfile.TarFile(self.tarfilename,'a') as tar_data:
                self._save_single_key(key,data,file_handle=tar_data)
    def save(self,key=None,purge_memory=False,file_handle=None):
        self.last_use = time.time()
        if len(self._data_not_synced) == 0:
            if purge_memory:
                self._in_memory = dict((k,False) for k in self.keys())
                self._data = {}
                self._meta_in_memory = dict((k,False) for k in self.keys())
                self._meta_data = {}
            return
        if key is None:
            for not_synced_key in self._data_not_synced:
                with tarfile.TarFile(self.tarfilename,'a') as tar_data:
                    self.save(not_synced_key,purge_memory=purge_memory,file_handle=tar_data)
            if purge_memory:
                self._in_memory = dict((k,False) for k in self.keys())
                self._data = {}
                self._meta_in_memory = dict((k,False) for k in self.keys())
                self._meta_data = {}
            if self.contains_duplicates:
                self.re_save()
            self._members_loaded_from_file = False
            if file_handle is None:
                self.load()
        else:
            if key not in self._data:
                self._in_memory[key] = False
                return
            if key not in self._data_not_synced:
                if purge_memory:
                    self._in_memory[key] = False
                    del self._data[key]
                return
            if self.save_meta_data:
                self._save_single_key(key+'.meta',data_to_metadata(self._data[key]),file_handle=file_handle)
            if key in self._members.keys():
                self.contains_duplicates = True
            self._save_single_key(key,self._data[key],file_handle=file_handle)
            self._data_not_synced.remove(key)
            if file_handle is not None:
                self.update_members(file_handle)
            if purge_memory:
                self._in_memory[key] = False
                del self._data[key]
            self._members_loaded_from_file = False
            if file_handle is None:
                self.load()
    def meta(self,key=None,default=None):
        self.load()
        if key is None:
            return self
        if key in self._data_not_synced:
            if key in self._data:
                return data_to_metadata(self._data[key])
        if key in self._meta_members and self._meta_members[key].isreg():
            if key in self._meta_data:
                return self._meta_data[key]
            with tarfile.TarFile(self.tarfilename,'r') as f:
                self._meta_data[key] = pickle.load(f.extractfile(self._meta_members[key]))
            return self._meta_data[key]
        return default
    def get_regex(self,regex,default=None):
        return [self.get(k,default=default) for k in self.keys() if re.search(regex,k)]
    def get_all(self,key=None,default=None):
        return [self.get(key,default=default)] + list(reversed([self.get(k,default=default) for k in self.keys() if k.startswith(key+'.') and re.search(r'^\d+$',k[len(key+'.'):])]))
    def get_all_keys(self,key=None,default=None):
        return [key] + list(reversed([k for k in self.keys() if k.startswith(key+'.') and re.search(r'^\d+$',k[len(key+'.'):])]))
    def get(self,key=None,default=None):
        self.load()
        if key is None:
            return self
        if key in self._data_not_synced:
            return self._data[key]
        if key in self._members and self._members[key].isreg():
            if self._in_memory[key]:
                return self._data[key]
            with tarfile.TarFile(self.tarfilename,'r') as f:
                self._data[key] = pickle.load(f.extractfile(self._members[key]))
                self._in_memory[key] = True
            return self._data[key]
        return default
    def __getitem__(self,key):
        return self.get(key)
    def set(self,key,data,preserve_old=False):
        print('writing '+str(key)+' in data...')
        self.last_use = time.time()
        self.empty = False
        if key in self.keys() and preserve_old:
            self.contains_duplicates = True
            i = 0
            while key+'.'+str(i) in self.keys():
                i += 1
            self._data[key+'.'+str(i)] = self.get(key)
            self._data_not_synced.append(key+'.'+str(i))
            if not self.opened_as_context_manager and self.save_on_every_change:
                self.save(key+'.'+str(i))
        self._data[key] = data
        self._in_memory[key] = True
        self._data_not_synced.append(key)
        self.synced_with_file = False
        #if not self.opened_as_context_manager and self.save_on_every_change:
        self.save(key)
    def append(self,key,data):
        self.set(key,data,preserve_old=True)
    def size(self):
        return os.path.getsize(self.tarfilename)/(1024.0*1024.0)
    def exists(self):
        if not os.path.exists(self.tarfilename) and len(self._data_not_synced)==0:
            # There is no file and we have no data in memory
            return False
        return True
    def to_html(self,key_filter=None):
        self.load()
        h =""
        if len(self.keys()) > 0:
            tree = {}
            # First we create a tree purely on the basis of the keys:
            def add_to_tree(node,path,full_path):
                if len(path) == 1:
                    if path[0] in node:
                        if '' in node[path[0]]:
                            node[path[0]][''].append(full_path)
                        else:
                            node[path[0]][''] = [full_path]
                    else:
                        node[path[0]] = {'': [full_path]}
                else:
                    if not path[0] in node:
                        node[path[0]] = {}
                    add_to_tree(node[path[0]], path[1:], full_path)
            for k in self.keys(no_old_versions=True):
                if key_filter is not None:
                    if not key_filter.startswith(k):
                        continue
                p = [str(pp) for pp in k.split("/")]
                add_to_tree(tree,p,k)

            # Then we recursively render it as html:
            def render_data_entry(d):
                if key_filter is not None:
                    if not key_filter == d:
                        return ""
                all_keys = self.get_all_keys(d)
                if len(all_keys) == 1:
                    meta = self.meta(d)
                    if meta is not None:
                        return "<div class='data "+str(d)+"'>"+meta["html"]+"</div>"
                else:
                    data_h = "<div class='data "+str(d)+"'>"
                    data_h += "<div class='data_pieces "+str(d)+"'>"
                    other_data_pieces = []
                    for data_piece in all_keys:
                        meta = self.meta(data_piece)
                        if meta is not None:
                            html_piece = meta["html"]
                            if not html_piece in other_data_pieces:
                                data_h += "<div class='data_piece'>"+html_piece+"</div>"
                                other_data_pieces.append(html_piece)
                            else:
                                data_h += "<div class='data_piece_duplicate'>"+html_piece+"</div>"
                    return data_h + "</div></div>"
                return ""
            def rec_tree(node,depth):
                _h = ""
                for k in sorted(node.keys()):
                    _h += "<div class='data_sub depth_"+str(depth)+"' data-depth='"+str(depth)+"'>"
                    _h += "<h2 class='data_title'>"+str(k)+"</h2>"
                    _h += "<div class='data_content'>"
                    if type(node[k]) is dict:
                        _h += rec_tree(node[k],depth+1)
                    else:
                        for n in node[k]:
                            _h += render_data_entry(n)
                    _h += "</div>"
                    _h += "</div>"
                return _h
            h += rec_tree(tree,0)
        return h
    def __str__(self):
        return self.to_html()

class TaskDataContainer(object):
    def __init__(self, data=None, name=''):
        self.last_use = None
        self.filename = name
        if not self.filename.endswith('.pkl'):
            self.filename+='.pkl'
        self.empty = True
        if data is not None:
            self.in_memory = True
            self.synced_with_file = False
            if os.path.exists(self.filename):
                self.synced_with_file = False
                self.empty = False
            else:
                self.synced_with_file = True
        else:
            self.in_memory = False
            if os.path.exists(self.filename):
                self.synced_with_file = False
                self.empty = False
            else:
                self.synced_with_file = True
                self.empty = True
        self.data = data
        self.opened_as_context_manager= False
    def loaded_objects(self):
        if self.in_memory:
            return 1
        return 0
    def get(self):
        self.last_use = time.time()
        if not self.in_memory or not self.synced_with_file:
            if os.path.exists(self.filename):
                self.load()
        return self.data
    def set(self,key,data):
        self.last_use = time.time()
        self.empty = False
        d = self.get()
        if type(d) == dict:
            d[key]=[data]
        elif d is None:
            d = {key:[data]}
        else:
            d = {key:data,'_':d}
        self.data = d
        self.in_memory = True
        self.synced_with_file = False
        if not  self.opened_as_context_manager:
            self.save()
    def append(self,key,data):
        self.last_use = time.time()
        d = self.get()
        if type(d) == dict:
            if key in d:
                if type(d[key]) is list:
                    d[key].append(data)
                else:
                    d[key] = [d[key],data]
            else:
                d[key]=[data]
        elif d is None:
            d = {key:[data]}
        else:
            d = {key:data,'_':d}
        self.data = d
        self.in_memory = True
        self.synced_with_file = False
        if not  self.opened_as_context_manager:
            self.save()
    def exists(self):
        if not os.path.exists(self.filename) and not self.in_memory:
            return False
        return True
    def size(self):
        return os.path.getsize(self.filename)/(1024.0*1024.0)
    def load(self):
        if not os.path.exists(self.filename):
            self.data = None
            return
        with open(self.filename,'r') as f:
            try:
                if module_verbose:
                    print("loading ",self.filename, " | size: ", os.path.getsize(self.filename)/(1024*1024)
," MB")
                self.data = pickle.load(f)
                self.synced_with_file = True
                self.in_memory = True
                self.last_use = time.time()
            except:
                return
    def save(self,purge_memory=False):
        self.last_use = time.time()
        if self.in_memory and not self.synced_with_file:
            if self.data is None:
                self.empty = True
                os.remove(self.filename)
            else:
                if module_verbose:
                    print("saving ",self.filename)
                with open(self.filename,'w') as f:
                    pickle.dump(self.data, f)
                    self.synced_with_file = True
                if module_verbose:
                    print("saved ",self.filename, " | size: ", os.path.getsize(self.filename)/(1024*1024)," MB")
        if purge_memory:
            self.data = None
            self.in_memory = False
            self.last_use = None
            if module_verbose:
                print("purging ",self.filename)
    def __enter__(self):
        self.last_use = time.time()
        self.opened_as_context_manager = True
        if module_verbose:
            print 'open',time.time()
            self.opened_at = time.time()
        return self
    def __exit__(self, type, value, tb):
        self.last_use = time.time()
        self.opened_as_context_manager = False
        if module_verbose:
            print 'close',time.time(), time.time()-self.opened_at 
        self.save(purge_memory=True)
    def savefig(self,key="", fig=None, close=True):
        self.last_use = time.time()
        import base64, urllib
        import StringIO
        imgdata = StringIO.StringIO()
        if fig is None:
            from matplotlib.pylab import gcf
            fig = gcf()
        fig.savefig(imgdata, format='png')
        imgdata.seek(0) 
        image = base64.encodestring(imgdata.buf) 
        self.append(key,"<img src='data:image/png;base64," + urllib.quote(image) + "'>")
    def display(self,key):
        from IPython.core.display import HTML, display
        def h(l):
            if type(l) is list:
                for ll in l:
                    display(HTML(ll))
            else:
                display(HTML(l))
        h(self.get()[key])
    def to_html(self,key_filter=None):
        self.load()
        h =""
        if type(self.data) is dict:
            tree = {}
            def add_to_tree(node,path,full_path):
                if len(path) == 1:
                    if path[0] in node:
                        if '' in node[path[0]]:
                            node[path[0]][''].append(full_path)
                        else:
                            node[path[0]][''] = [full_path]
                    else:
                        node[path[0]] = {'': [full_path]}
                else:
                    if not path[0] in node:
                        node[path[0]] = {}
                    add_to_tree(node[path[0]], path[1:], full_path)
            for k in self.data.keys():
                if key_filter is not None:
                    if not key_filter.startswith(k):
                        continue
                p = [str(pp) for pp in k.split("/")]
                add_to_tree(tree,p,k)
            def render_data_entry(d):
                _h = ""
                if key_filter is not None:
                    if not key_filter == d:
                        return ""   
                _h += "<div class='data "+str(d)+"'>"
                if len(self.data[d]) == 1:
                        _h += data_to_html(self.data[d][0])
                else:
                    _h += "<div class='data_pieces "+str(d)+"'>"
                    other_data_pieces = []
                    for data_piece in reversed(self.data[d]):
                        html_piece = data_to_html(data_piece)
                        if not html_piece in other_data_pieces:
                            _h += "<div class='data_piece'>"+html_piece+"</div>"
                            other_data_pieces.append(html_piece)
                        else:
                            _h += "<div class='data_piece_duplicate'>"+html_piece+"</div>"
                    _h += "</div>"
                _h += "</div>"
                return _h
            def rec_tree(node):
                _h = ""
                for k in sorted(node.keys()):
                    _h += "<div class='data_sub'>"
                    _h += "<h2 class='data_title'>"+str(k)+"</h2>"
                    _h += "<div class='data_content'>"
                    if type(node[k]) is dict:
                        _h += rec_tree(node[k])
                    else:
                        for n in node[k]:
                            _h += render_data_entry(n)
                    _h += "</div>"
                    _h += "</div>"
                return _h
            h += rec_tree(tree)
        return h
    def __str__(self):
        return self.to_html()

class Task(object):
    def __init__(self, name="New Task", parent=None,cmd=Command(), variables = {}, parallel=False, tags=[],path=''):
        self.subtasks = []
        self.blocked = False
        use_tar = True
        self._log = []
        if type(name) is dict:
            self.__dict__.update(name)
            for t in name['tasks']:
                self.subtasks.append(Task(t,parent=self))
            del self.__dict__['tasks']
            self.cmd = Command(**name['command'])
            self.data_path =name['data_path']
            self.blocked = name.get('blocked',False)
            self._log = name.get('log',[])
        else:
            self.name=name
            self.parallel = parallel
            self.variables = variables
            self.cmd = cmd
            self.tags = tags
            self.data_path = None
        self.parent = parent
        if self.parent is not None and path is '':
            path = self.parent.path+self.name
        self.path = path
        if path.endswith('/') and not os.path.exists(path):
            os.makedirs(path)
        if self.data_path is None:
            self.data_path = path+self.name+'_data'
        #fallback:
        if os.path.exists(self.data_path+'.pkl'):
            use_tar = False # can be removed if I dont use that data anymore
        if use_tar == True:
            self.data = TaskDataContainerTar(None,self.data_path)
        else:
            self.data = TaskDataContainer(None,self.data_path)
        self.running_start_time = None
    def is_running(self):
        if self.running_start_time is None:
            return False
        if time.time() - self.running_start_time < 0.5:
            return True
    def running_tick(self):
        self.running_start_time = time.time()
    def to_json(self):
        return {
            'name': self.name,
            'parallel': self.parallel,
            'variables': self.variables,
            'command': self.cmd.to_json(),
            'path':self.path,
            'tags':self.tags,
            'data_path': self.data.filename,
            'blocked':self.blocked,
            'tasks': [s.to_json() for s in self.subtasks],
            'log': self._log
        }
    def log_func(self, message, channel=''):
        #if module_verbose:
        #   print message
        self._log.append({'message':message,'time':time.time(), 'channel':channel})
        self.propagate_log(self,{'message':message,'time':time.time(), 'channel':channel})
    def propagate_log(self,sender,message_dict):
        pass
        #if self.parent is not None:
        #   self.parent.propagate_log(sender,message_dict)
    def __getitem__(self,key):
        if key in self.variables:
            return self.variables[key]
        if self.parent is not None:
            return self.parent.__getitem__(key)
        return None
    def keys(self):
        k = self.variables.keys()
        if self.parent is not None:
            k.extend(self.parent.keys())
        return list(unique(k))
    def to_html(self):
        h = """<h1>"""+self.name+"""</h1>"""
        variables = self.get_variables()
        h += "Tags: <span>"+"".join(["<b>"+str(t)+"</b> " for t in self.tags])+"</span>" 
        h += "Variables: <ul>"+"".join(["<li><b>"+str(k)+":</b> "+str(variables[k])+"</li>" for k in variables.keys()])+"</ul>" 
        return h
    def get_variables(self):
        d = {}
        if self.parent is not None:
            d.update(self.parent.get_variables())
        d.update(self.variables)
        return d
    def status(self):
        if self.complete():
            return "complete "+format_time(self.cmd.completion_time)
        if self.cmd.cmd is not None:
            if self.cmd.is_complete:
                return "ran in "+format_time(self.cmd.completion_time)
            else:
                if self.cmd.ret is not None:
                    return str(self.cmd.ret)
                if self.is_running():
                    return "running..."
            return "pending"
        return ""
    def complete(self):
        for t in self.subtasks:
            if not t.complete():
                return False
        return self.cmd.is_complete
    def pending(self):
        p = 0
        for t in self.subtasks:
            p += t.pending()
        if not self.cmd.is_complete:
            p += 1
        return p
    def count_runnable(self):
        p = 0
        for t in self.subtasks:
            p += t.count_runnable()
        if self.cmd.cmd is not None:
            p += 1
        return p
    def count(self):
        p = 0
        for t in self.subtasks:
            p += t.count()
        p += 1
        return p
    def run_one(self, verbose=False):
        if self.blocked:
            return None
        if not self.cmd.is_complete:
            variables = self.get_variables()
            variables['task'] = self
            if verbose:
                print(variables)
            self.run(variables,verbose=verbose)
            return self.cmd.ret
        for s in self.subtasks:
            r = s.run_one(verbose=verbose)
            if r is not None:
                return r
        return None
    def find_next_task(self):
        if self.blocked:
            return None
        if not self.cmd.is_complete:
            return self
        for s in self.subtasks:
            r = s.find_next_task()
            if r is not None:
                return r
        return None
    def run_all(self,verbose=True):
        while self.pending() > 0:
            if verbose:
                print("===",self.pending(),"/",self.count(),"===")
                showTasks("Tasks",self)
                print(self.tag_times())
            self.run_one(verbose=verbose)
    def find_runnable_tasks(self):
        if not self.cmd.is_complete:
            return [self]
        else:
            if self.parallel:
                ts = []
                for t in self.subtasks:
                    ts.extend(t.find_runnable_tasks())
                return ts
            else:
                ts = []
                for t in self.subtasks:
                    if t.cmd.is_complete:
                        ts.extend(t.find_runnable_tasks())
                    else:
                        ts.append(t)
                        break
                return ts
    def get_tasks(self,f = None, condition = None):
        """
            Creates a list of all subtasks

            f can be a function that is applied to each task to eg. retrieve the data:

                task.get_tasks(lambda x: x.data.get())
        """
        if condition is None or condition(self):
            if f is not None:
                ts = [f(self)]
            else:
                ts = [self]
        else:
            ts = []
        for t in self.subtasks:
            ts.extend(t.get_tasks(f,condition))
        return ts
    def get_tasks_as_generator(self,f = None, condition = None):
        """
            Creates a list of all subtasks

            f can be a function that is applied to each task to eg. retrieve the data:

                task.get_tasks(lambda x: x.data.get())
        """
        if condition is None or condition(self):
            if f is not None:
                yield f(self)
            else:
                yield self
        else:
            ts = []
        for t in self.subtasks:
            for tt in t.get_tasks(f,condition):
                if tt is not None:
                    yield tt
        raise 
    def display(self,key):
        for tt in self.get_tasks(lambda x: x.data, lambda x: not x.data.get() is None):
            tt.display(key)
    def get_data(self,key, condition = None):
        def f(task):
            data = None
            with task.data as d:
                if type(key) is list:
                    data = [d.get()[k] for k in key if k in d.get().keys()]
                    if data == []:
                        data = None
                else:
                    if key in d.get():
                        data = d.get()[key]
            return data
        return [t for t in [tt for tt in self.get_tasks(f, lambda x: not x.data.get() is None and (condition is None or condition(x)))] if t is not None and t is not []]
    def get_data_dict(self,key, condition = None, require_all=True, include_task_variables=True):
        def f(task):
            data = {}
            if include_task_variables:
                variables = task.get_variables()
                if type(key) is list:
                    data = dict([(k,[variables[k]]) for k in key if k in variables.keys()])
                else:
                    if key in variables.keys():
                        data = {key: [variables[key]]}
            with task.data as d:
                if type(key) is list:
                    data.update(dict([(k,d.get()[k]) for k in key if k in d.get().keys()]))
                else:
                    if key in d.get():
                        data = {key: d.get()[key]}
            if data == {}:
                return None
            if require_all and type(key) is list:
                for k in key:
                    if not k in data.keys():
                        return None
            if require_all and type(key) is not list:
                if not key in data.keys():
                    return None
            return data
        return [t for t in [tt for tt in self.get_tasks(f, lambda x: not x.data.get() is None and (condition is None or condition(x)))] if t is not None and t is not []]
    def get_data_dict_as_generator(self,key, condition = None, require_all=True, include_task_variables=True):
        def f(task):
            data = {}
            if include_task_variables:
                variables = task.get_variables()
                if type(key) is list:
                    data = dict([(k,[variables[k]]) for k in key if k in variables.keys()])
                else:
                    if key in variables.keys():
                        data = {key: [variables[key]]}
            with task.data as d:
                if type(key) is list:
                    data.update(dict([(k,d.get()[k]) for k in key if k in d.get().keys()]))
                else:
                    if key in d.get():
                        data = {key: d.get()[key]}
            if data == {}:
                return None
            if require_all and type(key) is list:
                for k in key:
                    if not k in data.keys():
                        return None
            if require_all and type(key) is not list:
                if not key in data.keys():
                    return None
            return data
        for t in self.get_tasks_as_generator(None, lambda x: not x.data.get() is None and (condition is None or condition(x))):
            if t is not None:
                yield f(t)
    def get_path(self):
        if self.parent is not None:
            return self.parent.get_path() + "/" + self.name
        else:
            return self.name
    def time(self):
        t = 0.0
        if self.cmd.endtime is not None and self.cmd.starttime is not None:
            t += self.cmd.endtime.real - self.cmd.starttime.real
        for s in self.subtasks:
            t += s.time()
        return t
    def tag_times(self):
        t = {}
        if self.cmd.endtime is not None and self.cmd.starttime is not None:
            for tag in self.tags:
                t[tag] = (self.cmd.endtime.real - self.cmd.starttime.real,1)
        for s in self.subtasks:
            ts = s.tag_times()
            for tk in ts.keys():
                if tk in t:
                    t[tk] = (t[tk][0] + ts[tk][0], t[tk][1] + 1)
                else:
                    t[tk] = (ts[tk][0],ts[tk][1])
        return t
    def list_tag_times(self):
        t = {}
        if self.cmd.endtime is not None and self.cmd.starttime is not None:
            for tag in self.tags:
                t[tag] = [self.cmd.endtime.real - self.cmd.starttime.real]
        for s in self.subtasks:
            ts = s.list_tag_times()
            for tk in ts.keys():
                if tk in t:
                    t[tk].extend(ts[tk])
                else:
                    t[tk] = ts[tk]
        return t
    def submit_data(self,key,data):
        self.data.append(key,data)
    def clone(self,changed_keys={},new_parent=None, name=None):
        if name is None:
            name = self.name
        new_cmd = Command(self.cmd.cmd,self.cmd.interpreter)
        if new_parent is None:
            new_parent = self.parent
        new_variables = dict([(k,self.variables[k]) for k in self.variables.keys()])
        for k in new_variables.keys():
            if k in changed_keys.keys():
                if changed_keys[k] == '++':
                    new_variables[k] = new_variables[k] + 1
                else:
                    new_variables[k] = changed_keys[k]
        clone = Task(name = name, parent = new_parent, cmd = new_cmd, variables = new_variables, parallel=False, tags=[],path='')
        if new_parent is not None:
            new_parent.subtasks.append(clone)
        for s in self.subtasks:
            s.clone(changed_keys, clone)
        return clone
    def run(self,variables={},verbose=False):
        if self.cmd.running:
            return "The Command is already running"
        self.log_func('starting')
        ret = self.cmd.run(variables,verbose,self.log_func)
        self.log_func('ended with ret: '+str(ret))      
        return ret

import sgui
def walk_task_tree(t, func):
    d = func(t)
    if d is None:
        return None
    d["subtasks"] = []
    for s in t.subtasks:
        a = walk_task_tree(s, func)
        if a is not None:
            d["subtasks"].append(a)
    return d

def showCompletedTasks(name,root_task):
    def f(a):
        if not a.cmd.is_complete:
            return None
        return {'name':a.name, 
                'path': a.get_path(), 
                'complete': a.cmd.is_complete,
                'completion_time': a.cmd.completion_time,
                'return_code': a.cmd.ret}
    sgui.startGUITable(name,[walk_task_tree(root_task, f)],
                        keys = {'path':'Path','complete':'Completed','completion_time':'Time to complete', 'return_code':'Return Code'})

def showUncompletedTasks(name,root_task):
    def f(a):
        #if a.complete():
        #   return None
        return {'name':a.name, 
                'path': a.get_path(),
                'cmd_complete': 'OK' if a.cmd.is_complete else '',
                'complete': 'OK' if a.complete() else ''}
    sgui.startGUITable(name,[walk_task_tree(root_task, f)],
                        keys = {'path':'Path','cmd_complete':'Completed Root Command', 'complete':'Completed'})

def format_time(time):
    if time == 0:
        time = ''
    elif time < 0.01:
        time = "< 0.01 s"
    elif time < 60:
        time = str(int(round(time*100.0)/100.0)) + " s"
    elif time < 60*60:
        time = str(int(round(time/60))) +'m ' + str(int(time%60)) + "s"
    else:
        time = str(int(round(time/3600))) +'h ' + str(int((round(time/60))%60)) +'m ' + str(int(time%60)) + "s"
    return time

def showTasks(name,root_task):
    def f(a):
        #if a.complete():
        #   return None
        time = a.time()
        if time == 0:
            time = ''
        elif time < 0.01:
            time = "< 0.01 s"
        elif time < 60:
            time = str(round(time*100.0)/100.0) + " s"
        elif time < 60*60:
            time = str(time/60) +'m ' + str(int(time%60)) + "s"
        else:
            time = str(time/360) +'h ' + str((time/60)%60) +'m ' + str(int(time%60)) + "s"
        return {'name':a.name, 
                'path': a.get_path(),
                'time': time if time != 0 else '',
                'cmd_complete': 'OK' if a.cmd.is_complete else '',
                'complete': 'OK' if a.complete() else '',
                'data': 'has data' if not a.data.empty else ''}
    sgui.startGUITable(name,[walk_task_tree(root_task, f)],
                        keys = {'time':'Time','complete':'Completed','data':'Data'})


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

"""
def showCompletedTasks(root_task):
    sgui.showGUITable([{'name':a.name, 
                        'path': a.get_path(), 
                        'complete': a.cmd.is_complete,
                        'completion_time': a.cmd.completion_time,
                        'return_code': a.cmd.ret} for a in root_task.get_tasks() if a.cmd.completion_time != None],
                        keys = {'path':'Path','complete':'Completed','completion_time':'Time to complete', 'return_code':'Return Code'})

def showUncompletedTasks(root_task):
    sgui.showGUITable([{'name':a.name, 
                        'path': a.get_path() } for a in root_task.get_tasks() if not a.cmd.is_complete],
                        keys = {'path':'Path'})
"""