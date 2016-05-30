import silver
import silver.numpy_aware_json as json
import numpy
import time
import resource
import weakref
import tarfile
import tempfile
import re
import time
import sys, StringIO, os
import traceback
import zmq
import subprocess
from  multiprocessing import Process, Queue
import uuid, base64
import hmac, hashlib
from IPython.utils.py3compat import (str_to_bytes, str_to_unicode, unicode_type,
                                     iteritems)

try:
    # We are using compare_digest to limit the surface of timing attacks
    from hmac import compare_digest
except ImportError:
    # Python < 2.7.7: When digests don't match no feedback is provided,
    # limiting the surface of attack
    def compare_digest(a,b): return a == b

module_debug = True

class FrontendCmdFunction(object):
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
    def __call__(self, *args, **kwargs):
        return self.parent.__attribute_call__(self.name, args, kwargs)

class FrontendObject(object):
    def __init__(self, relay, object_name):
        self.relay = relay
        self.object_name = object_name
        self.object_attributes = self.__request__({'dir':True})
    def __request__(self,req):
        return self.relay.request(self.object_name, req)
    def __dir__(self):
        return self.object_attributes.keys()
    def __getattr__(self, name):
        if name == "relay" or name == "object_name" or name == "object_attributes":
            return self.__dict__[name]
        if name in self.object_attributes.keys():
            if self.object_attributes[name] == True:
                return FrontendCmdFunction(self,name)
            return self.__request__({'call': name})
    def __attribute_call__(self, name, args=[], kwargs={}):
        return self.__request__({'call': name, "args": args, "kwargs": kwargs})
    def __setattr__(self, name, value):
        if name == "relay" or name == "object_name" or name == "object_attributes":
            self.__dict__[name] = value
        else:
            return self.__request__({'set': name, "value": value })
    def __repr__(self):
        return self.__attribute_call__('__repr__')
    def __str__(self):
        return self.__attribute_call__('__str__')
    def __getitem__(self,*args,**kwargs):
        return self.__request__({'call': '__getitem__', "args": args, "kwargs": kwargs})
    def __setitem__(self,*args,**kwargs):
        return self.__request__({'call': '__setitem__', "args": args, "kwargs": kwargs})

class FrontendSideRelay(object):
    def __init__(self,send_func):
        self.send_func = send_func
        self.objects = {}
    def request(self,object_name,something=None):
        if something is None:
            return FrontendObject(self,object_name)
        else:
            if type(something) is not str:
                something = json.dumps(something)
            r = self.send_func(object_name, something)
            #print ">>>",r
            if r[0] == "~":
                # special character for object references
                return FrontendObject(self,r[1:])
            try:
                j = json.loads(r)
                if type(j) is dict and 'type' in j.keys() and j['type'] == "SilverNode":
                    print "new silver Node"
                    return ZMQFrontendRelay([j['port']])
                return j
            except ValueError:
                print "Recieved non json message: ",r
                #raise


def __cmd__(obj,cmd):
    try:
        if type(cmd) is str:
            cmd = json.loads(cmd) # we expect a dict
        if 'dir' in cmd.keys():
            return dict([(k,callable(obj.__getattribute__(k))) for k in dir(obj)])
        if 'call' in cmd.keys():
            if hasattr(obj,cmd['call']):
                if callable(obj.__getattribute__(cmd['call'])):
                    if 'args' in cmd.keys() and 'kwargs' in cmd.keys():
                        r = obj.__getattribute__(cmd['call'])(*cmd['args'],**cmd['kwargs'])
                        return r
                    elif 'args' in cmd.keys():
                        r = obj.__getattribute__(cmd['call'])(*cmd['args'])
                        return r
                    elif 'kwargs' in cmd.keys():
                        r = obj.__getattribute__(cmd['call'])(**cmd['kwargs'])
                        return r
                    else:
                        r = obj.__getattribute__(cmd['call'])()
                        return r
                return obj.__dict__[cmd['call']]
        if 'set' in cmd.keys():
            # TODO
            if cmd['set'] in obj.__dict__.keys():
                return obj.__dict__[cmd['set']]
    except:
        if module_debug:
            raise
        else:
            pass
    return {}

class ObjectSideNamespace(dict):
    def __init__(self, *args, **kwargs):
        self.update(kwargs)
        self["namespace"] = self
    def eval(self,eval_string, output_variable=None):
        if module_debug:
            print "@evaluating",eval_string
        if output_variable is not None:
            self[output_variable] = eval(eval_string,{},self)
            return self[output_variable]
        else:
            return eval(eval_string,{},self)    
    def _exec(self,eval_string, output_variable=None):
        if module_debug:
            print "@exec",eval_string
        if output_variable is not None:
            self[output_variable] = eval(eval_string,{},self)
            return self[output_variable]
        else:
            exec eval_string in self
            if module_debug:
                print self.keys()


class ObjectSideRelay(object):
    def __init__(self, *args, **kwargs):
        self.objects = ObjectSideNamespace(**kwargs)
    def request(self,object_name,something):
        """ Expects valid json """
        if type(something) is not dict:
            something = json.loads(something)
        r = None
        if object_name == 'locals':
            return json.dumps(dict([(k,"~"+k) for k in self.objects.keys()]))
        if object_name in self.objects.keys():
            r = self.objects[object_name]
        if r is not None:
            r = __cmd__(r,something)
            try:
                return json.dumps(r)
            except:
                for k in self.objects.keys():
                    try:
                        assert(self.objects[k] == r)
                        return "~"+str(k)
                    except:
                        pass
                i = 0
                while str((object_name,i)) in self.objects.keys():
                    i += 1
                self.objects[str((object_name,i))] = r
                return "~"+str((object_name,i))
        else:
            return json.dumps({})

class ZMQSignedMessager(object):
    def __init__(self):
        self.key = "asdf"
        self.digest_history = []
        self.auth = hmac.HMAC(self.key, digestmod=hashlib.sha256)
    def sign(self, msg):
        if self.auth is None:
            return b''
        h = self.auth.copy()
        for m in msg:
            h.update(m)
        return str_to_bytes(h.hexdigest())
    def check(self,msg,signature):
        if self.auth is not None:
            if not signature:
                raise ValueError("Unsigned Message")
            if signature in self.digest_history:
                raise ValueError("Duplicate Signature: %r" % signature)
            self.digest_history.append(signature)
            if len(self.digest_history) > 100:
                self.digest_history.pop(0)
            check = self.sign(msg)
            if not compare_digest(signature, check):
                raise ValueError("Invalid Signature: %r" % signature)

class ZMQObjectRelay(ZMQSignedMessager):
    def __init__(self):
        super(ZMQObjectRelay,self).__init__()
        self.registered_nodes = {}
        self.process = None
        self.object_relay = ObjectSideRelay()
        self.class_names = { 'Experiment': silver.schedule.Experiment, 'Session': silver.schedule.Session }
        self.port = None
    def server(self,port="5556",port_return_queue=None):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        try:
            socket.bind("tcp://127.0.0.1:%s" % port)
            port_return_queue.put(port)
        except:
            if module_debug:
                print "Port in use... trying different port"
            actual_port = socket.bind_to_random_port("tcp://127.0.0.1", min_port=49152, max_port=65536, max_tries=100)
            port_return_queue.put(actual_port)
        while True:
            time.sleep(0.1)
            message = socket.recv()
            try:
                m = json.loads(message)
                if module_debug:
                    print 'loaded',m
                if type(m) == dict:
                    if 'signature' in m.keys() and 'request' in m.keys() and 'uuid' in m.keys():
                        self.check(m['uuid']+m['request'],m['signature'])
                    else:
                        if module_debug:
                            print "unsigned request!"
                    if 'create' in m.keys():
                        if not 'args' in m:
                            m['args'] = []
                        if not 'kwargs' in m:
                            m['kwargs'] = {}
                        if not 'name' in m:
                            m['name'] = str(uuid.uuid4())
                        if m['create'] in self.class_names:
                            self.object_relay.objects[m['name']] = self.class_names[m['create']](*m['args'],**m['kwargs'])
                            socket.send_string('~'+m['name'])
                        else:
                            socket.send_string(json.dumps("There was some mistake"))
                    elif 'object' in m.keys():
                        if 'request' in m.keys():
                            r = self.object_relay.request(m['object'],m['request'])
                        else:
                            r = self.object_relay.request(m['object'],m) # if the object name is one among many keys
                        socket.send_string(r)
                    else:
                        socket.send_string("wat?")
                else:
                    socket.send_string("wat?")
            except Exception as e:
                print "Error for:",message
                print str(e)
                socket.send_string("Exception "+str(e))
                if module_debug:
                    raise
    def createProcesses(self,port="5556"):
        if self.process is not None:
            return
        queue = Queue()
        self.process = Process(target=self.server, args=(port,queue))
        self.process.start()
        self.port = queue.get()

class CoreNode(ZMQObjectRelay):
    def __init__(self):
        super(CoreNode,self).__init__()
        self.open_sessions = []
        self.experiment_files = []
        self.processes = []
        import glob
        self.experiment_files.extend(glob.glob('Experiments/*.py'))
        self.namespace = self.object_relay.objects
        self.namespace["open_sessions"] = self.open_sessions
        self.namespace["experiment_files"] = self.experiment_files
        self.namespace["core"] = self
    def open_experiment(self,s):
        if s in self.namespace["experiment_files"]:
            self.namespace["experiment"] = silver.schedule.Experiment(s)
            return self.namespace["experiment"]
    def spawn_session(self,s):
        p = subprocess.Popen(["python","silver/zmq_node.py","session",s], stdout=subprocess.PIPE, bufsize=1)
        port = "None"
        for line in iter(p.stdout.readline, b''):
            if line.startswith('Port:'):
                port = int(line[len('Port:'):].strip())
                break
        self.processes.append({'process_id':p,'port':port})
        return {'type':'SilverNode','session':s,'port':port}


class SessionNode(ZMQObjectRelay):
    def __init__(self,session_file):
        super(SessionNode,self).__init__()
        self.this_is_a_session_node = True
        self.session = silver.schedule.Session(None,session_file)
        self.session.load()
        self.namespace = self.object_relay.objects
        self.namespace["session"] = self.session
        self.namespace["self"] = self
        self.processes = []
        self.threads = []
    def spawn_runner(self,uuid="None"):
        p = subprocess.Popen(["python","silver/zmq_node.py","runner",self.session.filename,uuid], stdout=subprocess.PIPE, bufsize=1)
        port = "None"
        for line in iter(p.stdout.readline, b''):
            if line.startswith('Port:'):
                port = int(line[len('Port:'):].strip())
                break
        self.processes.append({'process_id':p,'port':port})
        return {'type':'SilverNode','port':port}
    def run(self,uuid):
        task = self.session.root_task.find_uuid(uuid)
        #process = Process(target=task.run)
        #process.start()
        process = uuid
        self.threads.append(process)
        return {'process_id':self.threads.index(process)}
    def process_alive(self,pid):
        return self.threads[pid] != []#.is_alive()

class RunnerNode(ZMQObjectRelay):
    def __init__(self,session_file,uuid):
        super(RunnerNode,self).__init__()
        self.session = silver.schedule.Session(None,session_file)
        self.session.load()
        self.task_uuid = uuid
        self.namespace = self.object_relay.objects
        self.namespace["session"] = self.session
        self.namespace["self"] = self
    def run(self,task_id=None):
        if task_id is None or task_id is "None":
            next_task = self.namespace['session'].root_task.find_uuid(self.task_uuid)
            next_task.run_one()
            self.session.save()
        else:
            next_task = self.namespace['session'].root_task.find_uuid(self.task_id)
            next_task.run_one()
            self.session.save()
        return "Run complete."

class ZMQFrontendRelay(ZMQSignedMessager):
    def __init__(self, ports = ["5556"]):
        super(ZMQFrontendRelay,self).__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        for port in ports:
            self.socket.connect ("tcp://localhost:%s" % port)
        self.object_relay = FrontendSideRelay(self.send)
        self.namespace = self.object_relay.request("namespace")
    def send(self,object_name="",some_string=""):
        message_uuid = str(uuid.uuid4())
        self.socket.send_string(json.dumps( {"uuid":message_uuid, "object":object_name, "request":some_string , "signature": self.sign(message_uuid+str(some_string)) }) )
        return self.socket.recv()
    def Experiment(self,filename,name="e"):
        self.object_relay.request('',{'create':'Experiment', 'args': [filename], 'name':name})