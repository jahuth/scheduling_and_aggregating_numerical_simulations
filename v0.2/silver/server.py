import sys, os
os.environ['http_proxy']=''
from CGIHTTPServer import CGIHTTPRequestHandler
import BaseHTTPServer
from silver import schedule
import json
from silver import lazy


class LazyTemplateNode(object):
    def __init__(self, obj, template, depth = 0):
        self.obj = obj
        self.template = template
        self.html = None
        self.burn = True
        self.depth = depth
    def reference(self):
        return "<div class='reference'></div>"
    def render(self):
        if self.html is None:
            if type(self.template) == str:
                html = self.template
                template = self.template
                inherit_template = self.template
            elif type(self.template) == list:
                html = self.template[0]
                template = self.template[0]
                inherit_template = self.template[1:]
            elif type(self.template) == dict:
                template = self.template
                inherit_template = self.template
                if type(self.obj) in self.template:
                    html = self.template[type(self.obj)]
                else:
                    html = ""
            if type(self.obj) is dict:
                for k in self.obj.keys():
                    v = self.obj[k]
                    if "<"+str(k)+"/>" in html:
                        html = html.replace("<"+str(k)+"/>",str(v))
                    if "%"+str(k) in html:
                        html = html.replace("%"+str(k),str(v))
                if "<dict/>" in html:
                    dict_text = ""
                    for k in self.obj.keys():
                        v = self.obj[k]
                        if type(v) == dict:
                            v = LazyTemplateNode(v,inherit_template)
                        elif type(v) == list:
                            v = LazyTemplateNode(v,inherit_template)
                        elif hasattr(v, '__dict__'):
                            v = LazyTemplateNode(v.__dict__,inherit_template)
                        dict_text += "<div class='key_value_pair'><div class='key key_"+str(k)+"'>"+str(k)+"</div><div class='value key_"+str(k)+"'>"+str(v)+"</div></div>"
                    html = html.replace("<dict/>",dict_text)
            elif type(self.obj) is list:
                list_html = "<div class='list'>"
                for v in self.obj:
                    if type(v) == dict:
                        v = LazyTemplateNode(v,inherit_template)
                    elif type(v) == list:
                        v = LazyTemplateNode(v,inherit_template)
                    elif hasattr(v, '__dict__'):
                        v = LazyTemplateNode(v.__dict__,inherit_template)
                    list_html += "<div class='list_item'>"+str(v)+"</div>"
                list_html += "</div>"
                if "<list/>" in html:
                    html = html.replace("<list/>",list_html)
                else:
                    html = list_html
            self.html = html
            if self.burn:
                self.object = None
            return self.html
        else:
            return self.html
    def __str__(self):
        return self.render()

class SilverFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.type = None
        self.object = None
        if not os.path.exists(self.filename) and not os.path.exists(self.filename+'.json') and not os.path.exists(self.filename+'.tar'):
            self.exists = False
        else:
            self.exists = True
        if self.exists:
            self.shallow_check()
    def thorough_check(self):
        if not os.path.exists(self.filename) and not os.path.exists(self.filename+'.json') and not os.path.exists(self.filename+'.tar'):
            self.type = None
            return None
        if os.path.isdir(self.filename):
            self.type = 'directory'
        else:
            if self.filename.endswith('.tar'):
                # try to make it a tar data container
                try:
                    self.object = schedule.TaskDataContainerTar(None, name=self.filename)
                except:
                    raise
                else:
                    self.type = 'silver.schedule.TaskDataContainerTar'
            if self.type is None:
                try:
                    self.object = schedule.Session(None,self.filename)
                    self.object.check_json()
                except:
                    pass
                else:
                    self.type = 'silver.schedule.Session'
            if self.type is None:
                try:
                    self.object = schedule.Session(None,self.filename+'.json')
                    self.object.check_json()
                except:
                    pass
                else:
                    self.filename = self.filename+'.json'
                    self.type = 'silver.schedule.Session'
            if self.type is None:
                try:
                    self.object = schedule.Experiment(self.filename)
                    self.object.get_sandbox()
                except:
                    pass
                else:
                    self.type = 'silver.schedule.Experiment'
        return self.type
    def shallow_check(self):
        if not os.path.exists(self.filename) and not os.path.exists(self.filename+'.json'):
            self.type = None
            return None
        if os.path.isdir(self.filename):
            self.type = 'directory'
        else:
            if self.filename.endswith('.tar'):
                self.type = 'silver.schedule.TaskDataContainerTar'
            if self.filename.endswith('Session 1') or self.filename.endswith('Session 1.json'):
                self.type = 'silver.schedule.Session'
            if self.filename.endswith('.py'):
                self.type = 'silver.schedule.Experiment'
        print self.type
        return self.type
    def __str__(self):
        print "tostr/type:",type(self.object)
        self.thorough_check()
        if self.type == 'silver.schedule.Experiment':
            try:
                self.object = schedule.Experiment(self.filename)
                self.object.get_sandbox()
            except:
                pass
            else:
                print dir(self.object)
                sessions = [{"session":s} for s in self.object.load_sessions()]
                print sessions
                obj_text =  LazyTemplateNode(sessions,["<ul><list/></ul>", "<li><a href='/load/<session/>.json'><session/></a></li>"])
                return self.filename + " (" + str(self.type) + ")<br/>" + str(obj_text) + "<hr/>"
        elif self.type == 'silver.schedule.Session':
            self.object = schedule.Session(None,self.filename)
            self.object.load()
            print type(self.object)
            #obj_text = str(self.object.__dict__)#.to_html()
            ks = ['name','data_path']#,'cmd','command','variables','data_path','data','parallel','blocked']
            def r(node):
                _d = {}
                for k in ks:
                    if k in node.__dict__:
                        _d[k] = node.__dict__[k]
                if hasattr(node,'subtasks') and node.subtasks is not None:
                    _d['subtasks'] = []
                    for t in node.subtasks:
                        _d['subtasks'].append(r(t))
                return _d
            tree = r(self.object.root_task)
            obj_text = lazy.LazyPageObject({
                    'filename': self.object.filename,
                    'tasks': lazy.LazyPageObject(tree, template="<div class='task'><name/> (<a href='/load/<data_path/>'>load Data</a>)<br/><div class='subtasks'><subtasks/></div></div>"),
                    'root_task name': self.object.root_task.name,
                    'guess_total_time': self.object.guess_total_time(),
                    'commands': self.object.commands
                },"<dict/>")
            return str(obj_text)
        elif self.type == 'silver.schedule.TaskDataContainerTar':
            obj_text = str(self.object.__dict__.keys()) + self.object.to_html()
        else:
            obj_text = str(self.object) 
            with open(self.filename,'r') as f:
                obj_text = f.read()#lazy.LazyPageObject([obj_text, f.read()],"<list/>")
        return self.filename + " (" + str(self.type) + ")<br/>" + str(obj_text) + "<hr/>"
    def to_lazy(self):
        self.thorough_check()
        if self.type == 'silver.schedule.Experiment':
            try:
                self.object = schedule.Experiment(self.filename)
                self.object.get_sandbox()
            except:
                pass
            else:
                print dir(self.object)
                sessions = [{"session":s} for s in self.object.load_sessions()]
                print sessions
                obj_text =  LazyTemplateNode(sessions,["<ul><list/></ul>", "<li><a href='/load/<session/>.json'><session/></a></li>"])
                return lazy.LazyPageObject([], self.filename + " (" + str(self.type) + ")<br/>" + str(obj_text) + "<hr/>")
        elif self.type == 'silver.schedule.Session':
            self.object = schedule.Session(None,self.filename)
            self.object.load()
            print type(self.object)
            #obj_text = str(self.object.__dict__)#.to_html()
            ks = ['name','data_path']#,'cmd','command','variables','data_path','data','parallel','blocked']
            def r(node):
                _d = {}
                for k in ks:
                    if k in node.__dict__:
                        _d[k] = node.__dict__[k]
                if hasattr(node,'subtasks') and node.subtasks is not None:
                    _d['subtasks'] = []
                    for t in node.subtasks:
                        _d['subtasks'].append(r(t))
                return _d
            tree = r(self.object.root_task)
            obj_text = lazy.LazyPageObject({
                    'filename': self.object.filename,
                    'tasks': lazy.LazyPageObject(tree, template="<div class='task'><name/> (<a href='/load/<data_path/>.tar'>load Data</a>)<br/><div class='subtasks'><subtasks/></div></div>"),
                    'root_task name': self.object.root_task.name,
                    'guess_total_time': self.object.guess_total_time(),
                    'commands': self.object.commands
                },"<dict/>")
            return obj_text
        elif self.type == 'silver.schedule.TaskDataContainerTar':
            obj_text = str(self.object.__dict__.keys()) + self.object.to_html()
        else:
            obj_text = str(self.object) 
            with open(self.filename,'r') as f:
                obj_text = f.read()#lazy.LazyPageObject([obj_text, f.read()],"<list/>")
        return lazy.LazyPageObject([], self.filename + " (" + str(self.type) + ")<br/>" + str(obj_text) + "<hr/>")

class SilverExperiment(object):
    def __init__(self,path, name):
        self.path = path
        self.name = name
        self.id = name

class Server(object):
    def __init__(self):
        self.namespace = {}
        self.last_sessions = []
        self.open_experiments = {}
        self.open_pages = {}
        try:
            with open('last_sessions.json', 'r') as fp:
                self.last_sessions = json.load(fp)
        except:
            pass
    def load_file(self,path):
        path = path.replace("%20"," ")
        f = SilverFile(path)
        if f is not None:
            f.thorough_check()
            self.namespace[path] = f
        return f
    def get(self,path):
        #the_data_path = "/home/jacob/Projects/Silversight_beta/Tasks/angelo2_even_more_noise_0.75/.Stimulus 13 Phi:0.0  Speed:20.0 Noise:0.75 Contrast:0.75Trial 1 (0.025, 0.075, 10)Running NetworkRunning Network_data.tar"
        #the_data = schedule.TaskDataContainerTar(None,the_data_path,load=True)
        #return {'status':200, 'text': {'data':the_data.to_html()} }
        if path.startswith("/?"):
            p = path.split("/")
            print self.open_pages.keys()
            print p
            if p[2] in self.open_pages.keys():
                depth = 1
                try:
                    depth = int(p[4])
                except:
                    pass
                print depth, type(depth)
                return {'status':200, 'text': self.open_pages[p[2]].render(p[3],depth = depth) }
        if path.startswith("/Namespace"):
            return self.handle_namespace(path, "/Namespace")
        if path.startswith("/$"):
            return self.handle_namespace(path, "/$")
        if path.startswith("/load/"):
            import glob
            try:
                path = path[len("/load/"):].replace("%20"," ")
                if path == "":
                    return self.handle_directory(path)
                f = SilverFile(path)
                print "loading:",f.filename, f.exists, f.type
                if f.type == 'directory':
                    return self.handle_directory(path)
                else:
                    f = self.load_file(path)
                    if f is not None and f.type is not None:
                        print "Going over to namespace..."
                        self.namespace[path] = f
                        return self.handle_namespace("/"+path,"")
                    else:
                        print "using raw file..."
                        fr = ""
                        with open(path,'r') as f:
                            fr = f.read()
                        return self.boiler_plate("<textarea style='width: 100%; height: 90%;'>"+fr+"</textarea>")
            except:
                raise
                return self.boiler_plate("Not found: "+path)
        text = "Not Found: "+str(path) + "<br/><a href='/Namespace'>Namespace</a> | <a href='/Experiments'>Experiments</a>"
        text += "<ul>"
        for filename in self.last_sessions:
            w = SilverFile(filename)
            if w.exists:
                text += "<li><a href='/load/"+w.filename+"'>" + w.filename + "</a></li>"
            else:
                text += "<li><i>" + w.filename + "</i></li>"
        text += "</ul>"
        return self.boiler_plate(text)
    def handle_namespace(self,path,invocation="/Namespace"):
        if path == invocation:
            text = "<h1>Namespace:</h1>"
            text += "<ul>"
            for k in self.namespace.keys():
                k = k.replace("%20"," ")
                try:
                    text += "<li><b><a href='/Namespace/"+k+"'>"+k+"</a></b>: "+str(type(self.namespace[k]))+"</li>"
                except:
                    text += "<li><b>"+k+"</b>: -</li>"
            text += "</ul>"
            return self.boiler_plate(text)
        else:
            path = path[len(invocation):].replace("%20"," ")
            if path[1:] in self.namespace.keys():
                return self.boiler_plate(self.namespace[path[1:]])
            else:
                return self.boiler_plate("Not found")
    def handle_directory(self,path):
        if path == "":
            path = "./"
        if not path.endswith("/"):
            path = path + "/"
        text = ""
        text += "<a href='/load/'>#</a> "
        for i,p in enumerate(path.split("/")):
            text += "/ <a href='/load/"+("/".join(path.split("/")[:i+1]))+"'>"+p+"</a> "
        import os
        files = os.listdir(path)
        directory_list = [{'path': path+filename, 'filename': filename} for filename in files if os.path.isdir(path+"/"+filename)]
        file_list = [{'path': path+filename, 'filename': filename} for filename in files if not os.path.isdir(path+"/"+filename)]
        l = lazy.LazyPageObject({'path_navigation': text,
                            'directories':
                        lazy.LazyPageObject(directory_list,"<li><b><a href='/load/<path/>'><filename/></a></b></li>"), 
                            'files':
                        lazy.LazyPageObject(file_list,"<li><a href='/load/<path/>'><filename/></a></li>")
                        }, "<path_navigation/><ul><directories/></ul>\n<ul><files/></ul>")
        return self.boiler_plate(l)
    def boiler_plate(self,text):
        menu = lazy.LazyPageObject([
                {"url":"/$","title":"Namespace"},
                {"url":"/load/","title":"Load"},
                {"url":"/settings/","title":"Settings"},
                {"url":"/kernels/","title":"Kernels"}],
            ["<div class='menu'><list/></div>","<a href='%url'>%title</a>"])
        page = lazy.LazyPage()
        self.open_pages[str(page.uuid)] = page
        page_tree = {
                'menu': menu,
                'openfiles': "<div>" + " | ".join(["<a href='"+str(o)+"'>"+self.namespace[o].name+"</a>" for o in self.open_experiments.keys() if self.namespace[o].type == 'silver.schedule.Experiment']) + "</div>",
                'content': text,
                'additional content': {'some':{'more':{'more':{'more':{'more':'content'}}}}}
            }
        page.add(page_tree,["<menu/><openfiles/><content/><additional content/>", "<dict/>"])
        #pre_text += "<div>" + " | ".join(["<a href='"+str(o)+"'>"+self.namespace[o].name+"</a>" for o in self.open_experiments.keys() if self.namespace[o].type == 'silver.schedule.Experiment']) + "</div>"
        #post_text =""
        return {'status':200, 'text': page.render_page(depth = 4) }


static_files = ['/jquery-2.1.3.min.js','/silver.css','/silver.js']

class silverHandler(CGIHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        CGIHTTPRequestHandler.__init__(self, *args,**kwargs)
    def do_GET(self):
        global silver_server
        if self.path in static_files:
            self.send_response(200)
            if self.path.endswith('.html'):
                self.send_header('Content-type', 'text/html')
            elif self.path.endswith('.js'):
                self.send_header('Content-type', 'application/javascript')
            elif self.path.endswith('.css'):
                self.send_header('Content-type', 'text/css')
            elif self.path.endswith('.json'):
                self.send_header('Content-type', 'application/json')
            else:
                self.send_header('Content-type', 'text/plain')
            self.send_header('charset', 'utf-8')
            self.end_headers()
            with open(self.path[1:],'r') as f:
                self.wfile.write(f.read())
            return 
        if silver_server is None:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('charset', 'utf-8')
            self.end_headers()
            self.wfile.write("test")
        else:
            response = silver_server.get(self.path)
            self.send_response(response["status"])
            self.send_header('Content-type', 'text/html')
            self.send_header('charset', 'utf-8')
            self.end_headers()
            self.wfile.write(response["text"])

silver_server = None
def run_server(host = '127.0.0.1', port = 9090, parse_command_line = True):
    global silver_server
    cgi_directories = ["/cgi-bin/"]
    protocol = "HTTP/1.0"
    #host = '127.0.0.1'
    #port = 9090
    if parse_command_line:
        if len(sys.argv) > 1:
            arg = sys.argv[1]
            if ':' in arg:
                host, port = arg.split(':')
                port = int(port)
            else:
                try:
                    port = int(sys.argv[1])
                except:
                    host = sys.argv[1]
    server_address = (host, port)
    silver_server = Server()
    silverHandler.protocol_version = protocol
    httpd = BaseHTTPServer.HTTPServer(server_address, silverHandler)

    sa = httpd.socket.getsockname()
    print "Serving HTTP on", sa[0], "port", sa[1], "..."
    httpd.serve_forever()