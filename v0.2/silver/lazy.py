"""

    This module provides a lazy page management to keep large files in memory on the 
    server and only send pieces on request.

"""

import re
from uuid import uuid4

def html_dec(html, page_uuid=""):
    return """
    <html>
    <head>
    <style>

    </style>
    <link rel="stylesheet" type="text/css" href="/silver.css" />
    <script src="/jquery-2.1.3.min.js"></script>
    <script src="/silver.js"></script>
    <script>
        var depth_steps = 3;
        var page_uuid = '"""+str(page_uuid)+"""';
        $( document ).ready(function() {
            make_data_pieces(document);
            function make_reference(ref) {
                //ref.data("uuid", ref.html())
                ref.html("");
                var title = "";
                if (ref.data("title") != undefined && ref.data("title") != "") {
                    title = ref.data("title")
                }
                if (title == "") {
                    title = "Open";
                }
                var a = $("<a>open "+title+"</a>").appendTo(ref);
                a.click(function () {
                    $.ajax({
                        url: "/?/"""+str(page_uuid)+"""/"+ref.data("uuid")+"/"+depth_steps,
                        context: document.body
                    }).done(function(data) {
                        ref.html(data);
                        ref.find('.reference').each(function() {
                            make_reference($(this));
                        });
                    });
                });
            }
            $('.reference').each(function() {
                var ref = $(this)
                make_reference(ref);                
            });
        });
    </script>
    </head>
    <body>
    <div id='sidebar'>
    <ul class='toc'></ul>
    </div>
    <input style="display: none;" id="filter_input"/>
    <div id='main'>
    <div class='content'>
    """+html+  """
    </div>
    </div>
    </body>
    </html>
    """


class LazyPageCollector(object):
    def __init__(self):
        self.open_pages = {}
        self.page_paths = {}
    def open_page(self, path):
        if path in self.page_paths:
            return self.open_pages[self.page_paths[path]]
        else:
            return LazyPage()

class LazyPage(object):
    def __init__(self, uuid=None):
        if uuid is None:
            uuid = uuid4()
        self.uuid = uuid
        self.objects = {}
        self.root_object = None
    def add(self, obj, template=None):
        obj_uuid = uuid4()
        self.objects[str(obj_uuid)] = LazyPageObject(obj, template=template, uuid = uuid4())
        if self.root_object is None:
            self.root_object = obj_uuid
    def render_page(self, depth = 1):
        return html_dec(self.render(depth=depth), page_uuid=self.uuid)
    def render(self, object_uuid = None, depth = 1):
        if object_uuid is None:
            object_uuid = self.root_object
        #print "rendering ",object_uuid,"at",depth
        content = self.objects[str(object_uuid)].get()
        _html = content['html']
        objects = content['objects']
        self.objects.update(objects)
        if depth > 1:
            for k in objects.keys():
                obj_html = self.render(k,depth=depth-1)
                _html = re.sub("<div class='reference' data-uuid='"+str(k)+"'[^>]*><\/div>",obj_html,_html)
        return _html

class LazyPageObject(object):
    def __init__(self, obj, template=None, uuid=None, depth = 0):
        if uuid is None:
            uuid = uuid4()
        self.uuid = uuid
        self.obj = obj
        self.html = None
        self.template = template
        self.depth = depth
        self.content_objects = {}
    def __str__(self):
        return self.get_reference()
    def get_reference(self,title=''):
        return "<div class='reference' data-uuid='"+str(self.uuid)+"' data-title='"+str(title)+"'></div>"
    def get(self):
        if self.html is None:
            content_objects = {}
            html = ""
            if type(self.template) == str:
                html = self.template
                template = self.template
                inherit_template = self.template
            elif type(self.template) == list:
                html = self.template[0]
                template = self.template[0]
                if len(self.template) > 1:
                    inherit_template = self.template[1:]
                else:
                    inherit_template = self.template[0]
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
                        if type(v) == dict:
                            v = LazyPageObject(v,inherit_template, depth=self.depth + 1)
                        elif type(v) == list:
                            v = LazyPageObject(v,inherit_template, depth=self.depth + 1)
                        if hasattr(v,'to_lazy'):
                            v = v.to_lazy()
                        if hasattr(v,'uuid'):
                            if str(v.uuid) in content_objects:
                                print('UUID collision!')
                            content_objects[str(v.uuid)] = v
                            html = html.replace("<"+str(k)+"/>",v.get_reference(title=k))
                        else:
                            html = html.replace("<"+str(k)+"/>",str(v))
                    if "%"+str(k) in html:
                        if type(v) == dict:
                            v = LazyPageObject(v,inherit_template, depth=self.depth + 1)
                        elif type(v) == list:
                            v = LazyPageObject(v,inherit_template, depth=self.depth + 1)
                        if hasattr(v,'to_lazy'):
                            v = v.to_lazy()
                        if hasattr(v,'uuid'):
                            if str(v.uuid) in content_objects:
                                print('UUID collision!')
                            content_objects[str(v.uuid)] = v
                            html = html.replace("%"+str(k),v.get_reference(title=k))
                        else:
                            html = html.replace("%"+str(k),str(v))
                if "<dict/>" in html:
                    dict_text = ""
                    for k in self.obj.keys():
                        v = self.obj[k]
                        if type(v) == dict:
                            v = LazyPageObject(v,inherit_template, depth=self.depth + 1)
                        elif type(v) == list:
                            v = LazyPageObject(v,inherit_template, depth=self.depth + 1)
                        if hasattr(v,'to_lazy'):
                            v = v.to_lazy()
                        if hasattr(v,'uuid'):
                            if str(v.uuid) in content_objects:
                                print('UUID collision!')
                            content_objects[str(v.uuid)] = v
                            dict_text += "<div class='key_value_pair'><div class='key key_"+str(k)+"'>"+str(k)+"</div><div class='value key_"+str(k)+"'>"+v.get_reference(title=k)+"</div></div>"
                        else:
                            dict_text += "<div class='key_value_pair'><div class='key key_"+str(k)+"'>"+str(k)+"</div><div class='value key_"+str(k)+"'>"+str(v)+"</div></div>"
                    html = html.replace("<dict/>",dict_text)
            elif type(self.obj) is list:
                list_html = "<div class='list'>"
                for v in self.obj:
                    if type(v) == dict:
                        v = LazyPageObject(v,inherit_template, depth=self.depth + 1)
                    elif type(v) == list:
                        v = LazyPageObject(v,inherit_template, depth=self.depth + 1)
                    if hasattr(v,'to_lazy'):
                        v = v.to_lazy()
                    if hasattr(v,'uuid'):
                        if str(v.uuid) in content_objects:
                            print('UUID collision!')
                        content_objects[str(v.uuid)] = v
                        list_html += "<div class='list_item'>"+v.get_reference()+"</div>"
                    else:
                        list_html += "<div class='list_item'>"+str(v)+"</div>"
                list_html += "</div>"
                if "<list/>" in html:
                    html = html.replace("<list/>",list_html)
                else:
                    html = list_html
            self.html = html
            self.content_objects = content_objects
            #self.object = None
            return {'html':self.html,'objects':self.content_objects}
        else:
            return {'html':self.html,'objects':self.content_objects}
