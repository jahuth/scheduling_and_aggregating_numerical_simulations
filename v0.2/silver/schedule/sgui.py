import os
import Tkinter as tk
import ttk

import threading
import logging
import random


tree_app = {}

class App(tk.Frame):
    def __init__(self, name, master, list_of_dicts, keys={}):
        tk.Frame.__init__(self, master)
        master.wm_title(name)
        self.keys = keys
        self.tree = ttk.Treeview(master, columns=tuple(self.keys.keys()))
        ysb = ttk.Scrollbar(master, orient='vertical', command=self.tree.yview)
        xsb = ttk.Scrollbar(master, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscroll=ysb.set, xscroll=xsb.set)
        self.tree.heading('#0', text='Name', anchor='w')
        for i,k in enumerate(self.keys.keys()):
            self.tree.heading('#'+str(i+1), text=self.keys[k], anchor='w')
        
        self.process_dict('', list_of_dicts)

        #self.tree.grid(row=0, column=0)
        ysb.pack(side="right",fill="y")
        xsb.pack(side="bottom",fill="x")
        self.tree.pack(side="left",fill="both", expand=True)
        #ysb.grid(row=0, column=1, sticky='ns')
        #xsb.grid(row=1, column=0, sticky='ew')
        #self.grid()
        master.geometry('1000x600+0+0')
        tree_app[name] = self
        self.thread = None

    def set_keys(self, keys):
        self.keys = keys
        self.tree.configure(columns=tuple(self.keys.keys()))
        for i,k in enumerate(self.keys.keys()):
            self.tree.heading('#'+str(i+1), text=self.keys[k], anchor='w')
            self.tree.column('#'+str(i+1),minwidth=0,width=300, stretch=tk.YES)
    def clear_dict(self):
        for c in self.tree.get_children():
            self.tree.delete(c)

    def process_dict(self, parent, list_of_dicts):
        for d in list_of_dicts:
            oid = self.tree.insert(parent, 'end', text=d["name"], open=True, values=tuple([d[k] for k in self.keys.keys()]))
            if "subtasks" in d:
                self.process_dict(oid, d["subtasks"])


example_list_of_dicts = [
{'name':'Test1','status':'pending','completion_time':None},
{'name':'Test2','status':'pending','completion_time':None, 'subtasks': [
{'name':'SubTest1','status':'pending','completion_time':None},
{'name':'SubTest2','status':'pending','completion_time':None, 'subtasks': []},
{'name':'SubTest3','status':'pending','completion_time':None},
{'name':'SubTest4','status':'pending','completion_time':None}
]},
{'name':'Test3','status':'pending','completion_time':None},
{'name':'Test4','status':'pending','completion_time':None}

]

running_threads = {}


def showGUITable(name="",list_of_dicts=example_list_of_dicts,keys={'status':'Status','completion_time':'Completion Time'}):
    root = tk.Tk()
    app = App(name,root, list_of_dicts,keys)
    app.mainloop()


def f():
    print('thread function')
    return

import time 

def startGUITable(name='',list_of_dicts=example_list_of_dicts,keys={'status':'Status'}):
    if name in tree_app:
        if tree_app[name].thread is not None:
            if tree_app[name].thread.isAlive():
                tree_app[name].clear_dict()
                tree_app[name].set_keys(keys)
                tree_app[name].process_dict('',list_of_dicts)
                return
    t = threading.Thread(target=showGUITable,args=(name,list_of_dicts,keys))
    t.setDaemon(True)
    t.start()
    while not name in tree_app:
        time.sleep(0.1)
    tree_app[name].thread = t
