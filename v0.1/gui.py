from __future__ import print_function
import os

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.qt.manager import QtKernelManager
from IPython.kernel.multikernelmanager import MultiKernelManager
from IPython.lib import guisupport

use_qt4= True

import sys
if use_qt4:
    from PyQt4.QtGui import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QTreeView
    from PyQt4.QtCore import QUrl
    from PyQt4 import QtCore, QtGui
    from PyQt4.QtWebKit import QWebView
else:
    from PySide.QtGui import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QTreeView
    from PySide.QtCore import QUrl
    from PySide import QtCore, QtGui
    from PySide.QtWebKit import QWebView
import time
import json
import re
import subprocess
import datetime

from cgi import escape

sys.path.insert(0, ".")

qt_app = QApplication(sys.argv)

import silver.schedule as schedule

import misc_gui
misc_gui.use_qt4 = use_qt4
from misc_gui import *

class FileWidget(QtGui.QListWidgetItem):
    def __init__(self, parent, filename):
        QtGui.QListWidgetItem.__init__(self,filename)
        self.filename = filename
        self.type = None
        if not os.path.exists(self.filename) and not os.path.exists(self.filename+'.json'):
            self.exists = False
        else:
            self.exists = True
        if self.exists:
            self.shallow_check()
            #self.text(self.filename + " ("+str(self.type)+")")
        #self.type_label = QtGui.QLabel(str(self.type))
        #self.filename_label = QtGui.QLabel(self.filename)
        #layout = QVBoxLayout()
        #layout.addWidget(self.filename_label)
        #layout.addWidget(self.type_label)
        #self.setLayout(layout)
    def thorough_check(self):
        if not os.path.exists(self.filename) and not os.path.exists(self.filename+'.json'):
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
                    pass
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
                    self.object = schedule.Experiment(self.filename)
                    #self.object.get_sandbox()
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
        return self.type

class OpenGui(QWidget):

    '''  '''

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.selected_experiment = None
        self.layout = QVBoxLayout()
        self.view = Browser()
        html = '''<html>
        <head>
        <title>A Sample Page</title>
        </head>
        <body>
        <h1>No configuration selected</h1>
        
        </body>
        </html>'''

        self.view.setHtml(html)

        self.sessions_list = QtGui.QListWidget()

        self.last_session_list = QtGui.QListWidget()

        self.file_layout = QHBoxLayout()
        self.config_layout = QHBoxLayout()

        self.lbl = QtGui.QLabel('No file selected')
        self.file_layout.addWidget(self.lbl)

        btn = QtGui.QPushButton('Choose file', self)
        self.file_layout.addWidget(btn)
        self.last_sessions = []
        try:
            with open('last_sessions.json', 'r') as fp:
                self.last_sessions = json.load(fp)
        except:
            pass
        self.update_last_sessions()

        def open_file():
            fname = QtGui.QFileDialog.getOpenFileName(self, 'Select file')
            if fname:
                self.lbl.setText(fname)
                with open(fname) as f:
                    self.view.setHtml(f.read().replace('\n', '<br\>'))
                experiment = schedule.Experiment(fname)
                self.showsessions(experiment)
            else:
                self.lbl.setText('No file selected')
        self.connect(btn, QtCore.SIGNAL('clicked()'), open_file)
        self.layout.addLayout(self.file_layout)
        self.path = './'

        # create model
        model = QtGui.QFileSystemModel()
        model.setRootPath(self.path)
        self.session_layout = QtGui.QVBoxLayout()
        self.treeView = QtGui.QTreeView()
        # set the model
        self.treeView.setModel(model)
        self.treeView.setRootIndex(model.index(self.path))
        self.action_buttons = QtGui.QVBoxLayout()
        #self.connect(self.treeView.selectionModel(), QtCore.SIGNAL('selectionChanged(QItemSelection, QItemSelection)'), self.check_file)
        self.sessions_list.currentItemChanged.connect(self.load_session)
        self.last_session_list.currentItemChanged.connect(self.load_session)
        self.treeView.clicked.connect(self.check_file)
        self.config_layout.addWidget(self.last_session_list)
        self.config_layout.addWidget(self.treeView)
        self.config_layout.addLayout(self.session_layout)
        self.session_layout.addWidget(self.sessions_list)
        self.session_layout.addLayout(self.action_buttons)
        #self.config_layout.addWidget(self.view)
        self.layout.addLayout(self.config_layout)
        self.setLayout(self.layout)
        self.show()

    def save_last_sessions(self,additional_session = None):
        if additional_session is not None:
            self.last_sessions.append(additional_session)
            self.update_last_sessions()
        with open('last_sessions.json', 'wb') as fp:
            json.dump(self.last_sessions,fp)
    def update_last_sessions(self):
        try:
            with open('last_sessions.json', 'r') as fp:
                new_last_sessions = json.load(fp)
            self.last_sessions.extend(new_last_sessions)
        except:
            pass
        self.last_sessions = sorted(list(set(self.last_sessions)))
        self.last_session_list.clear()
        for filename in self.last_sessions:
            w = FileWidget(self,filename)
            if w.exists:
                self.last_session_list.addItem(w)
    def load_session(self, curr, prev):
        if curr is None:
            return
        self.last_session_list.setEnabled(False)
        if self.selected_experiment is not None:
            sess = self.selected_experiment.create_session(curr.text())
            sess.load()
            self.parent.open(sess)
        else:
            sess = schedule.Session(None,curr.text())
            sess.load()
            self.selected_experiment = sess.experiment
            self.parent.open(sess)            
        self.last_session_list.setEnabled(True)
    def showsessions(self, experiment):
        try:
            self.selected_experiment = experiment
            sessions = experiment.load_sessions()
            self.sessions_list.clear()
            for s in sessions:
                self.sessions_list.addItem(QtGui.QListWidgetItem(s))
            sand = experiment.get_sandbox()
            self.selected_sandbox = sand
            self.session_buttons = {}
            while self.action_buttons.count():
                item = self.action_buttons.takeAt(0)
                item.widget().deleteLater()
            for a in sand.actions.keys():
                new_button = QtGui.QPushButton( a )
                self.session_buttons[a] = new_button
                new_button.clicked.connect(self.func)
                self.action_buttons.addWidget(new_button)
        except Exception as e:
            print(e)
            pass
    def func(self):
        if self.sender() is not None:
            aa = self.sender().text()
            kwargs = self.selected_sandbox.actions[aa]['kwargs']
            """for k in kwargs:
                r, ok = QtGui.QInputDialog.getText(self, 'Input Dialog', 'Enter '+k+':')
                if ok and r != '':
                    kwargs[k] = r"""
            kwargs, ok = KwargDialog.getKwargs(self, kwargs)
            kwargs['experiment'] = self.selected_experiment
            ret = self.selected_sandbox.actions[aa]['function'](**kwargs)
            try:
                #print(ret.filename)
                #print(ret.walk_task_tree)
                ret.save()
                self.parent.open(ret)
                self.showsessions(self.selected_experiment)
            except:
                raise

    def check_file(self, index):
        from os.path import isdir, isfile, join
        indexItem = self.treeView.model().index(index.row(), 0, index.parent())
        # path or filename selected
        fileName = self.treeView.model().fileName(indexItem)
        # full path/filename selected
        filePath = self.treeView.model().filePath(indexItem)
        #print(filePath)
        if isdir(filePath):
            self.path = filePath
        else:
            experiment = schedule.Experiment(filePath)
            self.showsessions(experiment)

class SilverTaskRunnerThread(QtCore.QThread):
    def __init__(self, func, kwargs={}):
        super(SilverTaskRunnerThread , self).__init__()
        self.func = func
        self.kwargs = kwargs
    def run(self):
        print("runner started")
        try:
            ret = self.func(**self.kwargs)
        except:
            print("runner failed.")
            self.emit(QtCore.SIGNAL('update(QString)'), str(False))
        else:
            print("runner ended")
            self.emit(QtCore.SIGNAL('update(QString)'), str(ret))

class SilverResultsWidget(QtGui.QSplitter):

    def __init__(self,main_widget,session,session_widget):
        QtGui.QSplitter.__init__(self,QtCore.Qt.Horizontal)
        self.session = session
        self.main_widget = main_widget
        self.right_side_layout = QVBoxLayout()
        self.left_side_layout = QVBoxLayout()
        right_side_widget = QtGui.QWidget()
        right_side_widget.setLayout(self.right_side_layout)
        left_side_widget = QtGui.QWidget()
        left_side_widget.setLayout(self.left_side_layout)
        self.addWidget(right_side_widget)
        self.addWidget(left_side_widget)
        self.ipython = SilverIPython(kernel_manager=self.main_widget.kernel_manager)
        self.ipython.execute("""import silver
silver_session = silver.schedule.Session(None,filename='"""+session.filename+"""')
silver_session.load()""")
        self.ipython.push_vars(silver_session=session)
        self.ipython.push_vars(ipython_widget=self.ipython)
        self.right_side_layout.addWidget(self.ipython)
        self.session_buttons = {}
        sand = self.session.experiment.get_sandbox()
        for a in sand.result_actions.keys():
            new_button = QtGui.QPushButton( a )
            self.session_buttons[a] = new_button
            new_button.clicked.connect(self.func)
            self.left_side_layout.addWidget(new_button)
    def func(self):
        #sender = QtCore.QObject.sender()
        if self.sender() is not None:
            aa = self.sender().text()
            sand = self.session.experiment.get_sandbox()
            scipy = sand.result_actions[aa]['scipy']
            if 'kwargs' in sand.result_actions[aa]:
                kwargs = sand.result_actions[aa]['kwargs']
                kwargs, ok = KwargDialog.getKwargs(self, kwargs)
                if ok:
                    self.ipython.push_vars(**kwargs)
                    self.ipython.execute(scipy)
            else:
                kwargs = {}
                self.ipython.execute(scipy)

class SilverDebugWidget(QtGui.QSplitter):

    def __init__(self,main_widget,session,session_widget):
        QtGui.QSplitter.__init__(self,QtCore.Qt.Horizontal)
        self.session = session
        self.main_widget = main_widget
        self.right_side_layout = QVBoxLayout()
        self.left_side_layout = QVBoxLayout()
        right_side_widget = QtGui.QWidget()
        right_side_widget.setLayout(self.right_side_layout)
        left_side_widget = QtGui.QWidget()
        left_side_widget.setLayout(self.left_side_layout)
        self.addWidget(right_side_widget)
        self.addWidget(left_side_widget)
        self.ipython = SilverIPython(kernel_manager=None) # in process kernel
        self.ipython.execute("""import silver
silver_session = silver.schedule.Session(None,filename='"""+session.filename+"""')
silver_session.load()""")
        self.ipython.push_vars(silver_session=session)
        self.ipython.push_vars(ipython_widget=self.ipython)
        self.right_side_layout.addWidget(self.ipython)

class TaskFilterProxyModel(QtGui.QSortFilterProxyModel):
    def __init__(self, parent, model, filter_text=""):
        super(TaskFilterProxyModel, self).__init__(parent)
        self.setSourceModel(model)
        self.filter_text = filter_text
    def filter_function(self,task):
        if self.filter_text is None:
            return True
        if self.filter_text.strip() == '':
            return True
        for f in self.filter_text.split(' '):
            if '@done' in f.strip():
                if not task.cmd.is_complete:
                    return False
            elif '@complete' in f.strip():
                if not task.complete():
                    return False
            elif '@pending' in f.strip():
                if task.cmd.is_complete:
                    return False
            elif f.strip().startswith('#'):
                print("searching tag:",f.strip(),"in",task.tags)
                if len([ t for t in task.tags if f[:1] in t]) == 0:
                    return False
            else:
                if not f in task.name:
                    return False
        return True
    def filterAcceptsRow(self, source_row, source_parent):
        index0 = self.sourceModel().index(source_row, 0, source_parent)
        task = self.sourceModel().itemFromIndex(index0).data()
        if task is not None:
            return self.filter_function(task) or self.has_accepted_children(source_row, source_parent)
        return True
    def has_accepted_children(self, row_num, parent):
        ''' Starting from the current node as root, traverse all
            the descendants and test if any of the children match
        '''
        model = self.sourceModel()
        source_index = model.index(row_num, 0, parent)
     
        children_count =  model.rowCount(source_index)
        for i in xrange(children_count):
            if self.filterAcceptsRow(i, source_index):
                return True
        return False
class SilverTaskOverviewWidget(QtGui.QSplitter):

    '''  '''

    def __init__(self,main_widget,session,session_widget):
        QtGui.QSplitter.__init__(self,QtCore.Qt.Horizontal)
        self.main_widget = main_widget
        self.session = session
        self.session_widget = session_widget
        self.task_widget = None

        self.last_running_task = None
        self.notify_on_every_task = False
        self.right_side_layout = QVBoxLayout()
        self.left_side_layout = QVBoxLayout()
        self.runner_buttons = QHBoxLayout()
        self.model_initialized = 0
        self.task_progressbar = QtGui.QProgressBar()
        self.progressbar = QtGui.QProgressBar()
        self.progressbar.setMinimum(1)
        self.progressbar.setMaximum(100)
        self.left_side_layout.addWidget(self.task_progressbar)
        self.left_side_layout.addWidget(self.progressbar)
        self.status = QtGui.QLabel()
        self.left_side_layout.addWidget(self.status)
        self.run_button = QtGui.QPushButton('Run')
        self.run_button.clicked.connect(self.run)
        self.loop_id = 0
        self.auto_run = False
        self.auto_run_button = QtGui.QPushButton('Autorun OFF')
        self.auto_run_button.clicked.connect(self.toggle_autorun)
        self.runner_buttons.addWidget(self.run_button)
        self.runner_buttons.addWidget(self.auto_run_button)
        self.left_side_layout.addLayout(self.runner_buttons)
        self.task_filters = QHBoxLayout()
        self.filterTextBox = QtGui.QLineEdit()
        self.filterTextBox.editingFinished.connect(self.handleFilterEdit)
        self.filter_text = ''
        self.filter_reverse = False
        self.button_filter_pending = QtGui.QPushButton('Pending')
        self.button_filter_running = QtGui.QPushButton('Running')
        self.button_filter_done = QtGui.QPushButton('Done')
        self.button_filter_reverse = QtGui.QPushButton('Reverse')
        self.task_filters.addWidget(self.filterTextBox)
        self.task_filters.addWidget(self.button_filter_pending)
        self.task_filters.addWidget(self.button_filter_running)
        self.task_filters.addWidget(self.button_filter_done)
        self.task_filters.addWidget(self.button_filter_reverse)
        self.button_filter_pending.clicked.connect(lambda : self.setFilter('@pending'))
        self.button_filter_running.clicked.connect(lambda : self.setFilter('@running'))
        self.button_filter_done.clicked.connect(lambda : self.setFilter('@done'))
        self.button_filter_reverse.clicked.connect(lambda : self.setFilter('@reversed'))
        self.left_side_layout.addLayout(self.task_filters)
        self.taskTreeView = QTreeView()
        self.taskTreeView.setMinimumWidth(400)
        self.model = QtGui.QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Task','status'])
        #self.proxy_model = TaskFilterProxyModel(self,self.model,'')
        self.taskTreeView.setModel(self.model)
        self.left_side_layout.addWidget(self.taskTreeView)
        self.taskTreeView.clicked.connect(self.test)
        self.taskTreeView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.taskTreeView.customContextMenuRequested.connect(self.openTaskMenu)
        self.tasks = []
        self.ti = 0
        self.update()
        left_side_widget = QWidget()
        left_side_widget.setLayout(self.left_side_layout)
        self.addWidget(left_side_widget)
        self.show()
    def update(self):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(['Task','status'])
        def filter_function(task,filters):
            for f in filters.split('+'):
                if '@done' in f.strip():
                    if task.cmd.cmd is None or not task.cmd.is_complete:
                        return False
                elif '@complete' in f.strip():
                    if not task.complete():
                        return False
                elif '@running' in f.strip():
                    if not task.is_running():
                        return False
                elif '@pending' in f.strip():
                    if task.cmd.is_complete:
                        return False
                elif '::' in f.strip():
                    kv= f.split('::')
                    variables = task.variables
                    str_variables = dict((str(k), str(v)) for k, v in variables.items())
                    if not kv[0] in str_variables.keys():
                        return False
                    if not str(kv[1]).strip() in str(str_variables[kv[0]]).strip():
                        return False
                elif ':' in f.strip():
                    kv= f.split(':')
                    variables = task.get_variables()
                    str_variables = dict((str(k), str(v)) for k, v in variables.items())
                    if not kv[0] in str_variables.keys():
                        return False
                    if not str(kv[1]).strip() in str(str_variables[kv[0]]).strip():
                        return False
                elif '==' in f.strip():
                    kv= f.split('==')
                    variables = task.variables
                    str_variables = dict((str(k), str(v)) for k, v in variables.items())
                    if not kv[0] in str_variables.keys():
                        return False
                    if not str(kv[1]).strip() == str(str_variables[kv[0]]).strip():
                        return False
                elif '=' in f.strip():
                    kv= f.split('=')
                    variables = task.get_variables()
                    str_variables = dict((str(k), str(v)) for k, v in variables.items())
                    if not kv[0] in str_variables.keys():
                        return False
                    if not str(kv[1]).strip() == str(str_variables[kv[0]]).strip():
                        return False
                elif '#' in str(f):
                    if len([ t for t in task.tags if f[1:] in str(t)]) == 0:
                        return False
                else:
                    return False
                    if not f in task.name:
                        return False
            return True
        if self.session is not None:
            self.tasks = []
            self.ti = 0
            self.progressbar.setMinimum(0)
            self.progressbar.setValue(self.session.root_task.count_runnable()-self.session.root_task.pending())
            self.progressbar.setMaximum(self.session.root_task.count_runnable())
            self.progressbar.setFormat( "%p% ("+ schedule.format_time(self.session.guess_total_time())+")" )
            def addTasks(task):
                me = QtGui.QStandardItem(str(task.name))
                self.ti = len(self.tasks)
                self.tasks.append(task)
                me.setData(self.ti)
                compl = QtGui.QStandardItem('{}'.format(task.status()))
                compl.setData(self.ti)
                return [me, compl]
            def recAddTasks(task):
                [me, compl] = addTasks(task)
                if task.subtasks:
                    for s in task.subtasks:
                        me.appendRow(recAddTasks(s))
                return [me, compl]
            if self.filter_text is None or self.filter_text.strip() == "":
                self.model.appendRow(recAddTasks(self.session.root_task))
            else:
                #self.model.appendRow(recAddTasks(self.session.root_task))
                tasks = self.session.root_task.get_tasks(lambda x: x, lambda x: filter_function(x,self.filter_text))
                if self.filter_reverse:
                    tasks = reversed(tasks)
                for t in tasks:
                    self.model.appendRow(addTasks(t))
            self.taskTreeView.expandAll()
            self.taskTreeView.setColumnWidth(0, 250)
            def update_node(node):
                for index in range(node.childCount()):
                    parent = node.child(index)
                    for row in range(parent.childCount()):
                        child = parent.child(row)
                        if len(self.tasks) >= child.data():
                            task = self.tasks[child.data()]
                            if task is not None:
                                if filter_function(task,self.filter_text):
                                    parent.setExpanded(True)
                                else:
                                    parent.setExpanded(False)
            #update_node(root)
            #self.proxy_model.filter_text = filter_text
            #self.proxy_model.reset()
            #self.taskTreeView.setModel(self.proxy_model)
            self.taskTreeView.expandAll()
            root = self.model.invisibleRootItem()
            if 'debugWidget' in self.session_widget.__dict__:
                self.session_widget.debugWidget.ipython.push_vars(root_node=root)
                #self.session_widget.debugWidget.ipython.push_vars(proxy_model=self.proxy_model)
                self.session_widget.debugWidget.ipython.push_vars(taskTreeView=self.taskTreeView)
                self.session_widget.debugWidget.ipython.push_vars(model=self.model)
    def openTaskMenu(self, position):
        indexes = self.taskTreeView.selectedIndexes()
        item = self.model.itemFromIndex(indexes[0]).data()
        item = self.tasks[item]
        menu = QtGui.QMenu()
        sand = self.session.experiment.get_sandbox()
        task_actions = []
        try:
            for a in sand.task_context_actions:
                if sand.task_context_actions[a]['condition'](item) or (not 'condition' in sand.task_context_actions[a]):
                    task_actions.append([menu.addAction(self.tr(a)),sand.task_context_actions[a]])
        except:
            pass
        action_block = menu.addAction(self.tr("Block Task"))
        action_unblock = menu.addAction(self.tr("Unblock Task"))
        action_reset = menu.addAction(self.tr("Reset Task"))
        action_run = menu.addAction(self.tr("Run Task"))
        action_use_in_console = menu.addAction(self.tr("Use in console"))
        action = menu.exec_(self.taskTreeView.viewport().mapToGlobal(position))
        for act, task_action in task_actions:
            if action == act:
                if 'kwargs' in task_action:
                    kwargs, ok = KwargDialog.getKwargs(self, task_action['kwargs'])
                    task_action['function'](item,**kwargs)
                else:
                    task_action['function'](item)
                self.update()
        if action == action_block:
            item.blocked = True
            self.update()
        if action == action_unblock:
            item.blocked = False
            self.update()
        if action == action_reset:
            item.cmd.reset()
            self.update()
        if action == action_run:
            pass
            #item.cmd.reset()
        if action == action_use_in_console:
            self.session_widget.debugWidget.ipython.push_vars(_task = item)
            self.session_widget.debugWidget.ipython.input_buffer = '_task'
            self.session_widget.setCurrentIndex(2)
        self.session.save()
    def toggle_autorun(self):
        if self.auto_run:
            self.auto_run = False
            self.auto_run_button.setText('Autorun OFF')
        else:
            self.auto_run = True
            self.auto_run_button.setText('Autorun ON')
    def run(self):
        #self.session.root_task.run_one()
        self.status.setText('Running...')
        self.run_button.setEnabled(False)
        self.task_runner = SilverTaskRunnerThread(self.session.root_task.run_one)
        self.connect(self.task_runner , QtCore.SIGNAL('update(QString)') , self.run_done    )
        self.task_runner.start()
        next_task = self.session.root_task.find_next_task()
        self.last_running_task = next_task
        next_task.running_tick()
        predicted_time = self.session.guess_time(next_task)
        self.update()
        if predicted_time > 5:
            self.task_progressbar.show()
            self.task_progressbar.setMinimum(0)
            self.task_progressbar.setMaximum(round(predicted_time * 10.0))
            self.task_progressbar.setValue(0)
            def startLoop():
                this_loop_id = self.loop_id
                while True:
                    next_task.running_tick()
                    time.sleep(1.0)
                    value = self.task_progressbar.value() + 1
                    self.task_progressbar.setValue(value)
                    self.task_progressbar.setFormat( "%p% ("+ schedule.format_time((self.task_progressbar.maximum()-value)/10.0)+")" )
                    self.progressbar.setFormat( "%p% ("+ schedule.format_time(self.session.guess_total_time() - value/10.0)+")" )
                    QtGui.qApp.processEvents()
                    if self.loop_id is not this_loop_id:
                        break
                    if value >= self.task_progressbar.maximum():
                        break
            self.loop_id += 1
            QtCore.QTimer.singleShot(0, startLoop)

        #self.session_widget.ipython.push_vars(root_task= self.session.root_task)
        #self.session_widget.ipython.send_execute('root_task.run_one()')
    def run_done(self):
        failed = False
        if self.last_running_task.cmd.is_complete and not self.last_running_task.cmd.failed():
            if self.notify_on_every_task:
                if self.last_running_task.cmd.completion_time > 10:
                    notify("Task "+str(self.last_running_task.name)+" is done")
        else:
            notify("Task "+str(self.last_running_task.name)+" failed!")
            failed = True
        self.session.save()
        self.update()
        self.status.setText('Done.')
        self.loop_id += 1
        if self.auto_run and self.session.root_task.pending() > 0 and not failed:
            self.run()
        else:
            notify("Tasks are done.")
            self.task_progressbar.hide()
            self.run_button.setEnabled(True)
    def test(self, index):
        #ind = self.model.itemFromIndex(index)
        #ind = self.model.item(index.row(),0)
        ind = self.model.itemFromIndex(index)
        if ind is None:
            print("index "+str(index)+" not found")
            return
        task = self.tasks[ind.data()]
        if self.task_widget is not None:
            self.task_widget.hide()
            #self.removeWidget(self.task_widget)
            self.task_widget = None
        self.task_widget = SilverTaskWidget(task)
        self.addWidget(self.task_widget)
    def handleFilterEdit(self):
        self.filter_text = self.filterTextBox.text()
        self.update()
    def setFilter(self, text):
        if text == "@reversed":
            self.filter_reverse = not self.filter_reverse
        else:
            self.filter_text = text
        self.update()

class SilverLazyDataThread(QtCore.QThread):
    def __init__(self, data_widget):
        super(SilverLazyDataThread , self).__init__()
        self.data_widget = data_widget
    def run(self):
        self.data_widget.data.load(force_reload=True)
        html = self.data_widget.data.to_html(self.data_widget.key_filter)
        #self.data_widget.setHtml(html_dec(html))
        self.emit(QtCore.SIGNAL('update(QString)'), html)
        self.data_widget.loading = False 
        self.data_widget.loaded = True 

class SilverLazyDataWidget(Browser):

    '''  '''

    def __init__(self,data,key_filter=None):
        Browser.__init__(self)
        self.data = data
        self.key_filter = key_filter
        self.last_load = None
        self.loading = False
        self.loaded = False
    def lazy_show(self):
        if self.loading:
            return
        if self.last_load is not None:
            return
        self.setHtml(html_dec('<div class="loading"><p>Loading...</p><img src="data:image/gif;base64,R0lGODlhoAAUAKEAAAQCBAQGBP///wAAACH/C05FVFNDQVBFMi4wAwEAAAAh+QQICQAAACwAAAAAoAAUAAACZ4SPqcvtD6OctDaBs968+w+G4kiW5qkZAcq27gvHoCrX9o2/dM73fr77CYfEUbCITBKPyqbTxnxKp6co9YrtWLNc7LYLfn7DZOS4jP6d02zcug2PveN01ryOJ93z/I/lDxgoOEgoUQAAIfkECAkAAAAsAAAAAKAAFACDBAIENDY0HBocTE5MFBIUDAoMLCosZGJkBAYEREJEJCIkVFZU////AAAAAAAAAAAABKIQyEmrvTjrzbv/YIgdy5AEhkIQCMK8cCzPdG3feK7vfI9LJBNKISAUXL6kcslsOmfA0ilVPD6v2KzWFxVSjciteEx2dk8KIrjMbrtt52E1/K7bxfHUynrv+80AQVNqfH+Ghzp5hHSIjY4vinuMj5R/inOVmYaXa5qedpGdn6NsnIWkqFumk6mtTaGnrrJLq7O2tIFScqK3vYkiwMHCw8TFHREAIfkECAkAAAAsAAAAAKAAFACEBAIEdHZ0tLa0REJEXF5clJKUzM7MVFJUpKKkjIqMxMbETEpMbG5s3N7cfH58vL68REZEZGZknJqc1NbUVFZUrKqs////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABfcgII5kaZ5oqq5s675wjDaToTxCJUlF4gSMCIGyWEAGlqRyyWw6n9CodEqtWq8WEc2GqyB4iYAjODwYkdi0es1uq7W1W8UrSfiAQor56O77/4BTcFw5Xz0/ZHpngYyNjm8AWzeFdXeJe2iPmpucWZFxXYaWeWYDmZ2oqW6DkzqViEIERXyqtbZXrKFgsAQHmLfAwU+5lId4Eb6LwsvCxHR2vIqmzNS3zqK8ssrV3J3XBcZkybTd5Y/XPA7qZNoLp+bwf9/hQsnT8fh+6PQEsgf3+QKu2Teq1y+BCLHMG4Ws1LuEEKEQzDbrYcSLS2Ro3Mixo8ePLkIAACH5BAgJAAAALAAAAACgABQAhAQCBIyKjExKTMzKzCQiJOTm5KyurGxqbBQSFDQyNPTy9AwKDKSipFxeXNTW1MTCxHR2dAQGBJSWlFRSVCwqLOzu7LS2tBwaHDw+PPz6/Nze3Hx+fP///wAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxyNJdVVaE5w2MZDElgAzk0JgJMgnK5ICKRmnRKrVqv2Kx2y+1iRdKMIrd7+IBCohGppBCcUK98Tq/bveDajcw7B4dFDQJJSwQICAtRd4uMjY5UeTRifD0/EmmBbIVOiY+en6BbkTZjOn0/f2pHhEyciqGwsZ+je6aVaIBGg21NiK+ywMF0tKU7lQypmaxvvsLOz6IAYcWnuKqarc3Q29w1tDi2fphry67d59DElOK5q23MnejywOrhFsjjRxi85vP+od/WocqHDd6vfwgZ1TPG7lo5bQkj3llYLRk5fhAlapQT0B6+dgX7bRzJheIti+4olvSKR7LlF2l6qJ0kuG9TRpc4pXRkOBDkQ5Y5g3KQQbSo0aNIk7oIAQAh+QQICQAAACwAAAAAoAAUAIQEAgSMioxMSkzMyswkIiTk5uSsrqxsamwUEhQ0MjT08vQMCgykoqRcXlzU1tTEwsR0dnQEBgSUlpRUUlQsKizs7uy0trQcGhw8Pjz8+vzc3tx8fnz///8AAAAAAAAAAAAF/iAgjmRpnmiqrmzrvnCMcnRt21lVFZozPBYDQxLYQA6NiQCToFwuiEjkRq1ar9isdsvtVkXcjGLXewCFRCNSyaQQoFKvfE6v26ngbY7sOw+LRw0CS00ECAgLU3eLjI13eVpifD9BEmmBbIVQiY6dnp9fAGFjPH1Bf2pJhE6biqCvsI+iejqllGiASINtT4iuscDBWpBZkrZADKiYq2++ws/QeLORpD23ymurvZzR3c/EWHvHp5fZbc3c3uqw4FfG1n7lquet6/ag7VbvpsnyExi86t0b2ChfFXHwyOWap8kZwYd2DFLZd80fM4EQM3qReINivIWZWDnUSHLYtGK1NxL2A3lxZMmXoUZN+pgq5LZfMHPS4IijmilcNQE2TKdTJ88aCH9iY+gGY9GcMqJKnUq1qlUXIQAAIfkECAkAAAAsAAAAAKAAFACEBAIEjIqMTEpMzMrMJCIk5ObkrK6sbGpsFBIUNDI09PL0DAoMpKKkXF5c1NbUxMLEdHZ0BAYElJaUVFJULCos7O7stLa0HBocPD48/Pr83N7cfH58////AAAAAAAAAAAABf4gII5kaZ5oqq5s675wjHJ0bd94nVVVoTmDh8XAkAQ2kENjIsAkKJcLIhLJWa/YrHbL7VpFXmtG0fs9hEQjUsl0UghSanhOr9u14DtnVwaii0dJDQJNTwQICAtVeoyNjl8AjGN9QUMSaoJthlKKj56fenl3kz5+Q4BrS4VQnIugr7B4kXp8pZVpgUqEblGJrrHAwTSidqQ/lQyomatwvsLPwMR1xqa4qZqsztDbn9J0tcd/mGzMrdznjt5z1LfK5G7Nnejzduph7ELJ40sYvOb0AMPY8wKumjtV8P4FXChLEhlb4nIh3KSNoUVIDilFvLaql7yLIG0M7FIQ2UFs8S9+hQQ5kgu+U/uweVS50mLLLS+tZepH8WPNize1lNy4LGHFnzZlKF3KtKnTpy1CAAAh+QQICQAAACwAAAAAoAAUAIQEAgSMioxMSkzMyswkIiTk5uSsrqxsamwUEhQ0MjT08vQMCgykoqRcXlzU1tTEwsR0dnQEBgSUlpRUUlQsKizs7uy0trQcGhw8Pjz8+vzc3tx8fnz///8AAAAAAAAAAAAF/iAgjmRpnmiqrmzrvnCMcnRt33iOZ1VVaI7Bw2JgSAIbyKExEWASlMsFEYnortisdsvt3kTeMCej8AEfw+IxuWw+KYRpVUyv2+nge5ZnDqaNSEoNAk5QBAgIC1Z6jI2MeY47ZT9+RBJrgm6GU4qRnp9bkKBjk0BCRIBsTIVRnIujsLCioHyUp2qBS4RvUomvscCRs59kfacMqZmscb7Bzo7DnsW2f5htrL2dz9t20ZG1ptW5q2/M2tzoXt6O0+Go1uSbzen0oQDA7ZXI8BMYvK71AmJZ1whcJVyqNMEBKLDhl3ux8t1Kdq0cQ4cYCTKSKC4htosYG2rUY/AYxXgLL+eFFAkRFsd34xRm+7Wy3sg7LxFm8ifvXE2bLUeV7KjMosqf9GQoXcq0qdOnLkIAACH5BAgJAAAALAAAAACgABQAhAQCBIyKjExKTMzKzCQiJOTm5KyurGxqbBQSFDQyNPTy9AwKDKSipFxeXNTW1MTCxHR2dAQGBJSWlFRSVCwqLOzu7LS2tBwaHDw+PPz6/Nze3Hx+fP///wAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxydG3feK7rWVUVGsfgYTEwJIEN5NCYCDAJyuWCiER22Kx2y+3WRN5wNqP4BR9EI1LJdEIpBKpVTK/bd+C7vWcWpo9JSw0CT1EECAgLV3qMjVx5jl5kfUNFEmuCboZUipGen18AoFyTQH5FgGxNhVKci6Owd5CxPD6mlWqBTIRvU4mvtMFds8I2pUGVDKmZrHG/xdBaxNHHp7mqmq3P0dw509B8t3+Ybc2u3eg238XVuMvlb86d6enrwu1EyuRNGL3n9OjsBQuHbJyuVfH+AeQmkBY+VPuyyQO2sFjDWA+vMfO3rSK0i7AIntJ3UKJCj8IqQI7K+A5hFF/zUKYUlY5lxH6bOsqkpRKUSHcRzencCUuG0aNIkypd6iIEACH5BAgJAAAALAAAAACgABQAhAQCBIyKjExKTMzKzCQiJOTm5KyurGxqbBQSFDQyNPTy9AwKDKSipFxeXNTW1MTCxHR2dAQGBJSWlFRSVCwqLOzu7LS2tBwaHDw+PPz6/Nze3Hx+fP///wAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxydG3feK7ve1ZVBY1j8LAYGJLABnJoTASYBOVyQUQivKx2y+3SRN6wuJZRAIWP4jG5bD6jFEL1Oq7bw+C7Xuc7D9VISkwNAlBSBAgIC1h7jY4ceY+NZX5ERhJsg2+HVYuSn3WRoHaUQX9GgW1OhlOdjKOwWqKxYX2mlmuCTYVwVIqvtME3s8JbpUKWDKmarHK/xdBfANFdx6e5qputz9TCxN05tsiAmW7NruDB3+k21rjL5nDOnuyw6/Uc7kXK5U4YvejwgbpXT9w1eKvkBRQoiSA7faj6aZsHjGEjh+kgYmMGkJtFRxjBGUyGcOLCj3ssQnbTWJKVL3ooU06Lmc/MLXK6/Cn0SNOOSmojcWY7x7PnGBlIkypdyrSpixAAIfkECAkAAAAsAAAAAKAAFACEBAIEjIqMTEpMzMrMJCIk5ObkrK6sbGpsFBIUNDI09PL0DAoMpKKkXF5c1NbUxMLEdHZ0BAYElJaUVFJULCos7O7stLa0HBocPD48/Pr83N7cfH58////AAAAAAAAAAAABf4gII5kaZ5oqq5s675wjHJ0bd94ru/8nlWVgsYxeFgMDElgAzk0JgJMgnK5ICKRnnbL7dJE3rB4m1EEhw8jUsl0QqUUghU7rttz4Lte/DsT1UlLTQ0CUVMECAgLWXuNYXmOkT5mQn9HEmyDb4dWi5KfPJCgo2V+RUeBbU+GVJ2Mo7BfALGkQJWna4JOhXBViq+0oKLBjaW3Rgypmqxyv8Sjw893xkO4ym6svp7SktHcY33HqJnYcM3b343e6V7Ulrmqm3Gu7Oqz9XXup8nkTxi99PDdWSewR7hqgPrJOwes4KN7DrvoS6hrlbmAER9mlEgJ4biK8rQ13KiFIEkbBzMt8QPJDOPJkhBf6pj4MV42lzJ3mJRJE56mf5yc5eyx82VKawpbCh2qU4bTp1CjSp3aIgQAIfkECAkAAAAsAAAAAKAAFACEBAIEjIqMTEpMzMrMJCIk5ObkrK6sbGpsFBIUNDI09PL0DAoMpKKkXF5c1NbUxMLEdHZ0BAYElJaUVFJULCos7O7stLa0HBocPD48/Pr83N7cfH58////AAAAAAAAAAAABf4gII5kaZ5oqq5s675wjHJ0bd94ru98r2eVSkHjGDwsBoYksIEcGhMBJkG5XBCRiG/L7dZE3rB4bMsohMTHMblsPqNTCuGaJdu94Lt+Xw4Oi2tKTE4NAlJUBAgIC1p8jjd5j5JhZmiASBJthHCIV4yTkpGgozyVf0ZIgm5Qh1WejaR6orG0NUCWqGyDT4ZxVouwtWOzwrGmRKgMqputc8DFZMTQoMeXuqucrs/TYdLcj7engZpvza/feADopNW5y+Vxzp/rW970d+1HyuRQGL7n93rYCzgmHLJxu1jFA0hQx8CGXvKl4pdNXjCINh5i3CLxGrN/2zZmVCfSjsFL+zESVmRYkoPGlj/OiJuostWveTBfwsTR8V2/hSFb6tzZBxdCbOaClpTBtKnTp1CjuggBACH5BAgJAAAALAAAAACgABQAhAQCBIyKjExKTMzKzCQiJOTm5KyurGxqbBQSFDQyNPTy9AwKDKSipFxeXNTW1MTCxHR2dAQGBJSWlFRSVCwqLOzu7LS2tBwaHDw+PPz6/Nze3Hx+fP///wAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxydG3feK7vfO/jmUqloHEMHhYDQxLYQA6NiQCToFwuiEjkx+3eRN6weMzNKIbFB1LJdEKlVAoBqyXbceC7fl8WEo1rS01PDQJTVQQICAtbfHZ5jpGSHGZogEkSbYRwiFiMk2KQoKNklX9HSYJuUYdWno2kP6KxtD9Blqhsg1CGcVeLsLU7s8LFQGenSAyqm61zwMbDANHUN6ZFucxvrb+f1V/T39W3yama23HP3uI0xOy015e6q5xyr+/t4fjC8ajL51Ew+LqHz90+UOSwBQJYT10wcQYPSuq3cBerdATfRZToiKI5i/W6Pfy2keOehJcv/oF0lpFdSZN3PM5rNhBaQX0wJ8nUFhCjTY04c0ZCmY0hy58uZShdyrSp06ctQgAAIfkECAkAAAAsAAAAAKAAFACEBAIEjIqMTEpMzMrMJCIk5ObkrK6sbGpsFBIUNDI09PL0DAoMpKKkXF5c1NbUxMLEdHZ0BAYElJaUVFJULCos7O7stLa0HBocPD48/Pr83N7cfH58////AAAAAAAAAAAABf4gII5kaZ5oqq5s675wjHJ0bd94ru987/+1TKVS0DgGD4uBIQlsIIfGRIBJUC4XRCQC7OpE3rB4TLZlFETjI7lsPqPTKoWQ3Za94Lt+vxemj2xMTlANAlRWBAgIC1x8PXmOkZI9Z39IShJuhHGIWYyTXwCgo6QclUWASoJvUodXno2lNJCytXd+qJdtg1GGcliLsbK0tsVhp0aXDKubrnTBxcTG0z7IqbusnK/QttLU3zm4yYGacM6w0aLg6zrWuszmcs+f3ers90FouUnL5VIYv9DVw0fQ1JB9qvxpmyeslLeC09yR49VKnsBaDyEWk5iQojZg9DDa0/hNXKp+HivPcRNJch1HbM0Crhw2sqWxl/D+WZzpsKZNWybfKVQZkqaMo0iTKl3KlEUIACH5BAgJAAAALAAAAACgABQAhAQCBIyKjExKTMzKzCQiJOTm5KyurGxqbBQSFDQyNPTy9AwKDKSipFxeXNTW1MTCxHR2dAQGBJSWlFRSVCwqLOzu7LS2tBwaHDw+PPz6/Nze3Hx+fP///wAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxydG3feK7vfO//v0ylUtA4Bg+LgSEJbCCHxkSASVAuF0QkAvyJuuCweDzOKIjGR3LZfEanVQohuyXnvva8fh8cFo9rTE5QDQJUVgQICAtcfBx4jpGSdmZogEoSbYRwiFmMkZCToqM9lX9ISoJuUodXno18oaSztBxClqhsg1GGcViLsHuytcSSpkaoDKqbrXPAoADF0pPHl7qrnK7PjsPT3mV+yIGab82v0N/plGen47uscc6f3NHq9mDVyctvGL7n9PcC9sHlDpu5bbHqCVyYI19BZvH+JWRI8YbDVOTgWfk1b2LFj7faWVCWMZu8YHoqun28d/EaxI0ShSlcubDlPin9OiGUSZNiSHEY35mMmVKG0aNIkypd2iIEACH5BAgJAAAALAAAAACgABQAhAQCBIyKjExKTMzKzCQiJOTm5KyurGxqbBQSFDQyNPTy9AwKDKSipFxeXNTW1MTCxHR2dAQGBJSWlFRSVCwqLOzu7LS2tBwaHDw+PPz6/Nze3Hx+fP///wAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxydG3feK7vfO//wF2mUiloHIOHxcCQBDaQQ2MiwCQolwsiEgnSRN6weEwu1zKK4vGhZDqhUqqVQtByw2Czfs8fD9VIbU1PUQ0CVVcECAgLXV55fZGSkzRogElLEm+FcolajXgAlKOkZZZGgUuDcFOIWJ+OQZCltLVCRKiYboRSh3NZjLFAs7bFxadHmAyrnK51waHG0sdpuYKbca7AoI+i09+kf9aq2K1zz9yy3uDskcipu6yddLDR7fd778rMcRi/9d3wCTSFK9k1XuY8QQs4sKEXfQflOQOozqHFHxDJIZy3TdgPYhdD2hBn0MKycvMv0Hn0AVKkyIzxmv1bWNGlzUrVSsbsd47isHU3XxaExy8hPZo/ZShdyrSp06csQgAAIfkECAkAAAAsAAAAAKAAFACEBAIEjIqMTEpMzMrMJCIk5ObkrK6sbGpsFBIUNDI09PL0DAoMpKKkXF5c1NbUxMLEdHZ0BAYElJaUVFJULCos7O7stLa0HBocPD48/Pr83N7cfH58////AAAAAAAAAAAABf4gII5kaZ5oqq5s675wjHJ0bd94ru987//AoC1TqRQ0jsHDYmBIAhvIoTERYBKUywURieREwrB4TC7/Mgoj8rFsPqPT6pVC2Ha/ALN+z+8Pi0dJbU5QUg0CVlgECAgLXjhgfpKTlDtoaoJMEm+GcopbjniVo6SSl4FKTIRwVIlZoI83kaW0tWFEmKluhVOIc1qNsTaztsXGOadIqQyrna51waLH09TJmbusnq/RkHnU38W4qIOccc+w0uDqpNa6zeZz0KHd6/WV7UvM5VQYv+j09gL2EaeMHK9W8f7J8iawYRl8qvZpkyesBjGHGINAxObMH7eFGUNqBFTQgr6DEy4VDmMosqWOje8QYgE2D6TLmzhgSuz36eNKnEBrELwWM6VPizKSKl3KtKnTFiEAACH5BAgJAAAALAAAAACgABQAhAQCBIyKjMzKzExKTOTm5CQiJKyurGxqbPTy9DQyNKSipNTW1FxeXMTCxHR2dBwaHJSWlFRSVOzu7CwqLLS2tPz6/Dw+PNze3Hx+fP///wAAAAAAAAAAAAAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxmdG3feK7vfO//wCCwIpEQLgtBg2JQQAIYx4ERGVgSk8dDJOx6v+CwOFNBGJGNZfMZnVavk8IWMK7b7/gd8ZxUO6FSDANWWHJceYiJikFlfEpMEGyBb1hah4uYmZiNR31Mf21UhHFzmqand3udj2uAU4NwlnSotLVCnEiPCqCTo4aztsHCOLieraGUWaXDzMOquX6Sbr7Lzda0xay807HV19+Z2Uu70lQWcL/g6ptFq9Guot2X6/SpZu6f5cnp9f114vng7fPmr2CXZ57ICRwly6DDg/egBURGbd7Di3oiGttmTh4wjCCJtZN4rBe6OTIKUqpcybKlSxYhAAAh+QQICQAAACwAAAAAoAAUAIMEAgSMiozMyszk5uSsrqz08vSkoqTU1tTEwsR8fnyUlpTs7uy0trT8+vzc3tz///8EvBDISau9OOvNu/9giD1kaZ5oqq5s675wLKvNsgzOISAMYSiBhGRGLBqPyKSpUbjlEDwfUAhQWq/YLKzm1EV/waF2TC4fmd1dTzEVm9/w+LKJ8/bAVLl+P+bW1VJhVXyEhUZofzwGeG6Gjo8riDmAjIOQl5gPfpNfbZaZoIWSdoF5oad8o2qLnqiucpukla+0ZqqdgrW6Wrd3rbvASrGrs8HGh3ScvrnHzTK9pY3O05E2ictUItrb3N3e3xwRACH5BAgJAAAALAAAAACgABQAgAQCBP///wJmhI+py+0Po5y0soCz3rz7D4biSJbmqR3oyrbuC4NqTNf23c74zve37gsKhyIg8YgUGpPMJm3pjEpN0Kn1yqlit1Yt99v0gsdEMfncM6PXNjX7/XLD5yg5/T6y4/cei/8PGCg4OFEAACH5BAgJAAAALAAAAACgABQAgwQCBDQyNBwaHBQSFExKTAwKDCwqLAQGBDw+PCQiJFRSVP///wAAAAAAAAAAAAAAAASyEMhJq7046827/2CILWRpnmiqrmzrvnAst9IxCIIRIIQizcCgcEgspgCHwi2h4/kAxqh0Sp0hbQLmrverer9gI1KJayKe4bR6vaotm1woe04Hj2+5LbrO7xevb3pdfoSFL3dZcHuGjI0mgGWCco6UhohaToOVm3yQmHGcoXV3A5+Loqhhnoqaqa5Ul6yTr7RRkHmZs7W7QrGSvMC9B1imrcHHNEkDpbIizs/Q0dLTFAcXEQAh+QQICQAAACwAAAAAoAAUAIQEAgRsamw0MjSkoqQcGhyMioxMSkwUEhS0trQMCgx0dnQsKixcXlwEBgQ8PjysrqwkIiSUlpRUUlTEwsR8fnz///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAF8SAgjmRpnmiqrmzrvnCMVnRt33iu73zv/8CgsCJqHAiEhcBhkDACCkohMnggJqKhdsvtermARuIIUTKdUCnVigV83/C43Bc2EsrL5jM6HQyuWXOCg4RaYWNIZg5ofGuAboWRkpM0RWRmeml9bIGUnp9vh0dJeYxqVY+gqqtDdZele6eckKy1tjiid5imfX9tt8DAromwDI2ov8HKq7l4Z7GbqcvTnsPOmcez1NuSogfXvI7J3ORz1rvQ4p3l7F/N6JrqtO30W8OkzwEBstL1/kLvYMVDtu6fQR7nBGbrd7BhDm/g0hGUQbGixYsYKTaIEQIAIfkECAkAAAAsAAAAAKAAFACEBAIEjIqMTEpMzMrMJCIk5ObkrK6sbGpsFBIUNDI09PL0DAoMpKKkXF5c1NbUxMLEdHZ0BAYElJaUVFJULCos7O7stLa0HBocPD48/Pr83N7cfH58////AAAAAAAAAAAABf4gII5kaZ5oqq5s675wjHJ0bd94ru987//AYE4UQVwulARGMGkcIJuAhGGwPAYOTaGiyNREwrB4TC4LAZGFkZBcNp/RafXxyG67X4B5z+/7hxFFF2xKTE5QUgwMVlhaXF40YH+TlJU+aGpHbRhviHKMdo95lqSllkRrbYZwiXONd5Ackqa0tWSYRkiFnXFUVnWOeJF6tsXGl4Gpu4e9rqHCssTH09SRacpuzImLV8+xs9XhtmiCutkNnr503qPi7rS4g6q8raDB39Lv+pTk2Kvpzu6120fQDy4EhLKx+rRO4LCCEPn0k7dsobpXoh5G3DgmXsJ/zX6x08ixJJCJ5jRWHQjZzWE0kzCRZfpIjyEwWANj6rwxkaY2hhihgdtJ9KBPi3NuZowmo6nTp1CjooggdUQIACH5BAgJAAAALAAAAACgABQAhAQCBIyKjExKTMzKzCQiJOTm5KyurGxqbBQSFDQyNPTy9AwKDKSipFxeXNTW1MTCxHR2dAQGBJSWlFRSVCwqLOzu7LS2tBwaHDw+PPz6/Nze3Hx+fP///wAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxydG3feK7vfO//QJ8ogrhcKAmMYNI4QDYBCcNgeQwcmkJFkfmJguCweEwWRhZFAlLJdEKl1McDq+V6AeW8fs+nASJEF2pJS01PUQwMVVdZW11CeH2Sk5Q4f2hGaxhth3CLdI53laOke0Npa4VuiHGMdY89X6WztECXRUeEnG9TVXONdpC1w8SWgKi6hrytoMGxkcXRtbeCqbuIilbNsDyy0t+kf4G5bA0Nnb1y26Lg7ZTUg+WrntrA3Dve7vp64sjy6MzssdtHkMwtBPFUAfS1TljBh2H6VUs2L52rUA4hajSDKeE1er9eDdxIModEcqoxDiz7JDBjyZc14FlTxophy2cwc/o5NvHfynoiXeokedAjTZANn8lYyrSp06YRnroIAQAh+QQICQAAACwAAAAAoAAUAIQEAgSMioxMSkzMyswkIiTk5uSsrqxsamwUEhQ0MjT08vQMCgykoqRcXlzU1tTEwsR0dnQEBgSUlpRUUlQsKizs7uy0trQcGhw8Pjz8+vzc3tx8fnz///8AAAAAAAAAAAAF/iAgjmRpnmiqrmzrvnCMcnRt33iu73zv/0CaKIK4XCgJjGDSOEA2AQnDYHkMHJpCRZEJ6kTesHhMDgMiiyIBqWQ6oVLq44HVcstCAH7P7//ORBdrSUtNT1EMDFVXWVtdeGB+kpN9Z2lGbBhuh3GLdY57kZSjpEFDamyFb4hyjHaPZaKls7Q3lkVHhJtwU1V0jXeQerXEtYCouoa8rZ/BscPF0aO3gqm7iIpWzbBkstLffMeYyQ2cvXPbodDg7LFoyG3KrJ7A3GPe7fmmEYGD8audtNVTp6+gl1sI/KkyJ+fXK4IGI/YQp/BaQFeghEnc+OVdtWQAzznM+IyjSRviM3L9O7CM3kONJ09Sqygv4Ehn3dbFlEjRWs1zGHHe07nTIEKaIRumEyajqdOnUFdEiEo1BAAh+QQICQAAACwAAAAAoAAUAIQEAgSMioxMSkzMyswkIiTk5uSsrqxsamwUEhQ0MjT08vQMCgykoqRcXlzU1tTEwsR0dnQEBgSUlpRUUlQsKizs7uy0trQcGhw8Pjz8+vzc3tx8fnz///8AAAAAAAAAAAAF/iAgjmRpnmiqrmzrvnCMcnRt33iu73zv/zhRBHG5UBIYwaRxgGwCEobB8hg4NIWKIgPs2kTesHhMzgEiCyLhmFw2n9Hp43HNbss+MH7P79POQxdrSEpMTlAMDFRWWFpcfkEAkJOUP2dpRWwYbodxi3WOlTV6oqWlQmpshW+Icox2j6KkprSQl0RGhJxwUlR0jXenkrXEfYCpuoa8rqDBssPF0WS3gqq7iIpVzbGVs9LfQMeZyQ2dvXPbpt7g7DvUg23KrZ/A3JTr7flfEYHwq+bM6qmDpq/gKDQIEPi75gmdQGEGI/7hhyweq4avQkGUaPCdNXkNf8EayLEjxXEWMg8sozdyY8l8HpNdPCdS47OX+sQtBHkuo7NuBHF+u6Xw40w5NX/ek8G0qdOnQqBKVRECACH5BAgJAAAALAAAAACgABQAhAQCBIyKjExKTMzKzCQiJOTm5KyurGxqbBQSFDQyNPTy9AwKDKSipFxeXNTW1MTCxHR2dAQGBJSWlFRSVCwqLOzu7LS2tBwaHDw+PPz6/Nze3Hx+fP///wAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxydG3feK7vfO/vogjicqEkMIJJ4wDZBCQMg+UxcGgKFUXmx+1yRN6weBwGRBZDghGpZDqh0sfDitWS7zQwfs8vR4QXakdJS01PDAxTVVdZW31lAI+SkzhmaERrGG2GcIp0jZRceqGkfEFpa4Ruh3GLdY6lQJGxtGOWQ0WDm29RU3OMdrU5o8LFPWaAgmyFvK2fwcY1xNHUNreBqbuHiVTPsNTT1dXIuNkNDZy9ct7iX7Pt4tfKqunOwN/R4fDF5Ni6zKw83Wunb1+tWwjmaeu0bmC8dwaN9VMIkKErUA8jgjuD6t8qhr9eEYSo8eCfch4wDzQTKDJjSX4c/S37qC4kxnEkX5aaaI6mvZY4dQpDSNGnL3YPZShdyrRFhKZQm4YAACH5BAgJAAAALAAAAACgABQAhAQCBIyKjExKTMzKzCQiJOTm5KyurGxqbBQSFDQyNPTy9AwKDKSipFxeXNTW1MTCxHR2dAQGBJSWlFRSVCwqLOzu7LS2tBwaHDw+PPz6/Nze3Hx+fP///wAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxydG3feK7vfN+LEcTlQklgBJPGAbIJSBgGy2Pg0BQqioxvy+WKuuCw+BdZCAnFY3LZfEYfj+o1O657Afa8nhy8oI1ISkxODAxSVFZYWnuMNV+NkHoAZUJERhhrg26HcoqRjI+fomBAZ2mBbIRviHOLo3Whr7I7k2ZDp5ltUFJxiXSzYrHAwzSTfX9qgrqrnb/Ed8/RtaaAuYSGU82u0T943MPGlbgNDZq7cNrfPsLqotN+46mb2b7b7Tjs95Dh8NXKqpzq6dORb+CeaQiQoTL3plcrg/i8QdwX4Vg8hgEfTnQkceNBSv2SyTvn0JNHDgU6T4bhZ0nkgWUZTXpMqbLLO4XW5pV0tpFmzS38cP6bx0pmz44/g1FKeBEmOoEnZUidSrVEhKpYs6IIAQAh+QQICQAAACwAAAAAoAAUAIQEAgSMioxMSkzMyswkIiTk5uSsrqxsamwUEhQ0MjT08vQMCgykoqRcXlzU1tTEwsR0dnQEBgSUlpRUUlQsKizs7uy0trQcGhw8Pjz8+vzc3tx8fnz///8AAAAAAAAAAAAF/iAgjmRpnmiqrmzrvnCMcnRt33iu73zPiRHE5UJJYASTxgGyCUgYBstj4NAUKoqMb8vt7kTesNgLiCyEhOIxuWw+o49H9Zod2+81MH5/LwcvaUZISkxODAxSVFZYWnyOW3qPkj5lZ0NqGGyFb4lzjJOgOJGhpHllaGqDbYZwinSNpaCjsaGVQkSCmm5QUnKLdbSTs8GSfqi5hLutnsDEjsPOfLaAqbqGiFPMsNF40Nx9EX+4aw0Nm7xx2t973uti04Hkq5zZv9vuYe34XcbUyPPoXH3a9w4AwW5mECCIp+ocHF+vDuYzKHFMP4bW6AlsVpESxY5kEvqT57CXOpAePFGGFFdNyQFlneyp7KFvpg14LQE+PGkzR82eP8IdIwmzXkSgoj4i9ZlwYc6S6WQuvSmjqtUXEa5q3Xo1BAAh+QQICQAAACwAAAAAoAAUAIQEAgSMioxMSkzMyswkIiTk5uSsrqxsamwUEhQ0MjT08vQMCgykoqRcXlzU1tTEwsR0dnQEBgSUlpRUUlQsKizs7uy0trQcGhw8Pjz8+vzc3tx8fnz///8AAAAAAAAAAAAF/iAgjmRpnmiqrmzrvnCMcnRt33iu7zwtRojLhZLACCaNA2QTkDAMlsfAoSlUFJmedsvt6kTesBgHiCyCBKIRqWQ6oY8H1Yod2+9bMH7fKwMvaUVHSUtNDAxRU1VXWXyOe3qPkj5mQUNFGGyFb4lzjJOgXpGhfD9oaoNthnCKdI2ksF8AsaWVQqiabk9Rcot1tMA1o8FhfqeCuaudvq/EscPOXGVngLiEuohSnr/RsNDdfRF/l2sNDZu7cdvN4JPf7V+2geWqnNrM8KHv+TfG1cjXlN1zxc/drII8piFAMC8VOji9CCJ0tG+iv4bJ7LX6NLHUwY5k5Fmrly4iR5B3Oioi9Ecu1QFdrNahxKOyoMJ/9B7ykjnTTk1+F0fqHHiyZ7GPRhUyFApzJz6jYmRInWoiAtWrWLOeCAEAIfkECAkAAAAsAAAAAKAAFACEBAIEjIqMTEpMzMrMJCIk5ObkrK6sbGpsFBIUNDI09PL0DAoMpKKkXF5c1NbUxMLEdHZ0BAYElJaUVFJULCos7O7stLa0HBocPD48/Pr83N7cfH58////AAAAAAAAAAAABf4gII5kaZ5oqq5s675wjHJ0bd94ru+0GCGXCyWBEUwaB8gmIGEYLI+BQ1OoKDK8rHbL7epE3jAXEFkACcPiMblsPh+PafUqrtvvNzB+zyH/LmhERkhKTAwMUFJUVlh8jo85epBiZGZBaRhrhW6JcoyToI+SoWNkZ2mDbIZvinONpLCUALFjZUBCgpptTlBxi3S0wVujwpERf4FqhLusnsDF0DjE0TWVp7nLhohRzq/U1NPffreoRw2bvHDd3+zh4LaA5dmc3L/e7MXu0ePx2Kr0rT7h2zdrYA9bCJKlQvfGlyuD0PRFPHZN2b90AZ9BpCUxHzyFulb1WrcxWEdh/DdwWTzArJO9kiYLGrTWzyLDkS9hxjppkmLNhS3rPdQJiydHhCDnpXMokGgoGVBhRIhKtapVqiEAACH5BAgJAAAALAAAAACgABQAhAQCBIyKjExKTMzKzCQiJOTm5KyurGxqbBQSFDQyNPTy9AwKDKSipFxeXNTW1MTCxHR2dAQGBJSWlFRSVCwqLOzu7LS2tBwaHDw+PPz6/Nze3Hx+fP///wAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxydG3feK5zYoRcF0oCI5g0DpBNQMIwWB4Dh6ZQUWR22Kx2y+3uRN6wDRBZ/AhCohGpZDofDynVKq7b710wnkv2XdBDRUdJSwwMT1FTVVd7jY52eo9fZT9BQxhrhG6IcouSn6BfAKE5PWdpgmyFb4lzjKSwn5GxPJRAqJltTU9xinS0wHuzsX2ngbmrnL6vwc1hw7BkZn+4g7qHUJ2/ztx8o8HFt8cNDZq7cNrM3es60KTSxmrWydnL7Pft38DhgPKqm/Vc4Rs4Rh8teAj6pTL3ppdAggTdheJX7d+5Vp4gDpQICh61cRYbptO40SCxCH41LPk7oIvVSJL3OMqypRAZQIcZYa6TKYkiSIbKHursxvMRwprzbr4cyk2GUxMRnkqdSrXqiRAAIfkECAkAAAAsAAAAAKAAFACEBAIEjIqMTEpMzMrMJCIk5ObkrK6sbGpsFBIUNDI09PL0DAoMpKKkXF5c1NbUxMLEdHZ0BAYElJaUVFJULCos7O7stLa0HBocPD48/Pr83N7cfH58////AAAAAAAAAAAABf4gII5kaZ5oqq5s675wjHJ0bd84LkbIdVEJjGDSOEA2AQnDYHkMHJpCRZHJWa/YrHbL7dpE3iwgsugRgEKiEalkPh5QKTVMr9vvVjD+G+FdzkFDRUdJDAxNT1FTVXuNjo9fAJBjZT5oGGqEbYhxi5CfoHV6jjtmaIJrhW6Jcoyhr7B5ko+UPT+BmWxLTXCKc7HAwaONY36AaYO6q52/wc6hw3u1f6e5hYdOzK7P3KSzpH221Q0Nmrtv2t3qxN/EZKa4yaqcvtvr917ReMXwyKmb2erhG5ivnbR3CI6hMuemVyuCELXou8OPWrx/51h5isgxx0Q70xRaA+hwY8eTNDg+igpn6eIBZfQeokSpkk7IcRgbppt5smaYiiLlAdTYjGdEn/kQBs3Ja6fRozJcRIhKtarVq1ZDAAAh+QQICQAAACwAAAAAoAAUAIQEAgSMioxMSkzMyswkIiTk5uSsrqxsamwUEhQ0MjT08vQMCgykoqRcXlzU1tTEwsR0dnQEBgSUlpRUUlQsKizs7uy0trQcGhw8Pjz8+vzc3tx8fnz///8AAAAAAAAAAAAF/iAgjmRpnmiqrmzrvnCMcnRt37QYIddFJRjBpHGAbAIShsHyGDg0hYoig6tar9isdsvtWkVeDiCy4BF+wWHxmFw+Hs/oNEyv2+90sHe8u5yBQkRGSAwMTE5QUlR4jI2OeQBhY2U9aBhqg22HcYqPnp+gNXpdOmZogWuEbohyi6GvsJCSZDw+gJhsSkxwiXOxv8BYo1x8preCuaucvsHNzsNbk8ZpyISGTcuuztux0FrFlccNmbpv2dzosN5Z0n6nuKqbvdrp9Y3rwhF9f9Spmtjz7Am8F2kPLQT8UJFzw6vVwId28F0BlxDeP1adIGokVZAULXfH/JVrmHGjSWEdPInpq/WOyIFk8hyenFlF4pePFav9I8mMpk8xKaOtBNlvYcySP2narHkwp0iG55L6lFEiAtWrWLNq3YoiBAAh+QQICQAAACwAAAAAoAAUAIQEAgSMioxMSkzMyswkIiTk5uSsrqxsamwUEhQ0MjT08vQMCgykoqRcXlzU1tTEwsR0dnQEBgSUlpRUUlQsKizs7uy0trQcGhw8Pjz8+vzc3tx8fnz///8AAAAAAAAAAAAF/iAgjmRpnmiqrmzrvnCMcnTNiRFyXVSCCZPGAbIJSBgGy2Pg0BQqioxtSq1ar9isdsvtXkVVQGShI/R+wWHxmHw8ms+od06v2+9aMFWcu5h9QEJERgwMSkxOUFJ4jI2OjzR6U2JkO2cYaYNsh3CKkJ+goV8AYWJlZ4FqhG2IcYuisLGOkjaUOjyAmWtISm+JcrLBwnO0NXynuYK7rJ3Aw8/QpWFjyGjKhIZLza/R3dDFkRF9uNYNmrxu297rz+A31H6ouqucv9zs+KHux/HJqpva7OUbqI/UHmoI/lj7h86XK4IQH+0TVy3VOWYCI2q8M7GSQovLeqnbSJKYwUkUNS35OxAy4MOSMLd0rDgPoENPMXOOmtbnY010rXDqHFrrZC2EPq/ZHEm0aYsIMqJKnUq1qtUQACH5BAgJAAAALAAAAACgABQAhAQCBISGhERCRMzKzCQiJOTm5KyqrGRiZBQSFPTy9JyenExOTNze3DQ2NLy+vHx6fAwKDJSSlNTS1CwqLOzq7LSytGxubBwaHPz6/FRWVP///wAAAAAAAAAAAAAAAAAAAAX+ICCOZGmeaKqubOu+cIxCCEJMjbBkh/UEEYWh4hhIGIVCAqNpOp/QqHRKrVqv2KwWK6Jdbrld7xeJDIvHAmW5bbvf8Li8i/jidDwfUEgcDJBrTHKDhIWGV3R2YXlkCgp9aYGHk5SVb3Q2d2J6QWdGSEqClqOkpRqJYHhjAQF8aIBsprKzhKiajHsGBq8MkrS/wFq2i6tlnpGxwcrLUMOqnK6fvcnM1cHOAridRA7Iotbgs86bZGaQsN/h6qPj2o7natTr85Ptq63HoPL0/IP20PnipetH8BIAL6nI5YLnq6BDN/8s7HlU5I/Ahxi3/GsUsGHGj1Q2LnyVZB/Ik00LZKhcybKly5cuQgAAIfkECAkAAAAsAAAAAKAAFACEBAIEhIaEzMrM5ObkrKqs9PL0bG5snJ6c3N7cvL68lJKU1NLU7OrstLK0/Pr8fHp8////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABcwgII5kaZ5oqq5s675wjBpPoBxEkwgLMgwFB2RILBqPyKRyyWw6n9AoUkSzKXK73oARlHq/4LB4TKTWbliBwMcVkt/wuLxpth0OOh67O+/7/2B1aHlaQG6AiImKZQBVATiEe4eLlJVwgjgEWQgIbZafoIGNZ1eRW3yhqapKmGk9nairsrKtOglanrO6qoKlm6eTu8KUtb+5w8iKva4+hsnPiJh4xrHQ1pejdsywwdfeYXWPzMDf5WLFt5Lm617Spsfs8Uwy9PX29/j5LiEAIfkECAkAAAAsAAAAAKAAFACCBAIE3N7c9PL05Obk/Pr87Ors////AAAAA3kIutz+MMpJq704wzCGIEYojmRpnmiqrmzrvrDCFR9s33iu76w80CCecEgs6nxAo3LJXCJrzah02noGqdgs1artepXcr3iMC5PP6JQ5zW6v23DyO07vzuv46T3PdwJmUH2CTHuDhkd/P4GHjDuFjZA9GpOUlZaXmBUJADs="/></div>'))
        self.loading = True
        self.lazy_thread = SilverLazyDataThread(self)
        self.connect(self.lazy_thread , QtCore.SIGNAL('update(QString)') , self.change)
        self.lazy_thread.start()
        self.last_load = time.time()
    def change(self,html):
        self.setHtml(html_dec(html))

class SilverTaskLogWidget(QtGui.QTabWidget):

    def __init__(self,task):
        QtGui.QTabWidget.__init__(self)
        self.task = task
        self.log_box = QtGui.QTextEdit()
        self.log_box.setReadOnly(True)
        button = QtGui.QPushButton('reload',self)
        button.clicked.connect(self.update)
        layout = QVBoxLayout(self)
        layout.addWidget(button)
        layout.addWidget(self.log_box)
        self.setLayout(layout)
        self.update()
    def update(self):
        text = "Log:"
        for l in self.task._log:
            text += time.asctime( time.localtime(l['time']) ) + ' '
            if 'channel' in l and l['channel'] != '':
                text += '['+l['channel'] + '] '
            text += l['message'] + ' @' +  "\n"
        self.log_box.setPlainText(text)
        self.log_box.moveCursor(QtGui.QTextCursor.End)
    def lazy_show(self):
        self.update()

class SilverTaskWidget(QtGui.QTabWidget):

    '''  '''

    def __init__(self,task):
        QtGui.QTabWidget.__init__(self)
        self.task = task
        self.view = Browser()
        html = task.to_html()
        self.view.setHtml(html_dec(html))
        self.addTab(self.view, "Description")
        self.lazy_load_list = {}
        if task.data.exists():
            self.data_view = SilverLazyDataWidget(task.data)
            self.addTab(self.data_view, "Data")
            self.lazy_load_list[1] = self.data_view
        if task.data.exists():
            self.plot_view = SilverLazyDataWidget(task.data,'plots')
            self.addTab(self.plot_view, "Plots")
            self.lazy_load_list[2] = self.plot_view
        if task.cmd is not None:
            if task.cmd.cmd is not None:
                cmd_widget = QWidget()
                cmd_layout = QtGui.QVBoxLayout()
                if task.cmd.interpreter == "python":
                    cmd_box = QtGui.QTextEdit()
                    arguments_box = QtGui.QTextEdit()
                    cmd_box.setPlainText(task.cmd.cmd)
                    cmd_box.setReadOnly(True)
                    arguments_box.setPlainText(task.cmd.cmd)
                    arguments_box.setReadOnly(True)
                    cmd_layout.addWidget(cmd_box)
                    cmd_layout.addWidget(arguments_box)
                else:
                    cmd_box = QtGui.QLineEdit()
                    cmd_box.setText(task.cmd.cmd[0])
                    arguments_box = QtGui.QTextEdit()
                    arguments_box.setText("\n".join(task.cmd.cmd[1:]))
                    cmd_box.setReadOnly(True)
                    arguments_box.setReadOnly(True)
                    cmd_layout.addWidget(cmd_box)
                    cmd_layout.addWidget(arguments_box)
                self.addTab(cmd_widget, "Command")
                cmd_widget.setLayout(cmd_layout)
                self.log_widget = SilverTaskLogWidget(task)
                i = self.addTab(self.log_widget, "Log")
                self.lazy_load_list[i] = self.log_widget
        QtCore.QObject.connect(self,QtCore.SIGNAL("currentChanged(int)"),self.change)
    def change(self,i):
        for l in self.lazy_load_list:
            if l == i:
                self.lazy_load_list[l].lazy_show()

class SilverSessionWidget(QtGui.QTabWidget):

    '''  '''

    def __init__(self, main_widget, session):
        QtGui.QTabWidget.__init__(self)
        self.main_widget = main_widget
        self.taskWidget = SilverTaskOverviewWidget(main_widget, session,self)
        self.addTab(self.taskWidget, "Tasks")

        self.resultsWidget = SilverResultsWidget(main_widget, session,self)
        self.addTab(self.resultsWidget, "Results")
        self.debugWidget = SilverDebugWidget(main_widget, session,self)
        self.addTab(self.debugWidget, "Debug")

        #self.ipython = SilverIPython()
        #self.addTab(self.ipython, "Console")
        #self.ipython.push_vars(silver_session=session)


class SilverIPython(RichIPythonWidget):

    ''' Wraps an IPython kernel and provides IPython widgets for it '''

    def __init__(self, kernel_manager = None):
        RichIPythonWidget.__init__(self)
        self.external_kernel_manager = None
        if kernel_manager is None:
            self.kernel_manager = QtInProcessKernelManager()
            self.kernel_manager.start_kernel()
            self.kernel = self.kernel_manager.kernel
            self.kernel.gui = 'qt4'
        else:
            #with open(kernel_manager.get_new_connection_file(),'r') as f:
            self.external_kernel_manager = kernel_manager
            self.kernel_uid = kernel_manager.start_kernel()
            self.kernel_manager = QtKernelManager(connection_file=kernel_manager.get_new_connection_file(self.kernel_uid))
            self.kernel_manager.load_connection_file()
            self.kernel = self.kernel_manager.kernel#s.get(self.kernel_uid)
        #self.kernel.shell.push({'window': self, 'kernel': self.kernel})
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()
        self.kernel_client.execute('%pylab inline')
        self.exit_requested.connect(self.exit_requested_func)
    def push_vars(self, *args,**kwargs):
        if self.kernel is not None:
            self.kernel.shell.push(kwargs)
    def send_execute(self, *args,**kwargs):
        self.kernel_client.execute( *args,**kwargs )
    def exit_requested_func(self):
        self.kernel_client.stop_channels()
        #if self.external_kernel_manager is not None:
        #    self.external_kernel_manager.shutdown_kernel()
        try:
            self.kernel_manager.shutdown_kernel()
        except:
            pass
        qt_app.exit()

class SilverIPythonKernelManager_InfoThread(QtCore.QThread):
    def __init__(self, main_widget,uid,kernel):
        super(SilverIPythonKernelManager_InfoThread , self).__init__()
        self.main_widget = main_widget
        print(kernel)
        print(kernel.kernel.pid)
        self.uid = uid
        self.client = kernel.client()
        self.client.start_channels()
        self.command_queue = []
        self.stuff = []
        self.connect(self,QtCore.SIGNAL("add_cmd(QString)"), self.add_cmd)
    def add_cmd(self,cmd):
        #self.command_queue.append(cmd)
        self.stuff.append(self.client.execute(cmd))
        return self.stuff
    def run(self):
        while True:
            if len(self.command_queue) > 0:
                cmd = self.command_queue.pop()
                self.client.execute(cmd)
            while self.client.iopub_channel.msg_ready():
                io = self.client.get_iopub_msg()
                self.emit(QtCore.SIGNAL('io(QString)'), json.dumps({'uid':self.uid, 'channel':'io_pub', 'message':io},default=json_serial))
            while self.client.shell_channel.msg_ready():
                io = self.client.get_shell_msg()
                self.emit(QtCore.SIGNAL('io(QString)'), json.dumps({'uid':self.uid, 'channel':'shell', 'message':io},default=json_serial))
            time.sleep(0.1)


class SilverIPythonKernelManager(QtGui.QTabWidget):
    def __init__(self):
        QtGui.QTabWidget.__init__(self)
        self.kernel_manager = MultiKernelManager()
        self.kernels = {}
        self.kernel_status = {}
        self.threads = {}
        self.messages = {}
        self.update()
    def io(self, value):
        v= json.loads(value)
        #print (v['uid'], v['message'])
        if 'msg_type' in v['message'] and v['message']['msg_type'] == 'status':
            self.kernel_status[v['uid']] = str(v['message']['content']['execution_state'])
        self.messages[v['uid']] = self.messages.get(v['uid'],[]) + [v['message']]
        self.update()
    def start_kernel(self):
        uid = self.kernel_manager.start_kernel()
        kernel = self.kernel_manager.get_kernel(uid)
        self.kernels[uid] = kernel
        self.start_monitor_thread(uid)
        self.update()
        return uid
    def start_monitor_thread(self, uid):
        if uid in self.threads:
            return
        self.messages[uid] = [{'msg_type':'text/plain','content':{'data':'Started monitor thread'}}]
        info_thread = SilverIPythonKernelManager_InfoThread(self,uid,self.kernels[uid])
        info_thread.start()
        self.threads[uid] = info_thread
        #info_thread.emit(QtCore.SIGNAL('add_cmd(QString)'), '%pylab inline')
        self.connect(info_thread, QtCore.SIGNAL("io(QString)"), self.io)        
    def get_new_connection_file(self, uid = None):
        if uid is None:
            uid = self.kernel_manager.start_kernel()
        kernel = self.kernel_manager.get_kernel(uid)
        self.kernels[uid] = kernel
        self.update()
        return kernel.connection_file
    def execute(self,cmd,uid=None):
        if uid is None:
            if self.selected is not None:
                uid = self.selected
            else:
                uid = self.start_kernel()
        self.threads[uid].emit(QtCore.SIGNAL('add_cmd(QString)'), cmd)
        return uid,self.threads[uid].stuff[-1] # This is possibly a race condition!
    def update(self):
        self.emit(QtCore.SIGNAL("update()"))
    def shutdown(self):
        self.kernel_manager.shutdown_all()

class SilverIPythonKernelManagerFrontend(QWidget):
    def __init__(self,kernel_manager):
        QtGui.QTabWidget.__init__(self)
        self.kernel_manager = kernel_manager
        self.selected = None
        self.tree = QtGui.QTreeView()
        self.model = QtGui.QStandardItemModel()
        self.tree.setModel(self.model)
        self.log_box = Browser()
        self.cmd_box = QtGui.QLineEdit()
        self.cmd_box.returnPressed.connect(lambda *args,**kwargs: self.execute(self.cmd_box.text()))
        layout = QVBoxLayout(self)
        layout.addWidget(self.tree)
        layout.addWidget(self.cmd_box)
        layout.addWidget(self.log_box)
        self.setLayout(layout)
        self.update()
        self.tree.clicked.connect(self.click_tree)
        self.connect(self.kernel_manager, QtCore.SIGNAL("update()"), self.update)
    def execute(self,cmd,uid=None):
        return self.kernel_manager.execute(cmd,uid)
    def update(self):
        text = '<ul>'
        self.model.clear()
        self.model.setHorizontalHeaderLabels(['Kernel','connection file','state'])
        for uid in self.kernel_manager.kernels:
            kernel_row = QtGui.QStandardItem('Kernel ' +str(self.kernel_manager.kernels[uid].kernel.pid))
            kernel_row.setData(uid)
            self.model.appendRow([kernel_row,QtGui.QStandardItem(self.kernel_manager.kernels[uid].connection_file),QtGui.QStandardItem(self.kernel_manager.kernel_status.get(uid,'- idle -'))])
            text += '<li>Kernel: '+str(self.kernel_manager.kernels[uid].connection_file)+':'+str(self.kernel_manager.kernels[uid].kernel.pid)+'</li>\n'
        text += '</ul>'
        if not self.selected in self.kernel_manager.messages:
            self.selected = None
        if self.selected is None:
            self.log_box.setHtml(text)
        else:
            text = ''
            for m in reversed(self.kernel_manager.messages[self.selected]):
                if m['msg_type'] == 'stream':
                    text += "<div>[" +str(m['content']['name']) + "] " + escape(unicode(m['content']['data'])) + '</div>\n'
                elif m['msg_type'] == 'display_data':
                    for dtype in m['content']['data'].keys():
                        if dtype == 'text/plain':
                            text += "<div>" + escape(str(m['content']['data']['text/plain']))+ '</div>\n'
                        else:                        
                            text += "<div><img src=\"data:"+ str(dtype)+";base64," + str(m['content']['data'][dtype])+ '\"></div>\n'
                elif m['msg_type'] == 'execute_reply':
                    text += "<div>[" +str(m['content']['execution_count']) + "] " + str(m['content']['status'])+ '</div>\n'
                elif m['msg_type'] == 'pyout':
                    if 'text/plain' in m['content']['data']:
                        text += "<div>[" +str(m['content']['execution_count']) + "] " + escape(str(m['content']['data']['text/plain']))+ '</div>\n'
                    else:
                        text += "<div>[" +str(m['content']['execution_count']) + "] " + escape(str(m['content']['data']))+ '</div>\n'
                elif m['msg_type'] == 'execute_request' or m['msg_type'] == 'pyin':
                    text += "<div>[" +str(m['content']['execution_count']) + "] " + escape(str(m['content']['code']))+ '</div>\n'
                elif m['msg_type'] == 'status':
                    text += "<div>Status: " +str(m['content']['execution_state']) + '</div>\n'
                else:                    
                    text += '<div>\n\n----------------------------------\n\n' + str(m) + '\n\n----------------------------------</div>\n\n'
            self.log_box.setHtml(text)
    def click_tree(self, index):
        ind = self.model.itemFromIndex(index)
        if ind is None:
            return
        uid = ind.data()
        text = ''
        if uid in self.kernel_manager.messages:
            self.selected = uid
            self.update()
        else:
            self.kernel_manager.start_monitor_thread(uid)

class InfoThread(QtCore.QThread):
    def __init__(self, main_widget):
        super(InfoThread , self).__init__()
        self.main_widget = main_widget
        self.threshold = 1024*1024*3.0
    def kb_to_str(self,footprint_kb):
        if footprint_kb < 1024:
            return str(footprint_kb)+" kb"
        elif footprint_kb < 1024*1024:
            return str(0.1*int(round(10.0*footprint_kb/(1024))))+" Mb"
        elif footprint_kb < 1024*1024*1024:
            return str(0.1*int(round(10.0*footprint_kb/(1024*1024))))+" Gb"
        elif footprint_kb < 1024*1024*1024*1024:
            return str(0.1*int(round(10.0*footprint_kb/(1024*1024*1024))))+" Tb"
    def run(self):
        while True:
            session_count = schedule.meman.count_sessions()
            object_count = schedule.meman.count_existing()
            object_loaded_count = schedule.meman.count_in_memory()
            footprint_kb = schedule.meman.get_footprint()
            self.emit(QtCore.SIGNAL('update(QString)'), 'Memory %s | Max %s | %s Sessions, %s/%s Objects' % (
                                                                            self.kb_to_str(footprint_kb), 
                                                                            self.kb_to_str(self.threshold),
                                                                            session_count,
                                                                            object_loaded_count,
                                                                            object_count) )
            #self.main_widget.statusBar().showMessage('Memory %s | Max %s | %s Objects' % (self.kb_to_str(footprint_kb), self.kb_to_str(self.threshold),object_count) )
            time.sleep(1.0)
            schedule.meman.clear_oldest_until_below(self.threshold)

class SilverMainWindow(QtGui.QMainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.kernel_manager = SilverIPythonKernelManager()
        self.setWindowTitle('Silver Gui')
        self.exit = QtGui.QAction(QtGui.QIcon('icons/exit.png'), 'Exit', self)
        self.exit.setShortcut('Ctrl+Q')
        self.connect(
            self.exit, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))
        self.clear_memory = QtGui.QAction(QtGui.QIcon('icons/clear.png'), 'Clear Memory', self)
        def clearMemory():
            schedule.meman.clear_memory()
        self.connect(
            self.clear_memory, QtCore.SIGNAL('triggered()'), clearMemory)
        self.add_kernel = QtGui.QAction(QtGui.QIcon('icons/kernel.svg'), 'Kernel List', self)
        def addKernel():
            self.open_special('kernels')
        self.connect(
            self.add_kernel, QtCore.SIGNAL('triggered()'), addKernel)
        self.test_kernels = QtGui.QAction(QtGui.QIcon('icons/star.svg'), 'Test', self)
        def test_kernels():
            ret = self.kernel_manager.execute('"2"*3')
            #print('ret:')
            #print(ret)
        self.connect(
            self.test_kernels, QtCore.SIGNAL('triggered()'), test_kernels)
        self.statusBar()
        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(self.exit)
        filemenu.addAction(self.add_kernel)
        filemenu.addAction(self.clear_memory)
        self.main_widget = QWidget()
        main_widget_layout = QVBoxLayout(self.main_widget)
        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(self.exit)
        self.toolbar.addAction(self.clear_memory)
        self.toolbar.addAction(self.add_kernel)
        self.toolbar.addAction(self.test_kernels)
        #self.ipython = SilverIPython()

        self.tab_widget = QtGui.QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.connect(self.tab_widget, QtCore.SIGNAL('tabCloseRequested(int)'), self.closeTab)
        self.openWidget = OpenGui(self)

        self.tab_widget.addTab(self.openWidget, QtGui.QIcon("icons/star.svg"), "Open")
        #s = schedule.Session(None,'/home/jacob/Projects/Silversight/Tasks/ex1/Session 1')
        #s.load()
        ##self.tab_widget.addTab(SilverSessionWidget(s), "s.name")
        #self.tab_widget.addTab(SilverSessionWidget(None), "Experiment 2")
        #self.tab_widget.addTab(SilverSessionWidget(None), "Experiment 3")

        main_widget_layout.addWidget(self.tab_widget)
        self.setCentralWidget(self.main_widget)
        self.open_sessions = []
        #self.info_thread = InfoThread(self)
        #self.info_thread.start()
        #self.connect(self.info_thread, QtCore.SIGNAL('update(QString)'), self.statusBar().showMessage)
        self.connect(self, QtCore.SIGNAL('triggered()'), self.closeEvent)
    def closeEvent(self, event):
        self.kernel_manager.shutdown()
    def open(self, session):
        if session.filename in self.open_sessions:
            return
        self.open_sessions.append(session.filename)
        title = "New Session"
        if session is not None:
            title = session.filename
        self.openWidget.save_last_sessions(session.filename)
        self.tab_widget.addTab(SilverSessionWidget(self,session), title)
    def open_special(self, s):
        if 'special:'+s in self.open_sessions:
            return
        if s == 'kernels':
            self.kernel_manager_frontend = SilverIPythonKernelManagerFrontend(self.kernel_manager)
            self.tab_widget.addTab(self.kernel_manager_frontend, QtGui.QIcon("icons/kernel.svg"), "Kernels")
            self.open_sessions.append('special:'+s)
    def closeTab(self,i):
        if i > 0:
            self.open_sessions.remove(self.open_sessions[i-1])
            self.tab_widget.removeTab(i) 
    def add_experiment(self, experiment):
        title = "Experiment"
        if experiment is not None:
            title = experiment.title
        self.tab_widget.addTab(SilverExperimentGui(experiment), title)

def main():
    app = SilverMainWindow()
    app.show()
    app.setWindowIcon(QtGui.QIcon("icons/kernel.svg"));
    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]+'.json'):
            sess = schedule.Session(None,sys.argv[1])
            sess.load()
            app.open(sess)
    qt_app.exec_()


if __name__ == '__main__':
    main()
