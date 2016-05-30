from __future__ import print_function
import os

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

import sys
from PyQt4.QtGui import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QTreeView
from PyQt4.QtCore import QUrl
from PyQt4 import QtCore, QtGui
from PyQt4.QtWebKit import QWebView

sys.path.insert(0, ".")

qt_app = QApplication(sys.argv)

import silver.schedule

class Browser(QWebView):

    def __init__(self):
        QWebView.__init__(self)
        self.loadFinished.connect(self._result_available)

    def _result_available(self, ok):
        frame = self.page().mainFrame()
        # print unicode(frame.toHtml()).encode('utf-8')



class PythonHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, parent=None):
        super(PythonHighlighter, self).__init__(parent)

        keywordFormat = QtGui.QTextCharFormat()
        keywordFormat.setForeground(QtCore.Qt.darkGreen)
        keywordFormat.setFontWeight(QtGui.QFont.Bold)

        keywordPatterns = ["\\bchar\\b", "\\bclass\\b", "\\bconst\\b",
                "\\bdouble\\b", "\\benum\\b", "\\bexplicit\\b", "\\bfriend\\b",
                "\\binline\\b", "\\bint\\b", "\\blong\\b", "\\bnamespace\\b",
                "\\boperator\\b", "\\bprivate\\b", "\\bprotected\\b",
                "\\bpublic\\b", "\\bshort\\b", "\\bsignals\\b", "\\bsigned\\b",
                "\\bslots\\b", "\\bstatic\\b", "\\bstruct\\b",
                "\\btemplate\\b", "\\btypedef\\b", "\\btypename\\b",
                "\\bunion\\b", "\\bunsigned\\b", "\\bvirtual\\b", "\\bvoid\\b",
                "\\bvolatile\\b"] + [ '\\b'+t+'\\b' for t in ['and', 'assert', 'break', 'class', 'continue', 'def',
                'del', 'elif', 'else', 'except', 'exec', 'finally',
                'for', 'from', 'global', 'if', 'import', 'in',
                'is', 'lambda', 'not', 'or', 'pass', 'print',
                'raise', 'return', 'try', 'while', 'yield',
                'None', 'True', 'False']]

        self.highlightingRules = [(QtCore.QRegExp(pattern), keywordFormat)
                for pattern in keywordPatterns]

        classFormat = QtGui.QTextCharFormat()
        classFormat.setFontWeight(QtGui.QFont.Bold)
        classFormat.setForeground(QtCore.Qt.darkBlue)
        self.highlightingRules.append((QtCore.QRegExp("\\bQ[A-Za-z]+\\b"),
                classFormat))

        singleLineCommentFormat = QtGui.QTextCharFormat()
        singleLineCommentFormat.setForeground(QtCore.Qt.red)
        self.highlightingRules.append((QtCore.QRegExp("//[^\n]*"),
                singleLineCommentFormat))

        self.multiLineCommentFormat = QtGui.QTextCharFormat()
        self.multiLineCommentFormat.setForeground(QtCore.Qt.red)

        quotationFormat = QtGui.QTextCharFormat()
        quotationFormat.setForeground(QtCore.Qt.darkGreen)
        self.highlightingRules.append((QtCore.QRegExp("\".*\""),
                quotationFormat))
        self.highlightingRules.append((QtCore.QRegExp("\"\"\".*\"\"\""),
                quotationFormat))
        self.highlightingRules.append((QtCore.QRegExp("\'.*\'"),
                quotationFormat))
        self.highlightingRules.append((QtCore.QRegExp("\'\'\'.*\'\'\'"),
                quotationFormat))

        functionFormat = QtGui.QTextCharFormat()
        functionFormat.setFontItalic(True)
        functionFormat.setForeground(QtCore.Qt.blue)
        self.highlightingRules.append((QtCore.QRegExp("\\b[A-Za-z0-9_]+(?=\\()"),
                functionFormat))

        self.commentStartExpression = QtCore.QRegExp("/\\*")
        self.commentEndExpression = QtCore.QRegExp("\\*/")

    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            expression = QtCore.QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)

        startIndex = 0
        if self.previousBlockState() != 1:
            startIndex = self.commentStartExpression.indexIn(text)

        while startIndex >= 0:
            endIndex = self.commentEndExpression.indexIn(text, startIndex)

            if endIndex == -1:
                self.setCurrentBlockState(1)
                commentLength = text.length() - startIndex
            else:
                commentLength = endIndex - startIndex + self.commentEndExpression.matchedLength()

            self.setFormat(startIndex, commentLength,
                    self.multiLineCommentFormat)
            startIndex = self.commentStartExpression.indexIn(text,
                    startIndex + commentLength);



class OpenGui(QWidget):

    '''  '''

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
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

        self.file_layout = QHBoxLayout()
        self.config_layout = QHBoxLayout()

        self.lbl = QtGui.QLabel('No file selected')
        self.file_layout.addWidget(self.lbl)

        btn = QtGui.QPushButton('Choose file', self)
        self.file_layout.addWidget(btn)

        def open_file():
            fname = QtGui.QFileDialog.getOpenFileName(self, 'Select file')
            if fname:
                self.lbl.setText(fname)
                with open(fname) as f:
                    self.view.setHtml(f.read().replace('\n', '<br\>'))
                self.showsessions(fname)
            else:
                self.lbl.setText('No file selected')
        self.connect(btn, QtCore.SIGNAL('clicked()'), open_file)
        self.layout.addLayout(self.file_layout)
        self.path = './'

        # create model
        model = QtGui.QFileSystemModel()
        model.setRootPath(self.path)
        self.stuff = QtGui.QVBoxLayout()
        self.treeView = QtGui.QTreeView()
        # set the model
        self.treeView.setModel(model)
        self.treeView.setRootIndex(model.index(self.path))
        #self.connect(self.treeView.selectionModel(), QtCore.SIGNAL('selectionChanged(QItemSelection, QItemSelection)'), self.check_file)
        self.treeView.clicked.connect(self.check_file)
        self.config_layout.addWidget(self.treeView)
        self.config_layout.addLayout(self.stuff)
        self.stuff.addWidget(self.sessions_list)
        self.config_layout.addWidget(self.view)
        self.layout.addLayout(self.config_layout)
        self.setLayout(self.layout)
        self.show()

    def showsessions(self, filename):
        #import glob
        # self.sessions_list.clear()
        # for f in glob.glob(filename+'_*.session'):
        #    self.sessions_list.addItem(QtGui.QListWidgetItem(f))
        print('oepning...')
        import imp
        import os
        from os.path import split
        try:
            os.chdir(split(filename)[0])
            sand = imp.load_source('module.name', filename)
            t = sand.createTasks()
            print(t)
            self.view.setHtml(
                sand.description + "<br/><hr/>" + str(t.pending()) + " Tasks pending")
            for a in sand.actions.keys():
                new_button = QtGui.QPushButton( a )
                def func():
                    ret = sand.actions[a]['function'](**sand.actions[a]['kwargs'])
                    #print(ret.__class__)
                    #self.view.setHtml(str(ret))
                new_button.clicked.connect(func)
                self.stuff.addWidget(new_button)
        except Exception as e:
            print(e)
            pass

    def check_file(self, index):
        from os.path import isdir, isfile, join
        indexItem = self.treeView.model().index(index.row(), 0, index.parent())
        # path or filename selected
        fileName = self.treeView.model().fileName(indexItem)
        # full path/filename selected
        filePath = self.treeView.model().filePath(indexItem)
        print(filePath)
        if isdir(filePath):
            self.path = filePath
        else:
            self.showsessions(filePath)


class SilverTaskWidget(QWidget):

    '''  '''

    def __init__(self,session):
        QWidget.__init__(self)
        self.session = session
        self.setWindowTitle('Silver Gui')
        self.setMinimumWidth(400)

        self.layout = QHBoxLayout()
        self.right_side_layout = QVBoxLayout()
        self.left_side_layout = QVBoxLayout()

        # Create an in-process kernel
        # >>> print_process_id()
        # will print the same process ID as the main process

        self.view = Browser()
        html = '''<html>
        <head>
        <title>A Sample Page</title>
        </head>
        <body>
        <h1>Hello, World!</h1>
        <hr />
        I have nothing to say.
        </body>
        </html>'''

        self.view.setHtml(html)
        self.right_side_layout.addWidget(self.view)

        self.progressbar = QtGui.QProgressBar()
        self.progressbar.setMinimum(1)
        self.progressbar.setMaximum(100)
        self.left_side_layout.addWidget(self.progressbar)
        self.taskTreeView = QTreeView()
        self.taskTreeView.setMinimumWidth(400)
        self.model = QtGui.QStandardItemModel()
        self.taskTreeView.setModel(self.model)
        if self.session is not None:
            self.progressbar.setMinimum(self.session.root_task.pending())
            self.progressbar.setMaximum(self.session.root_task.count())
            silver_session.guess_total_time()

            def recAddTasks(task):
                me = QtGui.QStandardItem('{}'.format(task.name))
                me.setData(task)
                if task.subtasks:
                    for s in task.subtasks:
                        me.appendRow(recAddTasks( s))
                return me
                    
            self.model.appendRow(recAddTasks(self.session.root_task))
        self.taskTreeView.expandAll()
        #QtCore.QObject.connect(self.taskTreeView.selectionModel(), QtCore.SIGNAL('selectionChanged()'), self.test)
        self.left_side_layout.addWidget(self.taskTreeView)
        self.taskTreeView.clicked.connect(self.test)

        self.layout.addLayout(self.left_side_layout)
        self.layout.addStretch(1)
        self.layout.addLayout(self.right_side_layout)
        self.setLayout(self.layout)
        self.show()
    def test(self, index):
        ind = self.model.itemFromIndex(index)
        #print(ind.text())
        task = ind.data()
        self.view.setHtml(task.to_html())
    def run(self):
        # Show the form
        self.show()
        # Run the qt application
        qt_app.exec_()

class SilverSessionWidget(QtGui.QTabWidget):

    '''  '''

    def __init__(self, session):
        QtGui.QTabWidget.__init__(self)
        self.taskWidget = SilverTaskWidget(session)
        self.addTab(self.taskWidget, "Tasks")
        self.addTab(QtGui.QWidget(), "Data")
        self.addTab(QtGui.QWidget(), "Plots")

        self.ipython = SilverIPython()
        self.addTab(self.ipython, "Console")

        #self.setTabEnabled(1, False)
        #self.setTabEnabled(2, False)
        #self.setTabEnabled(3, False)

class SilverIPython(RichIPythonWidget):

    ''' Wraps an IPython kernel and provides IPython widgets for it '''

    def __init__(self):
        RichIPythonWidget.__init__(self)
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel

        self.kernel.gui = 'qt4'
        self.kernel.shell.push({'window': self, 'kernel': self.kernel})
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()
        self.kernel_client.execute('%pylab inline')
        self.exit_requested.connect(self.exit_requested_func)
    def send_execute(self, *args,**kwargs):
        self.kernel_client.execute( *args,**kwargs )
    def exit_requested_func(self):
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()
            qt_app.exit()
    
class SilverMainWindow(QtGui.QMainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle('Silver Gui')
        self.exit = QtGui.QAction(QtGui.QIcon('icons/exit.png'), 'Exit', self)
        self.exit.setShortcut('Ctrl+Q')
        self.connect(
            self.exit, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))
        self.statusBar()
        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(self.exit)
        self.main_widget = QWidget()
        main_widget_layout = QVBoxLayout(self.main_widget)
        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(self.exit)

        self.ipython = SilverIPython()

        self.tab_widget = QtGui.QTabWidget()

        self.openWidget = OpenGui(self)

        self.tab_widget.addTab(self.openWidget, "Open")
        s = schedule.Session(None,'/home/jacob/Projects/Silversight/Tasks/ex1/Session 1')
        s.load()
        self.tab_widget.addTab(SilverSessionWidget(s), "s.name")
        self.tab_widget.addTab(SilverSessionWidget(None), "Experiment 2")
        self.tab_widget.addTab(SilverSessionWidget(None), "Experiment 3")

        main_widget_layout.addWidget(self.tab_widget)
        self.setCentralWidget(self.main_widget)
    def add_experiment(self, experiment):
        title = "Experiment"
        if experiment is not None:
            title = experiment.title
        self.tab_widget.addTab(SilverExperimentGui(experiment), title)


def main():
    app = SilverMainWindow()
    app.show()
    qt_app.exec_()


if __name__ == '__main__':
    main()
