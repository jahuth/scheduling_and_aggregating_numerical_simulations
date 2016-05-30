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
    from PyQt4 import QtCore, QtGui, QtNetwork
    from PyQt4.QtWebKit import QWebView
else:
    from PySide.QtGui import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QTreeView
    from PySide.QtCore import QUrl
    from PySide import QtCore, QtGui, QtNetwork
    from PySide.QtWebKit import QWebView
import time
import json
import re
import subprocess
import datetime

from cgi import escape

import silver.schedule as schedule

class Browser(QWebView):

    def __init__(self):
        QWebView.__init__(self)
        self.loadFinished.connect(self._result_available)
        QtCore.QObject.connect(self, QtCore.SIGNAL("downloadRequested(const QNetworkRequest&)"), self.downloadRequested)
        self.netmanager = QtNetwork.QNetworkAccessManager(self)
    def downloadRequested(self,qrequest):
        orgFileName = QtCore.QFileInfo(qrequest.url().toString()).fileName()
        filename = QtGui.QFileDialog.getSaveFileName(self, "Save file as...", orgFileName)
        if filename.isEmpty():
            return
        nrequest = QtNetwork.QNetworkRequest(qrequest)
        nrequest.setAttribute(QtNetwork.QNetworkRequest.User, filename)
        self.files[filename] = self.netmanager.get(nrequest)
        QObject.connect(self.files[filename], QtCore.SIGNAL("finished()"), self.downloadFinished_Slot)
    def downloadFinished_Slot(self):
        pass
    def _result_available(self, ok):
        frame = self.page().mainFrame()
        # print unicode(frame.toHtml()).encode('utf-8')

def notify(message):
    try:
        p = subprocess.Popen(['notify-send',message], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    except:
        pass

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


class KwargDialog(QtGui.QDialog):
    def __init__(self,parent,kwargs):
        QtGui.QDialog.__init__(self,parent)
        self.kwargs = kwargs
        self.setupUi(self)
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(508, 300)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(150, 250, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.ks_box = QtGui.QVBoxLayout()
        self.kwarg_widget = QtGui.QTreeView()
        self.model = QtGui.QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Task','status'])
        self.kwarg_widget.setModel(self.model)
        for k in self.kwargs:
            k_box = [QtGui.QStandardItem('{}'.format(k)),QtGui.QStandardItem('{}'.format(str(self.kwargs[k]))),QtGui.QStandardItem('{}'.format(type(self.kwargs[k])))]
            self.model.appendRow(k_box)
        self.ks_box.addWidget(self.kwarg_widget)
        self.ks_box.addWidget(self.buttonBox)
        Dialog.setLayout(self.ks_box)
        self.model.itemChanged.connect(self.handleItemChanged)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    def handleItemChanged(self, item):
        parent = self.kwargs
        key = self.model.item(item.row(), 0).text()
        try:
            parent[key] = type(parent[key])(eval(item.text()))
        except:
            parent[key] = type(parent[key])(item.text())
    def get_kwargs(self):
        return self.kwargs
    @staticmethod
    def getKwargs(parent = None,kwargs=None):
        dialog = KwargDialog(parent,kwargs)
        result = dialog.exec_()
        kwargs = dialog.get_kwargs()
        return (kwargs, result == QtGui.QDialog.Accepted)

class SilverTaskRunnerThread(QtCore.QThread):
    def __init__(self, func, kwargs={}):
        super(SilverTaskRunnerThread , self).__init__()
        self.func = func
        self.kwargs = kwargs
    def run(self):
        print("runner started")
        ret = self.func(**self.kwargs)
        print("runner ended")
        self.emit(QtCore.SIGNAL('update(QString)'), str(ret))

def html_dec(html):
    return """
    <html>
    <head>
    <style>
    .data_pieces {
        overflow-x: scroll;
        white-space: nowrap;
        position: relative;
        top:0px;
    }
    .data_piece {
        display: inline-block;
        padding: 20px;
        margin: 20px;
    }
    .data_piece_duplicate {
        display: none;
    }
    .data_piece.data_piece_duplicate {
        display: inline-block;
    }
    .data_piece.selected {
        background: #ddd;
    }
    .data_piece:hover {
        background: #eee;
    }
    .data_piece td:hover {
        background: #aaa;
    }
    .data_piece td:hover {
        background: #ddd;
    }
    .data_content {
        padding-left: 20px;
    }
    .data_content.collapsed {
        height: 0px;
        border-top: 5px solid #aaa;   
        overflow: hidden;     
    }
    .data_content.hover {
        padding-left: 15px;
        border-left: 5px solid #aaa;
    }
    .arrow {
        display: inline-block;
        background: #ddd;
        color: #444;
        padding: 10px; 
        padding-top: 0px; 
        padding-bottom: 0px; 
        font-size: 24pt;
        cursor: hand;
    }
    .arrow:hover {
        color: #fff;
    }
    .arrow.left {
        border-bottom-left-radius: 10px;
    }
    .arrow.right {
        border-bottom-right-radius: 10px;
    }

    .matrix {
        padding: 10px;
    }
    .matrix td:hover {
        background: #ddd;
    }
    .mulit_matrix {
        overflow-x: scroll;
        white-space: nowrap;
    }
    .multi_matrix p {
        color: #888;
        font-size: 9pt;
        position: absolute;
        left: 0px;
    }
    .multi_matrix .matrix {
        padding: 10px;
        margin: 10px;
        margin-top: 5px;
    }
    .list {
        padding: 5px;
        border-left: 1px solid #aaa;
    }
    .list:hover {
        padding-left: 4px;
        border-left: 2px solid #aaa;
    }
    .list .list_info {
        color: #aaa;
    }
    .dict {
        
    }
    .dict_entry {
        padding-top: 5px;
        padding-left: 20px;
    }
    .dict .key {
        vertical-align: text-top;
        width: 120px;
        font-weight: bold;
        display: inline-block;
        padding-right:10px;
    }
    .dict .value {
        vertical-align: text-top;
        display: inline-block;
    }
    .dict .comma {
        display: inline-block;
    }
    .loading {
        margin: auto;
        margin-top: 200px;
        width: 200px;
        font-weight: bold;
        font-size: 18pt;
        background: #efefef;
        text-align: center;
        padding: 50px;
        padding-top: 20px;
    }
    #sidebar {
        position: absolute;
        top: 50px;
        left: 0px; 
        width: 200px;
        background: white;
        z-Index: 10;
    }

    body.body_with_toc #main {
        position: absolute;
        top: 0px;
        left: 200px;
        right: 0px;
        height: 100%;
        overflow-y: scroll;    
        z-Index: 5;
    }
    .toc {
        display: none;
    }
    body.body_with_toc .toc {
        display: block;
        padding: 0px;
        margin: 0px;    
    }
    .toc ul {
        list-style-type: none;
        padding: 0px;
        margin: 0px;
    } 
    .toc li {
        margin: 0; padding: 0; 
        padding-left: 5px;
    }
    .toc li a {
        font-size: 9pt;
        font-weight: bold;
        padding: 5px;
        cursor: pointer;
    }
    .toc a:hover {
        background: #ddd;
    }
    </style>
    <script src="jquery-2.1.3.min.js"></script>
    <script src="http://code.jquery.com/jquery-1.9.1.min.js"></script>
    <script>
        $( document ).ready(function() {
            $(".data_pieces").each(function (piece) {
                var container = $(this);
                var left = 0;
                var data_piece_index = -1;
                function select(ind) {
                        data_piece_index = ind;
                        if (data_piece_index < 0) {
                            data_piece_index = container.children('.data_piece').length-1;
                        } 
                        if (data_piece_index > container.children('.data_piece').length-1) {
                            data_piece_index = 0;                        
                        }
                        container.children('.data_piece').removeClass('selected');
                        if (container.children(".data_piece").length > 0) {
                            container.children('.data_piece:eq('+data_piece_index+')').addClass('selected');
                            left = container.scrollLeft()
                            left = left + (container.children('.data_piece:eq('+data_piece_index+')').position().left) + (container.children('.data_piece:eq('+data_piece_index+')').width() - container.width())/2;
                            container.animate({scrollLeft: left}, 800);
                        }
                }
                function go_left() {
                    data_piece_index = data_piece_index - 1;
                    select(data_piece_index)
                }
                function go_right() {
                    data_piece_index = data_piece_index + 1;
                    select(data_piece_index)
                }
                $(this).find(".data_piece").each(function (ind) {
                    var data_piece = $(this);
                    $(this).click(function () {
                        select(ind);
                    });
                });
                if ($(this).find(".data_piece").length > 1) {
                    $("<div class='arrow right'>&#8594;</div>").insertAfter(container).click(function () {
                        go_right();
                    });
                    $("<div class='arrow left'>&#8592;</div>").insertAfter(container).click(function () {
                        go_left()
                    });
                }
                if ($(this).find(".data_piece_duplicate").length > 0) {
                    $("<div>"+($(this).find(".data_piece_duplicate").length)+" duplicates</div>").appendTo(container);
                }
            });
            $('.data_title').each(function() {
                $(this).click(function(){
                    $(this).parent().find(".data_content").each(function () {
                        if ($(this).hasClass('collapsed')) {
                            $(this).removeClass('collapsed');
                        } else {
                            $(this).addClass('collapsed');
                        }
                    });
                });
                //if ($(this).parent().data('depth') > 3) {
                //    $(this).parent().find(".data_content").addClass('collapsed');
                //}
                $(this).mouseenter(function() {
                   $(this).parent().children(".data_content").addClass('hover'); 
                });
                $(this).mouseleave(function(){
                   $(this).parent().children(".data_content").removeClass('hover'); 
                });
            });
            $("#filter_input").change(function () {
                    if ($("#filter_input").val() == "") {
                        $(".data").show();
                    } else {
                        $(".data").hide();
                        $(".data").each(function () {
                            classes = $(this).attr('class').split(/\s+/);
                            var data_container = $(this);
                            $.each( classes, function(index, item){
                                if (item != 'data') {
                                   if (item.indexOf($("#filter_input").val()) > -1) {
                                        data_container.show();
                                   }
                                }
                            });
                        });
                    }
                });
            function makeTOC(node, elem, lv) {
                var t;
                if (node.children(".data_title").length > 0) {
                    if (node.children(".data_title").first().text().length > 0) {
                        t = $("<li class='level_"+lv+"'><a>"+node.children(".data_title").first().text()+"</a></li>").appendTo(elem);
                        t.find("a").data('node',node)
                        t.find("a").click(function () {
                            var top = $('#main').scrollTop() + node.offset().top - 20;
                            $('#main').animate({scrollTop: top},200);
                        });
                        var new_elem =$("<ul></ul>").appendTo(t);
                        $(node).children('.data_content').each(function () {
                            $(this).children('.data_sub').each(function () {
                                makeTOC($(this),new_elem,lv+1)
                            });
                        });
                    } 
                } 
            }
            if ($('.content').children('.data_sub').length > 0) {
                $('body').addClass('body_with_toc');
                $('.content').children('.data_sub').each(function () {
                    makeTOC($(this), $('.toc'), 0);
                });
            } else {
                $('body').removeClass('body_with_toc');
            }
        });
    </script>
    </head>
    <body>
    <div id='sidebar'>
    <ul class='toc'></ul>
    </div>
    <input id="filter_input"\>
    <div id='main'>
    <div class='content'>
    """+html+  """
    </div>
    </div>
    </body>
    </html>
    """

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, datetime.datetime):
        serial = obj.isoformat()
        return serial



class LazyThread(QtCore.QThread):
    def __init__(self, function):
        super(LazyThread , self).__init__()
        self.function = function
    def run(self):
        ret = self.function()
        self.emit(QtCore.SIGNAL('update(QString)'), ret)

def lazy(fun1,on_finish):
    lazy_thread = LazyThread(fun1)
    lazy_thread.connect(lazy_thread , QtCore.SIGNAL('update(QString)') , on_finish)
    lazy_thread.start()
    return lazy_thread
