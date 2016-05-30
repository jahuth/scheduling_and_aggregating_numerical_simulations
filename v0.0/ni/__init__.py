"""NI toolbox
.. module:: ni
   :platform: Unix
   :synopsis: Neuroinformatics Toolbox

.. moduleauthor:: Jacob Huth

"""

import tools
import model
import data

from tools.html_view import View, iView
from tools.statcollector import StatCollector
from data.data import Data, merge
from ni.tools.project import figure
from ni.tools.pickler import load

try:
    # This code can be put in any Python module, it does not require IPython
    # itself to be running already.  It only creates the magics subclass but
    # doesn't instantiate it yet.
    from IPython.core.magic import (Magics, magics_class, line_magic,
                                    cell_magic, line_cell_magic)
    from IPython.display import IFrame, HTML
    from IPython.utils.contexts import preserve_keys
    # The class MUST call this class decorator at creation time
    @magics_class
    class MyMagics(Magics):

        @line_cell_magic
        def niview(self, line, cell=None):
            if cell is None:
                #print("Called as line magic")
                return line
            else:
                #print("Called as cell magic")
                # line, cell
                options = line.split(' ')
                html_save_path = ''
                pickle_save_path = ''
                for o in options:
                    if o.endswith('.html'):
                        html_save_path = o
                    if o.endswith('.pkl'):
                        pickle_save_path = o
                v = View(line,capture = True)
                cell_ns = {'figure': v.figure,'add':v.add,'__view':v }
                if 'load' in options and pickle_save_path != '':
                    v.load(pickle_save_path,silent=True)
                with v:
                    exec(cell, self.shell.user_ns, cell_ns)
                if pickle_save_path != '':
                    v.save(pickle_save_path)
                cell_ns.pop('figure',None)
                cell_ns.pop('add',None)
                cell_ns.pop('__view',None)
                with preserve_keys(self.shell.user_ns, '__file__'):
                    self.shell.user_ns.update(cell_ns)
                return HTML(v.render_html())

    ip = get_ipython()
    ip.register_magics(MyMagics)
except:
    pass # No IPython? Well, no magics for you!


version = 0.2