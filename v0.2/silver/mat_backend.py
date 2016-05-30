from matplotlib.backends.backend_qt4 import *
import matplotlib.backends.backend_qt4 as old

def show(*args,**kwargs):
	print "this is not matplotlib"
	old.show(*args,**kwargs)