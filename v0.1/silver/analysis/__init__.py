import os as _os
import glob as _glob
import imp as _imp
_modules = _glob.glob(_os.path.dirname(__file__)+"/*.py")
_loaded_modules = {}
_functions = []
#__all__ = [ os.path.basename(_f)[:-3] for _f in _modules if _f is not '__init__']
for _f in _modules:
    _module_name = '.'.join((_f.split("/")[-1].split('.')[:-1]))
    if not _f.endswith('__init__.py'):
        with open(_f,'r') as _source_file:
            _loaded_modules[_module_name] = _imp.load_module(_module_name, _source_file, _f, ('.py','r',_imp.PY_SOURCE))
            locals()[_module_name] = _loaded_modules[_module_name]
            for _fun in dir(_loaded_modules[_module_name]):
                if callable(_loaded_modules[_module_name].__dict__[_fun]):
                    _doc = ""
                    try:
                        _doc = str(_loaded_modules[_module_name].__dict__[_fun].__doc__)
                    except:
                        pass
                    _functions.append({'module':_module_name, 'fun': _fun, 'doc': _doc})

def find(s,module=None):
    found = []
    for f in _functions:
        if s == f['module']+'.'+f['fun']:
            return _loaded_modules[f['module']].__dict__[f['fun']]
        if s in f['fun']:
            if module is None or module in f['module']:
                found.append(f)
    if len(found) == 0:
        return None
    if len(found) == 1:
        f = found[0]
        return _loaded_modules[f['module']].__dict__[f['fun']]
    else:
        print "Did you mean:"
        for f in found:
            print str(f['module']) + '.' +  str(f['fun'])
            if f['doc'] is not None:
                print '"""\n'+str(f['doc'])+'\n"""'