import numpy
import base64
import json

# Method I
class _NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Method II (with object hook)
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded
        """
        if isinstance(obj, numpy.ndarray):
            data_b64 = base64.b64encode(obj.data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)
def _json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return numpy.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

def dumps(*args,**kwargs):
    return json.dumps(*args, cls=_NumpyEncoder,**kwargs)
def dump(*args,**kwargs):
    return json.dump(*args, cls=_NumpyEncoder,**kwargs)

def loads(*args,**kwargs):
    return json.loads(*args, object_hook=_json_numpy_obj_hook,**kwargs)
def load(*args,**kwargs):
    return json.load(*args, object_hook=_json_numpy_obj_hook,**kwargs)