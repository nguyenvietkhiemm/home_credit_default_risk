# import all
from sitecustomize import ROOT

from .path import *
from .application import *
from .prev import *
from .bureau import *
from .other import *

def reload():
    # nếu có sự thay đổi trong config, chạy reload rồi import lại
    import importlib
    from . import application, prev, path, bureau, other
    importlib.reload(application)
    importlib.reload(prev)
    importlib.reload(path)
    importlib.reload(bureau)
    importlib.reload(other)
    globals().update(application.__dict__)
    globals().update(prev.__dict__)
    globals().update(path.__dict__)
    globals().update(bureau.__dict__)
    globals().update(other.__dict__)