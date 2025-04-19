# import all
from sitecustomize import ROOT

from .path import *
from .application import *
from .prev import *
from .bureau import *

def reload():
    # nếu có sự thay đổi trong config, chạy reload rồi import lại
    import importlib
    from . import application, prev, path, bureau
    importlib.reload(application)
    importlib.reload(prev)
    importlib.reload(path)
    importlib.reload(bureau)
    globals().update(application.__dict__)
    globals().update(prev.__dict__)
    globals().update(path.__dict__)
    globals().update(bureau.__dict__)
