# import all
from sitecustomize import ROOT

from .path import *
from .application import *
from .prev import *


def reload():
    # nếu có sự thay đổi trong config, chạy reload rồi import lại
    import importlib
    from . import application, prev, path
    importlib.reload(application)
    importlib.reload(prev)
    importlib.reload(path)
    globals().update(application.__dict__)
    globals().update(prev.__dict__)
    globals().update(path.__dict__)
