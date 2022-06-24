r"""Generic processing interfaces.

:mod:`audinterface` provides you classes
to apply user provided functions to signals or files.
This is very handy
if you want to design a user interface for your model.
You can just use :class:`audinterface.Process`
and provide your :meth:`model.predict` method
as ``process_func`` argument.

The :mod:`audinterface` classes implement all
or a selection of the following methods:

* :meth:`process_file`
* :meth:`process_files`
* :meth:`process_folder`
* :meth:`process_index`
* :meth:`process_signal`
* :meth:`process_signal_from_index`

You can inherit from the classes
or just instantiate them
to get some standard implementations of those methods.

"""
from audinterface import utils
from audinterface.core.feature import Feature
from audinterface.core.process import Process
from audinterface.core.process_with_context import ProcessWithContext
from audinterface.core.segment import Segment


# Disencourage from audinterface import *
__all__ = []


# Dynamically get the version of the installed module
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:  # pragma: no cover
    pkg_resources = None  # pragma: no cover
finally:
    del pkg_resources
