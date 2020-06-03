r"""Generic processing interfaces.

Collection of wrapper classes that provide a common set of methods to
apply them to signals or files, e.g.:

* :func:`process_signal`
* :func:`process_file`
* :func:`process_files`
* :func:`process_folder`

:mod:`audinterface` provides you classes that you can inherit or just
instantiate to get some standard implementations of those methods.

Example:
    >>> import numpy as np
    >>> def process_func(signal, sampling_rate):
    ...     return signal.shape[1] / sampling_rate
    ...
    >>> model = Process(process_func=process_func)
    >>> signal = np.array([1., 2., 3.])
    >>> model.process_signal(signal, sampling_rate=3)
    1.0

"""
from audinterface.core.process import (
    Process,
    ProcessWithContext,
)
from audinterface.core.segment import (
    Segment,
)


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
