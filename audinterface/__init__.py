"""Generic processing interfaces."""
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
