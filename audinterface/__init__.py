"""Generic processing interfaces."""

from audinterface import utils
from audinterface.core.feature import Feature
from audinterface.core.process import Process
from audinterface.core.process_with_context import ProcessWithContext
from audinterface.core.segment import Segment
from audinterface.core.segment_with_feature import SegmentWithFeature


# Disencourage from audinterface import *
__all__ = []


# Dynamically get the version of the installed module
try:
    import importlib.metadata

    __version__ = importlib.metadata.version(__name__)
except Exception:  # pragma: no cover
    importlib = None  # pragma: no cover
finally:
    del importlib
