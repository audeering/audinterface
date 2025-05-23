audinterface
============

.. automodule:: audinterface

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
The following classes are provided:

.. autosummary::
    :toctree:
    :nosignatures:

    Feature
    Process
    ProcessWithContext
    Segment
    SegmentWithFeature
