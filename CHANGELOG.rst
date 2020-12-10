Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 0.5.2 (2020-12-10)
--------------------------

* Fixed: :class:`audinterface.Feature` allow
  ``win_dur=None`` with ``unit='samples'``


Version 0.5.1 (2020-12-04)
--------------------------

* Changed: store ``Process.channels`` always as a list
* Changed: keep ``Feature.win_dur`` and ``Feature.hop_dur`` in original format


Version 0.5.0 (2020-12-03)
--------------------------

* Added: arguments ``channels`` and ``mixdown`` to
  :class:`audinterface.Process`,
  :class:`audinterface.ProcessWithContext`,
  :class:`audinterface.Feature`,
  :class:`audinterface.Segment`
* Removed: ``channel`` argument from all ``process_*`` functions


Version 0.4.3 (2020-11-24)
--------------------------

* Fixed: :meth:`audinterface.Feature.__call__`
  always returns :class:`numpy.ndarray`


Version 0.4.2 (2020-11-23)
--------------------------

* Changed: :meth:`audinterface.Process.process_unified_format_index` and
  :meth:`audinterface.Feature.process_unified_format_index`
  support filewise index


Version 0.4.1 (2020-11-20)
--------------------------

* Added: ``process_func_is_mono`` argument to
  :class:`audinterface.Feature`,
  :class:`audinterface.Process`
* Fixed: avoid nested progress bars


Version 0.4.0 (2020-10-21)
--------------------------

* Changed: make
  :class:`audinterface.Feature`,
  :class:`audinterface.Process`,
  :class:`audinterface.ProcessWithContext`,
  :class:`audinterface.Segment`,
  callable
* Changed: use ``name`` and ``params`` arguments
  in :class:`audinterface.Feature`


Version 0.3.2 (2020-09-21)
--------------------------

* Changed: switch to ``audeer.run_tasks``
* Changed: cut signal before resampling is applied


Version 0.3.1 (2020-09-18)
--------------------------

* Fixed: :class:`audinterface.Feature` raises an due to missing sampling rate
  now only if ``win_dur`` is given


Version 0.3.0 (2020-08-07)
--------------------------

* Changed: switch to :mod:`audsp` >=0.9.2, which fixes a critical resampling
  issue and introduces a new keyword arg name


Version 0.2.4 (2020-06-12)
--------------------------

* Fixed: description and keywords of package in :file:`setup.cfg`


Version 0.2.3 (2020-06-11)
--------------------------

* Fixed: syntax error in CHANGELOG


Version 0.2.2 (2020-06-11)
--------------------------

* Fixed: :meth:`audinterface.Process.process_file` was changing end times
  when process a segmented index


Version 0.2.1 (2020-06-10)
--------------------------

* Changed: ``utils.check_index`` ignores `datetime`


Version 0.2.0 (2020-06-10)
--------------------------

* Added: ``segment`` argument to :class:`audinterface.Process` and :class:`audinterface.Feature`
* Removed: ``name`` argument from :class:`audinterface.Feature`


Version 0.1.0 (2020-06-05)
--------------------------

* Added: initial release


.. _Keep a Changelog:
    https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning:
    https://semver.org/spec/v2.0.0.html
