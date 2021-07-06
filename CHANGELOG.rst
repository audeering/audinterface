Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 0.6.8 (2021-07-06)
--------------------------

* Fixed: CHANGELOG

Version 0.6.7 (2021-07-06)
--------------------------

* Added: ``utils.signal_index()`` to create a segmented index without file level
* Changed: use keyword argument with ``pandas.MultiIndex.set_levels()``
* Fixed: raise error if multiple frames are returned but ``win_dur`` is not set
* Fixed: remove ``num_channels`` from ``Feature`` docstring


Version 0.6.6 (2021-06-18)
--------------------------

* Added: tests on Windows and macOC


Version 0.6.5 (2021-06-08)
--------------------------

* Added: ``root`` argument to all functions processing file input


Version 0.6.4 (2021-06-07)
--------------------------

* Fixed: avoid using 'sec' as unit in ``pd.to_timedelta()`` for backward compatibility


Version 0.6.3 (2021-05-03)
--------------------------

* Fixed: empty API section in documentation


Version 0.6.2 (2021-04-28)
--------------------------

* Added: open source release on Github
* Changed: switch to MIT license


Version 0.6.1 (2021-04-16)
--------------------------

* Fixed: ``Process``, ``Feature``: do not apply segmentation twice when ``segment`` object is given


Version 0.6.0 (2021-04-15)
--------------------------

* Added: ``invert`` argument to ``Segment``
* Added: ``Segment.process_index()`` and ``Segment.process_signal_from_index()``
* Changed: ``Process.process_index()`` and ``Process.process_signal_from_index()`` do not ignore segment object
* Changed: ``Feature.process_index()`` and ``Feature.process_signal_from_index()`` do not ignore segment object


Version 0.5.5 (2021-02-17)
--------------------------

* Fixed: PyPI publishing pipeline
* Changed: use new tokenizer in CI pipeline


Version 0.5.4 (2021-02-17)
--------------------------

* Added: support for providing ``start`` and ``end`` time values
  in the same format as done in ``audformat``,
  e.g. as integer, floats, or ``pandas.Timedelta``
* Changed: improve speed of CI pipelines
* Fixed: ``audinterface.Feature`` handles empty index


Version 0.5.3 (2021-01-07)
--------------------------

* Changed: rename
  ``audinterface.Feature.process_unified_format_index``,
  ``audinterface.Process.process_unified_format_index``,
  ``audinterface.ProcessWithContext.process_unified_format_index``
  to
  ``audinterface.Feature.process_index``,
  ``audinterface.Process.process_index``,
  ``audinterface.ProcessWithContext.process_index``


Version 0.5.2 (2020-12-10)
--------------------------

* Fixed: ``audinterface.Feature`` allow
  ``win_dur=None`` with ``unit='samples'``


Version 0.5.1 (2020-12-04)
--------------------------

* Changed: store ``Process.channels`` always as a list
* Changed: keep ``Feature.win_dur`` and ``Feature.hop_dur`` in original format


Version 0.5.0 (2020-12-03)
--------------------------

* Added: arguments ``channels`` and ``mixdown`` to
  ``audinterface.Process``,
  ``audinterface.ProcessWithContext``,
  ``audinterface.Feature``,
  ``audinterface.Segment``
* Removed: ``channel`` argument from all ``process_*`` functions


Version 0.4.3 (2020-11-24)
--------------------------

* Fixed: ``audinterface.Feature.__call__``
  always returns ``numpy.ndarray``


Version 0.4.2 (2020-11-23)
--------------------------

* Changed: ``audinterface.Process.process_unified_format_index`` and
  ``audinterface.Feature.process_unified_format_index``
  support filewise index


Version 0.4.1 (2020-11-20)
--------------------------

* Added: ``process_func_is_mono`` argument to
  ``audinterface.Feature``,
  ``audinterface.Process``
* Fixed: avoid nested progress bars


Version 0.4.0 (2020-10-21)
--------------------------

* Changed: make
  ``audinterface.Feature``,
  ``audinterface.Process``,
  ``audinterface.ProcessWithContext``,
  ``audinterface.Segment``,
  callable
* Changed: use ``name`` and ``params`` arguments
  in ``audinterface.Feature``


Version 0.3.2 (2020-09-21)
--------------------------

* Changed: switch to ``audeer.run_tasks``
* Changed: cut signal before resampling is applied


Version 0.3.1 (2020-09-18)
--------------------------

* Fixed: ``audinterface.Feature`` raises an due to missing sampling rate
  now only if ``win_dur`` is given


Version 0.3.0 (2020-08-07)
--------------------------

* Changed: switch to ``audsp`` >=0.9.2, which fixes a critical resampling
  issue and introduces a new keyword arg name


Version 0.2.4 (2020-06-12)
--------------------------

* Fixed: description and keywords of package in ``setup.cfg``


Version 0.2.3 (2020-06-11)
--------------------------

* Fixed: syntax error in CHANGELOG


Version 0.2.2 (2020-06-11)
--------------------------

* Fixed: ``audinterface.Process.process_file`` was changing end times
  when process a segmented index


Version 0.2.1 (2020-06-10)
--------------------------

* Changed: ``utils.check_index`` ignores ``datetime``


Version 0.2.0 (2020-06-10)
--------------------------

* Added: ``segment`` argument to ``audinterface.Process`` and ``audinterface.Feature``
* Removed: ``name`` argument from ``audinterface.Feature``


Version 0.1.0 (2020-06-05)
--------------------------

* Added: initial release


.. _Keep a Changelog:
    https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning:
    https://semver.org/spec/v2.0.0.html
