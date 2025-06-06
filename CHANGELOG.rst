Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 1.3.1 (2025/06/05)
--------------------------

* Fixed: building release documentation


Version 1.3.0 (2025/06/05)
--------------------------

* Added: ``audinterface.SegmentWithFeature``
* Added: support for Python 3.13


Version 1.2.3 (2024/12/16)
--------------------------

* Added: support for Python 3.12
* Changed: update progress bar at least every second


Version 1.2.2 (2024/06/28)
--------------------------

* Added: ``audinterface.Segment.process_table()``,
  that segments a dataframe or series
  and expands its row values
  to the new segments
* Added: support for ``numpy`` 2.0
* Changed: depend on ``audmath>=1.4.1``


Version 1.2.1 (2024/03/27)
--------------------------

* Changed: default value of
  ``process_func_applies_sliding_window``
  argument of ``audinterface.Feature``
  from ``True`` to ``False``
* Fixed: ``multiprocessing`` for
  ``audinterface.Feature``,
  ``audinterface.Process``,
  ``audinterface.ProcessWithContext``,
  ``audinterface.Segment``
* Removed: deprecated ``unit`` argument
  from ``audinterface.Feature``


Version 1.2.0 (2024/03/21)
--------------------------

* Added: ``process_func_args`` argument to
  ``audinterface.Feature.process_file()``,
  ``audinterface.Feature.process_files()``,
  ``audinterface.Feature.process_folder()``,
  ``audinterface.Feature.process_index()``,
  ``audinterface.Feature.process_signal()``,
  ``audinterface.Feature.process_signal_from_index()``,
  ``audinterface.Process.process_file()``,
  ``audinterface.Process.process_files()``,
  ``audinterface.Process.process_folder()``,
  ``audinterface.Process.process_index()``,
  ``audinterface.Process.process_signal()``,
  ``audinterface.Process.process_signal_from_index()``,
  ``audinterface.ProcessWithContext.process_index()``,
  ``audinterface.ProcessWithContext.process_signal_from_index()``,
  ``audinterface.Segment.process_file()``,
  ``audinterface.Segment.process_files()``,
  ``audinterface.Segment.process_folder()``,
  ``audinterface.Segment.process_index()``,
  ``audinterface.Segment.process_signal()``,
  ``audinterface.Segment.process_signal_from_index()``.
  It allows to temporarily overwrite
  the ``process_func_args`` argument
  used when instantiating an interface
* Changed: depend on ``audeer>=1.18.0``
* Fixed: avoid deprecation warning
  by replacing
  ``pkg_resources``
  internally with
  ``importlib.metadata``


Version 1.1.0 (2023/07/28)
--------------------------

* Added: ``include_root`` argument to
  ``audinterface.Feature.process_folder()``,
  ``audinterface.Process.process_folder()``,
  ``audinterface.Segment.process_folder()``.
  Returns relative file path
  in index
  if set to ``False``.
  Default value is ``True``
* Changed: when ``audinterface.Feature``
  is instantiated with an ``audinterface.Segment``
  object that returns an empty index,
  ``audinterface.Feature.process_*()``
  no longer return ``Index([], dtype="object")``
  but ``MultiIndex([], names=["file", "start", "end"])``
* Fixed: preserve order of ``start`` and ``end`` values
  as returned by the segmentation callable
  in the index returned by ``audinterface.Segment``
  processing functions
* Fixed: precision of ``audinterface.utils.to_timedelta()``
  for ``pd.Timedelta`` objects as input,
  e.g.
  ``to_timedelta(pd.Timedelta("0 days 00:00:35.511437999"))``
  now returns
  ``Timedelta("0 days 00:00:35.511437999")``
  instead of
  ``Timedelta("0 days 00:00:35.511437")``.
  This also affects the output of
  ``audinterface.utils.signal_index()``
* Fixed: preserve requested ``start`` and ``end`` values in
  ``process_file()``,
  ``process_files()``,
  ``process_folder()``
  methods of ``audinterface.Process``
  and ``audinterface.Feature()``.
  Before they were rounded
  to the next sample
  in the returned index


Version 1.0.4 (2023/07/13)
--------------------------

* Changed: require ``audmath>=1.3.0``
* Changed: require ``audiofile>=1.3.0``
* Changed: always evenly round
  ``start`` and ``end``
  to samples
  in ``audinterface.utils.read_signal()``
* Fixed: process functions
  that work on segments
  are now always processing the identical signal
  independent if they get the signal as input,
  like ``audinterface.Process.process_signal()``,
  or a file,
  like ``audinterface.Process.process_file()``


Version 1.0.3 (2023/06/06)
--------------------------

* Added: ``cache_root`` argument
  to ``audinterface.Segment.process_index()``


Version 1.0.2 (2023/05/08)
--------------------------

* Added: support for Python 3.11
* Changed: require ``audformat>=1.0.1``
* Fixed: ``*.process_index()``
  keeps precision of ``end`` values
  when a segmented index
  is returned


Version 1.0.1 (2023/03/29)
--------------------------

* Fixed: do not truncate strings
  in values returned by process methods of
  ``audinterface.Feature``


Version 1.0.0 (2023/03/14)
--------------------------

* Added: ``preserve_index`` argument to
  ``audinterface.Process.process_index()``
  and ``audinterface.Feature.process_index()``
* Added: support for Python 3.10
* Changed: speed up processing methods
  dealing with multiple files/segments in
  ``audinterface.Feature``,
  ``audinterface.Process``,
  ``audinterface.ProcessWithContext``,
  and ``audinterface.Segment``
* Removed: deprecated ``kwargs`` argument
  from ``audinterface.Feature``,
  ``audinterface.Process``,
  ``audinterface.ProcessWithContext``,
  and ``audinterface.Segment``;
  use the ``process_func_args`` argument instead


Version 0.10.2 (2023/01/20)
---------------------------

* Fixed: require ``sphinx-audeering-theme>=1.2.1``
  to ensure the correct theme is used
  for the public documentation


Version 0.10.1 (2023/01/19)
---------------------------

* Changed: split API documentation into sub-pages
  for each function
* Fixed: typo in multi-channel usage example
  in the documentation


Version 0.10.0 (2022-10-24)
---------------------------

* Changed: ``audinterface.Feature``
  uses multi-level column format
  and names columns after channel ID
  when result has more than one channel


Version 0.9.2 (2022-10-07)
--------------------------

* Changed: require ``audformat>=0.15.3``
* Fixed: segmented indices are returned
  with ``file`` level
  as dtype ``string``
  as introduced with ``audformat`` 0.15.0
* Removed: support for Python 3.7


Version 0.9.1 (2022-07-22)
--------------------------

* Changed: support special arguments
  ``idx``,
  ``file``,
  ``root``
  in processing function


Version 0.9.0 (2022-07-11)
--------------------------

* Added: ``auinterface.utils.sliding_window()``
* Added: ``auinterface.utils.to_timedelta()``
* Added: argument
  ``process_func_applies_sliding_window``
  to
  ``audinterface.Feature``
* Added: arguments
  ``win_dur`` and ``hop_dur``
  to
  ``audinterface.Process``
* Added: arguments
  ``min_signal_dur`` and ``max_signal_dur``
  to
  ``audinterface.Feature``,
  ``audinterface.Process``,
  ``audinterface.Segment``
* Added: argument
  ``cache_root``
  to
  ``audinterface.Feature.process_index()``,
  ``audinterface.Process.process_index()``
* Changed: extend documentation with docstring examples and usage section
* Deprecated: ``unit`` argument of ``audinterface.Feature``,
  instead specify unit directly when providing duration arguments,
  compare ``audinterface.utils.to_timedelta()``
* Fixed: ``audinterface.ProcessWithContext``
  requires one output for every segment


Version 0.8.1 (2022-02-07)
--------------------------

* Added: ``audinterface.Feature`` does support
  specification of single feature as string
  in addition to list
* Changed: depend on ``audresample>=1.1``
* Fixed: handle empty and non-existing folder
  in ``process_folder()`` methods


Version 0.8.0 (2022-01-04)
--------------------------

* Added: Python 3.9 support
* Added: ``process_func_args`` argument to process methods
* Deprecated: ``**kwargs`` in process methods,
  use ``process_func_args`` argument instead
* Removed: deprecated ``process_unified_format_index()`` methods
* Removed: Python 3.6 support


Version 0.7.0 (2021-07-23)
--------------------------

* Changed: ``Feature.process_func()`` supports more shapes, namely
  ``(features)``
  ``(channels, features)``
  ``(features, frames)``
  ``(channels, features, frames)``
* Changed: ``Feature.process_func()`` supports more types
  (e.g. scalars, lists, strings, etc.)
* Fixed: ``Feature.__call__()`` always returns ``(channels, features, frames)``


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
  ``win_dur=None`` with ``unit="samples"``


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
