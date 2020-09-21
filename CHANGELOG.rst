Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 0.3.2 (2020-10-21)
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
