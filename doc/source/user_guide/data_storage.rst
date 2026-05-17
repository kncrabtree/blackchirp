.. index::
   single: Data Storage
   single: experiments/
   single: CSV format

.. _data-storage:

Data Storage
============

Blackchirp stores its data in the selected
:ref:`Data Storage Location <first-run-data-path>`. The location may be
changed from the **Settings → Data Storage** menu item.

.. image:: /_static/user_guide/first_run-savepathdialog.png
   :alt: Data storage location dialog

At this location, Blackchirp creates four subfolders:

* ``experiments``: Per-experiment data files, organized by experiment
  number (see below).
* ``log``: Application log files, plus the ``log/crashes`` diagnostic
  reports.
* ``rollingdata``: CSV files containing long-running monitoring data.
* ``textexports``: Default location for XY export files from plots.

All of Blackchirp's data files are written in plain-text CSV using the
separator character ``;``.

Each experiment is associated with an experiment number, and the location
of its data files within the ``experiments`` folder is derived from that
number. To avoid an excessive number of directories at a single level,
Blackchirp collects experiments in groups of 1000 and stores each group
under a single directory. For a given experiment number X, its data are
stored in ``experiments/Z/Y/X``, where ``Z = X//1000000`` and
``Y = X//1000`` (integer division, discarding the remainder). For example,
experiment 123456789 is stored in::

  experiments/123/123456/123456789

and experiment 480 is stored in::

  experiments/0/0/480

Every experiment writes a common set of files describing its program,
acquisition, and hardware state. CP-FTMW and LIF acquisitions add their
own data and configuration files within the same folder. The pages below
describe each group of files in detail.

.. toctree::
   :maxdepth: 1

   data_storage/experiment
   data_storage/ftmw
   data_storage/lif
   data_storage/other
