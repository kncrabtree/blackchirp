.. index::
   single: rollingdata/
   single: log/
   single: debug_log
   single: textexports/

.. _data-storage-other:

Other Data Files
================

Besides the per-experiment folders, Blackchirp keeps three subfolders at
the top of the :doc:`Data Storage Location </user_guide/data_storage>`
that are not tied to any single experiment.

rollingdata/
------------

This folder holds the long-running monitoring data plotted on the Rolling
Data tab. Data are organized into one folder per year, each containing one
folder per month; within a month, Blackchirp writes one CSV file per
rolling-data source. The file format, the identifier scheme, and the
history-trimming behavior are described on
:doc:`/user_guide/rolling-aux-data`. (The per-experiment ``auxdata.csv``
uses a closely related format and is documented on :doc:`experiment`.)

log/
----

This folder holds the application-wide log. All messages shown on the Log
tab are written here in semicolon-delimited CSV, one file per calendar
month named ``YYYYMM.csv``. When debug logging is enabled, a parallel
``debug_YYYYMM.csv`` collects the Debug-severity stream. The column layout
(``Timestamp;Epoch_msecs;Code;Message``) and the severity codes are
described on :doc:`/user_guide/log_tab`.

The ``log/crashes/`` subdirectory holds diagnostic reports written when
Blackchirp terminates unexpectedly; see :doc:`/user_guide/crash_reports`.

textexports/
------------

This folder is the default location for the XY export files produced by
the ``Export XY`` plot action (see :ref:`curve-configuration-options`).
Each export is a standalone text file whose column delimiter is chosen
beside the ``Export XY`` button — semicolon, comma, tab, or aligned
whitespace. Nothing in this folder is read back by Blackchirp.
