.. index::
   single: Crash Reports
   single: log/crashes/
   single: Diagnostics

.. _crash-reports:

Crash Reports
=============

If Blackchirp terminates unexpectedly because of a fault inside the
program (a segmentation fault, an unhandled exception, or a similar
internal error), it writes a diagnostic crash report to disk before the
process exits. It captures only what a developer needs to identify the
failing function in the source code; no acquired experiment data is
included (see `What a Report Contains`_).

Where Crash Reports Are Stored
------------------------------

Crash reports are written to ``<savePath>/log/crashes/`` where
``<savePath>`` is the data storage location chosen on the
:ref:`first run <first-run-data-path>` and shown on the
:ref:`Application Configuration dialog <application-config>`. Each report is a
small file named ``crash-<UTC timestamp>-<build SHA>.log`` (Linux,
macOS) or a pair ``crash-<UTC timestamp>-<build SHA>.dmp`` plus
``.log`` (Windows). Reports survive program restarts; the directory
is never cleaned automatically.

What a Report Contains
----------------------

The text portion of a crash report contains:

* The Blackchirp version and the build identifier (the git commit SHA
  the binary was compiled from).
* The Qt runtime version.
* The UTC timestamp of the crash.
* The signal or exception code that caused the termination, and the
  faulting memory address.
* The process ID and the active experiment number, if any.
* A stack trace listing the function call chain at the moment of the
  crash, in ``<module>(+0xoffset) [0xPC]`` form.

The reports do **not** contain personal information beyond the data
storage path and the active experiment number. They do not include
acquired FID data, hardware configurations, or any data acquired
during experiments. Open the file in any text editor to review it
before sending.

The Windows minidump (``.dmp``) is a binary file that lets a developer
load the full process state in a Windows debugger; its size is
typically tens of MB.

What To Do When a Crash Report Appears
--------------------------------------

When Blackchirp starts and finds one or more crash reports in
``<savePath>/log/crashes/`` that were not present at the previous
clean shutdown, it shows a notification dialog after the main window
appears. The dialog offers three actions:

* **Open Folder** opens the crash directory in the system file manager.
* **View Most Recent** opens the newest report in the default text
  editor.
* **Dismiss** closes the dialog. Reports remain in the directory.

To send a report to the developer, attach the file to an issue on the
`Blackchirp issue tracker`_ or to an email. Include a short
description of what you were doing when the crash occurred. The
developer can resolve the addresses in the stack trace to source-code
locations using the build identifier embedded in the report header.

If the same crash recurs reproducibly, attach the most recent report
and a description of the steps that trigger it.

.. _Blackchirp issue tracker: https://github.com/kncrabtree/blackchirp/issues
