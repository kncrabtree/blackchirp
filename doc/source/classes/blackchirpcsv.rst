.. index::
   single: BlackchirpCSV
   single: CSV; experiment storage
   single: BC::CSV; filename constants
   single: experiment; directory layout
   single: version.csv
   single: header.csv

BlackchirpCSV
=============

``BlackchirpCSV`` is the workhorse persistence class for experiment CSV I/O.
It owns the canonical experiment-directory layout and provides the static
write helpers, directory helpers, and format utilities that every storage
class in the persistence subsystem calls directly. The companion
:doc:`datastoragebase` class holds a ``BlackchirpCSV`` instance and delegates
all file access through it.

All CSV files produced by Blackchirp use the semicolon delimiter
(``BC::CSV::del``); the pipe character (``BC::CSV::altDel``) is reserved for
``QStringList`` values embedded within a single cell. The full set of
canonical filenames for an experiment directory is enumerated in the
``BC::CSV`` namespace: experiment-root files (``versionFile``,
``validationFile``, ``objectivesFile``, ``hwFile``, ``headerFile``,
``chirpFile``, ``markersFile``, ``clockFile``, ``auxFile``), FID artifacts
(``fidparams``, ``fidDir``), and LIF artifacts (``lifparams``, ``lifDir``).
Version-key constants (``majver``, ``minver``, ``patchver``, ``relver``,
``buildver``) are written to and read from ``version.csv``. The six
header-column name constants (``ok``, ``ak``, ``ai``, ``vk``, ``vv``,
``vu``) define the layout of ``header.csv``.

The class has two construction paths. The default constructor creates an
instance suitable for static-method use and for writing new experiments; the
delimiter defaults to ``BC::CSV::del``. The ``(num, path)`` constructor reads
``version.csv`` from the experiment directory identified by ``num`` and
``path``, populating an internal configuration map that the instance-level
``readLine`` and ``readFidLine`` methods then use to tokenize subsequent reads
with the correct delimiter. The version accessors (``majorVersion``,
``minorVersion``, ``patchVersion``, ``releaseVersion``, ``buildVersion``)
expose the version metadata loaded by this constructor.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: BlackchirpCSV
   :members:
   :undoc-members:
