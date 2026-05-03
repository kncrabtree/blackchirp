.. index::
   single: DataStorageBase
   single: storage; experiment data tree
   single: storage; lifecycle interface
   single: storage; advance/save/start/finish

DataStorageBase
===============

``DataStorageBase`` is the abstract root of Blackchirp's experiment
data-storage tree.  Every object that persists experiment data to disk
inherits from it.  An instance is identified by a non-negative
experiment number (``d_number``) and an optional base path
(``d_path``); passing ``-1`` for the number creates a transient
(peak-up or dummy) instance for which all disk I/O is silently skipped.

The four pure-virtual methods define the acquisition lifecycle contract:

- ``start()`` — called when acquisition begins; subclasses arm
  internal state.
- ``advance()`` — called at each segment boundary; subclasses flush the
  current in-progress accumulation and prepare for the next segment.
- ``save()`` — called to persist the current in-memory state.
- ``finish()`` — called when acquisition ends; subclasses clear their
  acquiring flag.

The protected ``pu_mutex`` guards mutable state accessed from the
acquisition and UI threads concurrently.  The protected ``pu_csv``
owns a :cpp:class:`BlackchirpCSV` helper scoped to the experiment
directory.  The ``writeMetadata`` and ``readMetadata`` helpers let
subclasses persist a key-value map to a named CSV file within (an
optional subdirectory of) the experiment directory; both delegate to
``pu_csv`` for the actual I/O.

The direct subclasses are:

- :doc:`fidstoragebase` (``FidStorageBase``) — FTMW FID waveforms and
  FT processing settings.
- :doc:`lifstorage` (``LifStorage``) — LIF trace data on a
  two-dimensional delay/laser grid.
- ``OverlayStorage`` — plot overlay annotations (see
  :doc:`overlaybase`).

:cpp:class:`AuxDataStorage`, which collects time-series auxiliary
hardware readings, has a similar role but does not inherit from
``DataStorageBase``; it is owned and driven directly by
:cpp:class:`Experiment`.

The on-disk layout is described in :doc:`/user_guide/data_storage`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: DataStorageBase
   :members:
   :protected-members:
   :undoc-members:
