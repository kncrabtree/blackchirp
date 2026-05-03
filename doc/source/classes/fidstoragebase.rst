.. index::
   single: FidStorageBase
   single: FID; storage
   single: FidSingleStorage
   single: FidMultiStorage
   single: FidPeakUpStorage
   single: FidStorageBase; cache
   single: FidStorageBase; processing settings
   single: BC::Key::FidStorage

FidStorageBase
==============

``FidStorageBase`` extends :doc:`datastoragebase` to manage
free-induction decay (FID) waveforms for CP-FTMW acquisitions.  It
holds a fixed number of FID records per segment (``d_numRecords``) and
implements the :cpp:class:`DataStorageBase` lifecycle methods.  The
constructor reads any existing ``fid/fidparams.csv`` into an internal
template list so that on-disk data can be accessed without re-reading
the full experiment.

Three concrete subclasses cover the standard acquisition modes:

- ``FidSingleStorage`` — a single persistent FID segment with optional
  backup snapshots (single-segment and target-shots modes).
- ``FidMultiStorage`` — multiple FID segments for LO-scan and DR-scan
  acquisitions.
- ``FidPeakUpStorage`` — a transient rolling-average store for
  peak-up mode (``d_number`` is ``-1``; no disk I/O).

These subclasses live in their own headers and are not documented on
this page; they each implement the pure-virtual ``loadDifferentialFidList``
and ``getCurrentIndex`` methods.

FID I/O surface
---------------

``addFids()`` co-averages an incoming :cpp:class:`Fid` list into the
current segment, applying an optional time-domain sample shift.
``setFidsData()`` replaces the current list unconditionally (used for
background-subtraction updates where the new list is not simply additive).
``getCurrentFidList()`` returns a mutex-protected copy of the in-progress
accumulation.  ``loadFidList()`` reads segment *i* from the in-memory
cache if available, or from ``fid/<i>.csv`` on disk; it is safe to call
from any thread.  The pure-virtual ``loadDifferentialFidList()``
returns the background-subtracted list and is implemented by each
concrete subclass.

Cache
-----

Loaded ``FidList`` objects are stored in ``d_cache``, keyed by segment
index, up to ``d_maxCacheSize`` bytes (approximately 200 MB by default).
The insertion order of segment indices is tracked in ``d_cacheKeys``; when
the cache is full, the oldest entry is evicted before inserting a new one.
Cache updates are serialized by an internal ``pu_baseMutex`` (separate
from ``pu_mutex`` in the base class, which guards the current FID list).

Backup interface
----------------

The virtual ``backup()`` method saves a point-in-time snapshot of the
current FID list.  The default implementation is a no-op; ``FidSingleStorage``
overrides it.  ``numBackups()`` returns the number of snapshots available.

Processing settings
-------------------

``writeProcessingSettings()`` serializes a
:cpp:class:`FtWorker`\::\ ``FidProcessingSettings`` struct to
``fid/processing.csv`` using the keys in ``BC::Key::FidStorage``:

+----------------------+-------------------------------------------+
| Key                  | Meaning                                   |
+======================+===========================================+
| ``fidStart``         | Processing window start (μs)              |
+----------------------+-------------------------------------------+
| ``fidEnd``           | Processing window end (μs)                |
+----------------------+-------------------------------------------+
| ``fidExp``           | Exponential apodization time constant (μs)|
+----------------------+-------------------------------------------+
| ``zpf``              | Zero-padding multiplier                   |
+----------------------+-------------------------------------------+
| ``rdc``              | DC removal flag                           |
+----------------------+-------------------------------------------+
| ``units``            | Output magnitude units                    |
+----------------------+-------------------------------------------+
| ``autoscaleIgnore``  | LO exclusion half-width (MHz)             |
+----------------------+-------------------------------------------+
| ``winf``             | Apodization window function               |
+----------------------+-------------------------------------------+

``readProcessingSettings()`` deserializes the same file back into a
``FidProcessingSettings`` struct.  ``getLORange()`` returns the
``[min, max]`` probe-frequency range across all stored FID segments,
which is used by LO-scan deconvolution in :cpp:class:`FtWorker`.

The FID data format and the broader FTMW acquisition workflow are
described in :doc:`/user_guide/data_storage`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: FidStorageBase
   :members:
   :protected-members:
   :undoc-members:
