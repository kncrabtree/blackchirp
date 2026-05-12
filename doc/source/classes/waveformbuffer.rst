.. index::
   single: WaveformBuffer
   single: WaveformEntry
   single: ring buffer; SPSC waveform
   single: digitizer; waveform data transfer
   single: FtmwConfig; WaveformBuffer pointer

WaveformBuffer and WaveformEntry
=================================

``WaveformBuffer`` is a bounded, single-producer/single-consumer (SPSC) ring
buffer that transfers waveform data from the digitizer hardware thread to the
acquisition manager thread. It provides bounded memory usage, drop-newest
overflow, and low-latency consumer notification via ``QSemaphore``.

``WaveformBuffer`` does not inherit ``QObject``. It is created and owned by the
``FtmwDigitizer`` hardware object; :cpp:class:`FtmwConfig` holds a non-owning
pointer to it (set via ``setWaveformBuffer()`` during acquisition setup and
retrieved by ``AcquisitionManager`` via ``waveformBuffer()``). The digitizer
data-flow design, including the rationale for the ring-buffer approach, is
described in :doc:`/developer_guide/ftmw_acquisition`.

Thread-safety discipline
------------------------

Exactly **one producer thread** (the digitizer hardware thread) and **one
consumer thread** (the acquisition manager thread) may access the buffer at any
time. Multi-producer or multi-consumer usage is undefined behavior. The producer
calls ``write()``, ``writeFlushMarker()``, and ``reset()``; the consumer calls
``read()``, ``drainAll()``, ``available()``, and ``waitForData()``.

Overflow policy
---------------

The producer never blocks. When the buffer is full, the incoming entry is
silently discarded (drop-newest policy) and an internal counter is incremented.
Call ``droppedCount()`` to retrieve the total number of dropped entries since
the last ``reset()``. The drop-newest approach is race-free because the
producer never modifies the read index.

When the producer detects a full buffer via ``isFull()``, it may optionally
begin accumulating incoming waveforms locally (pre-accumulation). Once a slot
becomes available, the accumulated result is written as a single entry with
``preAccumulated = true``, allowing the consumer to bypass the normal parse
pipeline for that entry.

Consumer notification
---------------------

The producer releases a ``QSemaphore`` after each successful write. The
consumer calls ``waitForData(timeoutMs)`` to block until at least one entry is
available, then drains the buffer with ``drainAll()`` or reads entries one at a
time with ``read()``. The timeout allows the consumer thread to perform periodic
housekeeping (autosave checks, abort polling) even when no waveform data
arrives.

WaveformEntry
-------------

Each ring slot holds a ``WaveformEntry`` struct with four fields:

- **data** — raw waveform bytes (empty for flush markers). When
  ``preAccumulated`` is ``true``, the bytes represent ``qint64`` values rather
  than raw digitizer samples.
- **shotCount** — number of shots or FIDs represented by this entry. For a
  single raw waveform this equals the digitizer's shot-increment value; for a
  pre-accumulated entry it equals the sum of all accumulated shot increments.
- **preAccumulated** — when ``true``, the consumer routes the entry through
  ``FtmwConfig::addPreAccumulatedFids()`` instead of the normal parse pipeline.
- **flushMarker** — when ``true``, the entry is a segment-boundary sentinel
  (``data`` is empty and ``shotCount`` is zero). The consumer drains all
  preceding entries for the current segment before advancing to the next one.

.. highlight:: cpp

API Reference
-------------

.. doxygenstruct:: WaveformEntry
   :members:
   :undoc-members:

.. doxygenclass:: WaveformBuffer
   :members:
   :undoc-members:
