.. index::
   single: AcquisitionManager
   single: acquisition; state machine
   single: FtmwProcessingResult

AcquisitionManager
==================

``AcquisitionManager`` is the object that drives a single :cpp:class:`Experiment`
from the moment hardware initialization completes through the end of the
acquisition loop. It owns the in-progress experiment shared pointer, accumulates
FID waveforms and auxiliary sensor readings into the per-experiment storage tree,
and emits the lifecycle signals that coordinate the main window, the hardware
layer, and the batch layer. All acquisitions in Blackchirp — single-experiment
runs as well as multi-experiment batch sequences — pass through
``AcquisitionManager``. The user-facing acquisition workflow is described in
:doc:`/user_guide/experiment_setup`.

The manager lives on a dedicated QThread (``"AcquisitionManagerThread"``) created
in ``MainWindow``. Its slots and signal emissions execute on that thread.
The GUI thread uses ``QMetaObject::invokeMethod`` to cross the thread boundary
when calling :cpp:func:`AcquisitionManager::beginExperiment`. Waveform
processing is further dispatched to the Qt thread pool via ``QtConcurrent::run``;
the processing result is handed back to the AM thread through a
``QFutureWatcher<FtmwProcessingResult>`` so that state mutations remain
thread-confined.

The immediate collaborators are:

- :doc:`experiment` — the in-progress experiment shared pointer; the AM writes
  FID data and aux readings into its storage objects and queries it for
  completion status.
- :doc:`hardwaremanager` (cross-thread) — receives ``beginAcquisition``
  and ``endAcquisition`` to start and stop hardware triggers; provides aux data
  via its ``auxData`` signal; applies clock settings in response to
  ``newClockSettings``.
- :doc:`fidstoragebase` — the FID storage layer embedded in the experiment;
  waveform data accumulate here.
- :doc:`auxdatastorage` — the auxiliary sensor storage embedded in the
  experiment; each timed aux-data point is committed here.
- :doc:`batchmanager` (GUI thread, queued connection) — receives
  ``experimentComplete`` after every acquisition loop ends and decides whether
  to start the next experiment.
- :doc:`lifstorage` (LIF mode only) — the LIF storage object embedded in the
  experiment when a LIF scan is configured.

.. _acquisitionmanager-state-machine:

State machine
-------------

``AcquisitionManager`` tracks its phase through the ``AcquisitionState`` enum.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - State
     - Meaning
   * - ``Idle``
     - No experiment is running. The manager is ready for a new
       ``beginExperiment`` call.
   * - ``Acquiring``
     - An experiment is active. Waveform data, aux readings, and validation
       checks are processed as they arrive.
   * - ``Paused``
     - Acquisition is suspended. Incoming data are dropped until ``resume()``
       is called.

The typical lifecycle of a single experiment is:

1. ``MainWindow::experimentInitialized`` receives ``HardwareManager::experimentInitialized``,
   calls ``Experiment::initialize()``, and invokes ``beginExperiment(exp)`` on
   the AM thread.
2. ``beginExperiment`` transitions to ``Acquiring``, emits ``newClockSettings``
   (FTMW experiments), emits ``beginAcquisition``, starts the aux-data interval
   timer, and starts the FTMW drain timer for buffered-waveform modes.
3. During the loop, the aux-data timer fires periodically; each firing calls
   ``auxDataTick``, which collects FTMW shot counts (and optional phase/chirp
   metrics), calls ``processAuxData``, and emits ``auxDataSignal`` so
   ``HardwareManager`` can read sensor values.
4. ``processAuxData`` stores the readings in ``AuxDataStorage`` and re-emits
   them as ``auxData(map, timestamp)`` for plot widgets.
5. ``processValidationData`` checks each incoming sensor value against its
   configured limit; a violation triggers ``abort``.
6. FTMW waveforms arrive through a ``WaveformBuffer`` polled by the drain
   timer. Batches of entries are dispatched to a worker via
   ``QtConcurrent::run``; the ``QFutureWatcher<FtmwProcessingResult>``
   delivers the result back to the AM thread. A processing failure calls
   ``abort``.
7. ``checkComplete`` is called after each processing batch. If
   ``Experiment::canBackup`` is true it launches a concurrent backup via
   ``QtConcurrent::run`` (the ``QFutureWatcher<void>`` emits ``backupComplete``
   when done). If ``Experiment::isComplete`` is true it calls
   ``finishAcquisition``.
8. ``pause`` and ``resume`` toggle between ``Acquiring`` and ``Paused`` without
   stopping hardware. While ``Paused`` the drain timer still fires but skips
   processing.
9. ``abort`` marks the experiment as aborted and calls ``finishAcquisition``.
10. ``finishAcquisition`` stops the drain timer, signals the worker to exit,
    waits for the worker to finish, emits ``endAcquisition``, transitions to
    ``Idle``, calls ``Experiment::finalSave``, and emits ``experimentComplete``.

.. note::

   ``AcquisitionManager::experimentComplete`` is a **signal**; the identically
   named ``BatchManager::experimentComplete`` is a **slot**. The two are
   connected via a queued connection that crosses the AM thread into the GUI
   thread, and this signal-to-slot handoff is what advances a batch from one
   experiment to the next.

Backups
-------

The acquisition loop launches periodic backups of the in-progress experiment
on a worker thread via ``QtConcurrent::run``. ``QFutureWatcher`` raises the
``backupComplete`` signal on the AM thread when each backup finishes, and
``FtmwViewWidget`` connects to it to refresh its list of available backups.

.. highlight:: cpp

API Reference
-------------

.. doxygenstruct:: FtmwProcessingResult
   :members:
   :undoc-members:

.. doxygenenum:: AcquisitionManager::AcquisitionState

.. doxygenclass:: AcquisitionManager
   :members:
   :protected-members:
   :undoc-members:
