.. index::
   single: BatchManager
   single: batch acquisition; lifecycle
   single: BatchSingle
   single: BatchSequence

BatchManager
============

``BatchManager`` is the abstract base class that controls the lifecycle of an
acquisition run from the main window's perspective. Every acquisition in
Blackchirp — whether it consists of a single experiment or a timed sequence of
experiments — is represented by a concrete ``BatchManager`` subclass and
submitted to ``MainWindow::startBatch``. The main window connects the batch's
signals to :doc:`acquisitionmanager` and the UI, then triggers the first
experiment via :doc:`/classes/hardwaremanager`. The user-facing workflow for
starting experiments and batch sequences is described in
:doc:`/user_guide/experiment_setup`.

``BatchManager`` lives on the main (GUI) thread. Its signals reach
``AcquisitionManager`` (which lives on a dedicated background thread) via queued
connections. Slots on ``BatchManager`` are called from the GUI thread by the
main window or by ``AcquisitionManager``'s ``experimentComplete`` signal.

The immediate collaborators are:

- :doc:`experiment` — each experiment in the batch is a shared pointer that
  ``currentExperiment()`` returns to the caller.
- :doc:`acquisitionmanager` — drives the hardware acquisition loop for each
  experiment; its ``experimentComplete`` signal is connected to
  ``BatchManager::experimentComplete`` for the duration of the batch.

The two built-in concrete subclasses are ``BatchSingle`` (wraps a single
experiment; ``BatchType::SingleExperiment``) and ``BatchSequence`` (repeats an
experiment template on a configurable interval; ``BatchType::Sequence``). The
batch type in use is identified by the ``BatchType`` enum stored in ``d_type``.

.. _batchmanager-state-machine:

State machine
-------------

The iterate-over-experiments loop proceeds as follows:

1. ``MainWindow::startBatch`` connects signals, then calls
   ``HardwareManager::initializeExperiment(currentExperiment())`` to start the
   first experiment. ``BatchManager`` emits ``beginExperiment()`` for subsequent
   experiments (see step 5).
2. ``HardwareManager`` configures hardware and emits ``experimentInitialized``.
   The main window calls ``Experiment::initialize()`` on success, then invokes
   ``AcquisitionManager::beginExperiment(exp)`` on the AM thread.
3. ``AcquisitionManager`` runs the acquisition loop and, when it ends (normally
   or via abort), emits its ``experimentComplete`` signal.
4. That signal is connected to ``BatchManager::experimentComplete`` (the slot).
   The slot:

   a. Logs the experiment result via ``Experiment::d_endLogMessage``.
   b. Calls ``processExperiment()`` if hardware and software initialization
      both succeeded.
   c. Evaluates ``isComplete()`` and whether the experiment was aborted.

5. If initialization succeeded, the experiment was not aborted, and
   ``isComplete()`` returns ``false``, the slot calls ``beginNextExperiment()``,
   which by default emits ``beginExperiment()`` immediately. The main window
   responds by invoking ``HardwareManager::initializeExperiment(currentExperiment())``
   for the next experiment (go to step 2).
6. If the batch is complete, or if the experiment was aborted, or if
   initialization failed:

   - ``abort()`` is called when the experiment was aborted or init failed.
   - ``writeReport()`` is called unconditionally.
   - ``batchComplete(aborted)`` is emitted, where ``aborted`` reflects
     ``Experiment::isAborted()``.

The connection between ``AcquisitionManager::experimentComplete`` (signal) and
``BatchManager::experimentComplete`` (slot) is the central handoff between the
two managers. The signal fires on the AM thread and is delivered to the BM slot
on the GUI thread via a queued connection.

Subclassing guide
-----------------

A new batch type inherits ``BatchManager`` and implements the following pure-virtual
methods:

``currentExperiment()``
    Return the experiment that is either in progress or about to be acquired.
    Called by the main window each time ``beginExperiment()`` is emitted, and by
    ``experimentComplete()`` to inspect the result.  Must never return a null
    pointer while the batch is active.

``isComplete()``
    Return ``true`` when no further experiments remain. Evaluated by
    ``experimentComplete()`` after ``processExperiment()`` returns.

``abort()``
    Mark the batch as complete so ``isComplete()`` returns ``true`` and release
    any pending timers or queued work. Called when the user clicks the abort
    button or when hardware initialization fails.

``processExperiment()``
    Perform any post-acquisition analysis or bookkeeping for the most recently
    completed experiment. Called only when initialization succeeded. May be a
    no-op.

``writeReport()``
    Write a summary report for the batch (file, log entry, or nothing). Called
    unconditionally after the last experiment ends, before ``batchComplete`` is
    emitted.

The optional override ``beginNextExperiment()`` is available when the default
behavior — emit ``beginExperiment()`` immediately — is not appropriate. For
example, ``BatchSequence`` overrides this method to start a ``QTimer`` and emit
``beginExperiment()`` only after the configured inter-experiment interval has
elapsed.

To register a new batch type with the main window, add a ``BatchType`` enumerator,
construct the concrete subclass in the appropriate ``MainWindow`` action handler,
and pass it to ``MainWindow::startBatch``.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: BatchManager
   :members:
   :protected-members:
   :undoc-members:
