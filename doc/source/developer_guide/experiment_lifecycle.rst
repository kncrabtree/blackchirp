.. index::
   single: experiment lifecycle
   single: BatchManager; lifecycle
   single: AcquisitionManager; lifecycle
   single: MainWindow; startBatch
   single: HardwareManager; initializeExperiment
   single: experimentInitialized
   single: experimentComplete; signal vs slot
   single: clock settings; runtime
   single: abort; experiment
   single: validation; abort path
   single: ExperimentObjective; hwReady

Experiment Lifecycle
====================

This page tells the cross-manager story of one experiment. The user
clicks *Start Experiment* (or *Quick Experiment*, or *Start Sequence*);
:cpp:class:`MainWindow` constructs a :cpp:class:`BatchManager`, hands it
to the hardware layer, and waits for the round-trip back from
:cpp:class:`AcquisitionManager` to know whether to start another
experiment or close out the batch. Most of that round-trip is spent
inside the two managers, but neither manager drives it alone — the
handoff between them is what this page covers.

The internal state machines of :cpp:class:`AcquisitionManager` and
:cpp:class:`BatchManager` are documented on their own API pages
(:ref:`acquisitionmanager-state-machine` and
:ref:`batchmanager-state-machine`). The waveform-processing pipeline
that runs *inside* the AM's acquisition loop is the topic of
:doc:`/developer_guide/ftmw_acquisition`; the LIF-side acquisition
mechanics are on :doc:`/developer_guide/lif_acquisition`. This page
treats those pipelines as black boxes and concentrates on the signals
that cross between :cpp:class:`MainWindow`,
:cpp:class:`HardwareManager`, :cpp:class:`AcquisitionManager`,
:cpp:class:`BatchManager`, :cpp:class:`Experiment`, and the live
:cpp:class:`HardwareObject` set.

The thread layout is the one introduced in
:doc:`/developer_guide/architecture`: :cpp:class:`MainWindow` and
:cpp:class:`BatchManager` live on the GUI thread,
:cpp:class:`HardwareManager` runs on its own
``HardwareManagerThread``, :cpp:class:`AcquisitionManager` runs on
``AcquisitionManagerThread``, and threaded
:cpp:class:`HardwareObject` instances each run on
``"<hwKey>Thread"``. Every cross-thread call described below uses a
queued connection or :cpp:func:`QMetaObject::invokeMethod`; this page
calls out the few that are blocking-queued because they need a return
value.

Starting a batch
----------------

Three menu actions land in three :cpp:class:`MainWindow` slots that
each construct a concrete :cpp:class:`BatchManager` subclass and pass
it to :cpp:func:`MainWindow::startBatch`:

- *Start Experiment* → :cpp:func:`MainWindow::startExperiment` opens
  :cpp:class:`ExperimentSetupDialog` (the full configuration wizard),
  then constructs :cpp:class:`BatchSingle` (``BatchType::SingleExperiment``).
- *Quick Experiment* → :cpp:func:`MainWindow::quickStart` opens
  :cpp:class:`QuickExptDialog` to repeat or reconfigure a previous
  experiment, then constructs :cpp:class:`BatchSingle`.
- *Start Sequence* → :cpp:func:`MainWindow::startSequence` opens
  :cpp:class:`BatchSequenceDialog` (count + interval), optionally
  re-runs the wizard or the quick-experiment dialog, then constructs
  :cpp:class:`BatchSequence` (``BatchType::Sequence``). The sequence
  re-uses one experiment template, cloning a fresh
  :cpp:class:`Experiment` from it for each iteration.

:cpp:func:`MainWindow::startBatch` is the wiring hub for everything
that happens during the batch:

.. code-block:: cpp

   connect(bm,&BatchManager::beginExperiment,[this,bm](){
       QMetaObject::invokeMethod(p_hwm,[this,bm](){
           p_hwm->initializeExperiment(bm->currentExperiment());
       });
   });
   connect(p_am,&AcquisitionManager::experimentComplete,
           bm,&BatchManager::experimentComplete);
   connect(bm,&BatchManager::batchComplete,this,&MainWindow::batchComplete);
   connect(p_hwm,&HardwareManager::abortAcquisition,
           p_am,&AcquisitionManager::abort,Qt::UniqueConnection);

The signals that span the per-batch lifecycle are installed here and
torn down in :cpp:func:`MainWindow::batchComplete` after the batch
ends. The static connections that survive across batches —
``experimentInitialized``, ``allClocksReady``, ``newClockSettings``,
``beginAcquisition``, ``endAcquisition``, ``auxData``,
``validationData``, and the LIF chain — were installed once during
:cpp:func:`MainWindow::MainWindow` and remain in place.

The first experiment in a batch is launched directly from the bottom
of :cpp:func:`MainWindow::startBatch`:

.. code-block:: cpp

   QMetaObject::invokeMethod(p_hwm,[this](){
       p_hwm->initializeExperiment(p_batchManager->currentExperiment());
   });

Subsequent experiments in a :cpp:class:`BatchSequence` are launched by
the connected ``BatchManager::beginExperiment`` signal — the
``connect`` shown above hands the next experiment to the hardware layer
without :cpp:class:`MainWindow` having to know that the batch advanced.

Hardware setup
--------------

:cpp:func:`HardwareManager::initializeExperiment` runs on the
``HardwareManagerThread`` and is the gateway every experiment passes
through before any data are acquired. Its three steps run in order:

1. **Configure clocks.**
   :cpp:func:`ClockManager::prepareForExperiment` reads the clock map
   from ``exp.ftmwConfig()->d_rfConfig.getClocks()``, calls
   :cpp:func:`ClockManager::configureClocks` to resolve each
   :cpp:enum:`RfConfig::ClockType` role to a :cpp:class:`Clock` and
   apply its multiplication factor, then writes the achieved
   frequencies back into the experiment with
   :cpp:func:`RfConfig::setCurrentClocks`. Non-FTMW experiments skip
   this step. Failure short-circuits the rest of the routine.

2. **Prepare every hardware object.** The manager iterates
   ``d_hardwareMap`` and dispatches
   :cpp:func:`HardwareObject::hwPrepareForExperiment` to each device.
   For threaded drivers the call goes across a
   :cpp:enumerator:`Qt::BlockingQueuedConnection` so the success
   value is delivered synchronously. The base wrapper does two
   things every driver gets for free: if the device is currently
   disconnected it tries :cpp:func:`HardwareObject::testConnection`
   once, and if that test fails on a critical device
   (``d_critical == true``, the default) it sets
   ``exp.d_errorString`` and returns ``false``. A non-critical
   disconnect is treated as success and the per-experiment
   :cpp:func:`HardwareObject::prepareForExperiment` override is
   bypassed for that device. The first failure breaks the loop and
   sets ``exp.d_hardwareSuccess = false``.

3. **Sanity-check LIF.** When ``exp.lifEnabled()`` is true the
   manager confirms that an active :cpp:class:`LifLaser` exists in
   :cpp:class:`RuntimeHardwareConfig`. A missing laser emits
   :cpp:func:`HardwareManager::lifSettingsComplete` with
   ``success = false`` and clears the experiment's hardware-success
   flag.

The routine ends with
:cpp:func:`HardwareManager::experimentInitialized` carrying the
populated :cpp:class:`Experiment`. Even on failure the signal is
emitted — the hardware-success flag tells the GUI-side handler what
happened.

For the per-class API of :cpp:func:`HardwareObject::hwPrepareForExperiment`,
:cpp:func:`HardwareObject::beginAcquisition`, and
:cpp:func:`HardwareObject::endAcquisition`, see
:doc:`/classes/hardwareobject`. The clock-routing model is on
:doc:`/classes/clockmanager`.

Registering and handing off the experiment
------------------------------------------

:cpp:func:`HardwareManager::experimentInitialized` is delivered to
:cpp:func:`MainWindow::experimentInitialized` on the GUI thread (the
connection is queued because the signal is emitted on the HM thread).
The slot does the GUI-thread side of the handoff:

- If ``exp.d_hardwareSuccess`` is false, the slot calls
  :cpp:func:`BatchManager::experimentComplete` directly so the batch
  is told the experiment is over before it ever started, restores
  the UI to ``Idle``, and returns.
- Otherwise it calls :cpp:func:`Experiment::initialize` synchronously
  on the GUI thread. That call does the *bookkeeping* a brand-new
  experiment needs: it increments the persisted experiment counter
  in :cpp:class:`SettingsStorage`, creates the on-disk directory via
  :cpp:func:`BlackchirpCSV::createExptDir`, walks the
  ``d_objectives`` set so each :cpp:class:`ExperimentObjective`
  initializes itself, and writes the initial CSVs (``version.csv``,
  ``header.csv``, ``objectives.csv``, ``hardware.csv``,
  ``chirps.csv``, ``clocks.csv``). Peak Up without LIF is the
  exception — the experiment is marked ``d_isDummy``, given
  experiment number ``-1``, and no directory or files are written.
- The slot prepares per-experiment GUI state (progress bars, the
  FTMW view widget, the aux-data widget, the LIF widget when LIF is
  enabled), opens the log entry for the experiment via
  :cpp:func:`LogHandler::beginExperimentLog` (or a one-line highlight
  for dummy experiments), and calls :cpp:func:`MainWindow::configureUi`
  with ``Acquiring`` so the toolbar buttons reflect the running
  state.
- Finally it crosses back into the AM thread:

  .. code-block:: cpp

     QMetaObject::invokeMethod(p_am,[this,exp](){
         p_am->beginExperiment(exp);
     });

  From this point :cpp:class:`AcquisitionManager` owns the experiment
  shared pointer for the duration of the acquisition loop.

Acquisition steady state
------------------------

:cpp:func:`AcquisitionManager::beginExperiment` is the AM-thread entry
point for the loop. It runs in this order:

1. Stores the experiment, transitions to ``Acquiring``, and emits
   ``statusMessage("Acquiring")`` for the status bar.
2. When ``exp->d_timeDataInterval > 0``, registers the per-timestamp
   FTMW aux keys (``Ftmw/Shots`` always; ``Ftmw/ChirpRMS`` when
   chirp scoring is on; ``Ftmw/ChirpPhaseScore`` and
   ``Ftmw/ChirpShift`` when phase correction is on), calls
   :cpp:func:`AcquisitionManager::auxDataTick` once to seed the
   first point, and starts a Qt timer using ``startTimer`` for the
   periodic aux refresh.
3. For FTMW experiments, emits
   :cpp:func:`AcquisitionManager::newClockSettings` carrying
   ``ftmwConfig->d_rfConfig.getClocks()``. The signal is consumed
   on the GUI thread by :cpp:func:`MainWindow::clockPrompt` (see the
   *Clock settings round-trip* note below); the GUI ultimately calls
   :cpp:func:`HardwareManager::setClocks`, which gates the
   :cpp:class:`FtmwDigitizer` while the clock frequencies change, walks
   :cpp:class:`ClockManager` to apply each value, ungates the scope,
   and emits :cpp:func:`HardwareManager::allClocksReady`. That signal
   is wired statically to
   :cpp:func:`AcquisitionManager::clockSettingsComplete`, which
   stores the achieved frequencies in
   ``ftmwConfig->d_rfConfig.setCurrentClocks`` and calls
   :cpp:func:`FtmwConfig::hwReady`. Calling ``hwReady`` clears
   ``d_processingPaused`` so waveform processing can begin.
4. Emits :cpp:func:`AcquisitionManager::beginAcquisition`. The signal
   is wired to :cpp:func:`HardwareManager::beginAcquisition`, which
   re-emits the broadcast onto every :cpp:class:`HardwareObject`'s
   ``beginAcquisition`` slot — devices start triggering, the
   digitizer's :cpp:class:`WaveformBuffer` becomes the data conduit.
5. For FTMW experiments with a live waveform buffer, allocates the
   processing :cpp:class:`QFutureWatcher`, starts the 20 ms drain
   timer, and connects its timeout to
   :cpp:func:`AcquisitionManager::drainFtmwBuffer`. The waveform
   pipeline that runs inside that timer is documented separately in
   :doc:`/developer_guide/ftmw_acquisition` and on
   :ref:`acquisitionmanager-state-machine`; this page treats the
   inner loop as a black box.
6. For LIF experiments, emits
   :cpp:func:`AcquisitionManager::nextLifPoint` to start the LIF
   parallel path described in the next section.

The aux-data tick keeps running on its own timer for the rest of the
acquisition. Each fire of :cpp:func:`AcquisitionManager::auxDataTick`
collects the FTMW shot count (and chirp metrics where enabled),
writes it through :cpp:func:`AcquisitionManager::processAuxData`, and
emits :cpp:func:`AcquisitionManager::auxDataSignal`. The signal is
wired to :cpp:func:`HardwareManager::getAuxData`, which fans the
``bcReadAuxData`` call out to every hardware object, prefixes each
returned key with the source's ``hwKey``, and re-emits the aggregate
as :cpp:func:`HardwareManager::auxData` and
:cpp:func:`HardwareManager::validationData`. The two arrive back on
the AM as :cpp:func:`AcquisitionManager::processAuxData` (which
records to :cpp:class:`AuxDataStorage` and re-emits to the plot
widgets) and :cpp:func:`AcquisitionManager::processValidationData`
(which checks each value against the experiment's
:cpp:class:`ExperimentValidator` set; an out-of-range reading calls
:cpp:func:`AcquisitionManager::abort`). The aux and validation paths
share the per-device ``readAuxData`` source — a device declares its
validation keys via :cpp:func:`HardwareObject::validationKeys` and
the same readings flow into both pipelines.

After every FTMW processing batch returns,
:cpp:func:`AcquisitionManager::checkComplete` checks
:cpp:func:`Experiment::canBackup`. When true, a backup is dispatched
through ``QtConcurrent::run`` and a :cpp:class:`QFutureWatcher`
emits :cpp:func:`AcquisitionManager::backupComplete` on the AM
thread when the worker finishes; :cpp:class:`FtmwViewWidget`
listens to that signal to refresh its list of available backups.
``checkComplete`` also calls :cpp:func:`Experiment::isComplete`; a
true return invokes :cpp:func:`AcquisitionManager::finishAcquisition`
to end the loop normally.

.. note::

   *Clock settings round-trip.* The chain
   ``newClockSettings → clockPrompt → setClocks → allClocksReady →
   clockSettingsComplete`` is the way Blackchirp resynchronizes when
   the experiment crosses an FTMW segment boundary that needs new LO
   frequencies. :cpp:func:`MainWindow::clockPrompt` sits between AM
   and HM because clock outputs marked manual-tune in
   :cpp:class:`SettingsStorage` need a user prompt before the next
   segment is allowed to begin; non-manual clocks pass through
   silently. The round-trip is also taken once at the start of every
   FTMW experiment — that is where ``hwReady`` first clears
   ``d_processingPaused`` so the drain timer's worker may run.

LIF parallel path
-----------------

When ``exp.lifEnabled()`` is true,
:cpp:func:`AcquisitionManager::beginExperiment` ends with one extra
emission: :cpp:func:`AcquisitionManager::nextLifPoint`, carrying the
first scan point's delay and laser position. The LIF parallel path
that follows — laser tuning, digitizer arming, shot accumulation,
point advancement — is the topic of
:doc:`/developer_guide/lif_acquisition`. The point of contact for
this page is the two queued connections that make the parallel path
visible to the rest of the system:

- :cpp:func:`HardwareManager::lifSettingsComplete` →
  :cpp:func:`AcquisitionManager::lifHardwareReady`. The HM emits
  this slot with ``success`` when the LIF laser and digitizer have
  been re-armed at the new point. ``lifHardwareReady`` calls
  :cpp:func:`LifConfig::hwReady` to clear LIF's own
  ``d_processingPaused`` flag — symmetric to FTMW's
  :cpp:func:`FtmwConfig::hwReady` after ``allClocksReady``. A
  ``success = false`` is treated as a fatal hardware error and
  triggers the abort path described below.
- :cpp:func:`HardwareManager::lifDigitizerShotAcquired` →
  :cpp:func:`AcquisitionManager::processLifDigitizerShot`. Each
  digitized waveform from :cpp:class:`LifDigitizer` is added to the
  active scan point through :cpp:func:`LifConfig::addWaveform`; when
  the point completes, ``advance`` returns true and the AM emits a
  fresh :cpp:func:`AcquisitionManager::nextLifPoint` for the next
  one. After every shot, ``checkComplete`` runs the same
  completion check FTMW uses.

Completion and abort paths
--------------------------

Every completion path ends in the same place:
:cpp:func:`AcquisitionManager::finishAcquisition` → emit
:cpp:func:`AcquisitionManager::experimentComplete` → queued connection
to :cpp:func:`BatchManager::experimentComplete` (the slot, GUI thread).
The four ways to get there:

- **Normal completion.** :cpp:func:`Experiment::isComplete` returns
  true (every :cpp:class:`ExperimentObjective` reports complete);
  ``checkComplete`` calls :cpp:func:`AcquisitionManager::finishAcquisition`.
- **User abort.** The user clicks the Abort button.
  :cpp:func:`MainWindow::startBatch` wired the button to
  :cpp:func:`BatchManager::abort` for the current batch — for
  :cpp:class:`BatchSingle` that just sets ``d_complete = true``; for
  :cpp:class:`BatchSequence` it stops the inter-experiment timer.
  The actual acquisition abort is the parallel
  ``HardwareManager::abortAcquisition`` → ``AcquisitionManager::abort``
  edge installed in ``startBatch`` and triggered the same way.
  :cpp:func:`AcquisitionManager::abort` calls
  :cpp:func:`Experiment::abort` (which sets ``d_isAborted`` and
  cascades :cpp:func:`ExperimentObjective::abort` to every
  objective) and then :cpp:func:`AcquisitionManager::finishAcquisition`.
- **Hardware-failure abort.** A previously-connected device emits
  :cpp:func:`HardwareObject::hardwareFailure`.
  :cpp:func:`HardwareManager::hardwareFailure` updates the
  connection-state map (see
  :doc:`/developer_guide/hardware_runtime`) and, if the failed device
  is critical, emits :cpp:func:`HardwareManager::abortAcquisition`.
  The ``startBatch``-installed connection routes that to
  :cpp:func:`AcquisitionManager::abort`. From there the path is
  identical to the user-abort case.
- **Validation-failure abort.** A reading delivered to
  :cpp:func:`AcquisitionManager::processValidationData` falls outside
  its configured range. The slot calls
  :cpp:func:`AcquisitionManager::abort` directly. Same downstream
  path.

:cpp:func:`AcquisitionManager::finishAcquisition` does the same five
things in every case: stop the drain timer; flip the
``d_abortProcessing`` atomic and wait for the in-flight worker to
exit; emit ``endAcquisition`` (broadcast through HM to every
hardware object's ``endAcquisition`` slot); transition to
``Idle``; and, for non-dummy experiments, call
:cpp:func:`Experiment::finalSave` to commit the closing artifacts to
disk. Then ``experimentComplete`` is emitted and the AM releases its
reference to the experiment.

.. note::

   :cpp:func:`AcquisitionManager::experimentComplete` is a **signal**;
   :cpp:func:`BatchManager::experimentComplete` is a **slot**. They
   share a name because the slot exists to consume the signal — the
   queued connection between them crosses the AM thread into the GUI
   thread and is what advances every batch from one experiment to the
   next.

Batch-level loop
----------------

:cpp:func:`BatchManager::experimentComplete` is the central
post-experiment hook. It logs the result through
:cpp:func:`LogHandler::logMessage` (using the experiment's
``d_endLogMessage``), evaluates ``initSuccess = exp->d_hardwareSuccess
&& exp->d_initSuccess``, and decides what to do next:

- ``initSuccess`` true: call :cpp:func:`BatchManager::processExperiment`
  on the just-completed experiment. For :cpp:class:`BatchSingle` this
  sets ``d_complete = true``; for :cpp:class:`BatchSequence` it
  increments ``d_experimentCount``. Other concrete subclasses can
  do post-acquisition analysis here.
- ``isAborted`` false **and** ``isComplete`` false **and**
  ``initSuccess`` true: call
  :cpp:func:`BatchManager::beginNextExperiment`. The default
  implementation emits ``beginExperiment`` immediately. The
  ``startBatch``-installed lambda picks it up and calls
  :cpp:func:`HardwareManager::initializeExperiment` for the next
  experiment in the batch — back to step 1 of the lifecycle.
  :cpp:class:`BatchSequence` overrides
  :cpp:func:`BatchSequence::beginNextExperiment` to run a one-shot
  ``QTimer`` for the configured inter-experiment interval, then emit
  ``beginExperiment`` from the timer's lambda.
- Otherwise (init failure, abort, or natural batch completion): call
  :cpp:func:`BatchManager::abort` if the experiment was aborted or
  init failed, then :cpp:func:`BatchManager::writeReport`
  unconditionally, then emit
  :cpp:func:`BatchManager::batchComplete` carrying ``isAborted``.

:cpp:func:`MainWindow::batchComplete` consumes that signal: it
disconnects the per-batch wiring set up in ``startBatch`` (the
``auxData`` connection to the aux-data view widget and the
``HardwareManager::abortAcquisition`` → ``AcquisitionManager::abort``
edge), refreshes the status bar and progress bar, and calls
:cpp:func:`MainWindow::configureUi` with ``Idle`` so the toolbar
returns to its non-acquiring state.

The lifecycle at a glance
-------------------------

The full single-experiment round-trip, from the moment
:cpp:func:`MainWindow::startBatch` runs to the moment
:cpp:func:`BatchManager::experimentComplete` returns:

.. mermaid::

   sequenceDiagram
       autonumber
       participant MW as MainWindow (GUI)
       participant BM as BatchManager (GUI)
       participant HM as HardwareManager (HM thread)
       participant CM as ClockManager
       participant HOs as HardwareObject set
       participant EX as Experiment
       participant AM as AcquisitionManager (AM thread)

       MW->>BM: new BatchSingle / BatchSequence
       MW->>MW: startBatch wires per-batch signals
       MW->>HM: invokeMethod(initializeExperiment(exp))
       HM->>CM: prepareForExperiment(exp)
       HM->>HOs: hwPrepareForExperiment(exp) per object
       HOs-->>HM: success / failure
       HM-->>MW: experimentInitialized(exp) [queued]
       MW->>EX: initialize() (creates dir, writes initial CSVs)
       MW->>AM: invokeMethod(beginExperiment(exp))
       AM->>AM: state Acquiring, start aux + drain timers
       AM-->>MW: newClockSettings(clocks)
       MW->>HM: invokeMethod(setClocks(clocks))
       HM-->>AM: allClocksReady -> clockSettingsComplete
       AM-->>HM: beginAcquisition (broadcast to HOs)
       loop steady state
           AM->>AM: drainFtmwBuffer / processLifDigitizerShot
           AM-->>HM: auxDataSignal -> getAuxData
           HM-->>AM: auxData / validationData
           AM->>AM: checkComplete
       end
       AM->>AM: finishAcquisition (endAcquisition, finalSave)
       AM-->>BM: experimentComplete [queued]
       alt batch continues
           BM->>BM: processExperiment, beginNextExperiment
           BM-->>MW: beginExperiment (next iteration)
       else batch ends
           BM->>BM: abort (if needed), writeReport
           BM-->>MW: batchComplete(aborted)
       end

Adding a new experiment mode
----------------------------

Two extension points exist for new experiment modes: a new
:cpp:class:`FtmwConfig` subclass (or a new :cpp:enum:`FtmwConfig::FtmwType`
that an existing subclass can switch on) for a new FTMW acquisition
strategy, or a new :cpp:class:`BatchManager` subclass for a new
batch-level iteration policy. Both are walked through end-to-end on
:doc:`/developer_guide/adding_an_experiment_mode`. The lifecycle this
page describes is what the new mode plugs into; the recipe page
covers what the subclass has to override and how to register it with
:cpp:class:`MainWindow`.
