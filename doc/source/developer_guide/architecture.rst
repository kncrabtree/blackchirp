.. index::
   single: architecture
   single: source tree
   single: orchestration singletons
   single: threading model
   single: GUI thread
   single: HardwareManager thread
   single: AcquisitionManager thread
   single: per-device threads
   single: QtConcurrent
   single: QFutureWatcher
   single: QMetaObject::invokeMethod
   single: queued connections
   single: MainWindow; wiring

Architecture
============

This page is the spine of the developer guide. The remaining sub-pages
assume the reader has absorbed three things from this one: which
top-level directory under ``src/`` owns which responsibility, the names
of the orchestration singletons that compose a running program, and the
threading model that connects them. Subsequent sub-pages reference the
directories, the singletons, and the threads by name without
re-introducing them.

The design is conventional Qt: a ``QApplication`` event loop on the GUI
thread, a small set of long-lived ``QObject`` orchestrators each on a
dedicated ``QThread``, signal/slot wiring between them, and a
``QtConcurrent`` worker pool for the few operations whose duration would
otherwise stall an event loop. The build system layout —
which library produces which directory — is on
:doc:`/developer_guide/build_system`; this page focuses on the
runtime layout. For per-method contracts, follow the
``:doc:`` cross-links to the API reference.

Source tree
-----------

The application source lives under ``src/``. Every sub-directory below
is one level deep from there.

``src/main.cpp``
   Application entry point. Sets the ``QApplication`` identity, loads
   font and save-path configuration, opens the
   :cpp:class:`ApplicationConfigDialog` and
   :cpp:class:`RuntimeHardwareConfigDialog` on first run, registers
   meta-types and catalog parsers, then constructs ``MainWindow`` and
   enters the event loop.

``src/acquisition/``
   The experiment-execution layer. ``acquisitionmanager.{cpp,h}``
   declares :cpp:class:`AcquisitionManager`, which drives an in-progress
   experiment (waveform pipeline, aux/validation timer, backups,
   finalization). The ``batch/`` sub-directory holds
   :cpp:class:`BatchManager` and its concrete subclasses
   :cpp:class:`BatchSingle` and :cpp:class:`BatchSequence`, which
   coordinate across-experiment state.

``src/data/``
   Data model, persistence, analysis, and application-wide singletons
   that are not hardware-specific.

   - ``data/experiment/`` — experiment configuration:
     :cpp:class:`Experiment`, the :cpp:class:`FtmwConfig` family,
     :cpp:class:`RfConfig`, :cpp:class:`ChirpConfig`, the
     :cpp:class:`DigitizerConfig` family,
     :cpp:class:`ExperimentObjective`, and
     :cpp:class:`ExperimentValidator`.
   - ``data/lif/`` — LIF-specific data: :cpp:class:`LifConfig`,
     :cpp:class:`LifDigitizerConfig`, :cpp:class:`LifStorage`,
     :cpp:class:`LifTrace`.
   - ``data/loadout/`` — :cpp:class:`HardwareLoadout`,
     :cpp:class:`LoadoutManager`, :cpp:class:`FtmwPreset`, and the
     loadout/preset snapshot helpers.
   - ``data/storage/`` — persistence: :cpp:class:`BlackchirpCSV`, the
     :cpp:class:`HeaderStorage` tree, :cpp:class:`DataStorageBase` and
     its concrete subclasses (:cpp:class:`FidStorageBase` and friends,
     :cpp:class:`LifStorage`, :cpp:class:`OverlayStorage`,
     :cpp:class:`AuxDataStorage`), :cpp:class:`SettingsStorage`,
     :cpp:class:`WaveformBuffer`, and
     :cpp:class:`ApplicationConfigManager`.
   - ``data/analysis/`` — :cpp:class:`FtWorker` (the FT processing
     worker), :cpp:class:`Analysis`, :cpp:class:`PeakFinder`,
     :cpp:class:`WaveformParser`, :cpp:class:`Ft`.
   - ``data/processing/`` — overlay processing/operations.
   - ``data/processing/parsers/`` — overlay/import file parsers
     (:cpp:class:`FileParser`, :cpp:class:`FileParserRegistry`,
     :cpp:class:`GenericXyParser`, :cpp:class:`SpcatParser`,
     :cpp:class:`XiamParser`, :cpp:class:`CatalogParser`).
   - ``data/model/`` — Qt item models for chirp, clock, marker, peak,
     and overlay tables.
   - ``data/settings/`` — the static key declarations
     ``hardwarekeys.h`` and ``guikeys.h``.
   - ``data/presentation/`` — :cpp:class:`CurveAppearance`.
   - ``data/loghandler.{cpp,h}`` — :cpp:class:`LogHandler` global
     diagnostic logging.
   - ``data/bcglobals.{cpp,h}`` — application-wide constants and
     persistent-key declarations.

``src/gui/``
   Qt Widgets layer. ``mainwindow.{cpp,h}`` is the application's
   central window and the wiring hub for all the orchestration
   singletons (see *MainWindow as the wiring hub* below).

   - ``gui/dialog/`` — modal dialogs (Add Profile, Application
     Config, Hardware Configuration, FTMW Configuration, Batch
     Sequence, Communication, About, etc.).
   - ``gui/expsetup/`` — the experiment-setup wizard pages and the
     dialog that hosts them.
   - ``gui/widget/`` — embeddable widgets (FTMW view, chirp config,
     hardware status boxes, digitizer config, pulse config, …).
   - ``gui/plot/`` — Qwt-based scientific plots:
     :cpp:class:`ZoomPanPlot` base, :cpp:class:`MainFtPlot`,
     :cpp:class:`FidPlot`, :cpp:class:`ChirpConfigPlot`,
     :cpp:class:`PulsePlot`, :cpp:class:`TrackingPlot`, plot-curve
     helpers, and the custom tracker/zoomer.
   - ``gui/lif/gui/`` — LIF-specific widgets and plots.
   - ``gui/overlay/`` — overlay configuration widgets and the
     unified overlay dialog.
   - ``gui/style/`` — :cpp:class:`ThemeColors` theme management.
   - ``gui/util/`` — small GUI utilities (numeric formatting).

``src/hardware/``
   Hardware abstraction and drivers.

   - ``hardware/core/`` — :cpp:class:`HardwareObject` (the abstract
     device base), :cpp:class:`HardwareManager`,
     :cpp:class:`HardwareRegistry`, the registration helpers in
     ``hardwareregistration.{cpp,h}`` and the aggregator headers
     ``hw_base.h`` / ``hw_h.h`` / ``hw_impl.h``,
     :cpp:class:`RuntimeHardwareConfig`,
     :cpp:class:`HardwareProfileManager`, the interface-class
     sub-directories (``clock/``, ``ftmwdigitizer/``,
     ``lifdigitizer/``, ``liflaser/``), and ``communication/``.
   - ``hardware/optional/`` — interfaces and drivers for
     hardware classes that are optional in any given experiment:
     ``chirpsource/`` (AWG), ``flowcontroller/``,
     ``gpibcontroller/``, ``ioboard/``, ``pressurecontroller/``,
     ``pulsegenerator/``, ``tempcontroller/``.
   - ``hardware/python/`` — Python trampoline classes (one per
     hardware type), the host process script ``python_hw_host.py``,
     and per-type template scripts.
   - ``hardware/library/`` — :cpp:class:`VendorLibrary` and concrete
     subclasses (:cpp:class:`LabjackLibrary`,
     :cpp:class:`SpectrumLibrary`).

``src/modules/``
   Optional compile-conditional modules. ``modules/cuda/`` holds
   ``GpuAverager`` — CUDA-accelerated FID accumulation gated by the
   ``BC_ENABLE_CUDA`` build option.

``src/resources/``
   Qt resource collection: icons, the ``resources.qrc`` manifest,
   and the udev rules file ``52-serial.rules``.

Orchestration singletons
------------------------

A running Blackchirp instance is held together by a small cast of
long-lived orchestrators. Most are application-wide singletons; the
acquisition trio (:cpp:class:`HardwareManager`,
:cpp:class:`AcquisitionManager`, :cpp:class:`BatchManager`) is owned
by ``MainWindow``. Each entry below is a one-paragraph orientation;
follow the cross-link for the per-method contract.

:cpp:class:`HardwareRegistry` (compile-time catalog)
   :doc:`/classes/hardwareregistry` is the static catalog of every
   hardware driver linked into the binary. Each driver
   registers itself before ``main()`` runs (factory function,
   supported communication protocols, per-driver settings,
   inheritance chain) using the ``REGISTER_HARDWARE_*`` macros from
   ``hardwareregistration.h``. The registry is the authoritative
   answer to "what drivers exist for hardware type *X*?" and
   is the only place that constructs :cpp:class:`HardwareObject`
   instances by key.

:cpp:class:`HardwareProfileManager` (profile metadata)
   :doc:`/classes/hardwareprofilemanager` owns the persistent
   profile records. A *profile* is an immutable
   ``(hardwareType, label, driver)`` triple with its own
   persisted settings and (for Python drivers) script path and
   class name; the ``<hardwareType>.<label>`` pair is the
   profile's identity, and the driver is fixed at
   creation time. Profiles are CRUD'd from
   :cpp:class:`RuntimeHardwareConfigDialog` (the Configure Hardware
   dialog) and stored via :cpp:class:`SettingsStorage`. The user
   workflow is documented in :doc:`/user_guide/hardware_config`.

:cpp:class:`RuntimeHardwareConfig` (active selection)
   :doc:`/classes/runtimehardwareconfig` records *which* profiles
   are active in the running session, keyed by profile identity
   (``<HardwareType>.<label>``). The driver key for each
   active profile is held as a denormalized copy of the profile's
   immutable value and validated against
   :cpp:class:`HardwareRegistry`. Read access is open from any
   thread (read/write-locked); write access is restricted to
   friend classes — primarily :cpp:class:`HardwareManager` and
   :cpp:class:`RuntimeHardwareConfigDialog`. The active set is what
   :cpp:func:`HardwareManager::initialize` consults to decide what
   to instantiate.

:cpp:class:`LoadoutManager` (named member-profile sets + FTMW presets)
   :doc:`/classes/loadoutmanager` persists named *loadouts* (a
   complete hardware selection — which AWG profile, which
   digitizer profile, which clock profiles) and the *FTMW presets*
   that ride on
   top of each loadout (an :cpp:class:`RfConfig` chain plus
   :cpp:class:`ChirpConfig` and digitizer config, named within the
   loadout). The user picks a loadout from the Loadout menu and an
   FTMW preset from the FTMW Preset menu;
   :cpp:func:`HardwareManager::applyHardwareMap` consumes the
   loadout's hardware map and pushes it through
   :cpp:class:`RuntimeHardwareConfig`.

:cpp:class:`HardwareManager` (live hardware owner)
   :doc:`/classes/hardwaremanager` owns every live
   :cpp:class:`HardwareObject` for the active loadout. It runs on
   its own thread, moves threaded hardware objects to per-device
   threads, mediates all cross-thread hardware calls, and fans out
   connection, auxiliary-data, and experiment-lifecycle signals.
   ``MainWindow`` constructs it before the event loop starts and
   triggers :cpp:func:`HardwareManager::initialize` from the
   thread's ``started()`` signal. The static
   :cpp:func:`HardwareManager::constInstance` accessor exposes
   read-only lookup of the hardware map for callers (such as
   :cpp:class:`HardwareObject` instances resolving GPIB controllers)
   that cannot hold a direct reference.

:cpp:class:`ClockManager` (RF clock subsystem)
   :doc:`/classes/clockmanager` is owned by
   :cpp:class:`HardwareManager` (``pu_clockManager``) and lives on
   the HardwareManager thread. It maps each
   :cpp:enum:`RfConfig::ClockType` role (``UpLO``, ``DownLO``,
   ``DRClock``, ``AwgRef``, …) to a physical :cpp:class:`Clock`
   device and output index, and gates the FTMW digitizer during
   clock transitions so partial frequency changes do not leak into
   acquired data. All cross-thread interaction with
   :cpp:class:`ClockManager` goes through
   :cpp:class:`HardwareManager`'s queued slots.

:cpp:class:`AcquisitionManager` (experiment driver)
   :doc:`/classes/acquisitionmanager` runs an in-progress experiment
   on its own thread: it holds the
   ``std::shared_ptr<Experiment>``, drives the FTMW waveform
   pipeline through a :cpp:class:`WaveformBuffer` and a
   ``QFutureWatcher``-tracked worker, services the auxiliary-data
   and validation timers, dispatches periodic backups, and
   finalizes the experiment when complete. Its public slots are
   invoked from the GUI thread via
   ``QMetaObject::invokeMethod``.

:cpp:class:`BatchManager` (across-experiment coordinator)
   :doc:`/classes/batchmanager` is an abstract base; the active
   subclass — :cpp:class:`BatchSingle` for a single experiment,
   :cpp:class:`BatchSequence` for a timed multi-experiment run —
   lives on the GUI thread inside ``MainWindow``. It coordinates
   what comes after each :cpp:class:`AcquisitionManager`
   ``experimentComplete()``: report writing, post-processing, and
   advancing to the next experiment (or ending the batch).

:cpp:class:`LogHandler`, :cpp:class:`ApplicationConfigManager`
   :doc:`/classes/loghandler` is the application-wide diagnostic
   sink that the ``bcLog`` / ``bcWarn`` / ``bcError`` macros forward
   to; it owns the per-experiment log file and the in-app log view.
   :doc:`/classes/applicationconfigmanager` exposes the user-level
   configuration that survives across runs — data save path, debug
   logging toggle, vendor library paths, font, LIF-enable flag — and
   emits change signals that other singletons subscribe to.

MainWindow as the wiring hub
----------------------------

``MainWindow`` is responsible for two things beyond hosting the UI:
constructing the orchestration singletons it owns, and wiring their
signals together. The constructor allocates :cpp:class:`HardwareManager`
and :cpp:class:`AcquisitionManager`, creates a dedicated ``QThread``
for each, calls ``moveToThread`` on the singleton, attaches the
thread's ``started()`` signal (HardwareManager only) to its
``initialize()`` slot, and wires ``deleteLater`` on ``finished()``.
The threads are not started until ``MainWindow::initializeHardware()``
emits ``startInit``.

The same constructor wires the inter-manager signal flow that the
acquisition lifecycle relies on:
:cpp:func:`AcquisitionManager::beginAcquisition` →
:cpp:func:`HardwareManager::beginAcquisition`,
:cpp:func:`AcquisitionManager::endAcquisition` →
:cpp:func:`HardwareManager::endAcquisition`,
:cpp:func:`HardwareManager::experimentInitialized` →
``MainWindow::experimentInitialized`` (which then queues
``AcquisitionManager::beginExperiment``),
:cpp:func:`HardwareManager::auxData` →
:cpp:func:`AcquisitionManager::processAuxData`,
:cpp:func:`HardwareManager::validationData` →
:cpp:func:`AcquisitionManager::processValidationData`,
:cpp:func:`HardwareManager::allClocksReady` →
:cpp:func:`AcquisitionManager::clockSettingsComplete`, and
:cpp:func:`AcquisitionManager::experimentComplete` →
:cpp:func:`HardwareManager::experimentComplete` and the
:cpp:class:`BatchManager` subclass. ``MainWindow`` also dispatches
user actions (toolbar buttons, menu items, dialog acceptance) to
whichever singleton owns them, almost always via
``QMetaObject::invokeMethod`` so the call hops onto the destination
thread.

The :cpp:class:`BatchManager` subclass is constructed lazily in
``MainWindow::startBatch`` when the user begins an experiment and
deleted the next time ``startBatch`` runs.

Subsequent sub-pages trace the specific signal chains in detail:
:doc:`/developer_guide/hardware_runtime` covers the hardware-side
wiring and the connection-test flow,
:doc:`/developer_guide/experiment_lifecycle` follows an experiment
from "Start Experiment" through to ``finalSave``, and
:doc:`/developer_guide/ftmw_acquisition` walks the FTMW data path.

Threading model
---------------

Blackchirp has four primary execution contexts and one shared worker
pool. The contexts and the rules for crossing between them are set
once here and assumed elsewhere.

GUI thread
   The ``QApplication`` main thread. Hosts ``MainWindow``, every
   dialog and widget, every plot, the active
   :cpp:class:`BatchManager` subclass, and the application-wide
   singletons :cpp:class:`LoadoutManager`,
   :cpp:class:`HardwareProfileManager`,
   :cpp:class:`RuntimeHardwareConfig`, :cpp:class:`LogHandler`, and
   :cpp:class:`ApplicationConfigManager`. (Those singletons are
   thread-safe for read access from any thread; writes are
   constrained.)

HardwareManager thread
   A dedicated ``QThread`` named ``HardwareManagerThread``,
   constructed in the ``MainWindow`` constructor before the event
   loop starts. :cpp:class:`HardwareManager` is moved onto it; the
   thread's ``started()`` signal triggers
   :cpp:func:`HardwareManager::initialize`, which loads the active
   profiles from :cpp:class:`RuntimeHardwareConfig`, instantiates
   the :cpp:class:`HardwareObject` instances, and brings them
   online. All :cpp:class:`HardwareManager` slots — including the
   acquisition-lifecycle slots and the per-hardware-type request
   slots — execute on this thread. :cpp:class:`ClockManager` is
   owned by :cpp:class:`HardwareManager` and runs on the same
   thread.

Per-device threads
   For each :cpp:class:`HardwareObject` whose ``d_threaded`` flag is
   ``true``, :cpp:class:`HardwareManager` creates a dedicated
   ``QThread`` named ``<key>Thread`` (e.g.,
   ``FtmwDigitizer.mainThread``), calls ``moveToThread`` on the object,
   and connects the thread's ``started()`` signal to
   :cpp:func:`HardwareObject::bcInitInstrument`. The flag is set in
   the driver's constructor and may be overridden per-profile via
   :cpp:class:`RuntimeHardwareConfig`. Two hard rules apply to
   threaded hardware:

   1. The driver constructor must pass ``nullptr`` as the
      ``QObject`` parent. ``moveToThread`` requires that the object
      have no parent (or share its parent's thread), and a parent
      assigned at construction time would prevent the move.
   2. The constructor must not create child ``QObject`` instances
      (timers, communication-protocol objects, child workers).
      Children would be parented to the not-yet-moved object on the
      caller's thread; once the object moves, the children are left
      behind. Construct children inside ``initialize()`` instead,
      which runs on the device thread after the move.

   Non-threaded hardware lives on the HardwareManager thread; for
   those objects :cpp:class:`HardwareManager` calls
   :cpp:func:`HardwareObject::bcInitInstrument` directly via
   ``QMetaObject::invokeMethod``.

AcquisitionManager thread
   A dedicated ``QThread`` named ``AcquisitionManagerThread``,
   constructed in the ``MainWindow`` constructor.
   :cpp:class:`AcquisitionManager` is moved onto it but the thread
   is not started until ``initializeHardware()``. The GUI thread
   calls :cpp:func:`AcquisitionManager::beginExperiment` via
   ``QMetaObject::invokeMethod``; from that point the FTMW drain
   timer, the aux-data timer, the experiment shared pointer, and
   every state mutation execute on this thread.

QtConcurrent worker pool
   Two operations on :cpp:class:`AcquisitionManager` run on the
   global Qt thread pool rather than the AM thread itself:

   - **Waveform parse-and-accumulate.** When the FTMW configuration
     uses a :cpp:class:`WaveformBuffer`, an internal 20 ms drain
     timer reads pending entries out of the buffer and dispatches
     them to ``QtConcurrent::run``. A
     ``QFutureWatcher<FtmwProcessingResult>`` carries the result
     back to the AM thread, where ``onProcessingComplete`` advances
     the segment, updates the progress signal, and restarts the
     drain timer. The drain timer is paused while the worker is
     in flight, so at most one waveform batch is in progress at a
     time. The full data flow is on
     :doc:`/developer_guide/ftmw_acquisition`.
   - **Periodic backups.** When
     :cpp:func:`Experiment::canBackup` returns ``true``,
     :cpp:func:`Experiment::backup` is dispatched to
     ``QtConcurrent::run`` and observed via a
     ``QFutureWatcher<void>``; the watcher's ``finished`` signal
     re-emits as ``backupComplete`` on the AM thread.

Cross-thread call patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

Blackchirp uses a small set of Qt patterns for crossing thread
boundaries. The same patterns appear throughout the codebase; new
code should follow them rather than invent variants.

Direct signal → slot connection
   ``Qt::AutoConnection`` (the default) is correct in almost every
   case. Qt picks queued delivery when the signal and slot live on
   different threads and direct delivery when they live on the
   same one. Most of the wiring in the ``MainWindow`` constructor
   is bare ``connect`` calls relying on this behavior.

``QMetaObject::invokeMethod``
   Used when a non-``QObject`` caller, or a caller that does not
   hold a matching signal, needs to invoke a slot on another
   thread. ``Qt::QueuedConnection`` for fire-and-forget invocations
   (the GUI thread asking
   :cpp:class:`HardwareManager` to start an experiment, an aux-data
   request, etc.). ``Qt::BlockingQueuedConnection`` only when the
   caller must wait for the result *and* is guaranteed to be on a
   different thread from the target — invoking
   ``BlockingQueuedConnection`` on the same thread deadlocks. The
   GUI uses it sparingly: ``MainWindow`` blocks briefly to read the
   current clock map before opening dialogs, for example.

``QFutureWatcher<T>``
   The canonical pattern for "do expensive work on the worker pool,
   hop the result back to my thread." :cpp:class:`AcquisitionManager`
   uses it for both waveform processing and backups; new code that
   wants to dispatch a worker should use the same pattern rather
   than rolling a thread of its own.

``QSemaphore`` (:cpp:class:`WaveformBuffer`)
   The FTMW digitizer thread is a producer and the AM thread is a
   consumer; the SPSC :cpp:class:`WaveformBuffer` between them uses
   a ``QSemaphore`` for low-latency notification — the producer
   ``release``-s after each shot, and the consumer can either drain
   on a timer (the default in
   :cpp:func:`AcquisitionManager::drainFtmwBuffer`) or block on
   ``waitForData``. The semaphore avoids the per-shot signal-queue
   overhead of routing every waveform through Qt's event loop. The
   full producer/consumer/pool story is on
   :doc:`/developer_guide/ftmw_acquisition`.

Static singletons
   :cpp:class:`LoadoutManager`, :cpp:class:`HardwareProfileManager`,
   :cpp:class:`RuntimeHardwareConfig`,
   :cpp:class:`HardwareRegistry`, :cpp:class:`LogHandler`, and
   :cpp:class:`ApplicationConfigManager` all expose ``instance()``
   and (where read-only access is meaningful) ``constInstance()``
   accessors. Each protects its own state — :cpp:class:`SettingsStorage`
   has an internal lock; :cpp:class:`RuntimeHardwareConfig` uses a
   ``QReadWriteLock``; :cpp:class:`LoadoutManager` uses a ``QMutex``
   — so calls from any thread are safe. The signals these
   singletons emit are delivered on the thread of the caller that
   triggered the change, so connect with ``Qt::AutoConnection`` and
   trust Qt to queue across thread boundaries.

Diagram
-------

The diagram below splits the four execution contexts into columns —
GUI, HardwareManager, AcquisitionManager, per-device — and shows
ownership (solid arrows) and the dominant inter-manager signal
flows (dashed arrows). The QtConcurrent worker pool is shown
attached to the AcquisitionManager thread, since that is the only
thread that dispatches to it.

.. mermaid::

   flowchart LR
       subgraph GUI[GUI thread]
           MW[MainWindow]
           BM[BatchManager<br/>BatchSingle / BatchSequence]
           Dialogs[Dialogs · plots · widgets]
           Singletons[Process-wide singletons:<br/>LoadoutManager · HardwareProfileManager<br/>RuntimeHardwareConfig · HardwareRegistry<br/>LogHandler · ApplicationConfigManager]
       end

       subgraph HMT[HardwareManager thread]
           HM[HardwareManager]
           CM[ClockManager]
           HMap[map&lt;key, HardwareObject*&gt;]
       end

       subgraph AMT[AcquisitionManager thread]
           AM[AcquisitionManager]
           Exp[shared_ptr&lt;Experiment&gt;]
           Drain[drainTimer · auxTimer]
           Pool[QtConcurrent pool:<br/>FTMW worker · backup worker<br/>tracked by QFutureWatcher]
       end

       subgraph DT[Per-device threads]
           HO1[HardwareObject A]
           HO2[HardwareObject B]
           HOn[...]
       end

       MW -- owns / moveToThread --> HM
       MW -- owns / moveToThread --> AM
       MW -- owns --> BM
       MW -- owns --> Dialogs
       HM -- owns --> CM
       HM -- owns --> HMap
       HMap -- moveToThread<br/>if d_threaded --> HO1
       HMap -- moveToThread<br/>if d_threaded --> HO2
       HMap -. moveToThread<br/>if d_threaded .-> HOn
       AM -- owns --> Exp
       AM -- owns --> Drain
       AM -- dispatches to --> Pool

       MW -. invokeMethod:<br/>initializeExperiment · setClocks · sleep .-> HM
       HM -. experimentInitialized .-> MW
       MW -. invokeMethod:<br/>beginExperiment · pause · abort .-> AM
       AM -. beginAcquisition · endAcquisition<br/>auxDataSignal · newClockSettings .-> HM
       HM -. auxData · validationData<br/>allClocksReady · lifSettingsComplete .-> AM
       AM -. experimentComplete .-> BM
       AM -. experimentComplete .-> HM
       HM -. queued slots:<br/>set / read / initialize / sleep .-> HO1
       HO1 -. updates · aux data<br/>connection results · failure .-> HM
       HO1 -. WaveformBuffer<br/>QSemaphore + SPSC ring .-> AM

Read the diagram top-down for ownership and left-to-right for the
acquisition lifecycle. The GUI launches everything; the
HardwareManager and AcquisitionManager threads exchange a small set
of signals that drive an experiment from setup to finalization; the
per-device threads carry per-instrument I/O; and the worker pool
absorbs the two operations whose duration would otherwise stall the
AM event loop.
