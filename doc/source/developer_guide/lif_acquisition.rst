.. index::
   single: LIF pipeline
   single: LIF; scan grid
   single: LifConfig; acquisition objective
   single: LifDigitizerConfig
   single: LifStorage; flattening convention
   single: LifTrace
   single: LifScanOrder
   single: LifCompleteMode
   single: AcquisitionManager; nextLifPoint
   single: HardwareManager; setLifParameters
   single: LifDigitizer; waveformRead
   single: LifLaser; setPosition
   single: integration gate; LIF
   single: LifProcessingWidget
   single: LifSpectrogramPlot
   single: LifSlicePlot
   single: LifTracePlot
   single: LifConversion; frequency-conversion topology
   single: LaserFreqConversionStage
   single: LifConversionSnapshot
   single: LifPreset
   single: liftopology.csv
   single: LifConversionTableModel
   single: LifConversionWidget
   single: ExperimentLifConfigPage
   single: HardwareManager; configureLifHarmonic

LIF Acquisition and Visualization
=================================

The LIF (laser-induced fluorescence) pipeline is fundamentally simpler
than FTMW. There is no ring buffer, no thread-pool drain loop, and no
parallel byte-unpacking — every shot is delivered as a Qt signal and
the integration math runs on the acquisition thread directly. The
complexity that this page documents lives elsewhere: in the
two-dimensional ``(delay, laser)`` scan grid, in the index arithmetic
that bridges that grid to flat per-cell storage, and in the
visualization that slices the same grid two different ways.

The LIF path is signal-based end-to-end because the data volumes and
trigger rates make signal overhead a non-issue. Each
:cpp:class:`LifDigitizer` shot is a small ``QVector<qint8>``; trigger rates
are bounded by the laser repetition rate (typically ≤ 100 Hz, i.e.
two to three orders of magnitude slower than the FTMW digitizer).
At those rates per-shot ``QMetaCallEvent`` allocation is invisible,
so the FTMW pipeline's ring-buffer plus drain-timer plus thread-pool
machinery would only add latency without buying anything back.

The LIF scan model
------------------

A LIF acquisition sweeps a two-dimensional grid:

- The **delay axis** is the time, in microseconds, between a triggering
  event (typically a discharge or other gas-pulse trigger) and the
  laser firing. Blackchirp programs the delay onto the LIF channel of
  every active :cpp:class:`PulseGenerator` via
  :cpp:func:`HardwareManager::setPGenLifDelay`.
- The **laser axis** is the excitation-beam position commanded to the
  active :cpp:class:`LifLaser`, gridded in the display unit selected on
  the laser's hardware profile (``BC::LifConv::LaserUnit`` —
  ``Cm1``, ``Nm``, ``GHz``, or ``eV`` — surfaced in the wizard through
  ``BC::Key::LifLaser::units`` and ``::decimals``). Internally, every
  value in the LIF pipeline is vacuum wavenumber (cm⁻¹); the display
  unit is a presentation-layer conversion applied only at the axis's
  read/write boundary (:cpp:func:`LifConfig::currentLaserPos`,
  :cpp:func:`LifConfig::laserRange`). When the experiment's
  frequency-conversion topology is non-identity, this axis is the
  *excitation* (output) beam reaching the sample, not necessarily the
  laser's own native tuning value — see *Frequency conversion* below.

At each ``(delay, laser)`` grid point the LIF digitizer records a
fluorescence trace. The magnitude inside a configurable integration
gate is computed by :cpp:func:`LifTrace::integrate` and becomes a
single point of the LIF spectrum. The digitizer trace itself is
preserved on disk so that gate position and filter parameters can be
re-applied without re-acquiring.

Three settings on :cpp:class:`LifConfig` govern how the grid is
traversed:

- :cpp:enum:`LifConfig::LifScanOrder` —
  ``DelayFirst`` cycles through every delay point at one laser
  position before stepping the laser. ``LaserFirst`` cycles through
  every laser position at one delay before stepping the delay. The
  scan order affects only the order in which points are visited; it
  does not change the on-disk layout (see *Storage* below).
- ``d_delayRandom`` — when ``true``, the delay axis is randomly
  permuted at the start of each sweep. The permutation is rebuilt by
  :cpp:func:`LifConfig::initialize` and reshuffled inside
  :cpp:func:`LifConfig::advance` whenever a delay sweep completes.
  Randomization helps decorrelate slow drifts (sample condition,
  background) from the delay coordinate. The laser axis is always
  stepped sequentially.
- :cpp:enum:`LifConfig::LifCompleteMode` — ``StopWhenComplete`` ends
  acquisition once the grid has been fully covered.
  ``ContinueAveraging`` reports completion (``perMilComplete() ==
  1000``) and continues sweeping for further averaging until the user
  aborts. The "indefinite" flag drives the
  :ref:`AcquisitionManager <acquisitionmanager-state-machine>`
  completion check.

The user-facing operation of the scan is documented in
:doc:`/user_guide/lif/experiment_setup` and
:doc:`/user_guide/lif/configuration`; the wiring that connects the
wizard widgets to the fields above is the topic of *Configuration UI*
below.

LifConfig and LifDigitizerConfig
--------------------------------

:cpp:class:`LifConfig` is the experiment objective and the runtime
cursor for a LIF acquisition. It inherits
:cpp:class:`ExperimentObjective` for the lifecycle interface
(:cpp:func:`initialize`, :cpp:func:`advance`,
:cpp:func:`hwReady`, :cpp:func:`isComplete`,
:cpp:func:`indefinite`, :cpp:func:`cleanupAndSave`) and
``HeaderStorage`` for header serialization. The class owns:

- A :cpp:class:`LifDigitizerConfig` (accessible via
  :cpp:func:`LifConfig::digitizerConfig`) that wraps the digitizer-side
  parameters: which analog channels carry the LIF signal and the
  optional reference signal, and the digitizer's
  :cpp:enum:`LifDigitizerConfig::ChannelOrder`. The digitizer-shared
  fields (record length, sample rate, trigger, vertical scaling) come
  from the :cpp:class:`DigitizerConfig` base.
- A :cpp:class:`LifStorage` (accessible via
  :cpp:func:`LifConfig::storage`) that persists raw traces and
  processing-gate settings.
- A ``LifTrace::LifProcSettings`` (``d_procSettings``) that holds the
  integration-gate bounds and waveform-filter parameters.

The current ``(delay, laser)`` cursor is tracked on
:cpp:class:`LifConfig` directly. ``d_currentDelayIndex`` and
``d_currentLaserIndex`` are the grid coordinates;
:cpp:func:`LifConfig::currentDelay` and
:cpp:func:`LifConfig::currentLaserPos` translate them to the physical
values by

.. code-block:: cpp

   currentDelay()    = d_currentDelayIndex * d_delayStepUs  + d_delayStartUs;
   // displayPos is gridded in the display LaserUnit (d_laserUnits);
   // currentLaserPos() converts it to output-beam cm⁻¹ at this boundary.
   displayPos        = d_currentLaserIndex * d_laserPosStep + d_laserPosStart;
   currentLaserPos() = BC::LifConv::toCm1(displayPos, d_laserUnits);

These two values are what the AM emits in the ``nextLifPoint`` signal
described next; ``currentLaserPos()`` always returns an output-beam
wavenumber in cm⁻¹, so every hardware-dispatch call downstream of it
(the laser, any conversion stage) works in cm⁻¹ without a further unit
lookup. Negative ``d_delayStepUs`` or ``d_laserPosStep`` is permitted
(the user can scan high-to-low); the storage indices remain 0-based,
but the visualization layer reverses indices on the fly so the
rendered axes are monotonic. See *Visualization* below. What turns an
output-beam value into a laser fundamental and a set of
conversion-stage setpoints is covered in *Frequency conversion* below.

The class-level contract — every method, every storage key — is on
:doc:`/classes/lifconfig` and :doc:`/classes/lifstorage`. This page
covers the cross-system flow.

Frequency conversion
---------------------

Many LIF setups do not aim the tunable laser directly at the sample:
the laser output passes through one or more optical conversion stages
— a doubling crystal, a sum- or difference-frequency mixing crystal —
before reaching the interaction region. Blackchirp models that optical
path as an assembled directed acyclic graph (DAG) so it can report the
actual excitation-beam frequency reaching the sample, not just the
laser's own tuning value, and drive any motorized conversion stages to
track the laser as it scans. The pipeline is cm⁻¹ end to end; the
display unit selected on the laser's hardware profile surfaces at
exactly one crossing point, :cpp:func:`LifConfig::currentLaserPos`
(see *The LIF scan model* above).

Architecture layers
~~~~~~~~~~~~~~~~~~~~

1. **Hardware stage** — :cpp:class:`LaserFreqConversionStage`
   (``hardware/optional/laserfreqconversion/laserfreqconversionstage.{h,cpp}``) is a
   :cpp:class:`HardwareObject` sibling of :cpp:class:`LifLaser`,
   registered like any other hardware type. It owns only
   device-identity state: the conversion operation
   (:cpp:func:`LaserFreqConversionStage::conversionOp`, NHG/SFG/DFG), the
   harmonic order for an NHG stage
   (:cpp:func:`LaserFreqConversionStage::harmonicOrder`), and a verify
   flag/tolerance pair used by
   :cpp:func:`LaserFreqConversionStage::setPosition` to confirm a move.
   A doubler or mixer *is* its operation by hardware identity, not a
   free choice, so a concrete driver such as :cpp:class:`SirahFcu`
   overrides :cpp:func:`LaserFreqConversionStage::conversionOp` to a
   constant while leaving the registered setting itself in place, so
   it stays snapshot-visible (see *Key invariants* below).
   ``VirtualLaserFreqConversionStage`` and ``FixedLaserFreqConversionStage``
   are the uncontrolled/CI implementations.
2. **DAG value type and assembly** — :cpp:class:`LifConversion`
   (``data/lif/lifconversion.{h,cpp}``) is a pure value type with no
   :cpp:class:`HardwareObject` or :cpp:class:`SettingsStorage`
   dependency. A ``BC::LifConv::Node`` describes one conversion node
   (``op``, harmonic ``n``, one or two ``BC::LifConv::InputRef``
   inputs, an ``isFinal`` marker); :cpp:func:`LifConversion::assemble`
   validates a node list and resolves it into an affine
   (``output = a·fundamental + b``) model per stage. Free helpers
   declared alongside :cpp:class:`LaserFreqConversionStage` join a node
   list from a settings snapshot without ever touching a live threaded
   device: ``lifConversionNodesFromSnapshot`` builds a fresh
   :cpp:class:`SettingsStorage` on each stage's hardware key to read
   ``op``/``n``, ``assembleLifConversion`` assembles the result, and
   ``assembleCurrentLifConversion`` resolves the current loadout's
   current ``LifPreset``, falling back to the identity
   conversion when none is selected.
3. **Per-experiment LifConfig** — ``d_conversionNodes`` on
   :cpp:class:`LifConfig` is the authoritative, already-joined node
   list for the experiment: op/harmonic order came from hardware,
   wiring came from the table.
   :cpp:func:`LifConfig::setConversionNodes` stores the list and
   rebuilds the cached :cpp:func:`LifConfig::conversion` via
   :cpp:func:`LifConversion::assemble`, falling back to identity on
   failure. :cpp:func:`LifConfig::writeTopologyFile` /
   :cpp:func:`LifConfig::readTopologyFile` are the disk boundary — see
   *On-disk format* below.
4. **Preset and loadout stack** — mirrors the FTMW
   :cpp:class:`RfConfig`/``FtmwPreset`` stack (see *The FTMW mirror*
   below), so a developer already familiar with FTMW presets can
   transfer that model directly. :cpp:struct:`LifConversionSnapshot`
   (``data/loadout/lifconversionsnapshot.{h,cpp}``) is the wiring-only
   persistable form (inputs/``isFinal`` plus the laser hwKey the
   wiring was captured against — never op/harmonic, which are
   hardware-owned); ``LifPreset`` (``hardwareloadout.h``)
   wraps a snapshot with a timestamp; :cpp:class:`LoadoutManager`
   performs per-loadout preset CRUD under
   ``Loadouts/<loadout>/lifPresets/<name>``.
5. **GUI** — :cpp:class:`LifConversionTableModel` /
   ``LifConversionTableDelegate``
   (``data/model/lifconversiontablemodel.{h,cpp}``) own the in-memory
   joined node list; :cpp:class:`LifConversionWidget`
   (``gui/lif/gui/lifconversionwidget.{h,cpp}``) hosts the table, the
   LIF preset bar, and a live preview footer; ``ExperimentLifConfigPage``
   (``gui/lif/gui/experimentlifconfigpage.{h,cpp}``) tabs the
   conversion widget alongside the acquisition control widget — see
   *Configuration UI* below.
6. **HardwareManager** — caches the currently assembled conversion as
   ``d_lifConversion``, re-derives it at experiment-prep and at
   connection-complete under different error-tolerance policies (see
   *Data flow by moment* below), and fans the resolved setpoints out
   to the laser and to every active conversion stage during
   acquisition.

Data flow by moment
~~~~~~~~~~~~~~~~~~~~

- **Config time** (wizard open/edit). :cpp:func:`Experiment::enableLif`
  seeds :cpp:class:`LifConfig` from the current ``LifPreset``
  of the current loadout, joining the preset's wiring with op/harmonic
  order read from each stage's own settings snapshot. It calls
  :cpp:func:`LifConversionSnapshot::toNodes` with local lambdas rather
  than the hardware-library free function
  ``lifConversionNodesFromSnapshot``, because ``data/`` cannot depend
  on ``hardware/`` (see the layering note in *Key invariants* below).
  This seeding runs before ``ExperimentLifConfigPage`` exists, so
  :cpp:class:`ExperimentTypePage`'s scan-axis bounds
  (:cpp:func:`ExperimentTypePage::updateLifLaserBounds`) have a real,
  non-identity output range on first paint. :cpp:class:`LifConversionWidget`'s
  own constructor performs the same self-seeding from the current
  preset, for the case where the widget is built directly rather than
  through this path. Table edits update :cpp:class:`LifConfig` only
  when ``ExperimentLifConfigPage::apply`` runs (on wizard **Next**);
  nothing is pushed to hardware from the table itself except a gated
  harmonic-order change (*Key invariants* below).
- **Prep time** (:cpp:func:`HardwareManager::initializeExperiment`).
  Re-assembles from ``exp->lifConfig()->conversionNodes()`` — the
  already-joined per-experiment list, not live device state — and
  **hard-fails** the experiment (``d_hardwareSuccess = false``) when
  assembly does not succeed. On success it caches ``d_lifConversion``
  and calls ``HardwareManager::pushLifConversionToLaser``.
  :cpp:func:`Experiment::initialize` then writes ``liftopology.csv``
  from the same node list.
- **Connection-complete** (:cpp:func:`HardwareManager::checkStatus` via
  ``HardwareManager::updateLifConversion``). Re-assembles from
  ``assembleCurrentLifConversion`` — the current **preset**, not the
  experiment — and **tolerates** assembly failure by falling back to
  the identity conversion, so the live jog/status path stays
  responsive even while the topology is mid-edit or unconfigured. Prep
  hard-fails; connection-complete tolerates. The two call sites share
  the same assembly machinery but apply opposite error policies — do
  not conflate them when documenting or debugging a topology error.
- **Acquisition.** :cpp:func:`HardwareManager::setLifParameters` (see
  *Acquisition flow* below) dispatches
  ``HardwareManager::setLifLaserPos``, then
  ``HardwareManager::setLifConversionStages``, then
  ``HardwareManager::setPGenLifDelay``. ``setLifConversionStages``
  resolves the fundamental from the requested output-beam value via
  :cpp:func:`LifConversion::outputToLaser`, computes each active
  stage's local input wavenumber via
  :cpp:func:`LifConversion::stageInput`, and dispatches every stage's
  :cpp:func:`LaserFreqConversionStage::setPosition` concurrently
  (``Qt::QueuedConnection`` plus a per-stage
  ``std::promise``/``std::future`` pair, not
  ``Qt::BlockingQueuedConnection``) so stages with independent motors
  move in parallel; the calling thread joins by waiting on every
  future and AND-combines the per-stage results.

The FTMW mirror
~~~~~~~~~~~~~~~~

The LIF preset/conversion stack was built as a structural mirror of
the FTMW :cpp:class:`RfConfig`/``FtmwPreset`` stack documented on
:doc:`/developer_guide/hardware_configuration` (*LoadoutManager —
named maps and FTMW presets*). A developer already familiar with one
side can read the other by substitution:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - FTMW
     - LIF
     - Notes
   * - :cpp:struct:`RfConfigSnapshot`
     - :cpp:struct:`LifConversionSnapshot`
     - LIF's snapshot carries wiring and laser-key provenance only —
       never op/harmonic, which stay hardware-owned.
   * - ``FtmwPreset``
     - ``LifPreset``
     - LIF's preset is narrower by design: conversion wiring only.
   * - ``.ftmwPresets`` / ``currentFtmwPresetName``
     - ``.lifPresets`` / ``currentLifPresetName``
     - Both live on ``HardwareLoadout``.
   * - ``getFtmwPreset`` / ``putFtmwPreset`` / … CRUD
     - ``getLifPreset`` / ``putLifPreset`` / ``removeLifPreset`` /
       ``renameLifPreset`` / ``lifPresetExists`` / ``lifPresetNames`` /
       ``currentLifPresetName`` / ``setCurrentLifPresetName`` /
       ``currentLifPreset``
     - On :cpp:class:`LoadoutManager`. No ``clearLifPresets``
       counterpart to ``clearFtmwPresets`` exists — the mirror is not
       total.
   * - ``rfConfigScalarsMap`` / ``rfConfigClocksArray`` /
       ``rfConfigSnapshotFromMaps``
     - ``lifConversionScalarsMap`` / ``lifConversionWiringArray`` /
       ``lifConversionSnapshotFromMaps``
     - Free functions in ``BC::Loadout``, declared alongside
       ``HardwareLoadout``.
   * - ``ClockTableModel`` / ``Delegate``
     - :cpp:class:`LifConversionTableModel` / ``LifConversionTableDelegate``
     -
   * - ``RfConfigWidget``
     - :cpp:class:`LifConversionWidget`
     - LIF's widget adds a live preview footer (assembled chain
       expression, output range, validation error).
   * - ``applyClocks`` → ``connectRfConfigWidget`` → ``configureClocks``
     - ``applyHarmonic`` → ``MainWindow::connectLifConversionWidget`` →
       ``HardwareManager::configureLifHarmonic``
     - Identical gated-setting channel; see *Key invariants* below and
       the *gated setting* callout on
       :doc:`/developer_guide/hardware_configuration`.
   * - ``ExperimentFtmwConfigPage::apply`` / ``commitFtmwPreset``
     - ``ExperimentLifConfigPage::apply`` /
       ``ExperimentLifConfigPage::commitLifPreset``
     - Both ``commit*Preset`` calls run from
       ``ExperimentSetupDialog::accept``.
   * - ``LOScanConfigWidget`` reads ``d_rfConfig``
     - :cpp:class:`ExperimentTypePage` reads
       ``exp->lifConfig()->conversion()``
     - Drives the scan-axis bounds (*Configuration UI* below).

One deliberate asymmetry beyond the table: FTMW's ``RfConfigWidget``
seeds itself from the GUI side (the widget constructor); LIF's seeding
happens in :cpp:func:`Experiment::enableLif`, on the data-layer side,
because the scan-axis bounds are needed before
:cpp:class:`LifConversionWidget` exists — see *Data flow by moment*
above.

Key invariants
~~~~~~~~~~~~~~

- **op/harmonic ownership is split from wiring ownership.** Conversion
  operation and harmonic order are hardware identity / registered
  settings, always read from a :cpp:class:`SettingsStorage` snapshot
  on the stage's hardware key — never from a live threaded device,
  never stored in a snapshot or preset. Wiring (``inputs``,
  ``isFinal``) is per-experiment/per-preset state. Nothing constructs
  a ``BC::LifConv::Node`` except ``lifConversionNodesFromSnapshot``,
  :cpp:func:`LifConversionSnapshot::toNodes`, or a caller-supplied
  joined list handed to :cpp:func:`LifConfig::setConversionNodes` —
  that is the one join point in the codebase.
- **A harmonic-order change always routes through the device.** This
  is the type's worked example of the *gated setting* pattern
  described on :doc:`/developer_guide/hardware_configuration`:
  :cpp:func:`LifConversionTableModel::requestHarmonicChange` emits
  ``applyHarmonic(stageKey, n)`` rather than editing the model;
  ``MainWindow::connectLifConversionWidget`` hops that signal onto the
  :cpp:class:`HardwareManager` thread into
  ``HardwareManager::configureLifHarmonic``, which calls
  :cpp:func:`LaserFreqConversionStage::setHarmonicOrder` on the device
  and only then emits ``lifHarmonicApplied(stageKey)``; the table
  model's own state updates only once that confirmation reaches
  :cpp:func:`LifConversionTableModel::harmonicApplied`. The Harmonic
  column is read-only in the table proper for exactly this reason.
- **Assembly is snapshot-only and safe from any thread.** Every
  assembly helper builds a fresh :cpp:class:`SettingsStorage` per
  stage hardware key rather than dereferencing a live
  :cpp:class:`HardwareObject` — conversion stages are always
  ``d_threaded`` — so GUI code and data-layer code can call
  ``assembleLifConversion`` / ``assembleCurrentLifConversion`` directly
  without cross-thread machinery.
- **Prep hard-fails; connection-complete tolerates.** Same assembly
  call, different error policy — see *Data flow by moment* above. Do
  not conflate the two.
- **Identity = empty node list = no file.** An empty
  ``d_conversionNodes`` is the universal "no conversion configured"
  sentinel: :cpp:func:`LifConfig::hasConversion` is ``false``,
  :cpp:func:`LifConfig::writeTopologyFile` writes nothing (returns
  ``true`` without touching disk), :cpp:func:`LifConversion::assemble`
  returns the identity conversion without requiring a FINAL node, and
  the Python reader's ``has_topology`` is ``False`` for the same case.
- **The scan axis is the one deliberate exception to
  cm⁻¹-everywhere.** Every other value in the conversion pipeline is
  cm⁻¹ end to end, but ``LifConfig::d_laserPosStart``/``d_laserPosStep``
  are stored and gridded in the display ``BC::LifConv::LaserUnit`` (a
  uniform cm⁻¹ grid would round unevenly on a laser whose actuator
  resolution is native-unit-limited). :cpp:func:`LifConfig::currentLaserPos`
  is the single crossing point back into output-beam cm⁻¹ for hardware
  dispatch — see *The LIF scan model* above.

On-disk format
~~~~~~~~~~~~~~

:cpp:func:`LifConfig::writeTopologyFile` writes ``liftopology.csv``
(``BC::CSV::lifTopologyFile``) into the experiment directory from
:cpp:func:`Experiment::initialize`, one row per conversion node:
``Index, StageKey, Op, Harmonic, IsFinal, Input0, Input1, OutCoeffA,
OutCoeffB``. ``OutCoeffA``/``OutCoeffB`` are the node's resolved
affine output mapping (``output = OutCoeffA·fundamental + OutCoeffB``,
cm⁻¹) — derived data, recomputed from two evaluations of
:cpp:func:`LifConversion::stageOutput` at write time, and re-derived
by :cpp:func:`LifConversion::assemble` on read rather than trusted
from disk, mirroring ``RfConfig::loadClockSteps``. An input token is
self-describing: ``Fixed:<cm-1>`` is a
``BC::LifConv::RefType::Fixed`` input; a token matching another row's
``StageKey`` is a ``BC::LifConv::RefType::Stage`` input; anything else
is a ``BC::LifConv::RefType::Laser`` input, and the token itself is
the laser's hardware key (captured into the config's conversion laser
key on read). The identity case (no conversion nodes) writes nothing,
and a missing file reads back as identity — the same "no file = no
conversion" sentinel described in *Key invariants* above.
:cpp:class:`Experiment`'s disk constructor reads the file alongside
:cpp:func:`LifConfig::loadLifData`, so ``blackchirp-viewer`` recovers
the topology for free.

The Python reader and the shared fixture format are documented on
:doc:`/developer_guide/python_module` and :doc:`/python/bclif`; the
user-facing column reference lives on
:doc:`/user_guide/data_storage/lif`.

Acquisition flow
----------------

The LIF acquisition flow is a strict signal-based ping-pong between
:cpp:class:`AcquisitionManager` (on ``AcquisitionManagerThread``),
:cpp:class:`HardwareManager` (on ``HardwareManagerThread``), and the
two LIF hardware objects (each on its own ``"<hwKey>Thread"``).

.. mermaid::

   flowchart LR
       AM["AcquisitionManager<br/>(AM thread)"]
       HM["HardwareManager<br/>(HM thread)"]
       LL["LifLaser<br/>(hw thread)"]
       CS["LaserFreqConversionStage(s)<br/>(hw thread each)"]
       PG["PulseGenerator<br/>(hw thread)"]
       LS["LifDigitizer<br/>(hw thread)"]
       AM -- "nextLifPoint" --> HM
       HM -- "BlockingQueued<br/>setPosition" --> LL
       HM -- "Queued+future<br/>setLifConversionStages" --> CS
       HM -- "BlockingQueued<br/>setLifDelay" --> PG
       HM -- "lifSettingsComplete" --> AM
       LS -- "waveformRead" --> HM
       HM -- "lifDigitizerShotAcquired" --> AM

The signals on the diagram are the Qt connections installed by
:cpp:func:`MainWindow::MainWindow` whenever
:cpp:func:`ApplicationConfigManager::isLifEnabled` is true:

.. code-block:: cpp

   connect(p_hwm, &HardwareManager::lifSettingsComplete,
           p_am,  &AcquisitionManager::lifHardwareReady);
   connect(p_hwm, &HardwareManager::lifDigitizerShotAcquired,
           p_am,  &AcquisitionManager::processLifDigitizerShot);
   connect(p_am,  &AcquisitionManager::nextLifPoint,
           p_hwm, &HardwareManager::setLifParameters);

The handshake at each grid point runs in five steps:

#. **AM emits** ``nextLifPoint(currentDelay, currentLaserPos)``.
   :cpp:func:`AcquisitionManager::beginExperiment` fires the first
   one if the experiment has LIF enabled (after FTMW setup, when both
   objectives are active);
   :cpp:func:`AcquisitionManager::processLifDigitizerShot` fires every
   subsequent one. The signal lands queued on the HM.

#. **HM gates the digitizer and reprograms the laser, any conversion
   stages, and the pulse generator.**
   :cpp:func:`HardwareManager::setLifParameters` calls
   :cpp:func:`LifDigitizer::setAcquisitionGated` to suppress any
   in-flight waveform, then issues a blocking-queued
   :cpp:func:`LifLaser::setPosition` call, dispatches
   ``HardwareManager::setLifConversionStages`` to move every active
   :cpp:class:`LaserFreqConversionStage` in parallel (see *Frequency
   conversion* above), and issues blocking-queued
   :cpp:func:`PulseGenerator::setLifDelay` calls (one per active pulse
   generator) — each step runs only after the previous one succeeds.
   After all three return, the digitizer's pre-trigger buffer is
   flushed via :cpp:func:`LifDigitizer::flushAcquisitionBuffer` and the
   gate is released. The blocking-queued idiom — and, for the
   conversion stages, waiting on every per-stage future before
   proceeding — is what guarantees that no shot from the previous grid
   point can leak into the new one.

#. **HM emits** ``lifSettingsComplete(success)``.
   :cpp:func:`AcquisitionManager::lifHardwareReady` is the slot. On
   ``success == false`` the AM logs an error and aborts. On success
   it calls :cpp:func:`LifConfig::hwReady`, which clears the
   ``d_processingPaused`` flag inherited from
   :cpp:class:`ExperimentObjective`.

#. **The next laser shot triggers the digitizer.** The
   :cpp:class:`LifDigitizer` subclass reads its acquired waveform from the
   instrument and emits ``waveformRead(QVector<qint8>)``, which the HM
   has wired (in :cpp:func:`HardwareManager::storeConnection` on the
   ``LifDigitizer`` branch) to its own
   :cpp:func:`HardwareManager::lifDigitizerShotAcquired`. The HM signal
   relays the same ``QVector<qint8>`` to
   :cpp:func:`AcquisitionManager::processLifDigitizerShot`.

#. **AM accumulates and advances.**
   :cpp:func:`processLifDigitizerShot` checks that the AM is in the
   ``Acquiring`` state and that ``d_processingPaused`` is clear, then:

   - Calls :cpp:func:`LifConfig::addWaveform`, which constructs a
     :cpp:class:`LifTrace` from the bytes (using the cached
     :cpp:class:`LifDigitizerConfig` for sample-rate and y-multiplier
     scaling) and forwards it to
     :cpp:func:`LifStorage::addTrace` for accumulation in the current
     cell.
   - Emits ``lifPointUpdate()`` for the GUI.
   - Calls :cpp:func:`LifConfig::advance`. ``advance`` returns
     ``true`` when the current cell has reached its shot target,
     handles the random-shuffle if a delay sweep just completed,
     advances ``d_currentDelayIndex`` / ``d_currentLaserIndex``
     according to ``d_order``, and calls
     :cpp:func:`LifStorage::advance` to flush the just-completed
     cell to disk. When ``advance`` returns ``true`` and the
     experiment as a whole is not yet complete, the AM emits a fresh
     ``nextLifPoint`` for the new cursor — the loop returns to step 1.
   - Emits ``lifShotAcquired(perMilComplete)`` to drive the main
     window's LIF progress bar.

The gate inside :cpp:func:`processLifDigitizerShot` —
``d_processingPaused`` — is what protects step 4's
``waveformRead`` from being mis-attributed to the previous grid
point: the AM ignores any shot that arrives before the matching
``lifHardwareReady(success)`` has cleared the flag. The blocking
queue inside :cpp:func:`HardwareManager::setLifParameters` makes the
flag's lifetime well-defined — the HM only emits
``lifSettingsComplete`` after both the laser move and the pulse-delay
write have returned.

Completion is the same predicate the FTMW path uses:
:cpp:func:`AcquisitionManager::checkComplete` consults
:cpp:func:`Experiment::isComplete`, which is true when every
enabled objective reports complete.
:cpp:func:`LifConfig::isComplete` returns ``true`` once the first
full sweep has finished (``d_complete = true`` set inside
:cpp:func:`advance`); under
:cpp:enum:`LifConfig::ContinueAveraging` the objective also reports
``indefinite() == true`` once ``perMilComplete()`` reaches 1000, so
the experiment continues until the user aborts. Under
:cpp:enum:`LifConfig::StopWhenComplete` the experiment finishes the
moment the first sweep ends; subsequent waveforms are dropped on the
floor by the early-return guard in :cpp:func:`LifConfig::addWaveform`.

Storage: the 2D grid and its flattening
---------------------------------------

:cpp:class:`LifStorage` extends :cpp:class:`DataStorageBase` and is
shared between the AM (writer) and the GUI (reader) via
``std::shared_ptr``. The grid dimensions are fixed at construction:
``d_delayPoints`` rows × ``d_laserPoints`` columns. Cells are
identified by a ``(delayIndex, laserIndex)`` pair.

Internally, :cpp:class:`LifStorage` keeps a ``std::map<int, LifTrace>``
keyed by a single flat integer. The flattening convention is
**laser-fastest, row-major**:

.. code-block:: cpp

   int LifStorage::index(int dp, int lp) const
   {
       return dp * d_laserPoints + lp;
   }

so iterating ``index`` from 0 upward visits all laser positions for
``delayIndex = 0`` before stepping ``delayIndex`` to 1. The same
integer is used as the on-disk filename stem (``lif/<index>.csv``),
which means the on-disk file ordering is **independent of the
``LifScanOrder``** chosen for the live traversal. ``DelayFirst`` and
``LaserFirst`` differ only in the order cells are visited and saved;
the file layout is identical.

The illustration below shows the cell visit order for a 3×4 grid
under each scan order; the bracketed numbers are the flat
``index`` values written to disk:

.. code-block:: text

   Grid layout (delay rows × laser columns)
   --------------------------------------
                   laser →
                   [0]   [1]   [2]   [3]
   delay [0]        0     1     2     3
   delay [1]        4     5     6     7
   delay [2]        8     9    10    11

   DelayFirst visit order:
       (0,0) (1,0) (2,0)   →   advance laser
       (0,1) (1,1) (2,1)   →   advance laser
       ...

   LaserFirst visit order:
       (0,0) (0,1) (0,2) (0,3)   →   advance delay
       (1,0) (1,1) (1,2) (1,3)   →   advance delay
       ...

Either order writes to the same flat index for a given
``(delayIndex, laserIndex)`` cell.

Inside the storage, three structures coexist:

- ``d_currentTrace`` — the trace currently being accumulated. The AM
  feeds shots into it via :cpp:func:`LifStorage::addTrace`.
- ``d_data`` — the completed-cell map. Cells move from
  ``d_currentTrace`` into ``d_data`` when
  :cpp:func:`LifStorage::advance` (and its internal
  :cpp:func:`save`) flushes the cell to disk.
- ``d_nextNew`` — a one-bit state flag. After ``advance``, the next
  call to :cpp:func:`addTrace` re-seeds ``d_currentTrace`` from
  ``d_data`` (if the cell already has prior shots, e.g. on a
  ``ContinueAveraging`` re-sweep) or installs the incoming trace as
  the seed.

The mutex on :cpp:class:`DataStorageBase` (``pu_mutex``) coordinates
the AM writer with GUI readers exactly as it does for FTMW: every
read or write takes the lock; readers ask for ``LifStorage`` results
by value, so they get a stable snapshot. The cost is one
:cpp:class:`LifTrace` copy per refresh — bounded, because
:cpp:class:`LifTrace`'s payload is wrapped in
``QSharedDataPointer<LifTraceData>``, so the copy is shallow until a
mutator modifies the data.

Processing-gate persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integration gate and waveform-filter parameters live in
``d_procSettings`` on :cpp:class:`LifConfig` and are persisted
separately from the trace files. When the experiment starts,
:cpp:func:`LifConfig::initialize` calls
:cpp:func:`LifStorage::writeProcessingSettings` to write the active
``LifTrace::LifProcSettings`` into ``lif/processing.csv`` (under the
keys declared in ``BC::Key::LifStorage`` —
``LifGateStartPoint``, ``LifGateEndPoint``, ``RefGateStartPoint``,
``RefGateEndPoint``, ``LowPassAlpha``, ``SavGolEnabled``,
``SavGolWindow``, ``SavGolPoly``). When an experiment is reopened,
:cpp:func:`LifConfig::loadLifData` reads the same file via
:cpp:func:`LifStorage::readProcessingSettings` and pushes the result
back onto ``d_procSettings``, so the viewer reproduces the same
spectrum without re-asking the user. The user-facing semantics of
each key are documented in :doc:`/user_guide/data_storage/lif`.

The recorded trace files on disk are never modified by the
processing-gate workflow. **Save** in the LIF tab's processing panel
overwrites only ``processing.csv``; **Reprocess All** re-integrates
the in-memory traces using the new gate but does not touch the
``lif/<index>.csv`` files. The trace-file format itself is
documented on :doc:`/user_guide/data_storage/lif`.

There is no LIF analog to :cpp:class:`FidPeakUpStorage`. A
"LIF peak-up" mode is implicit in :cpp:class:`LifControlWidget`'s
**Hardware → LIF Configuration** dialog: that dialog runs a live
:cpp:class:`LifTracePlot` rolling-average against fresh
``waveformRead`` shots without ever constructing a
:cpp:class:`LifStorage`, because the user is verifying gating and
laser alignment, not acquiring a scan. Production data is acquired
only inside an experiment, which always uses
:cpp:class:`LifStorage`.

Visualization
-------------

The LIF tab is implemented by :cpp:class:`LifDisplayWidget`
(``gui/lif/gui/lifdisplaywidget.{cpp,h}``). It hosts four plot areas
plus the processing panel:

- **LifTracePlot** — the most-recently-acquired raw trace as a
  function of sample time. The shaded zones show the LIF and
  reference integration gates. Drives the visual confirmation that
  the gate covers the fluorescence pulse correctly.
- **LifSlicePlot (delay slice)** — integrated LIF signal versus
  delay, at the laser-position column selected on the spectrogram.
- **LifSlicePlot (laser slice)** — integrated LIF signal versus
  laser position, at the delay row selected on the spectrogram.
- **LifSpectrogramPlot** — the full 2D map. Implemented as a Qwt
  ``QwtPlotSpectrogram`` driven by a ``QwtMatrixRasterData``. The
  matrix is laid out laser-fastest (``setValueMatrix(specDat,
  d_laserPosPoints)``), matching the storage's flattening convention
  so the matrix index can be computed as ``li + di * lp``.

The widget consumes the storage in two passes:

#. **Per-shot integration.** When the AM emits ``lifPointUpdate``,
   :cpp:func:`LifDisplayWidget::updatePoint` reads
   :cpp:func:`LifStorage::currentLifTrace`, integrates it with the
   active ``LifProcSettings``, and writes the integrated value into
   ``d_currentIntegratedData[li + di * lp]``. It also tells the
   spectrogram which cell is "live" so the live-cursor markers track
   the acquisition.
#. **Periodic redraw.** A :cpp:func:`QObject::startTimer` (interval
   from the **Refresh Interval** spin box, default 500 ms, persisted
   under ``BC::Key::LifDW::refresh``) ticks
   :cpp:func:`LifDisplayWidget::updatePlots`, which pushes
   ``d_currentIntegratedData`` into
   :cpp:func:`LifSpectrogramPlot::updateData` and refreshes the two
   slice plots and the trace plot. Decoupling per-shot integration
   from per-tick redraw keeps the AM thread responsive and bounds the
   GUI's draw cost.

When the user drags a cursor on the spectrogram or invokes the
right-click "Move cursor here" menu, the spectrogram emits
``laserSlice(int delayIndex)`` or ``delaySlice(int laserIndex)``;
the widget translates those into a fresh
:cpp:func:`LifStorage::getLifTrace` call to feed the trace plot and
into pre-computed slices through ``d_currentIntegratedData``. The
slice helpers ``laserSlice`` and ``delaySlice`` walk the same flat
buffer along the appropriate stride — laser slice steps by 1 inside a
fixed delay row, delay slice steps by ``d_laserPoints`` between rows.

Reverse-step axes
~~~~~~~~~~~~~~~~~

The user is allowed to configure a negative ``d_delayStepUs`` or
``d_laserPosStep`` (a high-to-low scan). Storage indices remain
0-based and ascending, so the display widget keeps two boolean
flags — ``d_delayReverse`` and ``d_laserReverse`` — and applies an
``index → (size - 1 - index)`` flip whenever it converts between a
storage index and a display index. The spectrogram itself always
draws with monotonic axes.

Reprocessing and processing settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The processing panel is :cpp:class:`LifProcessingWidget`
(``gui/lif/gui/lifprocessingwidget.{cpp,h}``). It owns spin boxes
for the LIF and reference gate bounds, a :math:`\\alpha` spin for the
single-pole IIR low-pass filter, an enable check plus window/order
spins for the Savitzky-Golay filter, and three buttons:

- **Reprocess All** triggers
  :cpp:func:`LifDisplayWidget::reprocess`, which re-integrates every
  ``(di, li)`` cell in storage with the current ``LifProcSettings``.
  The grid walk runs on a ``QtConcurrent`` worker tracked by a
  ``QFutureWatcher<QVector<double>>`` (``p_reprocessWatcher``), not
  on the UI thread: a 50×50 sweep would otherwise block the GUI for
  seconds. A modal ``QProgressDialog`` (200 ms minimum duration)
  reports progress and a Cancel button flips
  ``QPromise::isCanceled()`` so the worker bails between traces. The
  watcher's ``finished`` slot writes the result into
  ``d_currentIntegratedData`` and redraws on the main thread; a
  cancelled pass leaves the spectrogram empty. A re-entrancy guard
  ignores a second trigger while a pass is still in flight. The
  per-cell parameter lookup is O(1) because :cpp:class:`LifStorage`
  caches the parsed ``lifparams.csv`` on first trace load.
- **Reset** triggers :cpp:func:`LifDisplayWidget::resetProc`, which
  re-reads ``processing.csv`` via
  :cpp:func:`LifStorage::readProcessingSettings` and pushes the
  on-disk values back into the spin boxes.
- **Save** triggers :cpp:func:`LifDisplayWidget::saveProc`, which
  calls :cpp:func:`LifStorage::writeProcessingSettings` with the
  current :cpp:func:`LifProcessingWidget::getSettings`. This is the
  only path that mutates ``processing.csv``; the trace files are
  never rewritten.

The ``LifTracePlot`` instance hosted on the same tab subscribes to
:cpp:func:`LifProcessingWidget::settingChanged` and re-draws its
gate zones whenever the user nudges any of the spin boxes, so the
trace view tracks the panel without going through ``Reprocess
All``.

The user-facing behavior of the LIF tab — what each control does,
how the cursors interact with the slices — is documented in
:doc:`/user_guide/lif/lif_tab`; this page covers only the data flow
and the class wiring.

Configuration UI
----------------

The wizard exposes LIF in two places. They are separate pages because
they serve different concerns: the first defines the *experiment*
(scan grid plus traversal options); the second defines the *hardware
configuration* used to acquire it (digitizer settings, shots per
point, processing gate).

#. **Experiment-type page** —
   :cpp:class:`ExperimentTypePage`
   (``gui/expsetup/experimenttypepage.{cpp,h}``) is the wizard's first
   page and contains the FTMW group plus, when the LIF module is
   enabled in :cpp:class:`ApplicationConfigManager`, an LIF group.
   The LIF group hosts the **Delay** panel (start / step / points /
   read-only end), the **Laser** panel (start / step / points /
   read-only end, in the laser's display unit), and an **Options**
   panel (scan order, complete mode, auto-disable-flashlamp checkbox,
   randomize-delay checkbox). The Laser panel's bounds are not simply
   the laser hardware's native range: ``ExperimentTypePage::updateLifLaserBounds``
   passes the laser's native cm⁻¹ range through the experiment's
   current frequency-conversion topology
   (:cpp:func:`LifConversion::outputRange`) before converting to the
   display unit, so the spin boxes show the *excitation-beam* range
   reaching the sample rather than the laser's own tuning range — see
   *Frequency conversion* above.
   :cpp:func:`ExperimentTypePage::apply` writes every one of the
   panel's values onto the experiment's :cpp:class:`LifConfig`, already
   in the display unit (no cm⁻¹ conversion happens at this boundary —
   see the scan-axis exception in *Frequency conversion*):
   ``d_delayStartUs``, ``d_delayStepUs``, ``d_delayPoints``,
   ``d_laserPosStart``, ``d_laserPosStep``, ``d_laserPosPoints``,
   ``d_completeMode``, ``d_order``, ``d_disableFlashlamp``,
   ``d_delayRandom``.

#. **LIF configuration page** — ``ExperimentLifConfigPage``
   (``gui/lif/gui/experimentlifconfigpage.{cpp,h}``) is the per-LIF
   wizard page. It hosts a ``QTabWidget`` with two tabs:

   - **Acquisition** wraps a :cpp:class:`LifControlWidget`, shared with
     the live **Hardware → LIF Configuration** dialog (see
     :cpp:func:`MainWindow::launchLifConfigDialog`), hosting: the live
     :cpp:class:`LifTracePlot`, a :cpp:class:`DigitizerConfigWidget`
     keyed against the active :cpp:class:`LifDigitizer`, the laser
     control (:cpp:class:`LifLaserWidget`), a shots-per-point spin,
     :cpp:class:`LifProcessingWidget`, and Start / Stop / Reset
     buttons. :cpp:func:`LifControlWidget::toConfig` writes
     ``d_shotsPerPoint``, ``d_procSettings``, and the digitizer
     configuration onto :cpp:class:`LifConfig`;
     :cpp:func:`LifControlWidget::setFromConfig` is the inverse for
     loading a saved experiment.
   - **Conversion** wraps a :cpp:class:`LifConversionWidget`, with its
     Delete-preset button hidden — the wizard is the only place this
     widget is instantiated (with ``showDeleteButton = false``) — for
     editing the experiment's frequency-conversion topology; see
     *Frequency conversion* above for the widget's internals.

   ``ExperimentLifConfigPage::apply`` runs both tabs' ``toConfig`` on
   wizard **Next**. Accepting the wizard additionally calls
   ``ExperimentLifConfigPage::commitLifPreset`` from
   ``ExperimentSetupDialog::accept``, which — only when the Conversion
   tab has unsaved edits — prompts to overwrite the active LIF preset,
   save the edits as a new preset, or proceed without saving,
   mirroring ``ExperimentFtmwConfigPage::commitFtmwPreset``.

The Hardware → LIF Configuration dialog uses the same
:cpp:class:`LifControlWidget` outside of an experiment to drive the
laser and digitizer manually. The **Start Acquisition** button emits
``startSignal(LifConfig)`` which the HM converts to
:cpp:func:`HardwareManager::startLifConfigAcq`; incoming
``lifDigitizerShotAcquired`` shots flow into the embedded
:cpp:class:`LifTracePlot` for live alignment work, with no
:cpp:class:`LifStorage` constructed.

The status box on the main hardware status panel
(:cpp:class:`LifLaserStatusBox`) listens to
``lifLaserPosUpdate(double)`` and ``lifLaserFlashlampUpdate(bool)``
on the HM, which the HM in turn forwards from its connection to the
:cpp:class:`LifLaser`'s ``laserPosUpdate`` and
``laserFlashlampUpdate`` signals.

Pointers
--------

**LIF storage layout, on-disk format, base-36 encoding, and
``processing.csv`` semantics.** See :doc:`/user_guide/data_storage/lif`
and :doc:`/classes/lifstorage`.

**Per-class API contracts.** :doc:`/classes/lifconfig`,
:doc:`/classes/lifstorage`, :doc:`/classes/datastoragebase`,
:doc:`/classes/acquisitionmanager`. The LIF plot widgets do not
currently have dedicated API pages; they are documented inline on
this page only.

**Frequency-conversion API contracts and user-facing workflow.**
:doc:`/classes/lifconversion`, :doc:`/classes/lifconversionsnapshot`,
:doc:`/classes/laserfreqconversionstage`, :doc:`/classes/loadoutmanager`,
:doc:`/classes/hardwareloadout`. The *gated setting* pattern shared
with FTMW clock configuration is on
:doc:`/developer_guide/hardware_configuration`. The user-facing
Conversion tab and LIF preset bar are documented on
:doc:`/user_guide/lif/conversion` and :doc:`/user_guide/lif/presets`;
the ``liftopology.csv`` column reference is on
:doc:`/user_guide/data_storage/lif`.

**The cross-manager experiment lifecycle that surrounds the LIF
loop** — wizard apply, hardware initialization, and the
``experimentInitialized`` → ``beginAcquisition`` →
``experimentComplete`` round-trip — is on
:doc:`/developer_guide/experiment_lifecycle`. The LIF-specific path
covered here begins after ``beginAcquisition`` arrives at the AM and
ends when :cpp:func:`AcquisitionManager::checkComplete` reports
completion.
