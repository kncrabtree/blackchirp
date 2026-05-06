.. index::
   single: adding an experiment mode
   single: FtmwType; new mode
   single: FtmwConfig; subclass recipe
   single: BatchManager; subclass recipe
   single: BatchType; new value
   single: ExperimentTypePage; wiring
   single: BatchSequenceDialog; analog
   single: MainWindow; startBatch
   single: createStorage; FtmwConfig
   single: beginNextExperiment; override
   single: experiment mode; shot vs wall-clock vs indefinite
   single: SettingsStorage; batch configuration

Adding an Experiment Mode
=========================

Two extension points control how Blackchirp *runs* an acquisition: the
``FtmwType`` enum (chosen on the experiment-setup wizard) decides how a
single FTMW experiment terminates and whether it traverses multiple
segments, and the :cpp:class:`BatchManager` hierarchy decides how
many experiments make up a single user "Acquire" action and what
schedules them. This page is one recipe per extension point. The two
sections are independent — adding a new ``FtmwType`` does not touch
:cpp:class:`BatchManager`, and a new :cpp:class:`BatchManager`
subclass works with every existing ``FtmwType`` — but they share the
same shape (a small enum, a concrete subclass, a wizard or dialog
wiring, an API-page extension), so they are presented together. The
cross-manager experiment-lifecycle flow they both plug into is on
:doc:`/developer_guide/experiment_lifecycle`.

.. note::

   Larger additions that step outside these two extension points —
   a new ``ExperimentObjective`` peer alongside :cpp:class:`FtmwConfig`
   and :cpp:class:`LifConfig` (for example, a new spectroscopy modality
   that does not fit either), or any other change that touches the
   :cpp:class:`Experiment` aggregate, the :cpp:class:`AcquisitionManager`
   loop, or the persistence layer — go beyond the scope of this page.
   Those are coordinated multi-subsystem changes; please open a
   discussion on the Blackchirp Discord or a tracking issue on the
   GitHub issues board so the design can be sketched before
   implementation begins.

Section A — A new FtmwType
--------------------------

When to add a new type
''''''''''''''''''''''

The existing values of ``FtmwConfig::FtmwType`` are
``Target_Shots``, ``Target_Duration``, ``Forever``, ``Peak_Up``,
``LO_Scan``, and ``DR_Scan``. Each is the combination of a
*completion criterion* and a *segment-traversal pattern*. The
existing six divide along those two axes:

.. list-table::
   :header-rows: 1
   :widths: 18 30 22 30

   * - ``FtmwType``
     - Completion criterion
     - Segments
     - Concrete subclass
   * - ``Target_Shots``
     - Accumulated shot count reaches ``d_objective``.
     - Single
     - :cpp:class:`FtmwConfigSingle`
   * - ``Target_Duration``
     - Wall clock reaches the recorded target time.
     - Single
     - :cpp:class:`FtmwConfigDuration`
   * - ``Forever``
     - Never; ``indefinite()`` returns ``true``.
     - Single
     - :cpp:class:`FtmwConfigForever`
   * - ``Peak_Up``
     - Never; rolling average runs until the user stops.
     - Single (transient)
     - :cpp:class:`FtmwConfigPeakUp`
   * - ``LO_Scan``
     - All LO sweep steps and ``d_targetSweeps`` complete.
     - Multi
     - :cpp:class:`FtmwConfigLOScan`
   * - ``DR_Scan``
     - All DR scan steps complete one full sweep.
     - Multi
     - :cpp:class:`FtmwConfigDRScan`

The threshold for adding a new type rather than parameterizing an
existing one: if the new mode's completion logic can be expressed by
tuning ``d_objective`` on an existing subclass, parameterize. If it
requires a new ``_init()`` / ``createStorage()`` / ``isComplete()`` /
``perMilComplete()`` implementation, or a new segment-traversal
pattern, add a new type. Cross-link policies (a "scan a parameter
through values" mode that re-runs an entire experiment per parameter
value) belong on the :cpp:class:`BatchManager` side instead — see
*Section B*.

The touches: enum, subclass, factory, wizard, API page
''''''''''''''''''''''''''''''''''''''''''''''''''''''

A new ``FtmwType`` is a five-touch change. The first three live in
``data/experiment/``; the last two live in ``gui/expsetup/`` and
``doc/source/classes/``.

1. **Add the enumerator.** Append the new value to
   ``FtmwConfig::FtmwType`` in
   ``src/data/experiment/ftmwconfig.h``. The enum carries
   ``Q_ENUM(FtmwType)``; the wizard's ``QComboBox`` and Qt's
   metaobject system pick the new value up automatically. The
   serialized header field that stores the mode
   (``BC::Store::FTMW::ftType``) maps to the enumerator's integer
   value, so do not reorder existing enumerators when appending —
   reordering breaks experiment header round-trips.

2. **Add a concrete subclass.** Declare ``FtmwConfigMyMode`` in
   ``src/data/experiment/ftmwconfigtypes.h`` next to the existing
   six and implement it in the matching ``.cpp`` file. The subclass
   inherits :cpp:class:`FtmwConfig` and overrides:

   - :cpp:func:`FtmwConfig::_init` — initialize mode-specific state at
     acquisition start. For multi-segment modes, this is where you
     populate ``d_rfConfig`` with the segment list using the
     :cpp:class:`RfConfig` helpers (see :doc:`/classes/rfconfig`); for
     wall-clock modes, this is where you record the start time and
     compute the target time.
   - :cpp:func:`FtmwConfig::_prepareToSave` and
     :cpp:func:`FtmwConfig::_loadComplete` — header round-trip
     serialization for any mode-specific scalars. Use the
     ``HeaderStorage::store`` / ``HeaderStorage::retrieve`` helpers
     against keys declared in a per-mode ``BC::Store::FtmwMyMode``
     namespace, in the same style as ``BC::Store::FtmwLO`` and
     ``BC::Store::FtmwDR`` in ``ftmwconfigtypes.h``.
   - :cpp:func:`FtmwConfig::createStorage` — return a shared pointer
     to the right :cpp:class:`FidStorageBase` subclass for the mode.
     Pick from the three existing storage classes; see *Storage
     choice* below.
   - :cpp:func:`FtmwConfig::isComplete` — completion predicate.
   - :cpp:func:`FtmwConfig::perMilComplete` — progress in per-mille
     (0–1000); the value drives the GUI progress bar.
   - :cpp:func:`FtmwConfig::completedShots` — total shot count for
     progress reporting and the per-experiment ``Ftmw/Shots`` aux
     reading.

   Optional overrides:

   - :cpp:func:`FtmwConfig::indefinite` — return ``true`` to suppress
     the standard completion check; ``Forever`` is the only existing
     user.
   - :cpp:func:`FtmwConfig::bitShift` — return a non-zero shift to
     widen the rolling-average accumulator; ``Peak_Up`` returns 8 so
     each ADC sample is multiplied by 256 before accumulation.
   - :cpp:func:`FtmwConfig::advance` — multi-segment modes override
     to step the segment cursor and return ``true`` when a segment
     boundary was crossed. The drain-loop / flush-marker mechanics
     that drive ``advance()`` are documented in
     :doc:`/developer_guide/ftmw_acquisition`.

   Provide both constructors used by the existing concretes — one
   that takes the FTMW scope's hardware key, and one that takes a
   ``const FtmwConfig &`` for the deserialization path that constructs
   a typed subclass from a base-class value object.

3. **Update the factory.** :cpp:func:`Experiment::enableFtmw` in
   ``src/data/experiment/experiment.cpp`` is the single dispatch from
   ``FtmwType`` to a concrete subclass. Add a ``case`` for the new
   enumerator that constructs the new subclass with the discovered
   FTMW scope key, sets ``ps_ftmwConfig->d_type`` (the trailing
   assignment after the switch already handles this), and inserts it
   into ``d_objectives``. There is no second factory site;
   :cpp:class:`FtmwConfig`'s deserialization path constructs a
   ``FtmwConfigSingle`` from a header round-trip and then narrows it,
   so a header that records the new ``FtmwType`` will round-trip
   correctly as long as the factory case exists.

4. **Wire the type into the experiment-setup wizard.**
   :cpp:class:`ExperimentTypePage`
   (``src/gui/expsetup/experimenttypepage.{cpp,h}``) is the entry
   point. The constructor populates the ``Type`` ``QComboBox`` from
   the metaobject ``FtmwType`` enum, so a new enumerator is listed
   automatically. Two pieces still need editing:

   - The ``QStackedWidget`` page selector
     (:cpp:func:`ExperimentTypePage::configureUI`) routes each
     ``FtmwType`` to the widget that exposes its mode-specific
     parameters — the shots spinner, the duration spinner, an empty
     placeholder for ``Forever``, or a richer widget for
     ``LO_Scan`` / ``DR_Scan``. Add a ``case`` for the new mode and
     either route it to one of the existing widgets or construct a
     new ``MyModeConfigWidget``. ``LOScanConfigWidget`` and
     ``DRScanConfigWidget`` (in ``src/gui/expsetup/``) are the
     models for a richer per-mode page; both derive from
     :cpp:class:`ExperimentConfigPage` so they participate in the
     wizard's setting-storage round-trip.
   - :cpp:func:`ExperimentTypePage::apply` constructs the
     ``FtmwConfig`` by calling :cpp:func:`Experiment::enableFtmw` and
     populating ``d_objective``. Add a ``case`` for the new mode that
     reads its mode-specific spinner and sets ``d_objective``
     appropriately, and call your new config widget's ``apply()``
     hook if it has parameters of its own.

   The wizard's page-ordering logic
   (``ExperimentSetupDialog::pageVisited`` and friends in
   ``src/gui/expsetup/experimentsetupdialog.{cpp,h}``) walks a fixed
   set of pages in sequence; a mode that needs an additional standalone
   wizard page after the type page should insert that page into the
   ordering there as well.

5. **Extend the API page.** ``doc/source/classes/ftmwconfig.rst``
   already lists ``.. doxygenclass::`` directives for each of the six
   concrete subclasses on the same page. Append a new
   ``.. doxygenclass:: FtmwConfigMyMode`` block in the same style;
   if the new subclass introduces a per-mode keys namespace
   (analogous to ``BC::Store::FtmwLO``), let the Doxygen comments on
   the namespace surface through the page rather than duplicating
   them in prose. The Doxygen-comment style contract that the API
   ref enforces is :ref:`api-reference-style`.

Multi-segment vs. single-segment design
'''''''''''''''''''''''''''''''''''''''

The single-vs-multi axis decides which storage class to construct
and whether ``advance()`` does any work beyond the autosave hook
that the base class already provides.

- **Single-segment modes** (``Target_Shots``, ``Target_Duration``,
  ``Forever``, ``Peak_Up``) accumulate into one segment for the
  entire acquisition. The storage class is :cpp:class:`FidSingleStorage`
  (or :cpp:class:`FidPeakUpStorage` for the no-disk peak-up mode).
  ``advance()`` keeps the base-class behavior — periodic autosave
  driven by the experiment's ``d_backupIntervalMinutes`` — and does
  not return ``true`` for a segment boundary.
- **Multi-segment modes** (``LO_Scan``, ``DR_Scan``) populate
  ``d_rfConfig`` in ``_init()`` with a list of segments and use
  :cpp:class:`FidMultiStorage`, which stores each segment's FID
  data under a separate ``fid/<i>.csv`` file. The drain loop in
  :cpp:class:`AcquisitionManager` periodically calls
  ``FtmwConfig::advance()``; when ``advance()`` returns ``true`` the
  AM emits :cpp:func:`AcquisitionManager::newClockSettings` carrying
  the next segment's clock list, and the
  ``setAcquisitionGated`` + flush-marker round-trip described in
  :doc:`/developer_guide/ftmw_acquisition` quiesces the digitizer
  while the new clocks are programmed. ``LO_Scan`` and ``DR_Scan``
  are the canonical examples to model from; they extend
  :cpp:func:`FtmwConfig::createStorage` to call
  ``FidMultiStorage::setNumSegments`` after construction.

A multi-segment mode also typically writes a backup at every segment
boundary; the user-guide notes for ``LO_Scan`` and ``DR_Scan``
mention that the ``Backup Interval`` setting therefore has no effect
for those modes.

Completion: shot-based, wall-clock, indefinite
''''''''''''''''''''''''''''''''''''''''''''''

Three patterns cover every existing mode:

- **Shot-based.** :cpp:func:`FtmwConfig::isComplete` compares
  ``completedShots()`` against ``d_objective``;
  :cpp:func:`FtmwConfig::perMilComplete` returns
  ``1000 * completedShots() / d_objective``. The user-supplied target
  shot count is collected by :cpp:class:`ExperimentTypePage` from the
  ``Shots`` spinner and assigned to ``d_objective`` in
  :cpp:func:`ExperimentTypePage::apply`. ``Target_Shots`` is the
  single-segment example; ``LO_Scan`` and ``DR_Scan`` use the
  same ``d_objective`` field as a per-segment target and derive
  total progress from the :cpp:class:`RfConfig`'s segment counts.
- **Wall-clock.** Record the start time in ``_init()`` and compute
  the target time from ``d_objective`` (the units are mode-specific —
  ``Target_Duration`` uses minutes). :cpp:func:`FtmwConfig::isComplete`
  compares ``QDateTime::currentDateTime()`` against the target;
  :cpp:func:`FtmwConfig::perMilComplete` interpolates the elapsed
  fraction. ``Target_Duration`` is the canonical example.
  ``_prepareToSave()`` and ``_loadComplete()`` round-trip
  ``d_objective`` so the recorded duration is preserved on disk.
- **Indefinite.** :cpp:func:`FtmwConfig::indefinite` returns ``true``
  and :cpp:func:`FtmwConfig::isComplete` always returns ``false``.
  The :cpp:class:`AcquisitionManager` completion check skips the
  experiment, and the user must stop acquisition with the abort
  button. ``Forever`` is the existing example; ``Peak_Up`` is a
  variant that is also indefinite but tracks shots toward
  ``d_objective`` for progress display.

Storage choice
''''''''''''''

Pick the :cpp:class:`FidStorageBase` subclass that matches the
mode's segment shape and persistence requirements:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Mode shape
     - Storage class
     - Existing modes
   * - Single segment, on-disk
     - :cpp:class:`FidSingleStorage`
     - ``Target_Shots``, ``Target_Duration``, ``Forever``
   * - Single segment, transient (no disk I/O)
     - :cpp:class:`FidPeakUpStorage`
     - ``Peak_Up``
   * - Multi-segment, on-disk
     - :cpp:class:`FidMultiStorage`
     - ``LO_Scan``, ``DR_Scan``

A new mode picks whichever fits. If none fit — for example, a mode
that needs a non-FID raw-data accumulator, or per-segment files with
a layout the existing classes do not produce — consider whether a
new :cpp:class:`FidStorageBase` subclass is justified. That is rare,
and lives in ``src/data/storage/`` rather than this recipe's scope;
the :cpp:class:`DataStorageBase` lifecycle a new storage class
plugs into is documented on :doc:`/developer_guide/persistence`,
and the FTMW-specific pipeline that writes through it is on
:doc:`/developer_guide/ftmw_acquisition`.

Section B — A new BatchManager subclass
---------------------------------------

When to add a new subclass
''''''''''''''''''''''''''

Two concrete subclasses ship today:

.. list-table::
   :header-rows: 1
   :widths: 28 24 48

   * - Class
     - ``BatchType`` value
     - Policy
   * - :cpp:class:`BatchSingle`
     - ``SingleExperiment``
     - Run one experiment, then end the batch.
   * - :cpp:class:`BatchSequence`
     - ``Sequence``
     - Repeat one experiment template on a fixed interval until
       the configured count is reached or the user aborts.

The threshold for a new subclass: any policy that cannot be
expressed by varying the *interval* or *count* on
:cpp:class:`BatchSequence`. Examples that would justify a new
subclass: an "until N successful experiments" policy that filters
out aborted runs, a "scan a parameter through values" policy that
mutates each cloned experiment before launching it, an
externally-triggered "run on cue" policy that waits on a TCP
notification rather than a timer.

The touches: enum, subclass, dialog, MainWindow entry, API page
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

A new :cpp:class:`BatchManager` subclass is also a five-touch
change.

1. **Add the enumerator.** Append to ``BatchManager::BatchType`` in
   ``src/acquisition/batch/batchmanager.h``. The enumerator is
   passed to the base constructor and stored in ``d_type`` for
   downstream code that branches on the active batch type.
2. **Subclass** :cpp:class:`BatchManager` in
   ``src/acquisition/batch/<myname>.{cpp,h}``. The base class
   declares five pure virtuals; implement all of them. Each is
   documented in :doc:`/classes/batchmanager`; the per-method
   contracts in summary:

   - :cpp:func:`BatchManager::currentExperiment` — return the active
     experiment shared pointer. Must never be null while the batch
     is running. Called by the main window when emitting
     ``beginExperiment()`` and by
     :cpp:func:`BatchManager::experimentComplete` to inspect the
     just-finished experiment.
   - :cpp:func:`BatchManager::isComplete` — return ``true`` when no
     further experiments remain. Evaluated by
     :cpp:func:`BatchManager::experimentComplete` after
     :cpp:func:`BatchManager::processExperiment` returns.
   - :cpp:func:`BatchManager::abort` — mark the batch as complete so
     ``isComplete()`` returns ``true`` and release any pending
     timers or queued work. The user's abort button connects
     directly to this slot.
   - :cpp:func:`BatchManager::processExperiment` — post-acquisition
     bookkeeping for the experiment that just finished. May be a
     no-op (``BatchSingle`` flips its ``d_complete`` flag here;
     ``BatchSequence`` increments its count).
   - :cpp:func:`BatchManager::writeReport` — write any per-batch
     summary report. May also be a no-op; both shipping subclasses
     leave it empty (see *Persistence* below).

   Optional override:

   - :cpp:func:`BatchManager::beginNextExperiment` — defaults to
     emitting ``beginExperiment()`` immediately. Override when the
     next experiment should not start right away — for example
     :cpp:class:`BatchSequence` arms a single-shot ``QTimer`` and
     emits ``beginExperiment()`` from the timer's ``timeout`` lambda
     after cloning a fresh :cpp:class:`Experiment` from the
     template.
3. **Wire a configuration dialog.** Model on
   ``BatchSequenceDialog`` in ``src/gui/dialog/``. The dialog's
   responsibility is to collect the parameters the new policy needs
   (count, interval, value list, trigger source — whatever applies)
   from the user and to remember the last-used values via
   :cpp:class:`SettingsStorage`; the keys ``BC::Key::SeqDialog::key``
   and the ``numExpts`` / ``interval`` keys in
   ``batchsequencedialog.h`` are the convention to follow. Existing
   dialogs distinguish a *Quick* path (re-use a previous experiment)
   from a *Configure* path (run the full setup wizard); replicate
   that pattern only if the policy reasonably supports both.
4. **Add a MainWindow entry point.** Add a new menu action in the
   ``Acquire`` menu (the menu construction lives in
   :cpp:func:`MainWindow::MainWindow`) and connect it to a new
   :cpp:class:`MainWindow` slot that opens the dialog, builds the
   :cpp:class:`Experiment` (via :cpp:func:`MainWindow::createExperiment`
   and the experiment wizard, or via the quick-experiment dialog),
   constructs the new :cpp:class:`BatchManager` subclass, and calls
   :cpp:func:`MainWindow::startBatch`. :cpp:func:`MainWindow::startSequence`
   is the closest model — its branch structure is the same shape
   any new entry point will need.
5. **Extend the API page.** Add a ``.. doxygenclass:: MyBatch``
   directive to ``doc/source/classes/batchmanager.rst`` next to the
   existing ``BatchManager`` block. If the policy introduces a
   non-obvious lifecycle wrinkle, append a paragraph to the
   *Subclassing guide* section on that page; otherwise the Doxygen
   comments are sufficient.

Building the wiring
'''''''''''''''''''

The five overrides interact with the base class's
:cpp:func:`BatchManager::experimentComplete` slot, which is the
hub of the per-batch loop. The slot's decision tree (logged result
→ optional ``processExperiment`` → ``isComplete`` → either
``beginNextExperiment`` or ``writeReport`` + ``batchComplete``) is
documented at length in :ref:`batchmanager-state-machine`. Three
points are worth restating from the subclass author's perspective:

- The subclass *owns* the next experiment. Whether it stores a
  template and clones from it (``BatchSequence``), holds the only
  experiment shared pointer (``BatchSingle``), or constructs each
  experiment on demand from a parameter list, only the subclass
  decides how the next ``Experiment`` comes into existence. The
  base class never constructs an :cpp:class:`Experiment`.
- :cpp:func:`BatchManager::processExperiment` is the place to look
  at the just-completed experiment's data (numeric outputs,
  validation flags, derived values) and update aggregate state.
  No batch type does any data analysis here today; both shipping
  implementations only mutate the loop counter or the completion
  flag. A subclass that does want to inspect data should keep the
  work brief because the slot runs on the GUI thread (the comment
  in :cpp:func:`BatchManager::experimentComplete` flags this as a
  future cleanup).
- ``beginNextExperiment`` is the natural place to insert any
  inter-experiment delay or wait. ``BatchSequence`` uses a
  ``QTimer`` for a fixed interval; an externally-triggered batch
  would arm a TCP listener and emit ``beginExperiment()`` from
  the listener's slot; an interactive batch would pop a modal
  dialog and emit ``beginExperiment()`` on its accepted signal.
  Whatever waiting state the subclass enters, ``abort()`` must
  cancel it cleanly.

Coordination with the AcquisitionManager
''''''''''''''''''''''''''''''''''''''''

:cpp:class:`AcquisitionManager` does not know which batch type is
running. It emits :cpp:func:`AcquisitionManager::experimentComplete`
unconditionally at the end of every experiment, and the connection
:cpp:func:`MainWindow::startBatch` installs from that signal to the
:cpp:class:`BatchManager` slot of the same name is what advances
the batch. The signal fires on the AM thread; the slot runs on the
GUI thread; the connection is queued so the cross-thread
:cpp:class:`Experiment` access is delivered serially. Subclass
authors do not manage that connection — it is set up and torn down
by :cpp:func:`MainWindow::startBatch` and
:cpp:func:`MainWindow::batchComplete` for the duration of the
batch.

The cross-manager flow that fires that signal — hardware setup,
acquisition steady state, end-of-experiment teardown — is the
topic of :doc:`/developer_guide/experiment_lifecycle`. The
:cpp:class:`BatchManager` slot's internal decision tree is on
:ref:`batchmanager-state-machine`. This page does not duplicate
either; from the subclass author's perspective the AM is a black
box that calls ``experimentComplete()`` and the batch's job is to
either advance the loop or end it.

Persistence
'''''''''''

Two persistence questions arise for a new batch type. Both have
established conventions on the configuration side and an open
recommendation on the report side.

**Dialog configuration.** The batch's *configuration* — the
parameters the user chose in the dialog (count, interval, value
list, trigger details) — should persist across application
invocations so the dialog re-opens with the user's last choices.
The convention is the one ``BatchSequenceDialog`` uses: the dialog
inherits :cpp:class:`SettingsStorage`, declares a
``BC::Key::<MyBatch>`` namespace with one ``QLatin1StringView`` per
field, and reads/writes through ``get`` and ``set`` (or the
lower-level :cpp:func:`SettingsStorage::setDefault` /
:cpp:func:`SettingsStorage::save` for default-on-first-run
behavior). The persistence model that backs ``QSettings`` is
documented on :doc:`/developer_guide/persistence`.

**Report generation.** :cpp:func:`BatchManager::writeReport` is a
pure virtual on the base class, but neither shipping subclass
generates a report — both ``BatchSingle::writeReport`` and
``BatchSequence::writeReport`` are no-ops, and there is no
on-disk convention for where a batch report would live. A new
batch type that does want to generate a report needs both an
implementation and a destination, and the destination is currently
unspecified.

The recommended layout, when this becomes necessary, is a
``batch/`` top-level folder at the application's Data Storage
Location, peer to the per-experiment numeric directories (and
peer to the existing ``rollingdata/``, ``log/``, and
``textexports/`` auxiliary streams documented on
:doc:`/developer_guide/persistence`). The Data Storage Location is
created at first launch through ``BcSavePathWidget`` (driven by
``BcSavePathDialog`` at first run and reachable from the
application configuration thereafter); a new ``batch/`` peer
would need wiring into both the first-launch creation flow and
the change-of-DSL flow that ``ApplicationConfigManager``
coordinates. Until that landed, treat report generation as
genuinely unspecified — implement ``writeReport`` as a no-op (or
log via :cpp:func:`BatchManager::logMessage` /
:cpp:func:`BatchManager::statusMessage` for a transient summary)
rather than picking an ad-hoc on-disk location that future code
will have to migrate.
