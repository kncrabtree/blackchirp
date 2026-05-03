.. index::
   single: persistence; two-layer model
   single: SettingsStorage; persistence layer
   single: BlackchirpCSV; experiment directory
   single: HeaderStorage; configuration tree
   single: DataStorageBase; bulk-data lifecycle
   single: AuxDataStorage; per-experiment time series
   single: experiment directory; layout
   single: BC::Key; namespace
   single: BC::Store; namespace
   single: BC::CSV; namespace
   single: rolling data; on-disk
   single: log files; on-disk
   single: text exports
   single: file parsers; registry

Persistence
===========

Blackchirp keeps two kinds of state on disk and treats them as
separate problems. Configuration that survives across application
invocations — hardware profiles, loadouts, application-wide
preferences, GUI sizing — lives in ``QSettings`` and is read and
written through :cpp:class:`SettingsStorage`. Per-acquisition data —
the parameters that defined an experiment plus the waveforms it
recorded — lives in a per-experiment directory of semicolon-delimited
CSV files written through :cpp:class:`BlackchirpCSV`. The two layers
do not share keys, classes, or files. A contributor adding new
persistent state should consciously pick one: the default for
"remember between sessions" is ``SettingsStorage``; the default for
"save with the experiment" is :cpp:class:`HeaderStorage` plus, if
there is bulk data to go with it, a new :cpp:class:`DataStorageBase`
subclass.

This page explains the contract on both sides — how a class plugs
into ``SettingsStorage`` for configuration, how a class contributes
fields to an experiment header through ``HeaderStorage``, and how
bulk data flows through the ``DataStorageBase`` lifecycle. It then
walks the experiment-directory layout that the persistence
subsystem produces, names the auxiliary on-disk streams that share
the same data path (rolling data, application logs, text exports),
and closes with the file-parser registry that consumes external
data files. Per-class details (constructors, every public method,
every key) live on the API pages this page links out to; the
material here covers the cross-system shape.

Two layers: configuration vs. experiment data
---------------------------------------------

The split is sharp on purpose. Anything that needs to survive a
restart but is not a property of one specific acquisition belongs
in ``QSettings`` via ``SettingsStorage``: hardware connection
parameters, per-instance loadouts, the FTMW preset library, the
application data path, GUI panel sizes. Anything that should
travel with an experiment so that opening the experiment directory
months later reproduces the acquisition unambiguously belongs in
the experiment directory: the active hardware list, every
parameter that defined the chirp, every digitizer setting, and the
recorded waveforms themselves.

The boundary matters when a piece of state could plausibly live
either place. Two examples:

- A *user-tweaked default* (last-used pulse width, last-used
  zero-pad factor) is configuration. It belongs in
  ``SettingsStorage`` so the next experiment's setup wizard
  pre-fills it. The value also lands in the experiment header
  on the way out, but the source of truth between experiments
  is ``QSettings``.
- A *per-acquisition processing setting* (FT window function,
  autoscale ignore range) is experiment data. It belongs in the
  experiment directory under the relevant
  :cpp:class:`DataStorageBase` subclass, so re-opening the
  experiment restores the same view. The same field may appear in
  ``SettingsStorage`` as a remembered default for *new*
  experiments, but the experiment directory's copy wins when the
  experiment is loaded.

When the two could disagree, the experiment directory is the
authoritative record of what actually happened.

SettingsStorage
---------------

:cpp:class:`SettingsStorage` is an owning wrapper around a
``QSettings`` group. The constructor opens a group path (a list of
``beginGroup`` calls), reads every value, array, and sub-group into
in-memory caches, and from then on services reads from the cache.
Writes update the cache and either push to ``QSettings`` immediately
(if the optional ``write`` flag is true) or wait until ``save()`` —
which the destructor calls automatically unless
``discardChanges(true)`` was set.

The API is split by trust level. The ``get`` family
(:cpp:func:`SettingsStorage::get`,
:cpp:func:`SettingsStorage::getArray`,
:cpp:func:`SettingsStorage::getGroupValue`) is public and unrestricted:
any code may construct a transient ``SettingsStorage`` over a group
and read from it. The ``set`` family
(:cpp:func:`SettingsStorage::set`,
:cpp:func:`SettingsStorage::setArray`,
:cpp:func:`SettingsStorage::setGroupValue`,
:cpp:func:`SettingsStorage::setDefault`,
:cpp:func:`SettingsStorage::registerGetter`,
:cpp:func:`SettingsStorage::purge`, …) is **protected**: only a class
that *inherits* from ``SettingsStorage`` over a group can mutate it,
which by convention means the class that owns the data. UI code can
look up a hardware driver's persisted timeout; only the
:cpp:class:`HardwareObject` itself (or a declared friend) can change
it.

Three extensions recur throughout Blackchirp:

- **Array values** — vectors of ``SettingsMap``, mapping directly
  onto ``QSettings`` ``beginWriteArray`` / ``beginReadArray``. Used
  for repeated records like pulse-generator channels, AWG marker
  channels, and chirp segment definitions.
- **Group values** — nested ``SettingsMap`` blocks, each saved as
  its own ``beginGroup``. Used for protocol-specific configuration
  (e.g. one group per ``CommunicationProtocol`` type under a
  hardware key).
- **Getter registration** — ``registerGetter`` binds a key to a
  ``std::function<QVariant()>`` callback (typically a member
  function on the owning object). When the key is read or saved,
  the callback is invoked. Owners use getters to keep a key in
  sync with a live member variable or UI widget without having to
  call ``set()`` on every change. Owners that bind getters to UI
  widgets must call ``clearGetters()`` in their destructor before
  the widgets are torn down, or the destructor's automatic
  ``save()`` will dereference deleted objects.

Defaults for hardware settings come from the registry — the
``REGISTER_HARDWARE_*`` macros are processed at construction time by
:cpp:func:`HardwareObject::applyRegisteredSettings`. Hardware
subclasses should not call ``setDefault`` directly for any key the
registry already declares; use ``setDefault`` only for non-hardware
classes or for fields that the registry does not cover. The registry
itself is the topic of :doc:`/developer_guide/hardware_configuration`.

A class that owns a group inherits from ``SettingsStorage``:

.. code-block:: cpp

   class MyClass : public QObject, public SettingsStorage
   {
   public:
       MyClass(QObject *parent = nullptr) :
           QObject(parent),
           SettingsStorage({BC::Key::MyClass::group})
       { }
   };

The friend-helper pattern is the escape hatch for classes that
need to write into a group they do not own. ``LoadoutManager`` uses
it to maintain loadout entries across many hardware groups without
relinquishing the read-public / write-protected guarantee in those
groups: a tiny private subclass declares ``friend class
LoadoutManager`` and the manager constructs it on demand. Use the
pattern sparingly — it erodes the guardrail that the protected
``set`` family is there to enforce — and audit anywhere that a
single helper class starts being reused for many groups.

For the full API surface (constructors, the array/group/getter
helpers, ``readAll``, ``discardChanges``, the static ``purgeGroup``
helpers, the friend-helper template), see
:doc:`/classes/settingsstorage`.

Key namespaces
--------------

Persistent keys live in three ``BC::`` sub-namespaces, declared as
``inline constexpr QLatin1StringView`` constants per the Pattern B
convention from :doc:`/developer_guide/conventions`:

- ``BC::Key::`` — application-wide and hardware-related setting
  keys. Application-wide keys are declared in
  ``data/bcglobals.h`` (``BC::Key::savePath``, ``BC::Key::logDir``,
  ``BC::Key::exportDir``, ``BC::Key::trackingDir``, …);
  hardware-related keys are declared by hardware type in
  ``data/settings/hardwarekeys.h`` (``BC::Key::HW``,
  ``BC::Key::Comm``, ``BC::Key::Clock``, ``BC::Key::AWG``,
  ``BC::Key::Flow``, …). The ``BC::Key::AppConfig`` sub-namespace
  declares the keys handled by
  :cpp:class:`ApplicationConfigManager`; the
  ``BC::Key::FidStorage`` sub-namespace declares the FT processing
  settings serialized by :cpp:class:`FidStorageBase`.
- ``BC::Store::`` — header-storage object keys and value keys for
  data classes that contribute to ``header.csv``. Each owner
  declares its own sub-namespace next to the class
  (``BC::Store::Exp`` for :cpp:class:`Experiment`,
  ``BC::Store::RFC`` for :cpp:class:`RfConfig`,
  ``BC::Store::CC`` for :cpp:class:`ChirpConfig`,
  ``BC::Store::FtmwLO`` and ``BC::Store::FtmwDR`` for the FTMW
  scan-mode parameters, ``BC::Store::FlowConfig`` for the flow
  controller, ``BC::Store::LM`` for ``LoadoutManager``, and so on).
- ``BC::CSV::`` — canonical filenames and column-header constants
  for the CSV files produced by an experiment, declared once in
  ``data/storage/blackchirpcsv.h`` (``versionFile``, ``headerFile``,
  ``hwFile``, ``objectivesFile``, ``validationFile``, ``chirpFile``,
  ``markersFile``, ``clockFile``, ``auxFile``, ``fidparams``,
  ``fidDir``, ``lifparams``, ``lifDir``, plus the column-name
  constants ``ok``/``ak``/``ai``/``vk``/``vv``/``vu`` used by
  ``header.csv``).

Always reach for an existing namespace before introducing a string
literal at a call site. Adding a new key is a one-line edit to the
appropriate header.

BlackchirpCSV and the experiment directory
------------------------------------------

:cpp:class:`BlackchirpCSV` is the workhorse persistence class for
experiment I/O. It owns the canonical experiment-directory layout
and provides static write helpers, directory helpers, and format
utilities that every other storage class calls directly. Most of
its API is static — the static helpers do not require an
instance.

All CSV files Blackchirp writes use ``;`` (``BC::CSV::del``) as the
cell delimiter. The pipe character ``|`` (``BC::CSV::altDel``) is
reserved for ``QStringList`` values that are themselves serialized
inside a single cell, so the inner list never collides with the
outer delimiter.

Two construction paths cover the two roles the class plays:

- **Default constructor** — for static-method use and for writing
  new experiments. The instance carries no version metadata; the
  delimiter defaults to ``BC::CSV::del``.
- **``(num, path)`` constructor** — for reading an existing
  experiment. Opens ``version.csv`` in the experiment directory,
  detects the delimiter from the first line, and populates an
  internal configuration map. The instance-level
  :cpp:func:`BlackchirpCSV::readLine` and
  :cpp:func:`BlackchirpCSV::readFidLine` then tokenize subsequent
  reads with the correct delimiter — important because the
  delimiter has historically varied across file-format versions and
  ``version.csv`` is the source of truth for a loaded experiment.

Three directory helpers locate the per-experiment file tree:

- ``BlackchirpCSV::exptDir(num)`` returns the ``QDir`` for an
  experiment given its number, applying the
  thousand- and million-bucket layout described under
  :ref:`Experiment-directory layout <persistence-experiment-layout>`.
- ``BlackchirpCSV::createExptDir(num)`` creates the bucketed
  directory chain if it does not already exist.
- ``BlackchirpCSV::exptDirExists(num)`` is the existence check.

For the full set of static write helpers (``writeXY``,
``writeMultiple``, ``writeY``/``writeYMultiple``, ``writeHeader``,
``writeLine``, ``writeFidList``, ``writeVersionFile``) and the
formatting helpers (``formatInt64`` for the base-36 FID encoding,
the version accessors), see :doc:`/classes/blackchirpcsv`. The
auxiliary directories returned by ``logDir``, ``textExportDir``, and
``trackingDir`` are covered under
:ref:`Other on-disk streams <persistence-other-streams>`.

HeaderStorage: the configuration tree
-------------------------------------

Every parameter that defined an acquisition lands in
``header.csv`` via :cpp:class:`HeaderStorage`. The file is
human-readable and uses six fixed columns:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Column
     - Meaning
   * - ObjectKey
     - Identifier of the producing ``HeaderStorage`` object.
   * - ArrayKey
     - Name of the array this row belongs to (empty for scalars).
   * - ArrayIndex
     - Index within the array (empty for scalars).
   * - Key
     - The setting's key.
   * - Value
     - Stored value, formatted as a string.
   * - Unit
     - Unit of the value (empty if dimensionless).

The tree is rooted at :cpp:class:`Experiment`. Direct children
include :cpp:class:`FtmwConfig` (which itself owns the
:cpp:class:`RfConfig` and a :cpp:class:`DigitizerConfig`),
:cpp:class:`LifConfig` (which owns its
``LifDigitizerConfig``), the validator, and the
per-instance hardware configs (pulse generator, flow controller,
IO board, pressure controller, temperature controller). Each of
those nodes may add further grandchildren — :cpp:class:`RfConfig`,
for example, owns the :cpp:class:`ChirpConfig` plus the active
clock map.

Two virtuals are the entire subclass contract:

- ``storeValues()`` runs just before the header is written. Inside
  it, call ``store()`` once per scalar field and
  ``storeArrayValue()`` once per cell of any array fields. Each
  row is buffered in this object's cache.
- ``retrieveValues()`` runs after the header has been parsed and
  every matching row has been routed to this object. Inside it,
  call ``retrieve()`` and ``retrieveArrayValue()`` to extract
  cached values into your own members.

Children are declared by overriding ``prepareChildren()`` and
calling ``addChild()`` once per child. The framework rebuilds the
child list at the start of every read or write pass — children
that come and go with user choices (a freshly disabled FTMW or
LIF subsystem) are reflected automatically, and children
themselves do not call ``addChild`` on their parent.

The dispatch rule is the same on both directions. On write, the
root's ``getStrings()`` walks the tree depth-first, packs each
node's cached entries into the six-column form, and concatenates
the result. On read, each parsed CSV row is handed to the root's
``storeLine()``, which compares the row's first column against
``d_headerKey`` and either accepts it into its own cache or
forwards it to the children depth-first. Once every row has been
routed, the root's ``readComplete()`` invokes ``retrieveValues()``
on every node.

Object keys come from one of two sources:

- **Singleton-style objects** (``Experiment``, ``RfConfig``,
  ``ChirpConfig``, ``LifConfig``, the validator, the FTMW config
  flavors) pass a constant from their ``BC::Store::*`` namespace
  (``BC::Store::Exp::key``, ``BC::Store::RFC::key``, …).
- **Per-instance objects** (the hardware configs that may have
  several instances of the same type) pass the *hardware key* for
  the specific instance — e.g.
  ``"PulseGenerator.Default"`` or ``"FlowController.Main"``. This
  guarantees that experiments with multiple instances of the same
  hardware type produce distinguishable header rows.

Adding a new ``HeaderStorage`` child to ``Experiment`` is three
edits: (1) implement ``storeValues`` and ``retrieveValues`` on the
new class, (2) pick a unique object key (typically a constant in a
new ``BC::Store::*`` sub-namespace next to the class), and (3)
register the child in ``Experiment::prepareChildren()`` (or in the
parent node's equivalent if the new class is not a direct child of
``Experiment``).

For the call-sequence detail (the order of ``prepareToStore`` →
``storeValues`` → ``getStrings``, the cache-clearing rules, the
restriction that ``store`` may not be called outside
``storeValues``), see :doc:`/classes/headerstorage`. The
on-disk layout of ``header.csv`` and the example rows are
documented in :doc:`/user_guide/data_storage`.

DataStorageBase: the bulk-data lifecycle
----------------------------------------

:cpp:class:`DataStorageBase` is the abstract root for objects that
persist *bulk* experiment data: waveforms, traces, overlay
annotations. Each instance is identified by a non-negative
``d_number`` and an optional ``d_path``; passing ``d_number == -1``
creates a transient instance — used for peak-up and dummy
experiments — for which all disk I/O is silently skipped.

Four pure virtuals define the acquisition lifecycle that every
subclass implements:

- ``start()`` — acquisition begins. The subclass arms its
  internal acquiring flag and initializes any per-acquisition
  state.
- ``advance()`` — segment boundary. Flush the in-progress
  accumulation for the current segment, then prepare for the
  next.
- ``save()`` — persist the current in-memory state. May be called
  outside a segment boundary (e.g. periodic backups, or when the
  user requests an immediate save).
- ``finish()`` — acquisition ends. The subclass clears its
  acquiring flag.

The acquisition system drives the lifecycle: ``start`` and
``finish`` bracket each acquisition, ``advance`` fires at each
segment transition the experiment defines, and ``save`` is invoked
at the cadences each subclass defines. The protected ``pu_csv``
holds a :cpp:class:`BlackchirpCSV` instance scoped to the
experiment directory, and ``pu_mutex`` guards mutable state shared
across the acquisition and UI threads. The ``writeMetadata`` /
``readMetadata`` helpers serialize a key-value map to a named CSV
file within the experiment directory (or a subdirectory of it),
which subclasses use for their per-subdirectory ``processing.csv``
and similar metadata files.

Three direct subclasses cover the bulk-data domains:

- :cpp:class:`FidStorageBase` — FTMW FID waveforms. Three
  concrete subclasses cover the standard acquisition modes:
  ``FidSingleStorage`` (single segment),
  ``FidMultiStorage`` (multi-segment / LO scan),
  ``FidPeakUpStorage`` (peak-up / rolling-average mode). The
  in-memory cache and the FT processing-settings persistence are
  the topic of :doc:`/developer_guide/ftmw_acquisition`.
- :cpp:class:`LifStorage` — LIF trace data on the
  ``(delay, laser)`` scan grid. The flat-index encoding from grid
  cells to per-cell CSV files is documented in
  :doc:`/developer_guide/lif_acquisition`.
- ``OverlayStorage`` — plot overlay annotations for the
  experiment. Inherits ``DataStorageBase`` for compatibility with
  the data pipeline; only ``save()`` has a non-trivial
  implementation, and writes are dispatched asynchronously via
  ``QtConcurrent`` so the calling thread is not blocked. See
  :doc:`/classes/overlaybase`.

:cpp:class:`AuxDataStorage` plays a complementary role for
auxiliary time-series readings (pressure, flow, temperature, FTMW
shot count, phase-correction diagnostics) but does **not** inherit
``DataStorageBase``. It is owned and driven directly by
:cpp:class:`Experiment`. The split is deliberate: aux data is
accumulating time-series with its own cadence — keys are
registered by hardware objects via ``registerKey``, readings are
merged into the current point via ``addDataPoints``, and
``startNewPoint`` seals the point and appends a row to
``BC::CSV::auxFile`` — and does not fit the
start/advance/save/finish pattern of bulk data.

For the per-subclass cache models, processing-settings serialization,
and trace-access semantics, see :doc:`/classes/datastoragebase`,
:doc:`/classes/fidstoragebase`, :doc:`/classes/lifstorage`,
:doc:`/classes/overlaybase`, and :doc:`/classes/auxdatastorage`.

.. _persistence-experiment-layout:

Experiment-directory layout
---------------------------

One directory per experiment, named by the experiment number, lives
under the user's data path. To keep the file system from
accumulating a flat list of millions of entries, experiments are
bucketed by million and thousand: for an experiment number ``N``,
let ``M = N // 1_000_000`` and ``T = N // 1_000`` (both integer
division). The experiment lands at ``experiments/M/T/N``.
Experiment 480 lives in ``experiments/0/0/480``; experiment
123456789 lives in ``experiments/123/123456/123456789``.

Top-level files within an experiment directory (the ``BC::CSV``
constant for each filename appears in parentheses):

``version.csv`` (``versionFile``)
    Delimiter character on the first line; Blackchirp version
    constants on subsequent lines. The delimiter is the authority
    for parsing every other CSV in the directory.

``header.csv`` (``headerFile``)
    The :cpp:class:`HeaderStorage` tree, six columns.

``hardware.csv`` (``hwFile``)
    Active hardware map at acquisition time (key, subKey,
    hardwareType).

``objectives.csv`` (``objectivesFile``)
    Active acquisition objectives (FTMW/LIF and their termination
    criteria).

``validation.csv`` (``validationFile``)
    The validator's threshold map (which aux-data channels abort
    the experiment, and on what conditions).

``chirps.csv`` (``chirpFile``)
    Chirp segment definitions for the experiment.

``markers.csv`` (``markersFile``)
    AWG marker channel definitions. Not written when the active
    AWG has zero marker channels.

``clocks.csv`` (``clockFile``)
    The clock map (one row per clock per scan step).

``auxdata.csv`` (``auxFile``)
    Per-experiment auxiliary time-series readings written by
    :cpp:class:`AuxDataStorage`.

``log.csv`` (no constant)
    Per-experiment log file, opened by ``LogHandler`` between
    ``beginExperimentLog`` and ``endExperimentLog`` (see
    :ref:`Other on-disk streams <persistence-other-streams>`).

Subdirectories:

- ``fid/`` (``BC::CSV::fidDir``) — FTMW FID data. Contains
  ``fidparams.csv`` (``BC::CSV::fidparams``) plus the per-segment
  ``<i>.csv`` files in base-36 encoding (see
  :cpp:class:`BlackchirpCSV` for the format).
- ``lif/`` (``BC::CSV::lifDir``) — LIF data. Contains
  ``lifparams.csv`` (``BC::CSV::lifparams``) plus per-cell trace
  files keyed on the flat index from
  :cpp:class:`LifStorage`.
- ``backup/`` — periodic FTMW snapshots written by
  :cpp:func:`FidSingleStorage::backup`. Created only when the
  acquisition mode supports backups and the user requests them.
- ``overlays/`` — written by ``OverlayStorage``. Contains
  ``overlays.csv`` plus a pair of ``[label].settings.csv`` and
  ``[label].data.csv`` files per persistent overlay.

The user-guide page :doc:`/user_guide/data_storage` documents
the example contents of each file. Always refer to filenames
through the ``BC::CSV`` constants — never hard-code the
strings at a call site.

.. _persistence-other-streams:

Other on-disk streams
---------------------

Three auxiliary CSV streams share the user's data path but do not
live inside an experiment directory. They are created at first
launch by ``BcSavePathWidget`` and exposed through
``BlackchirpCSV`` directory helpers:

- **Rolling data** — ``rollingdata/`` (``BC::Key::trackingDir``;
  returned by :cpp:func:`BlackchirpCSV::trackingDir`). Continuously
  recorded hardware readings — temperatures, pressures, flows —
  written whenever a hardware object's rolling-data timer fires
  (see :doc:`/developer_guide/hardware_runtime` for the timer
  mechanism). ``RollingDataWidget`` lays the directory out as
  ``rollingdata/<YYYY>/<YYYYMM>/<identifier>.csv`` and appends one
  row per data point with timestamp, epoch seconds, and the
  scalar value. The user-facing description of the rolling-data
  format and configuration lives in
  :doc:`/user_guide/rolling-aux-data`.
- **Application logs** — ``log/`` (``BC::Key::logDir``; returned by
  :cpp:func:`BlackchirpCSV::logDir`). :cpp:class:`LogHandler`
  appends every Normal/Highlight/Warning/Error message to a
  monthly file ``YYYYMM.csv`` and, when debug logging is enabled,
  a parallel ``debug_YYYYMM.csv``. The same handler also writes
  the per-experiment ``log.csv`` listed in the experiment
  directory above; ``beginExperimentLog`` and ``endExperimentLog``
  open and close that per-experiment file. The user-facing log-tab
  description and severity levels are in
  :doc:`/user_guide/log_tab`.
- **Text exports** — ``textexports/`` (``BC::Key::exportDir``;
  returned by :cpp:func:`BlackchirpCSV::textExportDir`). Default
  destination for the "Export XY Data" action on plot curves
  (``ZoomPanPlot::exportCurve``) and similar one-shot exports such
  as the peak-list export dialog. These are user-initiated writes
  routed through the static ``BlackchirpCSV::writeXY`` /
  ``writeMultiple`` helpers.

All three of these streams use ``BlackchirpCSV::writeLine`` /
``writeXY`` for their CSV formatting, so the semicolon delimiter
and the column conventions match the rest of the persistence
subsystem.

File parsers
------------

External data files — spectroscopic line catalogs, generic XY
imports, anything else added later — feed Blackchirp through the
:cpp:class:`FileParser` hierarchy and the
:cpp:class:`FileParserRegistry` singleton.

:cpp:class:`FileParser` is the abstract interface. Each concrete
parser implements ``canParse`` (typically a suffix check followed
by a structural sniff of the first lines), ``formatName`` and
``formatDescription`` for user-visible labels, and
``fileExtensions`` for ``QFileDialog`` glob patterns. The protected
helpers (``isFileReadable``, ``hasMatchingExtension``,
``readFileHeader``) cover the file-system patterns shared by every
parser.

:cpp:class:`FileParserRegistry` is the process-wide singleton that
owns every registered parser. Registration happens during
application startup: ``main()`` constructs the registry via
``FileParserRegistry::instance()`` and calls ``registerParser``
once for each shipped parser. Registration order determines lookup
priority — the registry's ``findParser`` returns the first parser
whose ``canParse`` accepts a given file. The shipped parsers are
:cpp:class:`SPCATParser`, :cpp:class:`XIAMParser`, and
:cpp:class:`GenericXYParser`; the catalog families also share a
common :cpp:class:`CatalogParser` base. The overlay-creation
widgets (``CatalogOverlayWidget``, ``GenericXYOverlayWidget``) and
the overlay-replay code path consume the registry directly through
``findParser`` or the templated ``findParserOfType`` filter.

For the per-class API surface and the format-specific parse
methods, see :doc:`/classes/fileparser`,
:doc:`/classes/fileparserregistry`,
:doc:`/classes/genericxyparser`,
:doc:`/classes/spcatparser`,
:doc:`/classes/xiamparser`, and
:doc:`/classes/catalogparser`. The user-facing overlay workflow
that consumes these parsers is on :doc:`/user_guide/overlays`.

Adding new persistent state
---------------------------

A short decision-tree for the contributor about to add a new piece
of persistent state. Pick the matching row, then read the page it
points to for the per-mechanism detail.

- **Per-profile hardware setting** (timeout, baud rate, channel
  configuration) — declare the key in the appropriate
  ``BC::Key::*`` sub-namespace and register a default through
  ``REGISTER_HARDWARE_SETTINGS`` (or ``REGISTER_HARDWARE_BASE`` for
  fields shared across every instance of a base type). The
  registry plus :cpp:func:`HardwareObject::applyRegisteredSettings`
  does the rest. See :doc:`/developer_guide/hardware_configuration`.
- **Other configuration setting** that survives across sessions
  but is not a hardware property — declare a key in an existing or
  new ``BC::Store::*`` sub-namespace (or ``BC::Key::AppConfig::``
  if the setting belongs to the application configuration
  registry), and expose a ``SettingsStorage``-derived owner that
  reads and writes it. Reads are unrestricted; writes go through
  the owner's protected ``set`` family.
- **New experiment-level header field** — add ``store(...)`` and
  ``retrieve(...)`` calls to the relevant
  ``HeaderStorage::storeValues`` / ``retrieveValues`` override.
  Pick a key in an existing ``BC::Store::*`` namespace if the
  field belongs to an existing object, or add a new sub-namespace
  if the field belongs to a new ``HeaderStorage`` subclass that
  you are also adding. If you are adding a new subclass, register
  it as a child of ``Experiment`` (or of the appropriate parent)
  in the parent's ``prepareChildren`` override.
- **New experiment data file** — if the data is bulk waveform-like
  state with the start/advance/save/finish lifecycle, subclass
  :cpp:class:`DataStorageBase` and pick a filename constant in
  ``BC::CSV::``. If it is a one-shot configuration blob that
  travels with the experiment, treat it as a ``HeaderStorage``
  child instead.
- **New file parser** — subclass :cpp:class:`FileParser` (or
  :cpp:class:`CatalogParser` for catalog formats), declare the
  supported extensions, and add a
  ``FileParserRegistry::instance()->registerParser(...)`` call
  alongside the existing ones in ``main()``.

In every case, declare the new key constants in a header next to
the class that owns the data, never as raw string literals at the
call site.
