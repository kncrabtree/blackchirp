.. index::
   single: hardware configuration
   single: HardwareRegistry; configuration layer
   single: HardwareProfileManager; configuration layer
   single: RuntimeHardwareConfig; configuration layer
   single: LoadoutManager; configuration layer
   single: REGISTER_HARDWARE_META
   single: REGISTER_HARDWARE_SETTINGS
   single: REGISTER_HARDWARE_BASE
   single: REGISTER_HARDWARE_ARRAY
   single: REGISTER_LIBRARY
   single: REGISTER_CUSTOM_COMM
   single: HwSettingPriority
   single: HwSettingsWidget; create vs edit mode
   single: AddProfileDialog
   single: HWDialog
   single: RuntimeHardwareConfigDialog
   single: hardware profiles; persistence
   single: loadouts; configuration
   single: FTMW presets; configuration
   single: __LastUsed__
   single: drift detection

Hardware Configuration
======================

Blackchirp's hardware story has two halves. The *configuration* half — what
this page covers — describes the four singletons that decide which hardware
drivers exist, which user-visible profiles have been created from
them, which profiles are *active* in the running session, and which
profiles are grouped together as a named loadout. The *runtime* half —
:doc:`/developer_guide/hardware_runtime` — picks up where this page leaves
off: instantiating :cpp:class:`HardwareObject` instances, moving them onto
the right threads, opening their communication channels, and routing their
signals through :cpp:class:`HardwareManager`.

The configuration layer is built on four loosely coupled singletons. None
of them owns a live instrument; together they describe everything
:cpp:class:`HardwareManager` needs to know to bring instruments online.
The split reflects different read-access policies. The registry is
populated at static-initialization time and is everywhere-readable but
write-restricted to its own registration helpers. The profile manager
is the sole authority on per-profile metadata and settings. The runtime
configuration accepts reads from any thread but restricts writes to a
short list of friend classes. The loadout manager funnels all CRUD
through a single API.

The user-facing surface of the same four layers is the **Hardware
Configuration** dialog. The dialog's four panels are one-to-one with the
configuration singletons; orienting yourself to "panel X drives singleton
Y" is the fastest way into this part of the codebase. The user-guide
side of the same workflow is on :doc:`/user_guide/hardware_config`.

Compile-time to runtime: a layer at a time
------------------------------------------

A useful mental model is four layers of an onion, stacked from
compile-time outward:

1. **Catalog (compile time).** Every concrete driver registers itself
   with :cpp:class:`HardwareRegistry` at static-initialization time using
   the ``REGISTER_HARDWARE_*`` macros. This runs before ``main()``, so the
   catalog of factories, supported communication protocols, setting
   descriptors, and library dependencies is complete before any other
   code runs.

2. **Profile metadata (per process).** When the application starts,
   :cpp:class:`HardwareProfileManager` loads any persisted profile
   records from ``QSettings`` into its in-memory cache. A *profile* is
   a ``(hardwareType, label, driver)`` triple — fixed at
   creation time — together with its persisted settings and, for
   Python drivers, a script path / class name / environment path. The
   ``hardwareType`` and ``label`` together form the profile's identity
   (``"FtmwDigitizer.frontPanel"``, ``"FlowController.backup"``); the
   driver is immutable, so changing drivers means
   creating a new profile rather than re-pointing an existing one.

3. **Active selection (per loadout).** :cpp:class:`RuntimeHardwareConfig`
   records *which* profiles are active in the current runtime
   configuration, keyed by profile identity. This is the layer
   :cpp:class:`HardwareManager` reads when it instantiates objects.

4. **Named hardware maps (across sessions).** :cpp:class:`LoadoutManager`
   stores named *loadouts* — each a complete hardware map plus its
   collection of FTMW presets — so the user can switch between fully
   prepared instrument configurations without manually toggling each
   profile.

.. mermaid::

   flowchart LR
       subgraph Compile[Static initialization]
           Macros[REGISTER_HARDWARE_*<br/>in driver .cpp files]
       end

       subgraph Persist[QSettings on disk]
           QS[(QSettings:<br/>HardwareProfiles/...<br/>runtimeHardware/...<br/>Loadouts/...)]
       end

       subgraph Singletons[Process-wide singletons]
           Reg[HardwareRegistry<br/>factories · settingDefs<br/>protocols · libraries]
           PM[HardwareProfileManager<br/>profile metadata · per-profile settings]
           RC[RuntimeHardwareConfig<br/>active type.label → impl map]
           LM[LoadoutManager<br/>named hardware maps + FTMW presets]
       end

       Macros --> Reg
       PM <--> QS
       RC -. derived from active profiles .-> PM
       LM <--> QS
       LM -- applyHardwareMap --> RC

       Reg -. settingDefs .-> PM
       PM -. profile settings .-> HM[HardwareManager]
       RC -. active selections .-> HM

The four singletons are not collapsed into a single class because each
serves a different boundary. The registry is the only writer of
factories and setting descriptors and is read everywhere. The profile
manager is the only authority on what a profile identity means. The
runtime configuration mediates between the GUI's preview state and the
live hardware map. The loadout manager brackets all of the above into
named records the user can save, recall, and share.

HardwareRegistry — the catalog
------------------------------

:cpp:class:`HardwareRegistry` is a pure catalog. It does not check
availability, resolve dependencies, or fall back to alternatives; it
simply records what was registered and constructs instances on demand
through stored factory lambdas. See :doc:`/classes/hardwareregistry` for
the per-method API.

Each driver file registers itself by placing one or more macros at file
scope in its ``.cpp`` file. The macros are defined in
``hardware/core/hardwareregistration.h`` and run during
static-initialization. Eight macros cover the registration surface:

``REGISTER_HARDWARE_META(CLASS, DESC)``
   Primary registration. Declares the factory, the inheritance chain,
   and the description. Must appear before any other macro for the
   same class.

``REGISTER_HARDWARE_PROTOCOLS(CLASS, ...)``
   Lists the :cpp:enum:`CommunicationProtocol::CommType` values the
   driver supports (``Rs232``, ``Tcp``, ``Gpib``, ``Custom``,
   ``Virtual``).

``REGISTER_HARDWARE_SETTINGS(CLASS, ...)``
   Declares :cpp:struct:`HwSettingDef` descriptors for the
   driver's scalar settings: key, label, description,
   type-aware default, optional bounds, and
   :cpp:enum:`HwSettingPriority`.

``REGISTER_HARDWARE_BASE(CLASS, ...)``
   Same shape as ``REGISTER_HARDWARE_SETTINGS``, but for a
   non-instantiable base class (``Clock``, ``FtmwDigitizer``,
   ``HardwareObject`` itself). Settings are merged into every
   driver whose inheritance chain contains the base class.

``REGISTER_HARDWARE_ARRAY`` / ``REGISTER_HARDWARE_ARRAY_ENTRY``
   Declare an array setting and append entries to it. Each entry is a
   :cpp:type:`SettingsStorage::SettingsMap` of sub-key/value pairs
   (sample-rate tables, range tables, …).

``REGISTER_HARDWARE_BASE_ARRAY`` / ``REGISTER_HARDWARE_BASE_ARRAY_ENTRY``
   Array equivalents for base classes. Calling
   ``REGISTER_HARDWARE_BASE_ARRAY`` with no entries reserves the array
   key so it is always rendered in the settings dialog, even for
   drivers (Python-backed drivers, virtual drivers) that supply
   no entries of their own.

``REGISTER_LIBRARY(CLASS, LIBRARY_NAME)``
   Records a :cpp:class:`VendorLibrary` dependency, so the registry can
   answer "which drivers must be torn down before this library is
   reloaded?".

``REGISTER_CUSTOM_COMM`` / ``REGISTER_CUSTOM_COMM_BASE``
   Declare :cpp:struct:`CustomCommDef` field descriptors for drivers
   whose communication type is ``CommunicationProtocol::Custom``. The
   GUI reads these descriptors before construction so it can render the
   right input widgets without instantiating the driver.

Hardware-type and driver keys are derived from Qt's
``staticMetaObject`` rather than passed by hand. ``REGISTER_HARDWARE_META``
walks the metaobject ``superClass()`` chain to find the direct child of
:cpp:class:`HardwareObject` (the *type* key — ``Clock``, ``FtmwDigitizer``,
``Awg``, …), and uses ``CLASS::staticMetaObject.className()`` for the
*driver* key (the class name itself — ``Valon5009``,
``M4i2220x8``). Renaming a class therefore renames its registry key for
free; there is no parallel string table to update.

Base / driver override pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A setting registered with ``REGISTER_HARDWARE_BASE`` is shared by every
driver that inherits from the base. A driver that needs
a different default (or different bounds, or a different priority) for
the same key re-registers the key with ``REGISTER_HARDWARE_SETTINGS``.
:cpp:func:`HardwareRegistry::getSettingDefs` returns the
driver's entry first and skips the base-class entry for that
key, so no duplicate row appears in the UI and
:cpp:func:`HardwareObject::applyRegisteredSettings` writes the right
default. The same precedence applies to arrays.

This is the lever for tuning a driver's defaults without copy-pasting
the whole base-class set: register only what differs.

Priority levels and UI placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:enum:`HwSettingPriority` tags every setting with one of three
levels. The level determines where the setting appears in
:cpp:class:`HwSettingsWidget` — the shared widget used by both the
profile creation flow and the post-creation edit dialog.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Priority
     - UI placement
   * - ``Required``
     - Top form on the *Settings* tab. Editable in Create mode;
       read-only in Edit mode. Examples: digitizer channel count,
       pulse-generator channel count.
   * - ``Important``
     - Always-visible two-column table on the *Settings* tab.
       Editable in both modes. Examples: sample-rate tables, voltage
       ranges.
   * - ``Optional``
     - Collapsible *Advanced* tab. Editable in both modes. Examples:
       rolling-data interval, "critical hardware" flag.

Required settings carry a hard contract: they must be correct *before*
the hardware object is constructed, because the constructor reads them
to size internal arrays, allocate buffers, and so on. That is why
:cpp:class:`HwSettingsWidget` renders them read-only in Edit mode — the
profile must be deleted and recreated to change a Required value. See
:doc:`/classes/hwsettingswidget` for the per-mode behavior and
:doc:`/classes/hardwareregistry` for the full
:cpp:struct:`HwSettingDef` field reference.

HardwareProfileManager — profile metadata
-----------------------------------------

:cpp:class:`HardwareProfileManager` owns the persistent collection of
profiles. A profile is the tuple
``(hardwareType, label, driver, settings, [pythonScriptPath,
pythonClassName, pythonEnvPath])``. The
``(hardwareType, label, driver)`` triple is set when the
profile is created and is immutable thereafter — only the settings
and the per-profile metadata fields (description, threading override,
Python paths) can be edited later. To use a different driver,
create a new profile.

The pair ``<hardwareType>:<label>`` is the QSettings group root and the
profile's stable identity for everything outside the profile manager
(the runtime configuration, loadouts, the active hardware map). The
driver key is stored under that group as one of the fields, not
as part of the path, but it is fixed for the life of the profile. The
API is documented on :doc:`/classes/hardwareprofilemanager`.

Profile metadata persists under the ``HardwareProfiles`` group:

.. code-block:: text

   HardwareProfiles/
     <Type>/
       <label>/
         implementation = <impl-key>
         active         = true|false
         created        = <ISO timestamp>
         modified       = <ISO timestamp>
         description    = <user description>
         threaded       = true|false   # optional
         pythonScriptPath / pythonClassName / pythonEnvPath  # python only

The CRUD surface is small — ``createHardwareProfile``,
``deleteHardwareProfile``, ``activateHardwareProfile``,
``deactivateHardwareProfile``, plus the per-profile field accessors and
the Python-specific accessors. Every change updates the in-memory cache
under the manager's ``QReadWriteLock`` and is flushed back to
``QSettings``. Bulk import/export uses ``QByteArray`` payloads keyed by
``HardwareProfileData``.

Profile creation timing
~~~~~~~~~~~~~~~~~~~~~~~

The order in which a profile's settings appear is critical.
:cpp:class:`RuntimeHardwareConfigDialog` invokes
:cpp:class:`AddProfileDialog`, which presents an
:cpp:class:`HwSettingsWidget` in :cpp:enumerator:`HwSettingsMode::Create`,
populated from the registry's setting descriptors with their declared
defaults. On accept, ``AddProfileDialog`` writes the chosen
communication protocol, every scalar value, and every array entry to
``QSettings`` *before* calling
:cpp:func:`HardwareProfileManager::createHardwareProfile`. By the time
:cpp:class:`HardwareManager` later instantiates the
:cpp:class:`HardwareObject`, every setting it reads in its constructor
is already on disk; ``setDefault`` calls inside the constructor simply
fill in any keys the user did not touch.

Required settings are read-only after creation precisely because of
this contract: changing a Required value after construction would
desynchronize what the constructor read from what the rest of the
object expects. See
:doc:`/user_guide/hardware_config/profiles` for the user-facing
walkthrough.

System profiles
~~~~~~~~~~~~~~~

Some hardware types are required for Blackchirp to operate
(``FtmwDigitizer``, ``Clock``, plus the LIF types when LIF is enabled). The
manager guarantees a *system profile* — a profile labeled ``virtual``
backed by the corresponding virtual driver — for each required type via
:cpp:func:`HardwareProfileManager::ensureSystemProfiles`. System
profiles are flagged by :cpp:func:`HardwareProfileManager::isSystemProfile`
so the UI can prevent deletion of the only available profile of a
required type.

RuntimeHardwareConfig — active selections
-----------------------------------------

:cpp:class:`RuntimeHardwareConfig` records which profiles are currently
*active*, keyed by their ``"<HardwareType>.<label>"`` profile identity.
The driver key for each active profile is held alongside as a
denormalized field — a copy of the profile's immutable driver
that supports validation and drift detection — but the profile
identity is the source of truth. This is the layer
:cpp:func:`HardwareManager::initialize` consults to decide what to
instantiate. See :doc:`/classes/runtimehardwareconfig` for the full
API.

Read access is unrestricted: any thread can call
:cpp:func:`RuntimeHardwareConfig::constInstance` and query the active
map under the internal ``QReadWriteLock``. Write access is friend-only
— :cpp:class:`HardwareManager` and :cpp:class:`RuntimeHardwareConfigDialog`
are the only classes that can call ``setHardwareSelection``,
``applyConfiguration``, ``setThreaded``, and the related mutators. The
friend-class fence is the contract that "only the hardware management
layer changes the active map"; it is a stronger guarantee than a
``protected`` interface because it survives derived classes.

The string-based and template-based query APIs are interchangeable;
prefer the templates when the hardware type is statically known:

.. code-block:: cpp

   const auto &cfg = RuntimeHardwareConfig::constInstance();
   QStringList labels = cfg.getActiveLabels<FtmwDigitizer>();
   QString impl      = cfg.getHardwareImplementation<FtmwDigitizer>("default");

The template variants resolve the type key from
``T::staticMetaObject.className()`` so callers never spell raw strings.

:cpp:func:`RuntimeHardwareConfig::validateConfiguration` checks every
active selection against the registry and returns a
``QHash<QString, HardwareValidationResult>`` keyed by hardware type.
The static overload ``validateHardwareConfiguration(map)`` validates an
arbitrary configuration without consulting the singleton — used by
``RuntimeHardwareConfigDialog`` to validate the *preview* map before
the user commits. Neither method falls back automatically; errors must
be surfaced to the user (or the caller).

LoadoutManager — named maps and FTMW presets
---------------------------------------------

:cpp:class:`LoadoutManager` is the persistence layer above
:cpp:class:`RuntimeHardwareConfig`. A *loadout* is
``{name, hardwareMap, ftmwPresets, currentFtmwPresetName,
lastModified}``: a name, the set of profile identities
(``"<Type>.<label>"``) that make up this configuration, the FTMW
presets that ride on top of that set, a pointer to the loadout's
currently active preset, and a timestamp. See
:doc:`/classes/loadoutmanager` and :doc:`/classes/hardwareloadout` for
the data model and persistence helpers, and the user-guide pages
:doc:`/user_guide/hardware_config/loadouts` and
:doc:`/user_guide/hardware_config/ftmw_presets` for the workflow.

The hardware map inside a loadout records each member profile's
identity together with the driver that profile carried at the
time the loadout was last saved. The driver is denormalized —
the canonical value lives on the profile in
:cpp:class:`HardwareProfileManager` — but storing it lets the loadout
detect when a previously-saved member profile has been removed or
recreated under a different driver, and replay the active set
atomically when the loadout is applied. Loadouts persist under the
``Loadouts/`` QSettings subtree; the manager is the only writer of
that subtree. All public methods are thread-safe via an internal
``QMutex``.

FTMW presets
~~~~~~~~~~~~

An :cpp:struct:`FtmwPreset` is the named operating point that lives
inside a loadout:

- :cpp:struct:`RfConfigSnapshot` (RF chain scalars + clock table),
- ``ChirpConfig`` (chirp segments + markers),
- ``FtmwDigitizerConfig`` (record length, sample rate, channels,
  triggers),
- ``digiHwKey`` — the hardware key of the digitizer the preset was
  captured against.

Presets cannot exist outside a loadout; their lifetime is owned by
:cpp:class:`LoadoutManager`. AWG sample rate is intentionally *not*
stored: it is a hardware property and is reconstructed from the active
AWG profile when the preset is applied. Switching presets within a
single loadout swaps the FTMW operating point without touching the
hardware map.

The ``__LastUsed__`` sentinel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each loadout reserves the preset name ``__LastUsed__``
(:cpp:var:`BC::Store::LM::lastUsedFtmwPresetName`) for an automatic
sentinel preset that captures the last fully-accepted FTMW
configuration. The sentinel is updated only on two events:

- ``FtmwConfigDialog::accept`` — when the user dismisses the FTMW
  configuration dialog with **OK**;
- experiment start — when an experiment begins with the current FTMW
  configuration.

It is *not* updated on **Apply**, on **Cancel**, or on incidental edits
to the live configuration. The sentinel is hidden from every
user-facing dropdown; ``ftmwPresetNames`` excludes it unless its
``includeLastUsed`` flag is set.

Drift detection
~~~~~~~~~~~~~~~

A named FTMW preset is captured against a specific hardware map. If
the map drifts — for example, the user swaps the AWG profile, or the
FTMW digitizer — applying an existing preset against the new map can
silently mis-interpret captured frequencies and clock assignments.
:cpp:class:`RuntimeHardwareConfigDialog` guards against this by
comparing the FTMW-relevant subset of the preview hardware map against
the saved map of the loadout being saved. When the two differ and the
loadout has named presets, the dialog raises a three-button prompt:

- **Discard FTMW presets and save** — keep the new hardware map, drop
  every named preset (and the ``__LastUsed__`` sentinel) from the
  loadout.
- **Save As instead** — punt the change to a fresh loadout via the
  *Save As* flow, leaving the original loadout's presets intact.
- **Cancel** — leave both the loadout and the preview unchanged.

Drift detection runs on save, not on activate; the activate-time prompt
is a simpler "you have unsaved changes; save / discard / cancel"
dialog.

The Hardware Configuration dialog
---------------------------------

:cpp:class:`RuntimeHardwareConfigDialog` is the user-facing surface of
the entire configuration layer. It opens from **Hardware → Hardware
Selection** (and during the first-run onboarding sequence). The
*Hardware Configuration* tab carries four splitter panels — left to
right — and a *Library Status* tab sits alongside it. The mapping
below is the orientation point a contributor needs: "the user changed
panel X, so look at singleton Y."

**Loadout panel** (leftmost) — driven by :cpp:class:`LoadoutManager`.
   Lists every saved loadout and exposes the
   ``Activate``, ``Save``, ``Save As…``, ``Copy``, and ``Delete``
   buttons. Activating a loadout pushes its set of profile identities
   through :cpp:class:`RuntimeHardwareConfig`. Save / Save As trigger
   the drift-detection prompt described above.

**Configuration Overview panel** — driven by :cpp:class:`RuntimeHardwareConfig`.
   A tree of the *preview* active set: the in-memory copy of the
   currently active profile identities that the dialog edits before
   commit. The validation status bar at the bottom of the tab reports
   whether the preview is valid (every active profile resolves to a
   registered driver, every required type has at least one
   active profile).

**Hardware Browser panel** — driven by :cpp:class:`HardwareRegistry`.
   Lists the hardware types the binary supports and the count of
   profiles configured for each. Selecting a type populates the
   right-hand Configuration panel.

**Configuration panel** (rightmost) — driven by :cpp:class:`HardwareProfileManager`.
   For the type selected in the Hardware Browser, shows every profile
   (with its label and driver), the *Enable* checkbox for
   single-instance types, and *Add Profile* / *Remove Profile*
   buttons. *Add Profile* invokes :cpp:class:`AddProfileDialog`, which
   composes :cpp:class:`HwSettingsWidget` in
   :cpp:enumerator:`HwSettingsMode::Create` and writes settings to
   ``QSettings`` before
   :cpp:func:`HardwareProfileManager::createHardwareProfile` runs.
   The collapsible **Advanced** section inside the Configuration panel
   carries the *Run in own thread* override (which writes to
   :cpp:func:`RuntimeHardwareConfig::setThreaded` on accept — the
   runtime effect is on :doc:`/developer_guide/hardware_runtime`) and,
   for Python drivers, the script path / class name /
   environment fields that ``HardwareProfileManager`` persists.

The *Library Status* tab is hosted by :cpp:class:`LibraryStatusWidget`;
the build-time and runtime story for vendor libraries is on
:doc:`/developer_guide/vendor_libraries`.

The dialog is a *preview-state* editor: every change happens against
its in-memory ``d_previewRuntimeConfig`` and ``d_profileOverrides``
state. Hitting **Apply Configuration** validates the preview, calls
:cpp:func:`RuntimeHardwareConfig::applyConfiguration`, applies any
threading or Python overrides via the friend-class write path, and
emits the signals that drive the rest of the application to
re-initialize. Hitting **Cancel** drops the preview unchanged.

Where settings live
-------------------

Four kinds of "settings" share the same ``QSettings`` backing file but
land under disjoint groups, each with a different owner:

**Registry-declared defaults.**
   Live in process memory only. Applied to every newly constructed
   :cpp:class:`HardwareObject` by
   :cpp:func:`HardwareObject::applyRegisteredSettings`, which the base
   constructor calls. These defaults seed the per-profile group on
   first access; once a profile group has a value for a key, the
   stored value wins on subsequent loads.

**Per-profile values.**
   Stored under ``<Type>.<label>/`` in the QSettings file. Owned by
   :cpp:class:`HardwareProfileManager` (for profile metadata) and by
   the corresponding :cpp:class:`HardwareObject` instance (for the
   driver's own settings) via :cpp:class:`SettingsStorage`. Created at
   profile-creation time by ``AddProfileDialog`` and by the base
   constructor; modified by ``HWDialog`` in
   :cpp:enumerator:`HwSettingsMode::Edit` and by the driver itself during
   normal operation.

**Loadout snapshots.**
   Stored under ``Loadouts/<name>/`` in the QSettings file, including
   the loadout's set of member profile identities and every FTMW
   preset. Owned exclusively by :cpp:class:`LoadoutManager`; nothing
   else writes this subtree.

**Application-wide settings.**
   Stored under ``appConfig/``. Owned by
   :cpp:class:`ApplicationConfigManager` (data save path, vendor
   library paths, font, debug-logging toggle, LIF-enable toggle). Other
   singletons subscribe to its signals so a runtime toggle (e.g.,
   enabling LIF) propagates without a restart where possible. See
   :doc:`/classes/applicationconfigmanager` for the option registry and
   :doc:`/classes/settingsstorage` for the underlying key/value store.

The four owners share one file but never one group. Tracing a setting
back to its source therefore reduces to "which group does the key live
in?", which is the question a contributor opening
``~/.config/CrabtreeLab/Blackchirp.conf`` actually wants answered.
