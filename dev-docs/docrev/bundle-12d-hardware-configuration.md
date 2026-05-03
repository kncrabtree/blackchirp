# Bundle 12d — Developer Guide: Hardware Configuration

**Status:** complete

<!--
Status log:
- 2026-05-03: not started → complete. Single new page
  doc/source/developer_guide/hardware_configuration.rst introduces the
  four configuration singletons (HardwareRegistry,
  HardwareProfileManager, RuntimeHardwareConfig, LoadoutManager) and
  maps the Hardware Configuration dialog's four panels onto them.
  Page uses corrected profile-vocabulary: a profile is an immutable
  (hardwareType, label, implementation) triple, Type.label is the
  profile's identity (not a "slot"), and RuntimeHardwareConfig records
  which profiles are active rather than what is "bound" to a slot.
  No source-tree change taken (12d does not authorise one). Misleading
  "slot"/"bind" phrasing identified in five header Doxygen blocks
  (hardwareprofilemanager.h, hardwareloadout.h, loadoutmanager.h),
  dev-docs/loadout-system.md, bundle-12c-architecture.md (lines 125,
  130, 135), this bundle file (lines 36 and 115),
  developer_guide/architecture.rst (lines 181, 203), and
  classes/hardwareloadout.rst (line 12); flagged for a follow-up
  cleanup task that uses the corrected vocabulary across all of them.
  Content commit f859c72c.
-->

Sub-page of the Developer Guide chapter. Documents the four
configuration singletons that together describe Blackchirp's
hardware environment, and the way each maps to a panel of the
**Hardware Configuration** dialog (the user-facing surface).

This page is the configuration half of the hardware story; bundle
12e covers the runtime half (HardwareManager, lifecycle,
communication). Together they replace the original bundle's
`hardware_registration.rst` + `communication_protocols.rst` +
`settings_storage.rst` outline entries.

## Scope

Single Sphinx file:
`doc/source/developer_guide/hardware_configuration.rst`.

The page should answer the following for a contributor:

1. **Compile-time → runtime narrative.** Frame the four
   configuration singletons as four layers of an onion:

   - At compile time, `HardwareRegistry` is populated by the
     `REGISTER_HARDWARE_*` macros. Static-init runs before
     `main()`, so the registry catalog is complete before anything
     else exists.
   - At program start, `HardwareProfileManager` loads any saved
     profile metadata from `QSettings`. A *profile* is an
     immutable `(hardwareType, label, implementation)` triple
     with persistent settings; the `<hardwareType>.<label>` pair
     is the profile's identity, and the implementation is fixed
     at creation time (changing it requires creating a new
     profile).
   - `RuntimeHardwareConfig` records which profiles are *active*
     in the running session, keyed by profile identity. This is
     the layer `HardwareManager` reads when it instantiates
     objects.
   - `LoadoutManager` is the user-friendly layer on top: a
     *loadout* is a named set of member profile identities (the
     active set that should be applied to
     `RuntimeHardwareConfig`) plus its FTMW presets (RF chain +
     chirp + digitizer config snapshots). Switching loadouts
     swaps the active set.

   The four singletons are loosely coupled and have distinct
   responsibilities; they are not a single class because the
   read-access policies differ (the registry is everywhere-readable
   but write-restricted; the profile manager is the sole authority
   on profile metadata; the runtime config has friend-restricted
   write access; the loadout manager funnels every CRUD through a
   single API).

2. **`HardwareRegistry` — the compile-time catalog.** Cover:

   - The eight registration macros, declared in
     `hardware/core/hardwareregistration.h`:
     `REGISTER_HARDWARE_META`,
     `REGISTER_HARDWARE_PROTOCOLS`,
     `REGISTER_HARDWARE_SETTINGS`,
     `REGISTER_HARDWARE_BASE`,
     `REGISTER_HARDWARE_ARRAY` / `REGISTER_HARDWARE_ARRAY_ENTRY`,
     `REGISTER_HARDWARE_BASE_ARRAY` /
     `REGISTER_HARDWARE_BASE_ARRAY_ENTRY`,
     `REGISTER_LIBRARY`,
     `REGISTER_CUSTOM_COMM` / `REGISTER_CUSTOM_COMM_BASE`.
     One-line meaning for each; **forward-link** to
     `:doc:`/classes/hardwareregistry`` for parameter detail.
   - The metaobject-driven derivation: hardware-type key is the
     direct child of `HardwareObject` in the inheritance chain;
     implementation key is the class name. The macro uses
     `staticMetaObject` to look these up — no manual key
     management.
   - The base/implementation override pattern:
     `REGISTER_HARDWARE_BASE` declares a setting common to a
     family of implementations; an implementation can override the
     default by re-registering the same key with
     `REGISTER_HARDWARE_SETTINGS`. The registry returns the
     implementation's entry first, so no UI duplication occurs.
     Same pattern applies to arrays.
   - Priority levels (`Required` / `Important` / `Optional`) and
     their UI placement (top form, always-visible table,
     Advanced tab). Deep detail belongs on
     `:doc:`/classes/hardwareregistry`` and on
     `:doc:`/classes/hwsettingswidget``.
   - The `HwSettingsWidget` is the shared widget used by both
     `AddProfileDialog` (Create mode, before construction) and
     `HWDialog` (Edit mode, after construction); the registry
     drives both.

3. **`HardwareProfileManager` — profile metadata.** Cover:

   - A profile is `(hardwareType, label, implementation, settings,
     [pythonScriptPath, pythonClassName, pythonEnvPath])`. The
     `<hardwareType>:<label>` pair is the QSettings group root.
   - CRUD surface: `addProfile`, `removeProfile`, `renameProfile`,
     plus the Python-specific accessors. Every change persists to
     `QSettings` immediately.
   - Profile creation flow: `RuntimeHardwareConfigDialog` invokes
     `AddProfileDialog`, which presents an `HwSettingsWidget` in
     Create mode populated from the registry's setting
     descriptors. On accept, the manager writes the profile and
     its settings to `QSettings` *before* the hardware object is
     constructed, so when `HardwareManager` later instantiates
     the object, the settings are already there.
   - Cross-link to `:doc:`/classes/hardwareprofilemanager`` for
     the API surface and to
     `:doc:`/user_guide/hardware_config`` for the user-facing
     workflow.

4. **`RuntimeHardwareConfig` — active selections.** Cover:

   - Records the set of active profiles, keyed by profile
     identity (`<HardwareType>.<label>`). The implementation key
     is held alongside as a denormalized copy of the profile's
     immutable implementation; the canonical value lives on the
     profile in `HardwareProfileManager`. This is the data
     `HardwareManager` reads at startup and on loadout switches.
   - Validation: `validateConfiguration()` checks every active
     selection against `HardwareRegistry` and returns a per-type
     `HardwareValidationResult`. Errors are surfaced to the user;
     no automatic fallback.
   - Read access is unrestricted via `constInstance()`; write
     access is friend-restricted to `HardwareManager` and
     `RuntimeHardwareConfigDialog` so only the hardware
     management layer can change the active map.
   - The `getActiveLabels<T>()` and `getHardwareImplementation<T>()`
     template variants use Qt's `staticMetaObject` to derive the
     hardware-type key at compile time, so callers avoid raw
     strings.
   - Cross-link to `:doc:`/classes/runtimehardwareconfig``.

5. **`LoadoutManager` — named hardware maps + FTMW presets.**
   Cover:

   - A *loadout* is `{name, hardwareMap, currentFtmwPreset,
     ftmwPresetNames}`. The hardware map duplicates the
     `RuntimeHardwareConfig` data so loadouts can be activated
     atomically.
   - An *FTMW preset* is `{rfScalars, rfClocks, chirpScalars,
     chirpSegments, chirpMarkers, digiScalars, digiAnalog,
     digiDigital, digiHwKey}`. Presets cannot exist outside a
     loadout. Switching presets within a loadout swaps the FTMW
     operating point without touching the hardware map.
   - The `__LastUsed__` sentinel preset is updated on
     `FtmwConfigDialog::accept` and on experiment start; never on
     Apply, never on Cancel. It is hidden from user-facing
     dropdowns and is never user-deletable.
   - Drift detection: when a loadout is activated and its
     hardware map differs from the saved map of any of its named
     presets, `RuntimeHardwareConfigDialog` raises a
     Discard/Save-As/Cancel prompt before applying. This prevents
     silent reinterpretation of presets against an incompatible
     hardware map.
   - Cross-link to `:doc:`/classes/loadoutmanager``,
     `:doc:`/classes/hardwareloadout``, and the user-guide pages
     `:doc:`/user_guide/hardware_config/loadouts`` and
     `:doc:`/user_guide/hardware_config/ftmw_presets``.

6. **Mapping to the Hardware Configuration dialog.** Make the
   user-facing → developer-facing connection explicit:

   - The dialog
     (`gui/dialog/runtimehardwareconfigdialog.{cpp,h}`,
     `runtimehardwareconfigdialog_ui.h`) has four panels (or
     tab-equivalents). Confirm the exact panel names by reading
     the source — *the drafter must verify the panel labels in
     the running UI or the source rather than trusting this
     bundle file*.
   - Each panel is the user-facing surface of one configuration
     singleton:
     - The implementation/profile picker invokes
       `AddProfileDialog`, drives `HardwareProfileManager`, reads
       the registry's setting descriptors via
       `HwSettingsWidget` in Create mode.
     - The active-selection panel writes to
       `RuntimeHardwareConfig`.
     - The loadout panel reads/writes `LoadoutManager`.
     - The threading-override panel writes to
       `RuntimeHardwareConfig::setThreaded` (cross-link to 12e
       for the runtime effect).
   - This mapping is the orientation point a contributor needs:
     "I want to change panel X — I should look at the singleton
     it talks to."

7. **Where settings live.** Tie together:

   - Registry-declared defaults applied by
     `HardwareObject::applyRegisteredSettings()` from the base
     constructor.
   - Per-profile values stored in `QSettings` under
     `<hwType:label>/`.
   - Loadout-specific snapshots stored under `Loadouts/<name>/`.
   - Application-wide settings (data path, debug logging, vendor
     library paths) stored by `ApplicationConfigManager`.

   For class-level detail on `SettingsStorage` and how each layer
   reads from `QSettings`, forward-link to
   `:doc:`/classes/settingsstorage``.

## Out of scope

- The runtime hardware lifecycle (object construction, threading,
  comm-protocol selection at `bcInitInstrument` time) — that is
  bundle 12e.
- Vendor library integration — bundle 12k.
- Adding a new driver or new hardware type — 12l, 12m. This page
  describes the configuration model; the how-to pages reference
  it.
- The exact set of registration-macro parameters — covered in
  `:doc:`/classes/hardwareregistry``.
- The user-facing operational workflow for the dialog — covered in
  `:doc:`/user_guide/hardware_config`` and its sub-pages.

## Sources

### Related source files

- `src/hardware/core/hardwareregistration.h` — macro definitions.
- `src/hardware/core/hardwareregistry.{cpp,h}` — registry
  singleton.
- `src/hardware/core/hardwareprofilemanager.{cpp,h}` — profile
  metadata.
- `src/hardware/core/runtimehardwareconfig.{cpp,h}` — active
  selections and friend-class write policy.
- `src/data/loadout/loadoutmanager.{cpp,h}` — loadouts and
  presets.
- `src/data/loadout/hardwareloadout.{cpp,h}` — `HardwareLoadout`
  and `FtmwPreset` structs and the `BC::Store::LM` keys.
- `src/gui/dialog/runtimehardwareconfigdialog.{cpp,h}` — the
  Hardware Configuration dialog, to confirm panel labels and
  layout for the *Mapping to the Hardware Configuration dialog*
  section.
- `src/gui/dialog/addprofiledialog.{cpp,h}` — profile creation
  flow.
- `src/gui/widget/hwsettingswidget.{cpp,h}` — Create-mode and
  Edit-mode shared widget.
- `src/gui/dialog/hwdialog.{cpp,h}` — Edit-mode hardware
  settings dialog (links to 12e for the Control tab story).
- `src/data/settings/hardwarekeys.h` — canonical hardware-key
  namespaces.
- `src/data/storage/settingsstorage.{cpp,h}` — the storage
  layer underneath every setting.

### Related dev-docs

- `dev-docs/settings-registry.md` — research material for the
  registry, priority levels, base/impl override pattern.
- `dev-docs/loadout-system.md` — research material for loadouts,
  presets, the `__LastUsed__` sentinel, drift detection.

### Related user-guide pages

Forward-link, do not duplicate:

- `doc/source/user_guide/hardware_config.rst` and the
  `hardware_config/` sub-pages (`loadouts`, `ftmw_presets`,
  `profiles`, etc., as they exist after the user-guide bundles
  have landed).

### Related API reference pages

- `doc/source/classes/hardwareregistry.rst`
- `doc/source/classes/hardwareprofilemanager.rst`
- `doc/source/classes/runtimehardwareconfig.rst`
- `doc/source/classes/loadoutmanager.rst`
- `doc/source/classes/hardwareloadout.rst`
- `doc/source/classes/hwsettingswidget.rst`
- `doc/source/classes/settingsstorage.rst`
- `doc/source/classes/applicationconfigmanager.rst`
- `doc/source/classes/hardwareobject.rst` (for
  `applyRegisteredSettings`)

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/hardware_configuration.rst`.

## Page structure

H1 intro: 2–3 paragraphs introducing the four-singleton model and
its mapping onto the Hardware Configuration dialog.

H2 sections (`-` underlines):

- *Compile-time to runtime: a layer at a time*
- *HardwareRegistry — the catalog*
- *HardwareProfileManager — profile metadata*
- *RuntimeHardwareConfig — active selections*
- *LoadoutManager — named maps and FTMW presets*
- *The Hardware Configuration dialog* — panel-to-singleton mapping
- *Where settings live* — tie-together summary

A small Mermaid or ASCII diagram showing the four singletons and
their data-flow direction (registry → settings flow, profile
manager → QSettings, runtime config → HardwareManager, loadout
manager → runtime config + presets) is a helpful reinforcement
but is optional.

## Acceptance criteria

- The four configuration singletons are each introduced with a
  one-paragraph role description and forward-linked to their API
  ref page.
- The eight registration macros are listed with a one-line meaning
  each; the page does not re-document their parameter lists.
- The base/implementation override pattern is explained well
  enough that a contributor knows how to override a base default.
- The three priority levels are named and mapped to UI placement.
- The profile-creation timing is correct: the manager writes
  settings to `QSettings` *before* `HardwareManager` constructs
  the object.
- The `RuntimeHardwareConfig` read/write policy (unrestricted
  read, friend-restricted write) is documented.
- The loadout / FTMW-preset relationship is correct: presets
  cannot exist outside a loadout; `__LastUsed__` is updated on
  accept and experiment start only; drift detection prompt
  before applying.
- The Hardware Configuration dialog is mapped panel-by-panel to
  the singleton it talks to. Panel labels match the running UI.
- No content duplicates per-method API documentation.
- No rendered link points into `dev-docs/`.
