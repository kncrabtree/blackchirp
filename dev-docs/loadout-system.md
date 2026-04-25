# Hardware Loadout System

## Goal

A **loadout** captures everything needed to bring an instrument into a known operating
state for a given experimental configuration: the active hardware map plus the FTMW
operating settings (clock frequencies, RF chain parameters, chirp waveform, and
digitizer configuration). Users switch loadouts to swap between instrument
configurations (e.g. "S-band", "X-band", "DR scan setup") without re-entering
settings dialog by dialog.

## Current Status

**PROBLEM FOUND:** The design below embeds the Ftmw configuration inside the
HardwareLoadout. While it makes sense that the configuration needs to change
when the loadout changes, fundamentally these are separate concerns. This
matters when the user is configuring Ftmw configurations: we have a "load from"
feature that lets the user select settings from a different loadout with
compatible hardware. But this means a user might create two loadouts with
identical hardware but different ftmw configurations, which is confusing.
Instead, a "loadout" should be defined by its hardware map only. Each loadout
should support multiple ftmw configurations (and maybe later lif
configurations), but each configuration should be limited to a single loadout.
The loadout keeps track of all its configurations and its current (default)
configuration. When a user changes the configuration, either via the
FtmwConfigDialog or the ESD, they should be prompted about whether they wish
to overwrite the current configuration, create a new one, or proceed without
saving. When proceeding without saving, the config is saved to a sentinel
`__LastUsed__` configuration in the loadout which is used to populate the UI on
the next instance unless the user has changed the configuration. The UI for
loading Rf config, chirp config, and digitizer config from loadouts changes
since configs can only be loaded from other existing confgiurations within the
same loadout, and now shoudl support config creation, editing, renaming,
saving, etc. The `__LastUsed__` option is hidden in this context.

Considerations to address:

- What to call these "configurations" from the user perspective to distinguish
  them from "Loadouts". "Configuration" seems too vague-- "FTMW Preset"? that
  term will be used here, but consider changing it.
- Structure and ownership of "FTMW Preset" values in SettingsStorage.
- Add menu-based selector for FTMW presets similar to that for Loadouts?
- Add timestamps for last loadout and last preset change (including
  `__LastUsed__`).
- Integration with ESD becomes a bit cleaner: Look at Loadout and load settings
  from the most recent preset associated with the loadout, only falling back
  to widget persistence if no preset exists for the loadout.

Todo items after addressing considerations above:

1. Rewrite conceptual specification (second-level headers up until
   `##Implementation Plan`).
2. Evaluate each completed phase of the implementation plan for necessary
   changes. This will begin with changes to the data models, then propagate up
   through each phase of the plan. Write an updated implementation plan
   for each step to incorporate this design change.

**All of the following is the original plan before introduction of this preset concept**. Once the spec revision and planning are complete, remove this section.

## Data Model

### HardwareLoadout

```cpp
struct HardwareLoadout {
    QString name;                              // unique, user-visible
    std::map<QString, QString> hardwareMap;    // identical shape to RuntimeHardwareConfigDialog::d_previewRuntimeConfig
    std::optional<FtmwSnapshot> ftmw;          // RF + chirp + digitizer; absent for non-FTMW loadouts
};
```

`hardwareMap` keys are `Type.label` (BC::Key::hwKey form), values are implementation
class names — same convention used by `RuntimeHardwareConfig::getCurrentHardware()`. A
hardware type is "disabled" by being absent from the map; we do **not** add a separate
enable/disable flag.

`ftmw` is a single optional: a loadout either has a complete FTMW snapshot or none.
Partial states (e.g. RF set but chirp empty) are represented by the relevant fields
inside `FtmwSnapshot` being default-constructed but the snapshot itself still
present.

### FtmwSnapshot

```cpp
struct FtmwSnapshot {
    RfConfigSnapshot rfConfig;       // clock freqs + RF chain settings
    ChirpConfig chirpConfig;         // chirp segments, markers, timing (no sample rate)
    FtmwDigitizerConfig digitizer;   // analog/digital channels, trigger, record length, etc.
};
```

The three sub-configurations are bundled because they are tightly coupled —
digitizer record length and sample rate depend on chirp duration, which depends on
RF chain settings, which depend on clock frequencies. Editing them in lockstep
matches how users actually think about an FTMW operating point.

### RfConfigSnapshot

A pure data struct (no QObject, no SettingsStorage inheritance) holding the
**set-only** parameters that define the RF operating state. Read-back values like
`d_completedSweeps` and the clock-step list (used during scans) are deliberately
excluded — they are experiment runtime state, not loadout state.

```cpp
struct RfConfigSnapshot {
    bool commonUpDownLO{false};
    double awgMult{1.0};
    RfConfig::Sideband upMixSideband{RfConfig::UpperSideband};
    double chirpMult{1.0};
    RfConfig::Sideband downMixSideband{RfConfig::UpperSideband};

    // Clock template, keyed by ClockType.
    // Each ClockFreq carries hwKey (= Clock.<label>) + output index + freq + factor + op.
    QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks;
};
```

**Clock keying.** `ClockFreq::hwKey` already encodes the clock by *label*
(`Clock.<label>`), not by implementation class. A loadout that targets the
`awg-ref` clock will resolve to whatever implementation is currently bound to the
`Clock.awg-ref` profile. No additional indirection is required.

### ChirpConfig in Loadouts

Reuse the existing `ChirpConfig` class. The loadout stores the set-only fields:

- `numChirps`, `chirpInterval`
- `chirpList` (segments)
- `markerChannels`
- `allChirpsIdentical` (derivable from `chirpList`)

The AWG sample rate is **not** stored; it is read from the active AWG's
`SettingsStorage(BC::Key::AWG::key, Hardware)` block when the chirp is reconstructed
or rendered. This is already how `ChirpConfigWidget` works today.

`ChirpConfig`'s existing `writeChirpFile` / `readChirpFile` path operates on
experiment directories and is unsuitable here. The loadout uses an **independent**
serializer that writes the same fields into a SettingsStorage group.

### FtmwDigitizerConfig in Loadouts

Reuse the existing `FtmwDigitizerConfig` class (extends `DigitizerConfig`,
`HeaderStorage`). The loadout stores all of its persistent fields:

- Analog channel map (per-channel enabled, full scale, vertical offset)
- Digital channel map (per-channel enabled, input flag, role)
- Trigger channel, slope, delay, level
- Sample rate, record length
- Bytes per point, byte order
- Block average enabled + count
- Multi-record enabled + count
- FID channel index (`d_fidChannel`)

As with `ChirpConfig`, `HeaderStorage`'s `storeValues`/`retrieveValues` are aimed
at the experiment header writer; the loadout adds a parallel SettingsStorage-based
serializer in `src/data/loadout/`.

## Storage

**Decision: QSettings under the top-level `Loadouts/` group**, mirroring the pattern
used by `HardwareProfileManager` (`HardwareProfiles/<type>.<label>/...`). Rationale:

- Reuses existing SettingsStorage infrastructure (group keys, arrays, atomic save).
- Same lifecycle as hardware profiles, which are conceptually adjacent.
- Simpler than maintaining a second on-disk format.

Layout:

```text
Loadouts/
  currentLoadout = "<name>"           # top-level pointer to active loadout
  defaultLoadout = "Default"          # fallback loadout used when active is deleted
  <name>/
    hardwareMap/                      # array: each entry { hwKey, implementation }
    ftmw/                             # absent if loadout has no FTMW snapshot
      rfConfig/                       # group: scalar RF chain values + clocks array
        commonUpDownLO, awgMult, ...
        clocks/                       # array: { type, hwKey, output, op, factor, freqMHz }
      chirpConfig/                    # group: numChirps, interval, allIdentical, ...
        segments/                     # array: { chirpIndex, startFreq, endFreq, duration, alpha, empty }
        markers/                      # array: { name, role, timingMode, start, end, enabled }
      digitizer/                      # group: trigger/horizontal/averaging fields
        analog/                       # array: { index, enabled, fullScale, offset }
        digital/                      # array: { index, enabled, input, role }
        fidChannel = <int>
```

Portability follow-up: `LoadoutManager` will gain explicit `.ini` import/export
later (single-loadout share/backup). Not in the initial slice.

## UI Changes

The user-facing model is now three pieces:

1. **Hardware Configuration dialog** — hardware map + loadout management.
2. **FTMW Configuration dialog** (replaces standalone "RF Configuration") — RF /
   Chirp / Digitizer tabs. Edits target the active loadout.
3. **Hardware menu Loadout submenu** — quick switch between loadouts.

### Hardware Configuration Dialog (`RuntimeHardwareConfigDialog`)

Adds a **Loadout group** above the existing Hardware Configuration tab content:

- `QComboBox` listing all loadout names; current selection = active loadout.
- Buttons: **Save**, **Save As…**, **Delete**, **Set as Default**.

It does **not** gain RF/Chirp/Digitizer tabs. Rationale: those settings belong to a
distinct conceptual axis (FTMW operating point) and are managed in their own
dialog. The Hardware Configuration dialog stays focused on hardware selection.

**Design principle — runtime config is authoritative for hardware.** The dialog
always opens with `d_previewRuntimeConfig` initialised from
`RuntimeHardwareConfig::getCurrentHardware()`, not from any loadout. The runtime
hardware config already persists itself to QSettings independently; embedding it
inside a loadout would create a redundant, competing source of truth. Loadouts
are named presets that a user can explicitly apply or save to; they do not
silently track live hardware state. `d_activeLoadoutName` serves as a pointer to
the FTMW snapshot target and as the destination for the **Save** button — it is
not a constraint on what appears in the preview.

Behavior:

- On open: `d_previewRuntimeConfig` = current runtime config (unchanged from
  pre-loadout behavior). Combo populated from `LoadoutManager::loadoutNames()`,
  selection set to `currentLoadoutName()`. Editing the hardware map mutates
  `d_previewRuntimeConfig` as today.
- **Combo selection change**: selecting a different loadout is an explicit "load
  this preset" action. A confirmation prompt ("Load loadout `<name>`? This will
  replace your current hardware preview.") guards the destructive replacement of
  `d_previewRuntimeConfig`. On confirm, the chosen loadout's `hardwareMap` is
  copied into `d_previewRuntimeConfig` and the overview tree is refreshed;
  `d_activeLoadoutName` is updated. On cancel, the combo reverts to the previous
  selection without touching the preview.
- **Save**: persist `d_previewRuntimeConfig` as the `hardwareMap` of the active
  loadout (FTMW snapshot unchanged).
- **Save As**: prompt for a name; validate it is non-empty and not a duplicate
  (overwrite confirmation on conflict). Create the loadout with the current
  `d_previewRuntimeConfig` as its `hardwareMap` and **no** `FtmwSnapshot`.
  Update `d_activeLoadoutName` to the new name. Then prompt: *"Configure FTMW
  settings for this loadout? [Yes] [No]"*. If **Yes**, set `d_openFtmwConfigOnClose`
  so that `MainWindow` opens the FTMW Configuration dialog after this dialog
  closes. If **No**, no further action.
- **Delete**: confirm; remove. Does **not** alter `d_previewRuntimeConfig`.
  `d_activeLoadoutName` is reassigned to `LoadoutManager::currentLoadoutName()`
  after removal (the manager picks the default or first remaining). If all
  loadouts were deleted, recreate a "Default" loadout from `d_originalRuntimeConfig`.
- **Set as Default**: `LoadoutManager::setDefaultLoadoutName(currentSelection)`.
  Button is disabled when the active loadout is already the default.
- On dialog accept: apply hardware map (existing path). Call
  `LoadoutManager::setCurrentLoadoutName(d_activeLoadoutName)`. Then, if the
  active loadout has an `FtmwSnapshot`, push its clock frequencies via
  `HardwareManager::configureClocks` (clocks whose `hwKey` is not in the new
  hardware map are skipped with a warning).

### Prepopulation: per-component "Load from loadout"

A new loadout starts with an empty `FtmwSnapshot` (or omitted entirely if the
user declines the Save As FTMW prompt). Population is **per-component and
explicit**, driven from inside the FTMW Configuration dialog rather than at
creation time. Each of the dialog's three tabs exposes a **"Load from loadout:"**
combobox at the top whose entries are filtered to loadouts whose hardware
matches the relevant component:

- **RF Config tab** — source filter: source loadout's `hardwareMap` shares the
  same AWG `hwKey` as the active loadout's hardware map. On selection, copy
  the source's RF chain scalars (`awgMult`, `chirpMult`, sidebands,
  `commonUpDownLO`) wholesale, and copy each `clocks` entry whose `hwKey`
  matches a clock present in the active loadout's hardware map (per-clock
  match). Clocks without a matching hwKey are left at their existing values.
- **Chirp Config tab** — source filter: source has matching AWG `hwKey`. On
  selection, replace the chirp widget contents with the source's `ChirpConfig`.
- **Digitizer Config tab** — source filter: source has matching `FtmwDigitizer`
  `hwKey`. On selection, replace the digitizer widget contents with the
  source's `FtmwDigitizerConfig`.

Each combo's first entry is a sentinel (e.g. `"(no source — keep current)"`).
Selecting the sentinel does not modify the widget. Selecting a real loadout
overwrites only that tab's widget contents; other tabs are unaffected.

This mechanism is available whenever the dialog is open — both when editing an
existing loadout and immediately after Save As. There is no implicit
prepopulation at Save As time, so the user always sees the empty starting
state and chooses sources deliberately.

The `defaultLoadout` setting is retained as the **fallback target on delete**
(see Hardware Configuration Dialog → Delete) but is no longer the prepopulation
source.

### FTMW Configuration Dialog (new)

Replaces the existing Hardware menu **RF Configuration** entry. The QAction is
renamed to **FTMW Configuration**.

Layout: a tabbed dialog with three tabs. Each tab has a **"Load from loadout:"**
combobox at the top, followed by the editing widget:

- **RF Config** — `RfConfigWidget` + load-source combo filtered to loadouts
  whose hardware map shares the active loadout's AWG `hwKey`.
- **Chirp Config** — `ChirpConfigWidget` + load-source combo filtered as RF.
- **Digitizer Config** — `FtmwDigitizerConfigWidget` + load-source combo
  filtered to loadouts whose hardware map shares the active loadout's
  `FtmwDigitizer` `hwKey`.

Each combo's first entry is `"(no source — keep current)"`. Selecting a real
loadout overwrites only that tab's widget contents (per-component matching
rule from "Prepopulation" above); the other tabs are unaffected. The combo
returns to the sentinel after a load.

Behavior:

- On open: read the active loadout. If it has an `FtmwSnapshot`, populate all
  three widgets from it. If not, leave widgets in their default state and let
  the user populate via the per-tab load-source combos (or by manual edit).
- The standalone "Apply Clock Settings Now" button inside `RfConfigWidget` is
  preserved and continues to push clocks to hardware immediately.
- On **OK**: pull state from all three widgets into a fresh `FtmwSnapshot`,
  write it to the active loadout via `LoadoutManager::put`, and push clocks via
  `HardwareManager::configureClocks`. (No prompt — accept means save, in keeping
  with the dialog's narrow purpose.)
- On **Cancel**: discard.

The dialog can be opened from:

- Hardware menu → FTMW Configuration.
- Hardware Configuration dialog → Save As → "Yes, configure FTMW".
- (Optional, follow-up) right-click on the clock display box.

### Hardware menu — Loadout submenu

Add a submenu **Loadout** under the Hardware menu (positioned next to or just
under the existing **Hardware Selection** entry). Contents:

- A `QActionGroup` (exclusive). One `QAction` per loadout, label = loadout name,
  the active loadout's action checked.
- Repopulated when `LoadoutManager` emits `loadoutAdded` /
  `loadoutRemoved` / `loadoutChanged` / `currentLoadoutChanged`.

Behavior on click:

1. Confirmation: *"Switch to loadout `<name>`? This will reconfigure all
   hardware."* — `[Switch] [Cancel]`.
2. On confirm, perform the same activation sequence as accepting the Hardware
   Configuration dialog: apply the loadout's `hardwareMap`, then
   `configureClocks` for the loadout's RF snapshot. UI rebuild via
   `clearHardwareUI`/`buildHardwareUI` and `HardwareManager::syncWithRuntimeConfig`.

**Enable state:** mirror `actionRuntimeHardwareConfig`. The submenu (and all
its loadout actions) is enabled exactly when "Hardware Selection" is enabled,
i.e. only when the program state is `Disconnected` or `Idle`. Disable during
`Acquiring` and `Paused` to avoid mid-experiment hardware swaps.

## Experiment Setup Dialog

Defaults flow:

1. **Baseline** — active loadout's `FtmwSnapshot` if present.
2. **Override** — `RfConfigWidget` / `ChirpConfigWidget` /
   `FtmwDigitizerConfigWidget` retain their existing SettingsStorage-backed
   last-used state. When constructed without explicit data, they pull from
   settings as today, giving "repeat last experiment" behavior unchanged.
3. **Reset to Loadout Defaults** button on the RF, Chirp, and Digitizer pages.
   Restores the widget contents from the active loadout's snapshot (no-op if
   loadout has no FTMW snapshot).

Per-experiment edits in the wizard never write back to the loadout. They update
last-used SettingsStorage and the experiment object, exactly as today.

## Constraints

- Do **not** embed RF/chirp/digitizer data in `d_previewRuntimeConfig`.
- Do **not** change the type of `d_previewRuntimeConfig`.
- Do **not** store enabled/disabled as a separate flag — map presence is the source
  of truth.
- Do **not** store AWG sample rate in the loadout; it is hardware-derived.
- Loadout FTMW changes flow only through the FTMW Configuration dialog (or
  programmatically via `LoadoutManager`), never the experiment wizard.
- The Loadout submenu and quick-switch are gated to `Disconnected`/`Idle` states.
- Existing experiments on disk are unaffected; loadouts are a settings-layer concept.

## Migration

On first launch after this feature lands:

1. If `Loadouts/` group is empty, create a single loadout named **"Default"** from:
   - `RuntimeHardwareConfig::getCurrentHardware()` → `hardwareMap`.
   - Last-used `RfConfigWidget` settings + current clock state
     (`HardwareManager::getClocks()`) → `RfConfigSnapshot`.
   - Last-used `ChirpConfigWidget` settings → `ChirpConfig`.
   - Last-used `FtmwDigitizerConfigWidget` settings (or current digitizer
     settings via `SettingsStorage(FtmwScope hwKey)`) → `FtmwDigitizerConfig`.
2. Set `Loadouts/currentLoadout = "Default"` and `Loadouts/defaultLoadout = "Default"`.
3. Subsequent launches read the active loadout from `currentLoadout`.

---

## Implementation Plan

The plan is structured for an orchestrator that may dispatch Haiku/Sonnet agents per
task as needed. Repetitive and/or token-heavy tasks should be delegated, otherwise
can be executed by the orchestator. Each task lists target files, the contract to add
 or modify, and the acceptance check the orchestrator should run after completion.

The orchestrator (not any agents) runs builds and tests. Use:

```bash
cmake . -B build/Desktop-Debug/
make -C build/Desktop-Debug/ -j$(nproc)        # full app build (~3 min, 300000ms timeout)
cmake . -B build/tests
make -C build/tests tests -j$(nproc)
ctest --test-dir build/tests
```

Phases A and B are largely sequential. C, D, and E depend on A+B.

## Dispatch Notes for the Orchestrator

- Tasks marked **‖** within a phase can run in parallel agent sessions.
- Each agent task is self-contained: it lists files, signatures, and behavior.
  Avoid passing the whole spec; pass only the relevant section plus any updated
  context from prior phases.
- After every agent task: run the appropriate build target (debug first, then
  tests). Do not advance to the next phase until the current phase's gate is
  green.
- Phase A is well within Haiku's range (pure data + serialization). Phases B, C,
  and D involve UI wiring and inter-dialog handoffs — start with Sonnet for
  those, escalate from Haiku if needed.

### Phase A — Data Model & Persistence (no UI) — **DONE**

All source files live under `src/data/loadout/` and are wired into
`cmake/BlackchirpData.cmake`. The test suite is in
`tests/tst_loadoutmanagertest.cpp` (registered in the root `CMakeLists.txt`).

#### Design note: where SettingsStorage I/O lives

`setGroupValue`/`setArray` are protected on `SettingsStorage`. Rather than
friending many free functions, all QSettings I/O is concentrated in
`LoadoutManager`, which subclasses `SettingsStorage`. The sub-component helper
files provide pure data-conversion functions (no storage dependency) that
`LoadoutManager`'s private methods call when reading and writing.

#### `RfConfigSnapshot` (`rfconfigsnapshot.{h,cpp}`)

A plain struct holding the set-only RF operating parameters: five scalars
(`commonUpDownLO`, `awgMult`, `upMixSideband`, `chirpMult`,
`downMixSideband`) and a `QHash<RfConfig::ClockType, RfConfig::ClockFreq>
clocks`. Provides `fromRfConfig(const RfConfig&)` and `applyTo(RfConfig&)`
for converting to/from the full runtime `RfConfig`.

#### ChirpConfig loadout helpers (`chirpconfigloadout.{h,cpp}`)

Free functions in `namespace BC::Loadout` that convert a `ChirpConfig` into
the three `SettingsMap` / `vector<SettingsMap>` structures that
`LoadoutManager` writes to QSettings, and reconstruct a `ChirpConfig` from
them. AWG sample rate is not stored in the loadout; the inverse builder takes
it as a parameter so callers can pass the live hardware value.

#### FtmwDigitizerConfig loadout helpers (`ftmwdigitizerloadout.{h,cpp}`)

Same pattern as the chirp helpers, but for `FtmwDigitizerConfig`. Covers all
trigger, horizontal, averaging, and channel fields. The inverse builder takes
the digitizer `hwKey` so the returned object is correctly keyed.

#### `FtmwSnapshot` and `HardwareLoadout` (`hardwareloadout.{h,cpp}`)

`FtmwSnapshot` bundles `RfConfigSnapshot`, `ChirpConfig`, `FtmwDigitizerConfig`,
and `digiHwKey`. `HardwareLoadout` adds a `name`, `std::map<QString,QString>
hardwareMap`, and `std::optional<FtmwSnapshot> ftmw`.

`hardwareloadout.cpp` provides the remaining `BC::Loadout` free functions:
`rfConfigScalarsMap`, `rfConfigClocksArray`, `rfConfigSnapshotFromMaps`,
`hardwareMapArray`, `hardwareMapFromArray`, and the two per-component copy
helpers used by `FtmwConfigDialog` tabs:

- `copyClocksMatching(source, dest, allowedHwKeys)` — copies only the clock
  entries whose `ClockFreq::hwKey` is in `allowedHwKeys`; leaves other dest
  clocks untouched.
- `copyRfScalars(source, dest)` — copies the five RF scalar fields; does not
  touch `clocks`.

#### `LoadoutManager` (`loadoutmanager.{h,cpp}`)

A `QObject` + `SettingsStorage` singleton (keyed under `"Loadouts"`) that owns
the in-memory loadout map and is the single point of QSettings I/O. Key
behaviors:

- **CRUD** — `getLoadout`, `putLoadout`, `removeLoadout`, `loadoutExists`,
  `loadoutNames`. Mutations write through to QSettings immediately.
- **Current / default tracking** — `currentLoadoutName`,
  `setCurrentLoadoutName`, `currentLoadout` (and matching default variants).
  `removeLoadout` reassigns `current` away from the removed name automatically.
- **Filtering** — `loadoutsMatchingHwKey(hwKey)` returns the names of all
  loadouts whose `hardwareMap` contains `hwKey` as a key; used by the FTMW
  Configuration dialog to populate its per-tab "Load from loadout" combos.
- **Signals** — `loadoutAdded`, `loadoutRemoved`, `loadoutChanged`,
  `currentLoadoutChanged`, `defaultLoadoutChanged`.
- **First-run** — if the settings store is empty on construction, a default
  loadout named `"Default"` (no FTMW snapshot) is created, persisted, and set
  as both current and default.
- **Thread safety** — `QMutex` guards all mutations (same pattern as
  `HardwareProfileManager`).
- **Test isolation** — a private constructor
  `LoadoutManager(QAnyStringView org, QAnyStringView app)` bypasses the
  singleton for unit tests; `LoadoutManagerTest` is declared a friend.

> **Orchestrator gate after Phase A:** full build + all tests green before starting Phase B.

### Phase B — Hardware Configuration Dialog (loadout selector only)

Read the current dialog before starting:

- `src/gui/dialog/runtimehardwareconfigdialog.{h,cpp}`
- `src/gui/dialog/runtimehardwareconfigdialog_ui.h`

#### B1. Add Loadout group + Save / Save As / Delete / Set as Default — **DONE**

**Implementation summary:**

- `runtimehardwareconfigdialog_ui.h`: Added a `QGroupBox("Loadout")` above the
  hardware splitter containing `QComboBox *p_loadoutCombo` (stretchy) and four
  `QPushButton`s: `p_loadoutSave`, `p_loadoutSaveAs`, `p_loadoutDelete`,
  `p_loadoutSetDefault`.
- `runtimehardwareconfigdialog.h`: Added `QString d_activeLoadoutName`,
  `bool d_openFtmwConfigOnClose`, public `openFtmwConfigOnClose()` getter, and
  private methods: `populateLoadoutCombo`, `switchToLoadout`, `updateLoadoutButtonStates`,
  `ensureRequiredTypes`, `onLoadoutComboChanged`, `onLoadoutSave`, `onLoadoutSaveAs`,
  `onLoadoutDelete`, `onLoadoutSetDefault`.
- `runtimehardwareconfigdialog.cpp`:
  - Constructor: `d_previewRuntimeConfig` is still initialised from the current
    runtime config (unchanged). `d_activeLoadoutName` is set to
    `LoadoutManager::instance().currentLoadoutName()`. The auto-activate-required-types
    block was extracted into `ensureRequiredTypes()`. Loadout combo is populated and
    all five signals are wired after `populateHardwareBrowser()`.
  - `onLoadoutComboChanged`: confirmation prompt on every selection change; on
    confirm calls `switchToLoadout` which copies the loadout's `hardwareMap` into
    `d_previewRuntimeConfig` and refreshes all panels. On cancel reverts the combo.
  - `onLoadoutSave`: builds `HardwareLoadout` from `d_previewRuntimeConfig`,
    preserving any existing FTMW snapshot; calls `putLoadout`.
  - `onLoadoutSaveAs`: trims and validates name; prompts on duplicate; saves with no
    FTMW snapshot; updates `d_activeLoadoutName`; prompts for FTMW configuration and
    sets `d_openFtmwConfigOnClose` if yes.
  - `onLoadoutDelete`: removes the loadout; reassigns `d_activeLoadoutName` from
    `LoadoutManager::currentLoadoutName()`; recreates "Default" from
    `d_originalRuntimeConfig` if all loadouts were deleted. Does not alter the preview.
  - `onLoadoutSetDefault`: delegates to `LoadoutManager::setDefaultLoadoutName`.
  - `onDialogAccepted`: calls `LoadoutManager::setCurrentLoadoutName(d_activeLoadoutName)`
    before `accept()`. The `setCurrentLoadoutName` call in B2 is therefore already
    implemented; what remains in B2 is the clock push in the `MainWindow` `finished`
    lambda.

#### B2. Apply-on-accept wiring for clocks — **DONE**

**Implementation summary:**

- `onDialogAccepted` already calls `LoadoutManager::setCurrentLoadoutName(d_activeLoadoutName)`
  (delivered in B1).
- `src/gui/mainwindow.cpp`: a second `finished` lambda (added after the existing
  hardware-sync lambda) checks `result == QDialog::Accepted`, reads the current
  loadout's `FtmwSnapshot` from `LoadoutManager::instance().currentLoadout()`, and
  posts `p_hwm->configureClocks(clocks)` via `QMetaObject::invokeMethod` with
  `Qt::QueuedConnection`. Queue order (sync first, clocks second) is preserved
  because both are posted to `p_hwm`'s event queue in sequence.
- The FTMW Configuration dialog launch (`d_openFtmwConfigOnClose` flag) is deferred
  to Phase D, when the dialog itself is implemented.
- **First-run fix**: constructor now seeds the active loadout's `hardwareMap` from
  `d_originalRuntimeConfig` if it is empty, ensuring that loading "Default" back via
  the combo always restores a known hardware state.
- **Manual verification:** all loadout combo/save/delete/default operations verified.
  Clock push untested pending Phase C/D (no loadout yet carries an FtmwSnapshot).

> **Orchestrator gate after Phase B:** build the GUI app, launch it, manually verify:
>
> - Hardware Configuration dialog has the loadout combo + 4 buttons.
> - Switching loadouts updates the overview tree.
> - Save / Save As / Delete / Set as Default behave correctly.
> - Accepting the dialog applies clocks to hardware (visible in clock display box).

### Phase C — FTMW Configuration Dialog (new) — **DONE**

**Implementation summary:**

- **New files:** `src/gui/dialog/ftmwconfigdialog.{h,cpp,_ui.h}` (registered
  in `cmake/BlackchirpGui.cmake`). Three-tab dialog (`RfConfigWidget`,
  `ChirpConfigWidget`, `FtmwDigitizerConfigWidget`), each tab with a
  "Load from loadout:" combo filtered by AWG or digitizer hwKey.
- On open: populates from the active loadout's `FtmwSnapshot` if present;
  otherwise seeds widgets from `currentClocks`. On tab switch to Chirp,
  re-initializes `ChirpConfigWidget` from the current RF widget state so
  clock/sideband changes propagate without resetting chirp segments.
- Per-tab source combos use `copyRfScalars` / `copyClocksMatching` (RF tab)
  or direct widget setters (Chirp / Digitizer tabs); reset to sentinel after load.
- `accept()` builds a fresh `FtmwSnapshot` and calls
  `LoadoutManager::putLoadout`. Clocks are pushed to hardware by `MainWindow`
  via the forwarded `applyClocks` signal (on "Apply Now") and an
  `accepted`-connected lambda.
- `BC::Key::Ftmw::ftmwDialogKey` (`"FtmwConfigDialog"`) is the `d_openDialogs`
  re-entrancy key. `MainWindow::launchFtmwConfigDialog` replaces
  `launchRfConfigDialog`; both `actionRfConfig` and `clockBox::configureRequested`
  connect to it. Action display text changed to `"FTMW Configuration"`.
- Save-As-then-FTMW handoff: the `finished` lambda in
  `launchRuntimeHardwareConfigDialog` calls `launchFtmwConfigDialog()` when
  `d->openFtmwConfigOnClose()` is true.
- **Pre-existing bug fixed** (`clockmanager.cpp`): `configureClocks` now emits
  `clockHardwareUpdate(type, QString(), -1)` for clock roles that are removed,
  hiding their rows in `ClockDisplayBox`.

> **Orchestrator gate after Phase C:** build verified; all manual tests passed.

### Phase D — Hardware menu Loadout submenu — **DONE**

**Implementation summary:**

- `src/gui/mainwindow_ui.h`: added `QMenu *menuLoadout` inserted into
  `menuHardware` just after `actionRuntimeHardwareConfig`. Also renamed
  `actionRfConfig` → `actionFtmwConfig` for consistency.
- `src/gui/mainwindow.h`: added `QActionGroup *p_loadoutActionGroup`,
  `rebuildLoadoutMenu()`, and `onLoadoutActionTriggered(QAction*)`.
- `src/gui/mainwindow.cpp`: constructor creates the exclusive `QActionGroup`,
  connects all five `LoadoutManager` signals to `rebuildLoadoutMenu`, connects
  `menuLoadout::triggered` to `onLoadoutActionTriggered`, and calls
  `rebuildLoadoutMenu()` once. `rebuildLoadoutMenu` removes old actions from
  the group, clears the menu, then repopulates from `LoadoutManager::loadoutNames()`
  with checkable actions keyed by name. `onLoadoutActionTriggered` confirms the
  switch, then calls `HardwareManager::applyHardwareMap` (see below) via
  `BlockingQueuedConnection`, rebuilds the hardware UI, queues
  `syncWithRuntimeConfig`, and pushes loadout clocks if an `FtmwSnapshot` is
  present. `menuLoadout->menuAction()` is excluded from the dummy-acquiring
  enable-all loop so the submenu stays disabled during acquisition.
- `src/hardware/core/hardwaremanager.{h,cpp}`: added `applyHardwareMap(const
  std::map<QString,QString>&)` slot. As a friend of `RuntimeHardwareConfig` it
  calls `instance().applyConfiguration(...)`, avoiding the need for `MainWindow`
  to be a friend.

> **Orchestrator gate after Phase D:** build verified; all manual tests passed.

### Phase E — Experiment Setup Dialog Integration — **DONE**

The three separate FTMW ESD pages (`ExperimentRfConfigPage`,
`ExperimentChirpConfigPage`, `ExperimentFtmwDigitizerConfigPage`) are replaced
by a single `FtmwConfigWidget` used in both the standalone dialog and the ESD.
This is the cleanest solution to loadout-aware seeding: `FtmwConfigWidget`
inherits `SettingsStorage` and owns the `lastFtmwLoadout` key, so it can
compare loadout names and seed itself without any external coordination.

**Implementation summary:**

- **New files:** `src/gui/widget/ftmwconfigwidget.{h,cpp}` — three-tab widget
  (RF Config, Chirp Config, Digitizer Config), each with a "Load from loadout:"
  combo. Inherits `SettingsStorage` and tracks `lastFtmwLoadout`; seeds from
  the active loadout snapshot when the loadout changes, otherwise preserves
  last-used widget state. Provides `initializeFromSnapshot`, `toSnapshot`,
  `initializeFromExperiment`, and `resetToLoadout`. `LoadoutManager::removeLoadout`
  clears the stale `lastFtmwLoadout` key via a temporary `SettingsStorage`
  instance when the removed name matches.
- **`FtmwConfigDialog` refactored** to a thin shell: embeds `FtmwConfigWidget`,
  forwards `applyClocks`, calls `widget->toSnapshot()` + `putLoadout` on accept.
- **New `ExperimentFtmwConfigPage`** replaces the three former ESD pages
  (`experimentrfconfigpage`, `experimentchirpconfigpage`,
  `experimentftmwdigitizerconfigpage`). Embeds `FtmwConfigWidget`; for repeat
  experiments seeds from the existing `FtmwConfig`. `validate()` consolidates
  all FTMW checks and sets per-tab error indicators (colored text + icon) on the
  offending tab. `apply()` writes the snapshot into `p_exp->ftmwConfig()`.
  A **Reset to Loadout Defaults** button calls `resetToLoadout()` (disabled when
  the active loadout has no `FtmwSnapshot`).
- **Bugs fixed during testing:**
  - `SettingsStorage::setArray` now erases any stale `d_groupValues` entry for
    the same key before storing array data, preventing `save()` from letting an
    empty-array-read-as-group entry overwrite correctly written array data.
  - `ExperimentFtmwConfigPage` and `MainWindow::launchFtmwConfigDialog` now use
    `RuntimeHardwareConfig::hardwareTypeOf<FtmwScope>()` to identify the
    digitizer hardware key, replacing the stale `BC::Key::FtmwScope::ftmwScope`
    constant (`"FtmwDigitizer"`) that never matched the runtime key prefix
    (`"FtmwScope"`).
  - `ChirpConfigWidget::getChirps()` now syncs the chirp list, marker channels,
    and chirp interval from their respective models/widgets into
    `d_rfConfig.d_chirpConfig` before returning, so the returned config is
    correct whether or not the chirp tab was ever visited.

> **Orchestrator gate after Phase E:** build verified; all manual tests passed.

### Phase F — Cleanup & Documentation

#### F1. Remove obsolete code paths

- **Files:** `src/gui/mainwindow.{h,cpp}`.
- **Behavior:**
  - Delete the original `MainWindow::launchRfConfigDialog` (replaced by
    `launchFtmwConfigDialog` in Phase C).
  - Confirm no orphan references to the old function or to a "RfConfig" dialog
    title (Grep).

#### F2. Out of scope for the initial slice

- Loadout `.ini` import/export.
- Per-loadout dirty-state visual indicator beyond the prompt-on-switch.
- Migrating non-FTMW settings (pulse channel templates, flow setpoints) into
  loadouts.

---
