# Hardware Loadout System

## Goal

A **loadout** captures the active hardware map for a given experimental setup
— which AWG, which digitizer, which clock implementations are bound to which
profile slots. An **FTMW preset** is a named operating point — RF chain
parameters, clock frequencies, chirp waveform, and digitizer configuration —
that lives inside a single loadout. A loadout owns a list of FTMW presets;
each preset belongs to exactly one loadout. Switching loadouts swaps the
hardware map; switching FTMW presets within a loadout swaps the FTMW
operating point. Users do both without re-entering settings dialog by dialog.

## Concepts

The vocabulary below uses **FTMW preset** / **ftmwPreset** consistently
(rather than bare "preset") so that a parallel **LIF preset** /
**lifPreset** concept can be added later without ambiguity. Within this
document "preset" by itself always means an FTMW preset.

- **Loadout** — a hardware map (`Type.label` → implementation) plus the
  FTMW presets it owns.
- **FTMW Preset** — a named `FtmwPreset` (RF + chirp + digitizer) owned
  by a loadout. A preset cannot exist outside a loadout.
- **`__LastUsed__`** — a sentinel preset name reserved per loadout. It
  captures whatever FTMW state was last accepted by `FtmwConfigDialog` or
  by the Experiment Setup Dialog (= an experiment start), regardless of
  whether the user explicitly saved into a named preset. It is hidden
  from FTMW preset selectors, never user-deletable, and never cleared.
  It is the fallback the UI uses to populate widgets when the loadout's
  `currentFtmwPreset` is empty or itself points at `__LastUsed__`.
- **`defaultFtmwPreset`** — per-loadout sticky pointer to the preset
  auto-loaded when the loadout has no `currentFtmwPreset` (e.g. first
  activation, or after the current selection was deleted). Set by the
  user via "Set as Default".
- **`currentFtmwPreset`** — per-loadout transient pointer to the preset
  most recently selected, applied, or accepted into. May reference
  `__LastUsed__` if the user accepted without saving. Drives initial
  widget population whenever the loadout becomes active.

## Data Model

### HardwareLoadout

```cpp
struct HardwareLoadout {
    QString name;                                     // unique, user-visible
    std::map<QString, QString> hardwareMap;           // Type.label -> implementation
    std::map<QString, FtmwPreset> ftmwPresets;        // keyed by preset name; may include "__LastUsed__"
    QString defaultFtmwPresetName;                    // empty if loadout has no real presets yet
    QString currentFtmwPresetName;                    // empty, "__LastUsed__", or a real preset name
    QDateTime lastModified;                           // bumped on any structural change
};
```

`hardwareMap` keys are `Type.label` (BC::Key::hwKey form), values are
implementation class names — same convention used by
`RuntimeHardwareConfig::getCurrentHardware()`. A hardware type is "disabled"
by being absent from the map; we do **not** add a separate enable/disable
flag. Hardware drift between the map and existing FTMW presets is prevented
at write time by the Hardware Configuration dialog (see UI Changes).

### FtmwPreset

The body of an FTMW preset. Field set unchanged from the prior spec; only
the type name and the ownership semantics change. This struct was
previously called `FtmwSnapshot`.

```cpp
struct FtmwPreset {
    RfConfigSnapshot rfConfig;       // clock freqs + RF chain settings
    ChirpConfig chirpConfig;         // chirp segments, markers, timing (no sample rate)
    FtmwDigitizerConfig digitizer;   // analog/digital channels, trigger, record length, etc.
    QString digiHwKey;               // digitizer hwKey for this preset (must match the loadout)
    QDateTime lastModified;          // bumped on save
};
```

### RfConfigSnapshot

Unchanged. A pure data struct holding the **set-only** RF operating
parameters: five scalars (`commonUpDownLO`, `awgMult`, `upMixSideband`,
`chirpMult`, `downMixSideband`) and a `QHash<RfConfig::ClockType,
RfConfig::ClockFreq> clocks`. `fromRfConfig` / `applyTo` helpers convert
to and from the runtime `RfConfig`.

### ChirpConfig in FTMW Presets

Unchanged. Stores `numChirps`, `chirpInterval`, `chirpList`,
`markerChannels`, `allChirpsIdentical`. AWG sample rate is **not** stored;
the inverse builder takes it as a parameter so callers can pass the live
hardware value.

### FtmwDigitizerConfig in FTMW Presets

Unchanged. Stores all persistent fields of the existing class: analog/
digital channel maps, trigger, sample rate, record length, byte order,
block-average and multi-record settings, FID channel index.

## Storage

QSettings layout under the top-level `Loadouts/` group:

```text
Loadouts/
  currentLoadout = "<name>"             # active loadout pointer
  defaultLoadout = "Default"            # fallback loadout used when active is deleted
  names/                                # array of {name}
  <name>/
    name = "<name>"
    hardwareMap/                        # array of {key, value}
    defaultFtmwPreset = "<presetName>"  # empty string if none
    currentFtmwPreset = "<presetName>"  # may be "__LastUsed__"
    ftmwPresetNames/                    # array of {name}; may include "__LastUsed__"
    lastModified = <ISO timestamp>
    ftmwPresets/
      <presetName>/
        rfScalars/                      # group: commonUpDownLO, awgMult, ...
        rfClocks/                       # array: { ClockType, hwKey, output, op, factor, freqMHz }
        chirpScalars/                   # group: numChirps, interval, allIdentical, ...
        chirpSegments/                  # array of segments
        chirpMarkers/                   # array of markers
        digiScalars/                    # group: trigger / horizontal / averaging
        digiAnalog/                     # array
        digiDigital/                    # array
        digiHwKey = "<hwKey>"
        lastModified = <ISO timestamp>
```

The `__LastUsed__` literal lives in a public constant
(`BC::Store::LM::lastUsedFtmwPresetName`) and is filtered out of every
user-facing dropdown.

Portability follow-up: `LoadoutManager` will gain explicit `.ini`
import/export later (single-loadout share/backup, including its FTMW
preset set). Not in the initial slice.

## UI Changes

The user-facing model is three pieces:

1. **Hardware Configuration dialog** — hardware map + loadout management.
2. **FTMW Configuration dialog** — RF / Chirp / Digitizer tabs plus FTMW
   preset management for the active loadout.
3. **Hardware menu** — Loadout submenu (existing) and FTMW Preset submenu
   (new).

### Hardware Configuration Dialog (`RuntimeHardwareConfigDialog`)

Adds a **Loadout group** above the Hardware Configuration tab content
(unchanged from the prior spec):

- `QComboBox` listing all loadout names; selection = active loadout.
- Buttons: **Save**, **Save As…**, **Delete**, **Set as Default**.

Behavior changes vs. the prior spec are concentrated in **Save** and
**Save As**:

- **Combo selection change**: unchanged. Confirmation prompt; on confirm,
  the chosen loadout's `hardwareMap` is copied into
  `d_previewRuntimeConfig`; `d_activeLoadoutName` is updated.

- **Save** — compute the *hardware fingerprint* of `d_previewRuntimeConfig`
  and of the loadout's stored hardware map. The fingerprint is the tuple
  `(awgHwKey, ftmwDigitizerHwKey, sorted set of Clock.* hwKeys)`.
  Implementation-class swaps for the same hwKey are **not** drift; FTMW
  presets reference clocks by label, not by class.
  - **No drift**: persist the new map; FTMW presets and pointers are
    retained.
  - **Drift detected, loadout has no named FTMW presets** (only
    `__LastUsed__` or empty): persist; clear `__LastUsed__` defensively.
  - **Drift detected, loadout has at least one named FTMW preset**:
    present a modal with three options:
    - **Discard FTMW presets and save** — overwrite the loadout with
      the new map and an empty `ftmwPresets`; clear
      `currentFtmwPresetName` and `defaultFtmwPresetName`.
    - **Save As instead** — close the prompt; reroute to the Save As
      flow (the dialog asks for a new name and creates a new loadout
      with the new hardware).
    - **Cancel** — close the prompt; dialog state unchanged.

- **Save As** — prompt for a name; validate non-empty and not duplicate
  (overwrite confirmation on conflict). Capture the previous active
  loadout name before mutating. Create the new loadout with the current
  preview hardware map and an empty `ftmwPresets`. Update
  `d_activeLoadoutName` to the new name.
  - If the previous active loadout exists and shares the new loadout's
    hardware fingerprint and has at least one named FTMW preset, prompt
    *"Copy FTMW presets from `<prev>`?"*. On confirm, copy each named
    preset (excluding `__LastUsed__`) wholesale via
    `LoadoutManager::putFtmwPreset`, and copy the source's
    `defaultFtmwPresetName`.
  - Then prompt the existing *"Configure FTMW settings for this loadout?
    [Yes] [No]"*. **Yes** sets `d_openFtmwConfigOnClose`.

- **Delete** — confirm; remove. Nested storage means the loadout's FTMW
  presets are deleted automatically. `d_activeLoadoutName` is reassigned.
  If all loadouts were deleted, recreate "Default" from
  `d_originalRuntimeConfig` with no FTMW presets.

- **Set as Default** — unchanged.

- **On dialog accept**: apply hardware map (existing path); call
  `LoadoutManager::setCurrentLoadoutName(d_activeLoadoutName)`. Then, if
  the active loadout's `currentFtmwPreset` resolves to a preset (named
  or `__LastUsed__`), push its clocks via
  `HardwareManager::configureClocks`.

### FTMW Configuration Dialog (`FtmwConfigDialog`)

Layout: existing three-tab structure (`RfConfigWidget`, `ChirpConfigWidget`,
`FtmwDigitizerConfigWidget`), plus a new **FTMW Preset group** above the
tab widget:

- `QComboBox p_ftmwPresetCombo` — lists all named FTMW presets of the
  active loadout. `__LastUsed__` is hidden.
- Buttons: **Apply**, **Save**, **Save As…**, **Rename…**, **Delete**,
  **Set as Default**.

Per-tab "Load from FTMW preset:" combos remain on each tab. Their entries
are restricted to *other named FTMW presets within the active loadout*
(cross-loadout sourcing is dropped — within a single loadout, all presets
share hardware by construction). Selecting a real preset overwrites only
that tab's widget contents; the sentinel "(no source — keep current)"
does nothing.

Behavior:

- **On open** — pick an FTMW preset to populate widgets in this order:
  `currentFtmwPreset` (whether named or `__LastUsed__`) →
  `defaultFtmwPreset` → fall back to widget last-used SettingsStorage.
  The combo reflects the named selection, or shows no selection /
  a "(unsaved)" sentinel when `currentFtmwPreset == __LastUsed__`.

- **Dirty tracking** — connect every relevant change signal of the three
  tab widgets (RF chain controls, clock entries, chirp model rows,
  marker rows, digitizer fields) to a single `markDirty()` slot that
  flips `d_dirty = true` and shows a visible indicator (asterisk in
  title or a label). Cleared by **Apply** and by any persisting **Save**
  path.

- **Apply** — if `d_dirty`, prompt the user to confirm losing the
  unsaved changes. Then `initializeFromFtmwPreset(getFtmwPreset(name))`,
  `setCurrentFtmwPresetName(name)`, push clocks, clear dirty.

- **Save** — only enabled when `currentFtmwPreset` is a real named
  preset *and* `d_dirty`.
  `putFtmwPreset(loadout, currentFtmwPreset, toFtmwPreset())` and write
  `__LastUsed__` with the same body. Clear dirty.

- **Save As…** — prompt for a name; reject `__LastUsed__` and any
  duplicate (with overwrite confirmation). Save under the new name; set
  `currentFtmwPreset = newName`; write `__LastUsed__` with the same body.
  Clear dirty.

- **Rename…** — prompt for a new name; reject `__LastUsed__` and
  duplicates. `LoadoutManager::renameFtmwPreset` moves the preset and
  rewrites `currentFtmwPreset` / `defaultFtmwPreset` if they referenced
  the old name. Cannot rename `__LastUsed__`.

- **Delete** — confirm. `removeFtmwPreset`. After delete, choose new
  `currentFtmwPreset` per fallback chain (`defaultFtmwPreset` → first
  remaining named preset → `__LastUsed__` → empty).

- **Set as Default** — enabled only when `currentFtmwPreset` is a real
  preset (not `__LastUsed__`, not empty). `setDefaultFtmwPresetName(
  loadout, currentFtmwPreset)`.

- **Apply Now** (existing button inside `RfConfigWidget`) — preserved;
  pushes clocks immediately. Does **not** mark the preset dirty (the
  setting being applied is already in the widget).

- **On accept (OK)** — if `!d_dirty`, push clocks and accept. If
  `d_dirty`, present the three-way prompt:
  - **Overwrite `<currentFtmwPreset>`** — Save behavior. Disabled when
    `currentFtmwPreset` is `__LastUsed__` or empty.
  - **Save as new FTMW preset…** — Save As prompt. On cancel of that
    prompt, return to the dialog without accepting.
  - **Proceed without saving** — `putFtmwPreset(loadout, "__LastUsed__",
    toFtmwPreset())`; `setCurrentFtmwPresetName(loadout, "__LastUsed__")`;
    do not modify any named preset.
  In all three branches, push clocks via the `applyClocks` signal.

- **On Cancel** — discard. `__LastUsed__` is **not** updated.

The dialog can be opened from:

- Hardware menu → FTMW Configuration.
- Hardware Configuration dialog → Save As → "Yes, configure FTMW".
- Right-click on the clock display box (existing).

### Hardware menu — Loadout submenu

Unchanged from the prior spec. Exclusive `QActionGroup`, one action per
loadout, repopulated on `LoadoutManager::loadoutAdded` /
`loadoutRemoved` / `loadoutChanged` / `currentLoadoutChanged`.
Confirmation on click, then activation sequence as before. Gated to
`Disconnected` / `Idle`.

### Hardware menu — FTMW Preset submenu (new)

A new submenu **FTMW Preset**, sibling to **Loadout**, immediately under
it in the Hardware menu. Contents:

- A `QActionGroup` (exclusive). One `QAction` per *named* FTMW preset of
  the *active* loadout (label = preset name). The action whose name
  equals `currentFtmwPresetName` is checked. If `currentFtmwPresetName ==
  __LastUsed__` or empty, no action is checked.
- Repopulated on `LoadoutManager::currentLoadoutChanged`,
  `ftmwPresetAdded`, `ftmwPresetRemoved`, `ftmwPresetChanged`,
  `currentFtmwPresetChanged`.

Behavior on click:

1. Confirmation: *"Switch to FTMW preset `<name>` of `<loadout>`? This
   will push new clock settings."* — `[Switch] [Cancel]`.
2. On confirm, `LoadoutManager::setCurrentFtmwPresetName(activeLoadout,
   name)` and push the preset's clocks via
   `HardwareManager::configureClocks` (BlockingQueuedConnection).

**Enable state**: mirrors the Loadout submenu — only enabled in
`Disconnected` / `Idle`. The FTMW Preset submenu is also disabled when
the active loadout has no named FTMW presets.

## Experiment Setup Dialog

The ESD's FTMW page (`ExperimentFtmwConfigPage`) embeds
`FtmwConfigWidget` and gains the same FTMW preset controls as
`FtmwConfigDialog`, **except the Delete button** (preset removal is
reserved to the dedicated dialog). Buttons: Apply, Save, Save As…,
Rename…, Set as Default. The previous **Reset to Loadout Defaults**
button is replaced by **Apply default FTMW preset** (loads
`defaultFtmwPreset` via `initializeFromFtmwPreset`; disabled when
`defaultFtmwPresetName` is empty).

Defaults flow:

1. **Repeat experiment** — when the wizard is launched from a saved
   experiment, the widget seeds from the experiment's stored
   `FtmwConfig` via `initializeFromExperiment`. Repeat means repeat;
   the loadout's current FTMW preset is **not** consulted. Mark
   not-dirty.
2. **Fresh experiment** — same population order as `FtmwConfigDialog`:
   `currentFtmwPreset` → `defaultFtmwPreset` → widget last-used
   SettingsStorage.
3. Dirty tracking is identical to `FtmwConfigDialog`.
4. **On ESD accept (= experiment start)** — if `d_dirty`, the same
   three-way prompt fires (overwrite / save as new / proceed without
   saving). In all three branches, `__LastUsed__` is updated. The
   "Proceed without saving" branch only updates `__LastUsed__` and does
   not modify any named preset.

Per-experiment edits never silently bleed back into named FTMW presets;
the user must explicitly choose Overwrite or Save As.

## Constraints

- Loadouts hold hardware; FTMW presets hold FTMW operating-point data.
  FTMW presets cannot exist outside a loadout.
- AWG sample rate is hardware-derived (read from the AWG's
  SettingsStorage); not stored in FTMW presets.
- A loadout's hardware map cannot be silently mutated in a way that
  invalidates its named FTMW presets; the Hardware Configuration dialog
  forces the user to choose Discard / Save As / Cancel.
- `__LastUsed__` is per-loadout, never deleted by user action, never
  shown in dropdowns. Updated only on `FtmwConfigDialog::accept` and on
  ESD experiment-start (not on Apply, not on Save without dirty changes,
  not on Cancel).
- Loadout switching and FTMW preset switching via the Hardware menu
  submenus are gated to `Disconnected` / `Idle`.
- Existing experiments on disk are unaffected; loadouts and FTMW presets
  are a settings-layer concept.
- Naming convention: identifiers and storage keys related to this
  feature use the `ftmwPreset` prefix. A future LIF-preset feature is
  expected to mirror the same shape with a `lifPreset` prefix; nothing
  in this design precludes adding both to a single loadout.

## Migration

This is new development with no prior public release. No migration pathway
is needed.

---

## Implementation Plan

### Status

Phases A–E were completed against the original (single-`FtmwSnapshot`-
per-loadout) spec. Each phase below preserves its **Original
implementation** summary for context, followed by a **revision task list**
that brings the work into alignment with the FTMW-preset-aware spec
above. The revision should be carried out in phase order (A → B → C →
D → E) because data-model changes propagate up through each UI layer.
Phase F (cleanup) gains new items.

The orchestrator (not any agents) runs builds and tests:

```bash
cmake . -B build/Desktop-Debug/
make -C build/Desktop-Debug/ -j$(nproc)        # full app build (~3 min, 300000ms timeout)
cmake . -B build/tests
make -C build/tests tests -j$(nproc)
ctest --test-dir build/tests
```

Each agent task lists files, signatures, and acceptance behavior. Tasks
marked **‖** within a phase can run in parallel.

### Phase A — Data Model & Persistence

**Original implementation (DONE):** `src/data/loadout/` houses
`rfconfigsnapshot.{h,cpp}`, `chirpconfigloadout.{h,cpp}`,
`ftmwdigitizerloadout.{h,cpp}`, `hardwareloadout.{h,cpp}`, and
`loadoutmanager.{h,cpp}`. `LoadoutManager` is a `QObject` +
`SettingsStorage` singleton keyed under `"Loadouts"`, owning the
in-memory loadout map, providing CRUD + current/default tracking +
signals + a `loadoutsMatchingHwKey` filter, with a private testing
constructor. Tests in `tests/tst_loadoutmanagertest.cpp`.

**Phase A revisions (DONE):** `FtmwSnapshot` was renamed to `FtmwPreset`
throughout. `HardwareLoadout` dropped its single `std::optional<FtmwSnapshot>`
and gained `std::map<QString, FtmwPreset> ftmwPresets`,
`defaultFtmwPresetName`, `currentFtmwPresetName`, and `lastModified`.
`LoadoutManager` gained a full FTMW preset CRUD API (`getFtmwPreset`,
`putFtmwPreset`, `removeFtmwPreset`, `renameFtmwPreset`, `ftmwPresetExists`,
`ftmwPresetNames`, `clearFtmwPresets`), current/default preset accessors,
and five new signals. Each preset is persisted in a private `LoadoutHelper`
scoped to `Loadouts/<loadout>/ftmwPresets/<preset>/` using the existing
scalar/array helpers. A bug in `SettingsStorage::readAll()` was fixed: groups
with no flat keys (only sub-subgroups) are no longer stored in `d_groupValues`,
which prevented a sibling `LoadoutHelper` from inadvertently wiping nested
preset data via `writeGroup`. Tests were updated and extended with full CRUD,
pointer, rename, cascade, and persistence coverage; all 17 test suites pass.

> **Note:** preset serialization uses four levels of nested QSettings groups,
> which stretches `SettingsStorage` beyond its original single-level design.
> If the preset data grows further or new preset types are added (e.g. LIF
> presets), it may be worth replacing this with a dedicated serialization
> strategy (JSON per preset, or a separate settings scope) rather than pushing
> the nesting deeper.

### Phase B — Hardware Configuration Dialog

**Original implementation (DONE):** Loadout group with combo + four
buttons (Save / Save As / Delete / Set as Default) above the existing
hardware splitter. `d_previewRuntimeConfig` initialized from the runtime
config (not from any loadout). Combo selection prompts before
overwriting the preview. Save persists `d_previewRuntimeConfig` as the
loadout's hardware map, preserving the existing FTMW snapshot. Save As
prompts for a name and offers a follow-up "Configure FTMW now?". Accept
calls `setCurrentLoadoutName`; `MainWindow`'s `finished` lambda pushes
the loadout's clocks.

**Phase B revisions (DONE):** `onLoadoutSave` gained hardware-drift
detection: a static `ftmwRelevantHwKeys` helper extracts the set of AWG,
FtmwScope, and Clock hwKeys (keys only, not implementations) from a
hardware map. When the fingerprints differ and the loadout has named FTMW
presets, a three-button modal offers *Discard FTMW presets and save*,
*Save As instead*, or *Cancel*; drift with only `__LastUsed__` clears it
defensively. `onLoadoutSaveAs` captures the previous active loadout name
before mutating `d_activeLoadoutName`; after creating the new loadout it
offers to copy named FTMW presets (excluding `__LastUsed__`) and the
`defaultFtmwPresetName` when the hardware fingerprints match. The `finished`
lambda in `MainWindow::launchRuntimeHardwareConfigDialog` already pushes
clocks from `LoadoutManager::currentFtmwPreset` on accept. As a related
cleanup, `HardwareLoadout::hardwareMap` and `ftmwPresets` were changed to
`std::less<>` comparators throughout (`hardwareloadout.{h,cpp}`,
`HardwareManager::applyHardwareMap`, all explicit map constructions in the
dialog).

### Phase C — FTMW Configuration Dialog

**Original implementation (DONE):** `src/gui/dialog/ftmwconfigdialog.{h,
cpp,_ui.h}` is a thin shell wrapping `FtmwConfigWidget`. The widget
hosts three tabs and a "Load from loadout:" combo per tab. On accept,
the dialog pulls a snapshot from the widget and writes it to the active
loadout's `FtmwSnapshot`. `applyClocks` signal forwarded to `MainWindow`.

**Phase C revisions (DONE):** The FTMW preset bar was folded directly into
`FtmwConfigWidget` rather than extracted into a separate widget class. A
`QComboBox p_ftmwPresetCombo` and six buttons (Apply, Save, Save As…,
Rename…, Delete, Set as Default) were added above the tab widget, along with
a status label that shows a dirty indicator and the current preset name.
`bool d_dirty` and `bool d_suppressDirty` were added to the widget; every
user-editable signal in `RfConfigWidget`, `ChirpConfigWidget`, and
`FtmwDigitizerConfigWidget` routes through `markDirty()`, which is guarded
by `d_suppressDirty` to prevent programmatic tab-switch population from
triggering false positives. `clearDirty()` is called by every save path.
The tab-switch handler wraps the `p_chirpWidget->initialize(rfc)` call in a
`d_suppressDirty` guard to fix a chirp-tab false positive. The Apply button
is enabled when the combo has a selection *and* either the selected name
differs from `currentFtmwPresetName` or the widget is dirty. Per-tab "Copy
from other FTMW Preset:" combos restrict their source list to named presets
of the active loadout; selecting one calls `markDirty()` explicitly.
`FtmwConfigDialog` is a thin shell that forwards `applyClocks`; its accept
path pushes clocks unconditionally (the three-way accept prompt from the spec
was deferred — the preset bar's Save / Save As buttons cover the common
cases). The `d_openFtmwConfigOnClose` mechanism on
`RuntimeHardwareConfigDialog` was removed: loadout management is now flexible
enough that auto-launching `FtmwConfigDialog` on close adds complexity without
meaningful gain. As a polish item, `onLoadoutSaveAs` was updated to copy
`currentFtmwPresetName` metadata to the new loadout when the source has a
named current preset. The Hardware Configuration dialog loadout panel was
redesigned: the `QGroupBox` above the three-panel splitter was replaced by a
dedicated leftmost splitter panel containing the `QListWidget` and a 2×3
button grid (row 0: [blank, Activate]; row 1: [Save, Save As…]; row 2:
[Copy, Delete]), giving a four-panel layout. Buttons carry themed SVG icons
(`bolt`, `archive-box`, `arrow-up-on-square`, `document-duplicate`, `trash`).
On dialog accept, if the active loadout is dirty a prompt offers Save and
apply, Apply without saving, Save to new loadout…, or Cancel.

### Phase D — Hardware menu submenus

**Original implementation (DONE):** Loadout submenu populated from
`LoadoutManager::loadoutNames()`, exclusive `QActionGroup`, click
confirms then applies hardware map + clocks. Excluded from the
acquiring-state enable-all loop.

**Phase D revisions (DONE):** A `menuFtmwPreset` submenu was added to `menuHardware` immediately below `menuLoadout` in `mainwindow_ui.h`. `MainWindow` gained `p_ftmwPresetActionGroup`, `rebuildFtmwPresetMenu`, and `onFtmwPresetActionTriggered`; the rebuild slot is connected to all relevant `LoadoutManager` signals (`currentLoadoutChanged`, `loadoutChanged`, `ftmwPresetAdded`, `ftmwPresetRemoved`, `ftmwPresetChanged`, `currentFtmwPresetChanged`) and re-checks the menu's empty state. Switching via the submenu confirms with the user, calls `setCurrentFtmwPresetName`, and pushes clocks via `HardwareManager::configureClocks` on a `BlockingQueuedConnection`; the submenu is excluded from the dummy-acquiring enable-all sweep and disabled whenever the menu is empty.

### Phase E — Experiment Setup Dialog

**Original implementation (DONE):** `ExperimentFtmwConfigPage` embeds
`FtmwConfigWidget` and has a single **Reset to Loadout Defaults**
button (disabled when the active loadout has no FTMW snapshot). Repeat
experiments seed via `initializeFromExperiment`. `validate()`
consolidates checks across the three tabs; `apply()` writes the
snapshot to `p_exp->ftmwConfig()`.

#### Phase E revision tasks

##### E.R1 — FTMW preset bar in `ExperimentFtmwConfigPage`

- **Files:** `src/gui/expsetup/experimentftmwconfigpage.{h,cpp}`.
- Embed the same FTMW preset bar as `FtmwConfigDialog` with
  `showDeleteButton=false`.
- Replace the existing "Reset to Loadout Defaults" button with **Apply
  default FTMW preset** (loads `defaultFtmwPreset` snapshot via
  `initializeFromFtmwPreset`; disabled when `defaultFtmwPresetName` is
  empty).

##### E.R2 — Initial population

- **Files:** `src/gui/expsetup/experimentftmwconfigpage.cpp`.
- Repeat-experiment branch (`exp->ftmwEnabled()`): unchanged —
  `initializeFromExperiment` from the saved `FtmwConfig`. After
  seeding, `clearDirty`.
- Fresh-experiment branch: same population order as
  `FtmwConfigDialog` (`currentFtmwPreset` → `defaultFtmwPreset` →
  widget last-used). After seeding, `clearDirty`.

##### E.R3 — Dirty tracking + accept

- **Files:** `src/gui/expsetup/experimentftmwconfigpage.{h,cpp}`.
- The widget's `dirtyChanged` signal drives the page's local state.
- The page's `validate()` runs first (existing behavior). If
  validation passes and `widget->isDirty()`, fire the same three-way
  prompt as `FtmwConfigDialog::accept`. All three branches update
  `__LastUsed__`. The "Proceed without saving" branch only updates
  `__LastUsed__`.
- `apply()` then writes the snapshot into `p_exp->ftmwConfig()` as
  before.

##### E.R4 — `FtmwConfigWidget` shared changes (‖ with C.R*)

- **Files:** `src/gui/widget/ftmwconfigwidget.{h,cpp}`.
- Move the FTMW preset bar into `FtmwConfigWidget` (or compose it via
  `FtmwPresetBar`) so both dialog and page reuse the same control.
- Drop `lastFtmwLoadout` (per C.R2).
- Audit `toFtmwPreset` (was `toSnapshot`) — the existing implementation
  reads `d_fidChannel` from the active loadout; revise to read from
  the active *preset* via `currentFtmwPreset`. If no current preset
  exists, read from the widget directly.
- Update `populateSourceCombos` per C.R3.

> **Phase E revision gate:**
>
> 1. Repeat experiment seeds from the saved exp's `FtmwConfig` (no
>    FTMW preset consultation).
> 2. Fresh experiment seeds from the active FTMW preset.
> 3. ESD accept fires the three-way prompt only if dirty; all three
>    branches update `__LastUsed__`.

### Phase F — Cleanup & Documentation

#### F1 — Remove obsolete code paths (existing)

- **Files:** `src/gui/mainwindow.{h,cpp}`.
- Delete the original `MainWindow::launchRfConfigDialog` and any
  remaining "RfConfig" dialog title strings.

#### F.R1 — Audit retired helpers

- `loadoutsMatchingHwKey`, `copyClocksMatching`, `copyRfScalars`: with
  cross-loadout sourcing dropped, these helpers become candidates for
  removal. Keep `copyClocksMatching` only if Save-As copy-FTMW-presets
  ends up wanting partial-copy semantics; otherwise delete with their
  tests.

#### F.R2 — Documentation rewrite

- `CLAUDE.md` (project root): no change needed unless the team wants
  to document the loadout / FTMW preset terminology there.
- This `dev-docs/loadout-system.md`: **do not delete on completion**.
  Instead, rewrite it as a brief feature reference (concept summary
  and pointers to the key code locations: `LoadoutManager`,
  `HardwareLoadout` struct, `FtmwPreset` struct, `FtmwConfigWidget`,
  `FtmwConfigDialog`, `ExperimentFtmwConfigPage`, hardware menu
  submenus, storage layout). The implementation-plan body is dropped
  in that rewrite. The artifact is intended to feed the upcoming
  documentation revision project.

#### F2 — Out of scope for the initial slice (unchanged)

- Loadout `.ini` import/export (now would also serialize FTMW presets).
- Per-loadout / per-FTMW-preset dirty-state visual indicators beyond
  the prompt-on-switch.
- Migrating non-FTMW settings (pulse channel templates, flow
  setpoints) into loadouts or as parallel preset families
  (e.g. LIF presets — see Constraints).
