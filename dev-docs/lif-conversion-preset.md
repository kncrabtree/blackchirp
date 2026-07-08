# LIF conversion topology — per-experiment config + Preset refactor

Move the LIF frequency-conversion **topology (DAG wiring)** from per-device
hardware settings to a **per-experiment `LifConfig`**, edited in a table UI and
sourced through the existing **Preset** system — mirroring the FTMW
`RfConfig` + `FtmwPreset` paradigm. This **supersedes plan §1/§7 of
[`sirah-cobra-refresh.md`](sirah-cobra-refresh.md)** (which assembled the DAG at
prep from per-device node descriptors). The already-landed work —
`liftopology.csv` serialization and `LifConversion::stageOutput` (commit
`10db1824`) — stays; this refactor rewires the source of truth and adds the
preset + table UI.

## Orientation (for a fresh orchestrator session)

- **Status:** plan approved, **implementation not started** — no code from this
  refactor exists yet. Branch `feature/sirah-cobra-refresh`.
- **Already landed, and kept as-is:** commit `10db1824` (per-experiment
  `liftopology.csv` write + `LifConversion::stageOutput`). This refactor makes
  `LifConfig` the *authoritative* source that file is written from, and adds a
  read counterpart (C-5).
- **How to pick up — delegate in dependency order (do not fan out all at
  once):** send **Task 1 (keystone)** to a Sonnet agent first; it authors and
  freezes the C-1/C-2/C-3/C-5 interfaces. Only once Task 1 compiles, run Tasks
  **2** and **4** in parallel (both consume the frozen interface), then **3**,
  then **5**. See "Task breakdown" below.
- **What this supersedes:** §1/§7 of
  [`sirah-cobra-refresh.md`](sirah-cobra-refresh.md) (DAG assembled at prep from
  per-device hardware settings). In that project's
  [contract](sirah-cobra-refresh-contract.md), §B's `isFinal`/`inputs` node
  settings are removed by C-4; `op`/`harmonic`/`verify`/`tolerance` remain.
- **The template being cloned** is the FTMW config/preset stack — read these
  before implementing: `data/loadout/` (`hardwareloadout.*`,
  `rfconfigsnapshot.*`, `loadoutmanager.*`), `data/model/clocktablemodel.*`,
  `gui/widget/rfconfigwidget.*` and `ftmwconfigwidget.*`, and
  `gui/expsetup/experimentftmwconfigpage.*` / `loscanconfigwidget.*`. The mirror
  table below maps each FTMW piece to its LIF counterpart.
- **Standing decisions captured in memory:**
  `project_lif_topology_per_experiment.md`. Build/test per `AGENTS.md`.
- The user may edit this plan between sessions; treat the committed version as
  the current source of truth.

## Why

The per-device-hardware-settings model is a different storage paradigm from
everything else in Blackchirp. `RfConfig` is the natural analog: an
infrequently-changing yet per-experiment-flexible instrument config, decoupled
from the hardware identity via the Preset/Loadout system. Consistency with that
established, robust pattern is worth the refactor.

## What is hardware vs. per-experiment

The conversion DAG has two kinds of per-stage data with different owners:

| Datum | Owner | Rationale |
|---|---|---|
| **op** (`NHG`/`SFG`/`DFG`) | **Hardware — device identity** | A doubler *is* an NHG device. Not a free choice. Expressed as a virtual `conversionOp()` returning a constant for concrete drivers; read-only in the table. |
| **harmonic** (N, NHG only) | **Hardware — registered setting** | Some FCUs support multiple harmonics, so it is a real setting — but not freely edited. Changed via a **gated** table context-menu action that writes the hardware setting and pushes to the device. |
| **verify / tolerance** | Hardware — device-local | Move-verification behavior; unchanged. |
| driver phase-match calibration | Hardware — driver | e.g. `SirahFcu` sine-bar `stages`; unchanged. |
| **input refs** (`RefType` + stageKey / fixed cm⁻¹) | **Per-experiment `LifConfig`** | The DAG wiring — how beams are connected; editable in the table. |
| **isFinal** marker | **Per-experiment `LifConfig`** | Which stage's output is the excitation beam; can vary per experiment (scan doubled vs. tripled). Exclusive selection in the table. |

The assembled `BC::LifConv::Node` **joins the two sources**: `op`/`n` from the
stage's hardware, `inputs`/`isFinal` from `LifConfig`.

## Architecture mirror (FTMW → LIF)

| FTMW (exists) | LIF (to build) |
|---|---|
| `RfConfigSnapshot` (`data/loadout/rfconfigsnapshot.*`) | `LifConversionSnapshot` (wiring only) |
| `FtmwPreset` (`data/loadout/hardwareloadout.h`) | `LifPreset` (= conversion snapshot + `lastModified`) |
| `HardwareLoadout.ftmwPresets` / `currentFtmwPresetName` | `.lifPresets` / `currentLifPresetName` |
| `LoadoutManager` `…FtmwPreset…` CRUD + `Loadouts/` QSettings subtree | `…LifPreset…` CRUD (clone `BC::Store::LM` keys + `BC::Loadout` flatten helpers) |
| `ClockTableModel` / `ClockTableDelegate` (`data/model/`) | `LifConversionTableModel` / `Delegate` |
| `RfConfigWidget` (table + "Apply … Now") | `LifConversionWidget` (table + preview + preset bar) |
| `FtmwConfigWidget` preset bar (combo + Apply/Save/SaveAs/Rename/Delete) | same bar on `LifConversionWidget` |
| `ExperimentFtmwConfigPage::apply()` → `FtmwConfig.d_rfConfig` | `ExperimentLifConfigPage::apply()` → `LifConfig` nodes |
| `LOScanConfigWidget` reads `exp->ftmwConfig()->d_rfConfig` for scan bounds | scan-axis widget reads `exp->lifConfig()` conversion for laser bounds |

## Frozen interface contract

Downstream tasks must not redesign these without escalation.

### C-1. `LifConversionSnapshot` (mirror `RfConfigSnapshot`)

Home: `data/loadout/lifconversionsnapshot.{h,cpp}`, added to `cmake/BlackchirpData.cmake`.
Captures the **wiring only** — no op/harmonic (those come from hardware).

```cpp
namespace BC::LifConv {
// Per-stage wiring, keyed by the stage's hwKey.
struct StageWiring {
    QString stageKey;
    std::vector<InputRef> inputs;  // NHG->1, SFG/DFG->2 (validated at assembly)
    bool    isFinal{false};
};
}

struct LifConversionSnapshot {
    std::vector<BC::LifConv::StageWiring> wiring;
    QString laserKey;              // active LifLaser hwKey the wiring was captured against
    // Round-trip helpers (mirror RfConfigSnapshot::fromRfConfig/applyTo):
    static LifConversionSnapshot fromNodes(const std::vector<BC::LifConv::Node> &nodes,
                                           const QString &laserKey);
    // Fill wiring into a node list; op/n are supplied by the caller (from hardware).
    std::vector<BC::LifConv::Node> toNodes(
        const std::function<BC::LifConv::Op(const QString&)> &opOf,
        const std::function<int(const QString&)> &harmonicOf) const;
};
```

### C-2. `LifPreset` + `HardwareLoadout` (mirror `FtmwPreset`)

In `data/loadout/hardwareloadout.h`:

```cpp
struct LifPreset {
    LifConversionSnapshot conversion;
    QDateTime lastModified;
};
// on HardwareLoadout, beside ftmwPresets:
std::map<QString, LifPreset> lifPresets;
QString currentLifPresetName;
```

Add `BC::Loadout` flatten/reconstruct free functions (mirror
`rfConfigClocksArray` / `rfConfigSnapshotFromMaps`): the wiring is an array,
one entry per stage with subkeys `stageKey`, `isFinal`, and the ordered inputs
(`in0Type/in0Key/in0Fixed/in1Type/in1Key/in1Fixed`).

### C-3. `LoadoutManager` LIF API (mirror the FTMW methods)

Add to `data/loadout/loadoutmanager.{h,cpp}`: `getLifPreset` / `putLifPreset`
(insert-or-replace, stamps `lastModified`, refuses to remove the active one) /
`removeLifPreset` / `renameLifPreset` / `lifPresetExists` /
`lifPresetNames(includeLastUsed=false)` / `currentLifPresetName` /
`setCurrentLifPresetName` / `currentLifPreset`; signals
`lifPresetAdded/Changed/Removed`; `__LastUsed__` sentinel. Add `BC::Store::LM`
keys `lifPresetsKey`, `lifPresetNamesKey`, `currentLifPresetKey`,
`lastUsedLifPresetName`, and private `p_readLifPreset` / `p_writeLifPreset` /
`p_syncLifPresetIndex` / `p_writeLifPresetPointers`.

### C-4. `LifFreqConversionStage` changes

`hardware/core/liflaser/liffreqconversionstage.{h,cpp}`:

- **Add** `virtual BC::LifConv::Op conversionOp() const;` — base default reads
  the `op` registered setting (for generic Fixed/Virtual stages); concrete
  drivers override to a constant. **Add** `int harmonicOrder() const { return
  get(BC::Key::LifConvStage::harmonic, 2); }`.
- **Add** `virtual bool setHarmonicOrder(int n);` — the driver hook for a gated
  harmonic change. Base default persists the `harmonic` setting (via `set` +
  `save`) and returns success; a unit that can retune its harmonic output in
  firmware **overrides** to issue the hardware command (and update the setting).
  This is the seam the gated context-menu action drives (C-7), so changing
  harmonic is never a raw setting poke — it always passes through the device.
- **Keep registered:** `harmonic`, `verify`, `tolerance`. **Keep** the `op`
  registered setting on the base *only* for the generic stages that read it.
- **Remove:** the `isFinal` scalar setting, the `inputs` array registration
  (`REGISTER_HARDWARE_BASE_ARRAY`/`_ENTRY`), `nodeFromSettings()`,
  `conversionNode()`, and `assembleActiveLifConversion()` in its current
  hardware-settings-sourced form (replaced per C-6). The `BC::Key::LifConvStage`
  `isFinal`/`inputs`/`refType`/`refKey`/`refFixedCm1` key constants are
  removed; `op`/`harmonic`/`verify`/`tolerance` remain.
- **Concrete drivers:** `SirahFcu` overrides `conversionOp()` → `Op::NHG`, keeps
  its `harmonic` default override, and **drops** the `isFinal` default override.
  `VirtualLifFreqConversionStage`/`FixedLifFreqConversionStage` keep the base
  `op`-setting path so an uncontrolled stage can stand in for any device.
- Orphaned on-disk keys in existing hardware profiles are harmless and left
  in place (no migration framework); note it.

### C-5. `LifConfig` authoritative topology API

`data/lif/lifconfig.{h,cpp}`. `d_conversionNodes` becomes the authoritative
per-experiment DAG (op/n snapshotted from hardware at config time, wiring from
the table). Replace `setConversionTopology(nodes, conv, laserKey)` with:

```cpp
void setConversionNodes(std::vector<BC::LifConv::Node> nodes, const QString &laserKey);
const std::vector<BC::LifConv::Node> &conversionNodes() const;
const LifConversion &conversion() const;   // cached assembled (rebuilt in setter)
bool hasConversion() const;                // !nodes.empty()
```

Add a **read counterpart** to `writeTopologyFile()`:
`bool readTopologyFile();` (mirror `RfConfig::loadClockSteps`) so loaded/edited
experiments reconstruct the DAG. `writeTopologyFile()` is unchanged in format.
`retrieveValues`/`storeValues` (header.csv) are untouched — the topology stays
in `liftopology.csv`.

### C-6. Assembly join (replaces `assembleActiveLifConversion`)

A free helper in `hardware/core/liflaser/liffreqconversionstage.*` (or a small
new TU) that builds a node list by joining hardware op/n with a wiring source:

```cpp
// op/n read from each active stage's hardware settings snapshot; wiring from `snap`.
LifConversion::AssemblyResult assembleLifConversion(const LifConversionSnapshot &snap);
// Convenience: wiring from the current LIF preset (LoadoutManager).
LifConversion::AssemblyResult assembleCurrentLifConversion();
```

Consumers re-pointed:
- **`HardwareManager::initializeExperiment`** (prep): build nodes from
  `exp->lifConfig()->conversionNodes()` (already joined), `assemble`, abort on
  `!ok`, cache `d_lifConversion`, `pushLifConversionToLaser`.
- **`HardwareManager::updateLifConversion`** (live/connect): source from
  `assembleCurrentLifConversion()` (current preset), not hardware settings.
- **`experimenttypepage.cpp`** scan-axis bounds: read the conversion from
  `exp->lifConfig()` (seeded from the current preset at `enableLif`), re-derive
  laser bounds in the page's `initialize()`/show path.
- **`liflaserwidget.cpp`** live bounds: `assembleCurrentLifConversion()`.

### C-7. Table UI (mirror `ClockTableModel`/`RfConfigWidget`)

`data/model/lifconversiontablemodel.{h,cpp}` +
`gui/lif/gui/lifconversionwidget.{h,cpp}` (+ delegate). Rows = active
`LifFreqConversionStage` keys (from `RuntimeHardwareConfig`). Columns:

| Col | Content | Editable? |
|---|---|---|
| Stage | hwKey | read-only |
| Op | `conversionOp()` label | read-only |
| Harmonic | `harmonicOrder()` | **gated** — context menu "Change harmonic…" only |
| Input 0 | combo: `Laser` / other active stages / `Fixed`(+value) | editable |
| Input 1 | as Input 0; enabled only for SFG/DFG | editable |
| Final | exclusive single-select | editable |

`flags()` sets read-only columns to `Qt::ItemIsEnabled` only. Wiring/final edits
update per-experiment config and fire `edited()` — **never** auto-pushed to
hardware. A gated harmonic change emits `applyHarmonic(stageKey, n)` →
`HardwareManager` (mirror `applyClocks`→`setClocks`), which calls the stage's
`setHarmonicOrder(n)` driver hook (C-4) thread-aware (stages are `d_threaded`),
so a firmware-retunable unit changes its harmonic by command rather than a raw
setting write. `LifConversionWidget` carries the
preset bar (combo + Apply/Save/SaveAs/Rename/Delete → `LoadoutManager` LIF API)
and a preview footer live-assembling the chain expression + `outputRange` +
validation error. `setFromConfig(const LifConfig&)`/`toConfig(LifConfig&)`
bridge the model to the experiment.

### C-8. Page host

`ExperimentLifConfigPage` gains a `QTabWidget`: existing acquisition/control
content on one tab, `LifConversionWidget` on a "Conversion" tab (mirrors
`FtmwConfigWidget`'s tabbed layout). `apply()` writes the conversion into
`exp->lifConfig()`; the ctor seeds via `setFromConfig` (existing experiment) or
the current LIF preset (new). `enableLif()` (`experiment.cpp`) seeds
`exp->lifConfig()` conversion from the current LIF preset so the ExperimentType
scan-axis page has bounds before the LIF page is visited.

## Task breakdown (for delegation)

1. **Backend/data (keystone — blocks 3, 4).** C-1, C-2, C-3, C-5 (LifConfig API
   + `readTopologyFile`). Unit tests: snapshot `fromNodes`/`toNodes` round-trip,
   preset persistence round-trip through `LoadoutManager`, `liftopology.csv`
   write/read round-trip.
2. **Hardware strip + assembly helper (parallel with 1 after C-1/C-4 frozen).**
   C-4 (`conversionOp()`/`harmonicOrder()`, remove isFinal/inputs/nodeFromSettings,
   driver updates) + C-6 assembly helpers.
3. **HardwareManager + consumer rewire (depends on 1, 2).** C-6 consumer
   re-pointing: prep, `updateLifConversion`, `experimenttypepage`,
   `liflaserwidget`; add `applyHarmonic`→`setClocks`-style channel.
4. **Table UI (depends on 1).** C-7 model/delegate/widget + preset bar.
5. **Page host + wiring (depends on 3, 4).** C-8 tabbed page, `enableLif`
   seeding, `apply()`.

## Testing

- Unit (no hardware): snapshot round-trip; preset CRUD/persistence; assembly
  join (op/n from a stubbed hardware source + wiring) producing correct
  `laserToOutput`/`stageInput`; `liftopology.csv` write→read→assemble identity.
- Integration (virtual hardware): a virtual laser + virtual conversion stage,
  configure wiring in the table, run through prep, assert the joined
  `d_lifConversion` and the scan-axis bounds.
- Manual (bench): OH-band doubler co-tuning; gated harmonic change pushes to
  hardware; preset save/apply across sessions.

## Open items

- **`op` representation** — resolved toward a virtual `conversionOp()` constant
  on concrete drivers (baked into identity, hidden from the per-device settings
  UI), base reading the `op` setting only for generic Fixed/Virtual stages.
  Reconsider only if a real driver needs a user-selectable op.
- **Harmonic push semantics** — the gated change always routes through the
  `setHarmonicOrder(n)` driver hook (C-4). The base persists the setting; a
  firmware-retunable unit overrides to issue the command. Whether such a unit
  also re-tunes immediately vs. at next prep is per-driver.
- **Preset scope** — conversion wiring only for now; LIF digitizer/processing
  could join `LifPreset` later via the same machinery with no rework.
