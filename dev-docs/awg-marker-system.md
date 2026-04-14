# Generalized AWG Marker System

## Motivation

The current marker system is hardcoded to exactly 2 markers (protection and gate), with a
third (trigger) added as a special case for the AWG5204. AWGs support variable numbers of
marker channels (0 to 8+), and users need to control all of them for purposes beyond
protection/gate -- e.g., triggering digitizers or other devices timed relative to the chirp.

## Current State

- `ChirpConfig::getMarkerData()` returns `QVector<QPair<bool,bool>>` -- 2 booleans per sample
  (`src/data/experiment/chirpconfig.cpp:406-469`)
- `ChirpConfig::getTriggerData()` is a separate method with hardcoded 0.1 us lead time (AWG5204 special case)
  (`src/data/experiment/chirpconfig.cpp:471-520`)
- 4 timing parameters in `d_markers` struct: preProt, postProt, preGate, postGate
  (`src/data/experiment/chirpconfig.h:83-88`)
- Each AWG packs markers into implementation-specific bit positions (see per-implementation
  details in the Bit Remapping section below)
- AWG capability reported via 2 boolean settings keys: `BC::Key::AWG::prot` and
  `BC::Key::AWG::amp` (`src/data/settings/hardwarekeys.h:135-136`)
- Plot shows 2 fixed curves: "Protection" and "AmpEnable"
  (`src/gui/plot/chirpconfigplot.cpp:12-32`, `src/gui/plot/chirpconfigplot.cpp:39-91`)
- UI has 4 fixed spinboxes driving the 4 timing parameters
  (`src/gui/widget/chirpconfigwidget.cpp:12-115`)

## Marker Data Model

```cpp
enum class MarkerRole { Protection, Gate, Trigger, Custom };

struct MarkerChannel {
    QString name;              // user label, e.g. "Protection", "Dig Trigger"
    enum TimingMode { Absolute, ChirpRelative };
    TimingMode timingMode;     // Phase 1: always ChirpRelative
    double startTime;          // us -- relative to chirp start (negative = before)
    double endTime;            // us -- relative to chirp end (positive = after)
    bool enabled;
    MarkerRole role;           // drives safety validation
};
```

- `MarkerRole::Protection` and `MarkerRole::Gate` enable safety warnings: if protection is
  disabled while gate or chirp is active, warn the user (protection pulse prevents high-power
  chirp from reaching sensitive amplifier)
- `MarkerRole::Trigger` replaces the hardcoded `getTriggerData()` special case
- `MarkerRole::Custom` for any other use

### Timing Model

`startTime` is relative to the chirp start (negative = before chirp start). `endTime` is
relative to the chirp end (positive = after). The waveform lead time before each chirp in an
interval is `max(0, max(-m.startTime) for all enabled markers)`. Tail time after each chirp
end is `max(0, max(m.endTime) for all enabled markers)`.

Old parameters map to the new model as follows:
- Protection: `startTime = -(preProt + preGate)`, `endTime = postProt`
- Gate: `startTime = -preGate`, `endTime = postGate`

Default values when `markerCount >= 2`: both Protection and Gate use `startTime = -0.5`,
`endTime = 0.5` (matching the previous 0.5 μs default).

## Marker Generation API

`ChirpConfig` provides two output formats:

```cpp
// Structured: N markers x M samples -- clear, easy to inspect
QVector<QVector<bool>> getMarkerData() const;

// Packed: one quint32 per sample
// Bit ordering: bit 0 = marker channel 0, bit 1 = marker channel 1, etc. (LSB = channel 0)
// Convenience for AWGs that can use this bit ordering directly
QVector<quint32> getPackedMarkerData() const;
```

The `quint32` uses **LSB = channel 0**: bit 0 is marker channel 0, bit 1 is marker channel 1,
and so on. This is a logical ordering; hardware-specific remapping is done in each AWG
implementation (see Bit Remapping section).

`quint32` provides up to 32 marker channels -- far beyond current needs but avoids artificial
limits with negligible memory cost relative to the chirp waveform data.

`getTriggerData()` is removed. Add a channel with `MarkerRole::Trigger` instead.

## AWG Capability Reporting

Replace the `prot` and `amp` boolean settings keys with a single `markerCount` integer.
Each AWG implementation reports how many physical marker outputs it supports:

For Tektronix AWGxxxx models, the last digit of the model number equals the marker channel
count (e.g., AWG70002A → 2, AWG7122B → 2, AWG5204 → 4). These instruments use
MSB-first encoding: physical output 1 is the most significant used bit.

| Implementation | markerCount | Notes |
|----------------|-------------|-------|
| AWG70002A | 2 | Markers on physical outputs 1,2 → data bits 7,6 |
| AWG7122B | 2 | Markers on physical outputs 1,2 → data bits 7,6 |
| AWG5204 | 4 | Markers on physical outputs 1,2,3,4 → data bits 7,6,5,4 |
| M8195A | 2 | Markers on physical outputs 1,2 → data bits 0,1 |
| M8190 | 2 | Currently unused but hardware supports them |
| AD9914 | 0 | DDS -- no markers |
| VirtualAWG | 4 | For testing flexible marker counts |

When `markerCount >= 2`, the UI defaults to offering Protection on channel 0 and Gate on
channel 1. Users may disable, reassign, or leave them as-is.

## Bit Remapping in AWG Implementations

`getPackedMarkerData()` uses logical bit ordering (bit 0 = marker channel 0). Each AWG
implementation must extract bits from the `quint32` and pack them into whatever byte/word
format the hardware expects. The mapping between marker channel index and physical output
channel is fixed (channel 0 → physical output 1, channel 1 → physical output 2, etc.) because
that is how all existing hardware is physically wired. The remapping below accounts only for
the difference between the logical `quint32` bit ordering and the hardware's binary data
format.

### AWG70002A (`src/hardware/optional/chirpsource/awg70002a.cpp:201-330`)

Waveform and marker data sent as separate commands. MSB-first encoding: bit 7 = physical
output 1 (channel 0), bit 6 = physical output 2 (channel 1). Unused bits are zero.

Remapping from packed data:
```cpp
quint8 byte = ((packed >> 0) & 1) << 7   // channel 0 → bit 7
            | ((packed >> 1) & 1) << 6;  // channel 1 → bit 6
```

### AWG7122B (`src/hardware/optional/chirpsource/awg7122b.cpp:214-373`)

Waveform and marker interleaved as 5 bytes per sample (4-byte float + 1-byte marker). MSB-first
encoding: bit 7 = physical output 1 (channel 0), bit 6 = physical output 2 (channel 1).

Remapping from packed data (same as AWG70002A):
```cpp
quint8 byte = ((packed >> 0) & 1) << 7   // channel 0 → bit 7
            | ((packed >> 1) & 1) << 6;  // channel 1 → bit 6
```

### AWG5204 (`src/hardware/optional/chirpsource/awg5204.cpp:204-335`)

Waveform and marker data sent as separate commands. Has 4 marker outputs. MSB-first encoding:
bit 7 = physical output 1 (channel 0), bit 6 = physical output 2 (channel 1),
bit 5 = physical output 3 (channel 2), bit 4 = physical output 4 (channel 3). The trigger was
previously hardcoded as a third channel via `getTriggerData()` with a 0.1 μs lead time; it is
now a regular `MarkerRole::Trigger` channel (markerCount updated from the previously assumed 3
to the correct hardware value of 4).

Remapping from packed data:
```cpp
quint8 byte = ((packed >> 0) & 1) << 7   // channel 0 → bit 7
            | ((packed >> 1) & 1) << 6   // channel 1 → bit 6
            | ((packed >> 2) & 1) << 5   // channel 2 → bit 5
            | ((packed >> 3) & 1) << 4;  // channel 3 → bit 4
```

### M8195A (`src/hardware/optional/chirpsource/m8195a.cpp:68-230`)

Waveform (qint8) and marker (qint8) interleaved, 2 bytes per sample. Marker byte format:
bit 0 = physical output 1 (channel 0), bit 1 = physical output 2 (channel 1).

The logical `quint32` bit ordering already matches the hardware layout for the first 2
channels: bits 0 and 1 map directly to physical outputs 1 and 2. No remapping needed:
```cpp
qint8 markerVal = static_cast<qint8>(packed & 0x03);
```

### M8190 (`src/hardware/optional/chirpsource/m8190.cpp:67-257`)

Waveform interleaved as qint16 with markers in the lower bits. Marker code is currently
commented out; non-zero sample values have the last bit hardcoded to 1. With the new system,
marker channel 0 maps to the sample marker bit (bit 0 of the 16-bit word after the
chirp value shift). The exact bit layout should be verified against the M8190 programming
guide when re-enabling.

### AD9914 (`src/hardware/optional/chirpsource/ad9914.cpp`)

`markerCount = 0`. No marker data is read or written.

### VirtualAWG (`src/hardware/optional/chirpsource/virtualawg.cpp`)

`markerCount = 4`. No data is written to hardware; waveform data is computed but discarded.
Used for testing flexible marker counts in the UI.

## Storage

Add `static const QString markersFile{"markers.csv"};` to `BC::CSV` namespace in
`src/data/storage/blackchirpcsv.h` (alongside `chirpFile` on line 37).

Marker definitions stored in `markers.csv`, following the pattern of `clocks.csv`
and `chirps.csv`:

| Column | Type | Description |
|--------|------|-------------|
| Channel | int | Marker channel index (0-based) |
| Name | string | User-defined label |
| Role | string | Protection, Gate, Trigger, or Custom |
| TimingMode | string | ChirpRelative or Absolute |
| StartUs | double | Start time in us (relative to chirp start; negative = before) |
| EndUs | double | End time in us (relative to chirp end; positive = after) |
| Enabled | bool | Whether marker is active |

Read in `ChirpConfig::readMarkersFile()` (new, analogous to `readChirpFile` at
`src/data/experiment/chirpconfig.cpp:30`). Written in `ChirpConfig::writeMarkersFile()`
(new, analogous to `writeChirpFile` at `src/data/experiment/chirpconfig.cpp:84`).

The 4 old timing keys in `BC::Store::CC` (`preProt`, `postProt`, `preGate`, `postGate`) are
removed from `storeValues()`/`retrieveValues()` (`src/data/experiment/chirpconfig.cpp:676-698`).
Marker data is stored exclusively in `markers.csv`. `storeValues` keeps only `interval`,
`sampleRate`, and `sampleInterval`.

## UI Changes

- Replace the 4 fixed protection/gate spinboxes in `ChirpConfigWidget`
  (`src/gui/widget/chirpconfigwidget.cpp:50-102`) with a marker table (similar to
  `ChirpTableModel`) showing all configured markers
- Add/remove marker rows up to the active AWG's `markerCount` (read via
  `SettingsStorage(BC::Key::AWG::key, SettingsStorage::Hardware).get(BC::Key::AWG::markerCount, 0)`)
- Each row: channel index, name (editable), role (combo), start time (spinbox), end time
  (spinbox), enabled (checkbox)
- `ChirpConfigPlot` (`src/gui/plot/chirpconfigplot.cpp:12-91`) draws one curve per enabled
  marker (dynamically, not hardcoded 2)
- Safety validation: warn if Protection role is disabled while Gate or chirp is active

## Python Hardware Impact

When the generalized marker system is implemented, `PythonAwg` will need updates:

- **`configParams()` (`src/hardware/python/pythonawg.cpp:22-24`)**: Add a `markerCount`
  integer param (replacing the `prot`/`amp` booleans) so the user can declare how many
  physical marker channels their AWG has.

- **`prepareForExperiment()` IPC payload (`src/hardware/python/pythonawg.cpp:112-195`)**:
  Serialize the marker channel definitions (name, role, timing mode, start/end times, enabled)
  into `config['chirp']['markers']` as a compact list. Do **not** pre-compute sample arrays
  in C++ -- follow the same design as the chirp waveform (send parameters, compute in Python).

- **`python_awg_template.py` (`src/hardware/python/python_awg_template.py:88-162`)**:
  The `_compute_markers()` helper currently implements the hardcoded 2-channel
  (protection/gate) logic. It must be updated to iterate over `config['chirp']['markers']`
  and compute one boolean array per channel from the timing definitions. The
  `_compute_waveform()` helper is unaffected.

## Implementation Phases

**Phase 1: Generalized markers (chirp-relative, global)**

Step 1 — Data model (`chirpconfig.h/.cpp`, `hardwarekeys.h`, `blackchirpcsv.h`,
`chirpconfigplot.h/.cpp`, `chirpconfigwidget.cpp`, `ftmwconfig.cpp`): **COMPLETE**
- [x] Add `MarkerRole` enum and `MarkerChannel` struct to `chirpconfig.h`
- [x] Replace `d_markers` struct with `QVector<MarkerChannel> d_markerChannels`
- [x] Remove old `preChirpProtectionDelay()` / `postChirpProtectionDelay()` / etc. accessors;
  update all callers in `chirpconfigplot.cpp`, `chirpconfigwidget.cpp`, and `ftmwconfig.cpp`
- [x] Rewrite `getMarkerData()` → `QVector<QVector<bool>>`
- [x] Add `getPackedMarkerData()` → `QVector<quint32>` (bit 0 = channel 0)
- [x] Remove `getTriggerData()`
- [x] Update `totalDuration()` to derive lead/tail from marker startTime/endTime
- [x] Update `storeValues()`/`retrieveValues()` -- remove the 4 old keys
- [x] Add `readMarkersFile()` / `writeMarkersFile()`
- [x] Add `BC::Key::AWG::markerCount` to `hardwarekeys.h`; remove `prot` and `amp`
- [x] Add `BC::CSV::markersFile` to `blackchirpcsv.h`
- [x] Add `leadTimeUs()` / `tailTimeUs()` public accessors; add `findEnabledMarkerByRole()`
- [x] `ChirpConfigPlot::newChirp()`: dynamic curve per marker channel, labeled
  "Marker N" or "Marker N (Role)"; `ftmwconfig.cpp` uses Trigger marker timing with
  fallback to `leadTimeUs()`

Step 2 — AWG implementations (all 7 + PythonAwg): **COMPLETE**
- [x] Replace `prot`/`amp` SETTING_DEF entries with `markerCount` in each constructor
- [x] Switch each `writeWaveform`/`prepareForExperiment` to call `getPackedMarkerData()` and
  apply the per-implementation bit remapping shown above
- [x] AWG5204: remove `getTriggerData()` call; trigger is now a regular channel
- [x] PythonAwg: serialize marker channel definitions into `config['chirp']['markers']` JSON array

Step 3 — UI: **COMPLETE**
- [x] Add `MarkerTableModel` (`src/data/model/markertablemodel.h/.cpp`), analogous to
  `ChirpTableModel`; inherits `SettingsStorage` for last-used persistence across sessions
- [x] Replace 4 fixed spinboxes in `chirpconfigwidget.ui` + `ChirpConfigWidget` with a
  `QTabWidget`: "Chirp Segments" tab (existing controls) and "Markers" tab (`markerTable`)
- [x] Marker tab hidden when AWG `markerCount == 0`
- [x] `ExperimentChirpConfigPage::validate()` emits warnings when protection marker is
  absent/disabled, does not cover the full chirp, or does not enclose the amp enable pulse

Step 4 — Python AWG:
- `_compute_markers()` in `python_awg_template.py`

**Phase 2: Absolute timing and per-chirp markers**
- Enable `TimingMode::Absolute` in UI
- Extend marker storage to per-chirp definitions (marker table becomes segment-aware,
  similar to chirp segment table)
- Per-chirp marker overrides with an "apply to all" default
