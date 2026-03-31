# Generalized AWG Marker System

## Motivation

The current marker system is hardcoded to exactly 2 markers (protection and gate), with a
third (trigger) added as a special case for the AWG5204. AWGs support variable numbers of
marker channels (0 to 8+), and users need to control all of them for purposes beyond
protection/gate -- e.g., triggering digitizers or other devices timed relative to the chirp.

## Current State

- `ChirpConfig::getMarkerData()` returns `QVector<QPair<bool,bool>>` -- 2 booleans per sample
- `ChirpConfig::getTriggerData()` is a separate method with hardcoded 0.1 us lead time (AWG5204 special case)
- 4 timing parameters in `d_markers` struct: preProt, postProt, preGate, postGate
- Each AWG packs markers into implementation-specific bit positions
- AWG capability reported via 2 booleans: `prot` and `amp`
- Plot shows 2 fixed curves: "Protection" and "AmpEnable"

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

## Marker Generation API

ChirpConfig provides two output formats:

```cpp
// Structured: N markers x M samples -- clear, easy to inspect
QVector<QVector<bool>> getMarkerData() const;

// Packed: one quint32 per sample, marker 0 = bit 0, marker 1 = bit 1, etc.
// Convenience for AWGs that can use this bit ordering directly
QVector<quint32> getPackedMarkerData() const;
```

AWG implementations are responsible for remapping bits to their hardware-specific layout
if the default packing doesn't match (e.g., Tektronix AWGs use bits 6/7, not 0/1).

`quint32` provides up to 32 marker channels -- far beyond current needs but avoids artificial
limits with negligible memory cost relative to the chirp waveform data.

## AWG Capability Reporting

Replace the `prot` and `amp` boolean settings keys with a single `markerCount` integer.
Each AWG implementation reports how many physical marker outputs it supports:

| Implementation | markerCount | Notes |
|----------------|-------------|-------|
| AWG70002A | 2 | Markers on bits 6,7 |
| AWG7122B | 2 | Markers on bits 6,7 |
| AWG5204 | 3 | Markers on bits 5,6,7 (trigger was bit 5) |
| M8195A | 2 | Markers on bits 0,1 |
| M8190 | 2 | Currently unused but hardware supports them |
| AD9914 | 0 | DDS -- no markers |
| VirtualAWG | 4 | For testing flexible marker counts |

When `markerCount >= 2`, the UI defaults to offering Protection on channel 1 and Gate on
channel 2. Users may disable, reassign, or leave them as-is.

## Storage

Marker definitions stored in `markers.csv` (new file), following the pattern of `clocks.csv`
and `chirps.csv`:

| Column | Type | Description |
|--------|------|-------------|
| Channel | int | Marker channel index (0-based) |
| Name | string | User-defined label |
| Role | string | Protection, Gate, Trigger, or Custom |
| TimingMode | string | ChirpRelative or Absolute |
| StartUs | double | Start time in us |
| EndUs | double | End time in us |
| Enabled | bool | Whether marker is active |

## UI Changes

- Replace the 4 fixed protection/gate spinboxes in `ChirpConfigWidget` with a marker table
  (similar to `ChirpTableModel`) showing all configured markers
- Add/remove marker rows up to the active AWG's `markerCount`
- Each row: channel, name, role (combo), timing mode (combo), start, end, enabled (checkbox)
- `ChirpConfigPlot` draws one curve per enabled marker (dynamically, not hardcoded 2)
- Safety validation: warn if Protection role is disabled while Gate or chirp is active

## Implementation Phases

**Phase 1: Generalized markers (chirp-relative, global)**
- Replace `d_markers` struct with `QVector<MarkerChannel>`
- Implement `getMarkerData()` returning `QVector<QVector<bool>>`
- Implement `getPackedMarkerData()` returning `QVector<quint32>`
- Remove `getTriggerData()` -- fold into general marker system
- Replace `prot`/`amp` AWG settings with `markerCount`
- Update all AWG implementations to use new API and remap bits as needed
- Update UI: marker table, dynamic plot curves, role-based warnings
- Storage: `markers.csv` for experiment data
- Default Protection (channel 0) and Gate (channel 1) when `markerCount >= 2`

## Python Hardware Impact

When the generalized marker system is implemented, `PythonAwg` will need updates:

- **`configParams()`**: Add a `markerCount` integer param (replacing the `prot`/`amp`
  booleans) so the user can declare how many physical marker channels their AWG has.
  The dialog will write this value to QSettings before construction, mirroring the
  pattern used by other trampolines for constructor-time parameters.

- **`prepareForExperiment()` IPC payload**: Serialize the marker channel definitions
  (name, role, timing mode, start/end times, enabled) into `config['chirp']['markers']`
  as a compact list. Do **not** pre-compute sample arrays in C++ — follow the same
  design as the chirp waveform (send parameters, compute in Python).

- **`python_awg_template.py`**: The `_compute_markers()` helper currently implements
  the hardcoded 2-channel (protection/gate) logic. It must be updated to iterate over
  `config['chirp']['markers']` and compute one boolean array per channel from the
  timing definitions. The `_compute_waveform()` helper is unaffected.

**Phase 2: Absolute timing and per-chirp markers**
- Enable `TimingMode::Absolute` in UI
- Extend marker storage to per-chirp definitions (marker table becomes segment-aware,
  similar to chirp segment table)
- Per-chirp marker overrides with an "apply to all" default
