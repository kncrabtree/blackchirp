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
count (e.g., AWG70002A → 2, AWG7122B → 2, AWG5204 → 4). The bit encoding differs by
model: AWG70002A and AWG7122B place physical output 1 in bit 6 and output 2 in bit 7;
AWG5204 uses MSB-first ordering (bit 7 = output 1, bit 6 = output 2, etc.).

| Implementation | markerCount | Notes |
|----------------|-------------|-------|
| AWG70002A | 2 | Markers on physical outputs 1,2 → data bits 6,7 |
| AWG7122B | 2 | Markers on physical outputs 1,2 → data bits 6,7 |
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

Waveform and marker data sent as separate commands. Bit 6 = physical output 1 (channel 0),
bit 7 = physical output 2 (channel 1). Unused bits are zero.

Remapping from packed data:
```cpp
quint8 byte = ((packed >> 0) & 1) << 6   // channel 0 → bit 6
            | ((packed >> 1) & 1) << 7;  // channel 1 → bit 7
```

### AWG7122B (`src/hardware/optional/chirpsource/awg7122b.cpp:214-373`)

Waveform and marker interleaved as 5 bytes per sample (4-byte float + 1-byte marker). Bit 6 =
physical output 1 (channel 0), bit 7 = physical output 2 (channel 1).

Remapping from packed data (same as AWG70002A):
```cpp
quint8 byte = ((packed >> 0) & 1) << 6   // channel 0 → bit 6
            | ((packed >> 1) & 1) << 7;  // channel 1 → bit 7
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

Step 4 — Python AWG: **COMPLETE**
- [x] Rewrite `_compute_markers()` in `python_awg_template.py` to iterate over
  `config['chirp']['markers']` and return `(indices, arrays)`: one bool array per
  enabled channel plus the corresponding original channel indices
- [x] Add `_compute_markers_packed()`: packs the per-channel arrays into a `uint32`
  array (LSB = channel 0), matching `ChirpConfig::getPackedMarkerData()`
- [x] Update `_compute_waveform()` to derive lead/tail times from the markers list
  (replaces the removed `pre_chirp_*`/`post_chirp_*` keys)
- [x] Update all docstrings to reflect the new `markers` list format and return types

## Phase 2 — Absolute Timing and Per-Chirp Markers

### Status

**Deferred.** Phase 1 covers the concrete known use cases (generalized channel count,
trigger-as-marker, flexible role assignment). Phase 2 adds two independent but related
capabilities — absolute timing and per-chirp overrides — that together roughly double
the surface area of the marker system. The plan below is preserved so the work can be
resumed once a user requests a feature Phase 1 cannot express. Do not implement it
speculatively.

The two capabilities have very different costs and should be considered separately:

1. **Absolute timing alone** — moderate scope, mostly isolated to `ChirpConfig`,
   `ChirpConfigPlot`, validation, and the Python template. Viable as a standalone phase.
2. **Per-chirp marker overrides** — large scope, touches the storage schema, the
   UI model, validation, waveform hashing, and "all chirps identical" semantics. The
   bulk of Phase 2's risk lives here.

### Known Use Cases (to justify implementation)

Before building Phase 2, collect at least one concrete user scenario that Phase 1
cannot express. Candidates:

- **Absolute**: fire a single digitizer-arm pulse at a fixed waveform offset (e.g.
  8 µs after the first chirp begins) without wanting it to repeat per chirp. Phase 1
  always repeats markers per chirp interval.
- **Per-chirp**: in a multi-chirp waveform with different per-chirp durations, drive
  a device-specific trigger on only one chirp, or with different timing per chirp.
  Phase 1 forces identical marker timing across all chirps.

If neither is actually requested, leave Phase 2 deferred.

### Feature 2A — Absolute Timing

#### Data Model Changes

The `MarkerChannel::timingMode` field already exists from Phase 1; only the
`Absolute` branch needs to be honored.

Define the origin: **absolute t = 0 is the first chirp start**, not waveform sample 0.
This matches the existing chirp-relative convention (`startTime` is relative to the
chirp start), keeps the relationship "ChirpRelative with start = 0 == Absolute with
start = 0 on chirp 0" intuitive, and means an absolute marker's window is
`[startTime, endTime]` in "elapsed time since first chirp start" coordinates.

Invariants:

- An absolute marker fires **once** over the whole waveform, regardless of
  `numChirps`.
- Negative `startTime` means "before the first chirp begins" — this extends the
  waveform lead.
- `endTime` > `(numChirps - 1) * chirpInterval + lastChirpDur` extends the waveform
  tail.

#### `leadTimeUs` / `tailTimeUs` / `totalDuration`

These three functions become the bulk of the algorithmic complexity.

```cpp
double ChirpConfig::leadTimeUs() const
{
    double lead = 0.0;
    for (const auto &m : d_markerChannels) {
        if (!m.enabled) continue;
        if (m.timingMode == MarkerChannel::ChirpRelative)
            lead = qMax(lead, -m.startTime);
        else // Absolute
            lead = qMax(lead, -m.startTime);  // same formula, different meaning
    }
    return qMax(0.0, lead);
}
```

Chirp-relative `startTime` is relative to each chirp start, and the first chirp
starts at `lead`. Absolute `startTime` is already measured from "first chirp start"
(= `lead`). So `-m.startTime` still yields the required pre-roll for both modes,
and `leadTimeUs` collapses to the existing Phase 1 implementation. **No change
needed** — this was the reason for choosing "first chirp start" as the origin.

```cpp
double ChirpConfig::tailTimeUs() const
{
    double chirpRelTail = 0.0;
    double absExtra = 0.0;
    double lastChirpEndAbs = (numChirps() - 1) * d_chirpInterval + chirpDurationUs(numChirps() - 1);
    for (const auto &m : d_markerChannels) {
        if (!m.enabled) continue;
        if (m.timingMode == MarkerChannel::ChirpRelative) {
            chirpRelTail = qMax(chirpRelTail, m.endTime);
        } else {
            // Absolute endTime is waveform-absolute (from first chirp start).
            // Tail beyond the last chirp end is (m.endTime - lastChirpEndAbs).
            absExtra = qMax(absExtra, m.endTime - lastChirpEndAbs);
        }
    }
    return qMax(0.0, qMax(chirpRelTail, absExtra));
}
```

`totalDuration` continues to work as-is (`lead + (N-1)*interval + lastChirpDur + tail`)
because both modes collapse into the same `leadTimeUs` / `tailTimeUs` outputs.

#### `getMarkerData` Algorithm

The current per-interval loop handles chirp-relative markers. Absolute markers need
to be written **after** the per-interval loop as a direct sample-range fill per
channel, since they are waveform-global and do not repeat:

```cpp
// (existing interval loop for chirp-relative markers — unchanged)

// Overlay absolute markers
double firstChirpStartAbs = leadTimeUs();
for (int ch = 0; ch < numChannels; ++ch) {
    const auto &m = d_markerChannels.at(ch);
    if (!m.enabled || m.timingMode != MarkerChannel::Absolute) continue;

    double absStart = firstChirpStartAbs + m.startTime;  // in waveform sample time
    double absEnd   = firstChirpStartAbs + m.endTime;
    int s0 = getFirstSample(absStart);
    int s1 = getLastSample(absEnd);
    s0 = qMax(0, s0);
    s1 = qMin(numSamples - 1, s1);
    for (int s = s0; s <= s1; ++s)
        out[ch][s] = true;
}
```

The per-interval loop skips channels with `timingMode == Absolute` so the two
branches do not overwrite each other.

#### `ChirpConfigPlot::newChirp`

The existing plot draws one rectangle per chirp per marker. For absolute markers,
draw a single rectangle at `(firstChirpStartAbs + m.startTime, firstChirpStartAbs + m.endTime)`
instead of iterating per-chirp. Add a branch on `m.timingMode` inside the existing
channel loop.

#### Validation

`ExperimentChirpConfigPage::validate` currently compares `prot->startTime` /
`prot->endTime` directly against the chirp (assumes chirp-relative). For absolute
protection markers, translate the window into per-chirp coverage and check that
*every* chirp is fully enclosed:

```cpp
for (int i = 0; i < cc.numChirps(); ++i) {
    double chirpStartAbs = i * cc.chirpInterval();         // relative to first chirp start
    double chirpEndAbs   = chirpStartAbs + cc.chirpDurationUs(i);
    bool covered = false;
    for (const auto &m : cc.markerChannels()) {
        if (!m.enabled || m.role != MarkerRole::Protection) continue;
        if (m.timingMode == MarkerChannel::ChirpRelative) {
            if (m.startTime < 0.0 && m.endTime > 0.0) { covered = true; break; }
        } else {
            if (m.startTime <= chirpStartAbs && m.endTime >= chirpEndAbs) { covered = true; break; }
        }
    }
    if (!covered)
        emit warning(QString("Chirp %1 is not fully enclosed by an enabled protection marker.").arg(i + 1));
}
```

The gate-enclosure check needs the same treatment, pairwise against any active gate.

#### Python IPC and Template

`PythonAwg::prepareForExperiment` adds a `timing_mode` string (`"chirp_relative"` or
`"absolute"`) to each marker dict in `config['chirp']['markers']`.

`_compute_markers()` in `python_awg_template.py` splits the enabled markers into
two lists and handles them in parallel. `_compute_waveform()`'s lead/tail derivation
already uses `max(0, -m['start_us'])` / `max(0, m['end_us'])`, which remains correct
because of the "first chirp start = 0" origin choice.

#### Storage

`markers.csv` already has a `TimingMode` column from Phase 1 (read/write code is in
`readMarkersFile`/`writeMarkersFile`). No schema change needed for Feature 2A alone.

#### Hash

`waveformHash` must include `m.timingMode` so two configs that differ only by timing
mode hash differently. Add one line:

```cpp
c.addData(QByteArray::number(static_cast<int>(m.timingMode)));
```

#### UI

Add a "Mode" column to `MarkerTableModel` (Absolute / ChirpRelative combo box).
Update the tooltips on the Start (µs) and End (µs) columns to say what the value
means in each mode — this is the hardest UX part. Suggested wording:

- ChirpRelative Start: "µs relative to each chirp's start (negative = before)"
- ChirpRelative End:   "µs relative to each chirp's end (positive = after)"
- Absolute Start:      "µs from the first chirp's start (negative = before)"
- Absolute End:        "µs from the first chirp's start"

Consider making the headers change ("Start (rel)" / "End (abs)") when the active
row's mode changes, or add a per-cell tooltip that reflects the row's current mode.

Default when the user switches a row from ChirpRelative to Absolute: compute the
equivalent absolute window for chirp 0 so the marker does not visually jump.

#### Feature 2A Scope Summary

| Area | LOC | Risk |
|------|-----|------|
| `chirpconfig.h/cpp` (data + marker gen + hash) | ~80 | Medium — edge cases in multi-chirp absolute coverage |
| `chirpconfigplot.cpp` | ~30 | Low |
| `experimentchirpconfigpage.cpp` (validation) | ~60 | **High** — safety-critical; must be tested against real hardware |
| `markertablemodel.h/cpp` (mode column + delegate) | ~80 | Medium — UX clarity |
| `pythonawg.cpp` + `python_awg_template.py` | ~50 | Low |
| Unit tests | ~150 | Should add coverage for absolute + mixed cases |

**AWG implementations (AWG70002A, AWG7122B, AWG5204, M8195A, M8190, VirtualAWG):
no changes.** The packed `quint32` interface insulates them completely from the
timing-mode dimension. This is by design and a strong argument for the current
Phase 1 abstraction.

**Estimated effort for Feature 2A only: 2–3 days.**

### Feature 2B — Per-Chirp Marker Overrides

This is where the scope explodes. Proceed with caution; consider whether Feature 2A
alone satisfies the user's actual need before tackling per-chirp overrides.

#### Data Model

The cleanest model is a **default + overrides** structure per channel:

```cpp
struct MarkerChannel {
    QString name;
    MarkerRole role{MarkerRole::Custom};
    bool enabled{true};
    // Default entry applied to every chirp unless overridden
    MarkerTiming defaultTiming;
    // Sparse per-chirp overrides keyed by chirp index
    QMap<int, MarkerTiming> overrides;
};

struct MarkerTiming {
    enum Mode { Absolute, ChirpRelative };
    Mode mode{ChirpRelative};
    double startTime{-0.5};
    double endTime{0.5};
    // An override may also disable the marker for that specific chirp
    bool enabled{true};
};
```

Rationale over the alternative (`QVector<QVector<MarkerChannel>>` — one per-chirp
full channel set):

- Preserves the notion that a channel is a physical output with a fixed role and
  name, independent of chirp.
- Keeps the common case (same timing on every chirp) compact.
- Validation reasons across "chirp i's effective timing" rather than searching a
  flat list for matches.
- Matches the UX model of "default + exceptions" that users of the chirp segment
  table already know (`d_allIdentical` + `currentChirpBox`).

Trade-off: an extra indirection when generating marker samples (for each chirp i,
look up `overrides.value(i, defaultTiming)`).

#### Interaction with Absolute Mode

**Decision:** per-chirp overrides only apply to ChirpRelative timing. Absolute
markers are by definition waveform-global and cannot have per-chirp overrides
(they would not have a meaningful "chirp index" to key on). Enforce this in the
UI (overrides column is disabled when `defaultTiming.mode == Absolute`) and in
validation (`overrides.isEmpty()` required when default mode is Absolute).

#### `allChirpsIdentical`

Currently this function checks only segment shape. With per-chirp markers it must
also consider marker overrides — otherwise the "apply to all" checkbox semantics
break. Add a parallel `allChirpMarkersIdentical()` helper, or extend the existing
one.

#### Waveform Generation

`getMarkerData`'s per-interval loop must fetch the effective timing for channel
`ch` on interval `currentInterval` on every iteration:

```cpp
const MarkerChannel &m = d_markerChannels.at(ch);
const MarkerTiming &t = m.overrides.value(currentInterval, m.defaultTiming);
if (!t.enabled) continue;
markerStartSample[ch] = getFirstSample(chirpStartTime + t.startTime);
markerEndSample[ch]   = getLastSample(chirpEndTime + t.endTime) - 1;
```

This is a small code change but the validation story is what grows.

#### `leadTimeUs` / `tailTimeUs`

Must also walk `overrides`:

```cpp
double ChirpConfig::leadTimeUs() const {
    double lead = 0.0;
    for (const auto &m : d_markerChannels) {
        if (!m.enabled) continue;
        if (m.defaultTiming.enabled)
            lead = qMax(lead, -m.defaultTiming.startTime);
        for (const auto &t : m.overrides)
            if (t.enabled)
                lead = qMax(lead, -t.startTime);
    }
    return qMax(0.0, lead);
}
```

Absolute markers still behave as described in Feature 2A; they do not participate
in overrides.

#### Storage Schema

Add a `ChirpIndex` column to `markers.csv`; -1 means "default for all chirps", 0..N-1
means "override for chirp i". An entry with `ChirpIndex != -1` and
`TimingMode == "Absolute"` is an error on read and should be rejected with a log
warning.

**Backward compatibility:** Phase 1 `markers.csv` files have no `ChirpIndex` column.
The reader must detect the old format (6 columns + header without `ChirpIndex`) and
treat every row as a default entry for its `Channel`. Add a test case for this.

#### Validation

Protection validation must check **each chirp index separately**:

```cpp
for (int i = 0; i < cc.numChirps(); ++i) {
    // Find effective timings for protection and gate on chirp i
    // (default + override lookups across all channels)
    // Check: protection exists, covers chirp i, encloses gate on chirp i
}
```

This is where the validation logic stops fitting on a screen. Expect the validator
to grow from ~70 lines to ~150–200 lines. It is safety-critical and should have
dedicated unit tests (table-driven: N chirps × M channels × pathological overrides).

#### UI

**This is the biggest single piece of Phase 2B work.** The marker table must
become chirp-aware while remaining readable.

Recommended approach: mirror the chirp segment table UX.

- Add an "Apply to all chirps" checkbox next to the marker table.
- When checked (default), `MarkerTableModel` shows a single marker row per channel
  editing `defaultTiming` only. Overrides are hidden and cleared on commit.
- When unchecked, reuse the existing `currentChirpBox` (or add a marker-scoped one)
  to choose which chirp's effective timing is shown. Rows edit
  `overrides.value(currentChirp, defaultTiming)`; an "Override" column lets the user
  mark a row as overridden, at which point edits write to the override map instead
  of the default.
- Rows with no override appear greyed out and display "(default)" in the override
  column.

Alternative (rejected): give each chirp its own tab. Too many chirps makes this
unusable, and it loses the visual equivalence between chirps.

`ChirpConfigPlot::newChirp` walks chirp intervals already; change the inner
marker loop to use the chirp-specific effective timing. Absolute markers still
draw a single rectangle in waveform-absolute coordinates.

#### Python IPC

Two options:

1. **Preserve flat list, annotate with chirp index** — each marker dict gains
   `chirp_index` (-1 for default, 0..N-1 for override). Simple to serialize.
2. **Nested structure** — each channel dict contains `default: {...}` and
   `overrides: [{chirp, ...}]`. More explicit, slightly more work.

Option 1 is simpler and lets the Python template iterate a single list.
`_compute_markers()` becomes:

```python
for i in range(num_chirps):
    for ch_idx in range(num_channels):
        timing = _effective_timing(markers, ch_idx, i)  # override or default
        if not timing['enabled']: continue
        # fill arrays[ch_idx][sample_range] = True
```

The template's docstring and return shape (indices + per-channel arrays) remain
unchanged — the per-chirp complexity is fully hidden inside the helper.

#### `waveformHash`

Must include `overrides`:

```cpp
for (const auto &m : d_markerChannels) {
    // ... existing default fields ...
    for (auto it = m.overrides.cbegin(); it != m.overrides.cend(); ++it) {
        c.addData(QByteArray::number(it.key()));
        // hash the override timing
    }
}
```

#### Feature 2B Scope Summary

| Area | LOC | Risk |
|------|-----|------|
| `chirpconfig.h/cpp` (data model + marker gen + hash + lead/tail) | ~150 | Medium |
| `experimentchirpconfigpage.cpp` (validation) | ~150 | **Very high** — safety-critical; per-chirp combinatorics expand the bug surface |
| `markertablemodel.h/cpp` + delegate (override editing, override column, apply-to-all) | ~250 | **High** — biggest piece of work; UX ambiguity |
| `chirpconfigwidget.cpp` (sync currentChirp between tabs) | ~40 | Medium — easy for users to miss which chirp they are editing |
| `chirpconfigplot.cpp` | ~20 | Low |
| `blackchirpcsv.h` + read/write markers.csv (+ back-compat) | ~60 | Low–Medium |
| `pythonawg.cpp` + `python_awg_template.py` | ~80 | Medium |
| Unit tests (per-chirp validation, round-trip storage, back-compat read) | ~300 | Must add |

**AWG implementations: still no changes.** The `quint32` packed interface remains
the isolation boundary. This is the single biggest reason Phase 2B remains
feasible at all.

**Estimated effort for Feature 2B: 4–6 days**, dominated by the validator rewrite
and the UI model.

### Risks

1. **Validation is safety-critical.** Protection markers prevent damage to the
   amplifier. Every change to `validate()` must be covered by a table-driven unit
   test. The Phase 2B validator is large enough that bugs are likely; allocate
   time for real-hardware verification, not just unit tests.

2. **UI for per-chirp overrides is easy to misuse.** A user editing "chirp 3"'s
   markers while the chirp segment tab still shows chirp 1 can create silent
   mismatches. Mitigation: either sync the "current chirp" selector between the
   segment tab and the marker tab, or forbid override editing entirely when
   "apply to all" is checked.

3. **Absolute timing semantics are subtle.** The origin choice ("absolute 0 =
   first chirp start") is correct and keeps Phase 1 behavior unchanged, but users
   will expect "absolute 0 = waveform sample 0". Document prominently and
   consider a tooltip or example on the UI. Explicit unit tests for the origin
   choice.

4. **Backward compatibility of `markers.csv`.** Any Phase 2 schema change must
   read Phase 1 files. Add a regression test that loads a Phase 1 `markers.csv`
   and produces the expected Phase 2 data.

5. **`waveformHash` regression.** If any new field is forgotten in the hash,
   experiments with the same chirp segments but different new-field markers will
   collide. Add a test.

6. **`allChirpsIdentical` semantic drift.** Callers of this function (e.g.
   `chirpconfigwidget.cpp:136`) expect segment identity; changing it to include
   markers may alter the "apply to all" checkbox state unexpectedly. Consider a
   separate `allChirpMarkersIdentical()` method instead.

7. **Python template backward compatibility.** Users with customized
   `python_awg_template.py` scripts will need to port their changes when
   `_compute_markers()`'s helper shape evolves. Keep the return type stable
   (`(indices, arrays)`) even if the internal implementation changes.

### Recommendation

**Defer Phase 2 until a specific user request justifies it.** Phase 1's generalized
channel count with chirp-relative timing already addresses:

- Flexible use of all AWG marker outputs (up to 4 on AWG5204, more elsewhere)
- Trigger-as-marker (the AWG5204 special case is gone)
- Custom marker channels with user-defined names and roles
- Safety validation for protection and gate enclosure
- Per-AWG capability reporting via `markerCount`

The remaining Phase 2 capability — **absolute timing** and **per-chirp
overrides** — is elegant but speculative. Neither has a currently-known user
asking for it. The implementation cost is ~6–9 days total (2–3 for 2A + 4–6 for
2B), and the long-tail risk lives in the safety validator.

When a real use case appears:

- **If the use case needs absolute timing only** (e.g., "fire one pulse once
  per waveform, not once per chirp"), implement Feature 2A. It is self-contained
  and comparatively low-risk.
- **If the use case needs per-chirp markers** (e.g., "different trigger on chirp
  3 vs chirp 1 in a multi-chirp waveform"), implement 2A first, then 2B. Do not
  try to ship 2B without 2A — absolute timing is a much cleaner way to express
  "once over the waveform" than a per-chirp override workaround.

### Things That Are Explicitly *Not* Phase 2

- Per-segment markers (markers whose timing is relative to segment boundaries
  within a chirp). Not part of this plan. Would need a different data model
  again.
- Programmable marker pulse trains within a window (pulse width / rep rate /
  count). Out of scope; use the pulse generator for that.
- Independent enable toggles per chirp for a channel *without* changing timing.
  Already covered by `overrides[i].enabled = false` in the 2B model.
