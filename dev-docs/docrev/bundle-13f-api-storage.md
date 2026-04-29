# Bundle 13f — API Reference: Storage Classes

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Adds API reference pages for the storage subsystem.

## Scope

New pages under `doc/source/classes/`:

- `datastoragebase.rst` ← `src/data/storage/datastoragebase.h`
- `fidstoragebase.rst` ← `src/data/storage/fidstoragebase.h`
- `blackchirpcsv.rst` ← `src/data/storage/blackchirpcsv.h`
- `waveformbuffer.rst` ← `src/data/storage/waveformbuffer.h`
  (`WaveformBuffer`, `WaveformEntry`).
- `auxdatastorage.rst` ← `src/data/storage/auxdatastorage.h`
- `overlaystorage.rst` (if a separate header exists; otherwise
  fold into `overlaybase.rst` in bundle 13e and skip).
- `loghandler.rst` ← location of `LogHandler` (likely
  `src/data/loghandler.h`).
- `applicationconfigmanager.rst` ←
  `src/data/storage/applicationconfigmanager.h`.

`SettingsStorage` and `HeaderStorage` are already covered in
13a and are not touched here.

## Out of scope

- The data classes themselves (bundle 13e).

## Sources

- `dev-docs/digitizer-data-flow.md` — primary for
  `WaveformBuffer` semantics.
- `dev-docs/string-usage.md` — for the `LogHandler` API surface
  (severity guidelines).
- The header files.

## Sphinx file deltas

**Created:** one per page above.

**Possibly modified (Doxygen comment refresh):**
- All headers listed above.

## Acceptance criteria

- `WaveformBuffer` page documents the SPSC discipline,
  drop-newest overflow, the `WaveformEntry` schema (data,
  shotCount, preAccumulated), the QSemaphore notification, and
  the `isFull()` backpressure check.
- `LogHandler` page documents the free-function API
  (`bcLog`/`bcDebug`/`bcWarn`/`bcError`/`bcHighlight`) and the
  `hw*` helpers used inside `HardwareObject` subclasses.
- `BlackchirpCSV` page enumerates the canonical filename
  constants (`chirpFile`, `markersFile`, `clockFile`, etc.).
- `ApplicationConfigManager` page documents the runtime LIF
  toggle and the runtime debug-logging toggle.
