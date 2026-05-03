# Bundle 13f — API Reference: Storage Classes

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-05-02: drafted → complete. Content commit 9619aef1. User review surfaced class-level Doxygen / RST-intro duplication on `DataStorageBase` and `AuxDataStorage`; both class-level blocks trimmed to a tight `\brief` (RST owns orientation prose, header carries the per-method spec). User also flagged broader drift between earlier bundles (notably 13a `SettingsStorage`, header-heavy) and the later RST-intro-heavy convention; bundle 14 picks up a new scope item ("API page intro / header-comment harmonization") to align all `doc/source/classes/` pages and update `api_style.rst` accordingly.
- 2026-05-02: in progress → drafted. Three parallel Sonnet drafter/verifier pairs landed eight new RST pages and six refreshed headers; a final coherence reviewer surfaced cross-page asymmetries. Punch-list items resolved: `auxdatastorage.rst` falsely listed `TimePointData` under `Q_DECLARE_METATYPE` (only `AuxDataMap` is registered) — fixed; `waveformbuffer.rst` "earlier" temporal marker — fixed; `datastoragebase.rst` listed `AuxDataStorage` as a direct subclass — fixed (it is a member-of `Experiment`); `ftmwconfig.rst` (bundle 13e) gained the deferred `:cpp:class:` link to `WaveformBuffer` per the 13e status-log note. Doc build surfaced three more: `\section` markers in `fidstoragebase.h` and `lifstorage.h` collided with RST H2 labels via Sphinx `autosectionlabel` (class-level Doxygen blocks trimmed to remove the section structure, RST keeps the structured front-door content); `fidstoragebase.rst` table cell column-width caused an inline-literal warning on the `autoscaleIgnore` row — column widened. Group A drafter judgment calls preserved: `LifTrace` referenced in prose without `:cpp:class:` link (no `liftrace.rst` page exists); private `d_currentSegment` field left undocumented (apparently dead). Group C drafter found four `hw*` helpers on `HardwareObject` (`hwLog`, `hwWarn`, `hwError`, `hwDebug` — no `hwHighlight`); `loghandler.rst` matches. Coherence reviewer's RST-table-vs-header-key duplication concern reviewed and rejected: tables surface namespace constants that `.. doxygenclass::` does not auto-include, so they remain in the RST as the authoritative key reference. Doc build clean apart from pre-existing warnings in earlier bundles (chirpconfig/rfconfig nested-struct duplicates, todo extension not loaded, etc.).
- 2026-05-02: not started → in progress. Three parallel Sonnet drafter/verifier pairs split the eight pages into Group A (datastoragebase / fidstoragebase / lifstorage — storage hierarchy), Group B (blackchirpcsv / auxdatastorage — CSV persistence helpers), and Group C (waveformbuffer / loghandler / applicationconfigmanager — runtime infrastructure singletons). A final coherence reviewer runs after all group revisions are accepted. Scope adjustment: dropped `overlaystorage.rst` (already covered on `overlaybase.rst` in bundle 13e); added `lifstorage.rst` (`LifStorage` is a `DataStorageBase` subclass important to LIF acquisition).
-->

Adds API reference pages for the storage subsystem.

## Scope

New pages under `doc/source/classes/`:

- `datastoragebase.rst` ← `src/data/storage/datastoragebase.h`
- `fidstoragebase.rst` ← `src/data/storage/fidstoragebase.h`
- `lifstorage.rst` ← `src/data/lif/lifstorage.h`
- `blackchirpcsv.rst` ← `src/data/storage/blackchirpcsv.h`
- `waveformbuffer.rst` ← `src/data/storage/waveformbuffer.h`
  (`WaveformBuffer`, `WaveformEntry`).
- `auxdatastorage.rst` ← `src/data/storage/auxdatastorage.h`
- `loghandler.rst` ← `src/data/loghandler.h`.
- `applicationconfigmanager.rst` ←
  `src/data/storage/applicationconfigmanager.h`.

`SettingsStorage` and `HeaderStorage` are already covered in
13a and are not touched here. `OverlayStorage` is covered on
`overlaybase.rst` in bundle 13e and is not touched here.

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
