# Bundle 13e — API Reference: Data / Experiment Classes

Adds API reference pages for the experiment data model and analysis
helpers.

## Scope

New pages under `doc/source/classes/`:

- `experiment.rst` ← `src/data/experiment/experiment.h`
- `chirpconfig.rst` ← `src/data/experiment/chirpconfig.h`
  (`ChirpConfig`, `MarkerChannel`, `MarkerRole`, `ChirpSegment`).
  These three are tightly coupled and benefit from sharing a page.
- `rfconfig.rst` ← `src/data/experiment/rfconfig.h`
- `ftmwconfig.rst` ← `src/data/experiment/ftmwconfig.h`
- `lifconfig.rst` ← `src/data/lif/lifconfig.h`
- `fid.rst` ← `src/data/analysis/fid.h` (or wherever `Fid` lives)
- `ft.rst` ← `src/data/analysis/ft.h`
- `ftworker.rst` ← `src/data/analysis/ftworker.h`
- `overlaybase.rst` ← `src/data/experiment/overlaybase.h`
  (`OverlayBase`, `OverlayStorage`).

This is the largest API bundle; if effort runs over, splitting it
along data-experiment / data-analysis lines into two sessions is
acceptable.

## Out of scope

- Storage classes (`DataStorageBase`, `FidStorageBase`,
  `BlackchirpCSV`, `WaveformBuffer`, `LogHandler`) — bundle 13f.
- File parsers — bundle 13h.
- Any GUI-side companion (`ChirpConfigPlot`, `MarkerTableModel`,
  etc.) — bundle 13g.

## Sources

- `dev-docs/awg-marker-system.md` — for `MarkerChannel` /
  `MarkerRole` documentation.
- `dev-docs/digitizer-data-flow.md` — for any `WaveformBuffer`
  references on `FtmwConfig` (cross-link to bundle 13f for the
  buffer itself).
- The header files.

## Sphinx file deltas

**Created:** (one per page above).

**Possibly modified (Doxygen comment refresh):**
- All headers listed above.

## Acceptance criteria

- `ChirpConfig` page enumerates `MarkerChannel` fields with
  `\brief` comments and links to the user-guide chirp setup page
  (bundle 07) and the developer-guide adding-experiment-objective
  page (bundle 12).
- `FtmwConfig` page documents the `waveformBuffer()` setter/
  getter and notes that the buffer pointer is non-owning.
- `OverlayBase` page documents the type discriminator (Catalog /
  GenericXY / BCExperiment).
- Every public method has at least a `\brief`.
