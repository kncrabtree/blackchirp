# Bundle 10 — Rolling/Aux, Log Tab, Blackchirp-Viewer

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Refreshes the rolling/aux-data page, adds a new Log tab page, and
adds a chapter for the standalone Blackchirp-viewer application.

LIF is intentionally excluded from this bundle — see
`bundle-15-lif.md` for the LIF chapter scope.

## Scope

### Rolling/Aux refresh

- Refresh `doc/source/user_guide/rolling-aux-data.rst` (light):
  - Verify identifier format is current.
  - Verify aux-data sources mentioned (FTMW shots, phase
    correction info) match the current `Experiment` aux-data
    registration.

### Log tab

- New page at `doc/source/user_guide/log_tab.rst`:
  - What appears on the Log tab and where messages come from
    (`bcLog`, `hwLog` family of free functions).
  - The runtime debug-logging toggle in Application Configuration
    (commit `308e00ed`).
  - Log severities (Normal, Highlight, Warning, Error, Debug) and
    what triggers each.
  - On-disk log location and rotation (per-month CSV under
    `log/`).
  - The per-experiment log captured under each experiment folder.

### Blackchirp-viewer

- New page at `doc/source/user_guide/viewer.rst`:
  - What `blackchirp-viewer` is (standalone application installed
    alongside Blackchirp from the same package; CMake target
    `blackchirp-viewer`).
  - How to launch it and how it relates to the main program.
  - The View Experiment workflow inside the main program vs.
    using the standalone viewer.
  - Limitations relative to the main program (read-only, no
    hardware, no acquisition).

## Out of scope

- LIF documentation in any form (bundle 15).
- The Log tab's relationship to `LogHandler` API (bundle 13f).

## Sources

- Source: `src/data/storage/applicationconfigmanager.{h,cpp}` —
  confirm runtime-debug toggle key.
- `cmake/BlackchirpViewerApplication.cmake` — confirm the viewer
  executable target name and packaging.
- `cmake/BlackchirpViewerGui.cmake` — confirm which widgets the
  viewer pulls in (this informs the "limitations" wording).
- `src/gui/log/` (or wherever the Log tab widget lives) — confirm
  the `bcLog`/`hwLog` call sites and severity enum.
- Existing `doc/source/user_guide/rolling-aux-data.rst` — for the
  light refresh.

## Sphinx file deltas

**Modified:**

- `doc/source/user_guide/rolling-aux-data.rst` — light refresh.

**Created:**

- `doc/source/user_guide/log_tab.rst`
- `doc/source/user_guide/viewer.rst`

## Toctree delta

In `user_guide.rst`:

```
   user_guide/log_tab
   user_guide/viewer
```

(LIF is unchanged in the top-level toctree; it remains a single
entry pointing at `user_guide/lif` whose body changes under
bundle 15.)

## Screenshots

Drafter leaves TODO markers with these filenames:

- `_static/user_guide/log_tab/log_tab.png` — Log tab with a mix of
  severities visible.
- `_static/user_guide/log_tab/debug_toggle.png` — Application
  Configuration showing the debug-logging toggle.
- `_static/user_guide/viewer/main_window.png` — Blackchirp-viewer
  main window viewing an experiment.

## Acceptance criteria

- The Log tab page documents the runtime debug toggle and the
  on-disk log file layout.
- The viewer chapter clarifies the relationship between
  `blackchirp` and `blackchirp-viewer` and notes that the viewer
  ships in the same binary package.
- Rolling/Aux page no longer has stale references.
- No LIF content is added or modified by this bundle.
