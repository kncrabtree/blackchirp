# Bundle 10 — LIF, Rolling/Aux, Log Tab, Blackchirp-Viewer

Replaces the LIF stub with a full chapter, refreshes the
rolling/aux-data page, adds a new Log tab page, and adds a chapter
for the standalone Blackchirp-viewer application.

## Scope

### LIF chapter

- Replace the one-paragraph `doc/source/user_guide/lif.rst` stub
  with a real chapter. Move it into a sub-folder structure:
  - `doc/source/user_guide/lif.rst` (intro + toctree)
  - `doc/source/user_guide/lif/configuration.rst` — LIF and
    Reference channel concepts, gate setup, integration windows,
    laser power normalisation, the runtime LIF toggle in
    Application Configuration.
  - `doc/source/user_guide/lif/acquisition.rst` — laser-frequency
    and delay-axis setup, randomized delay points (commit
    `09b05738`), shots-per-point, initialization from previous
    acquisition.
  - `doc/source/user_guide/lif/visualization.rst` — LIF Display
    tab tour: time trace, delay plot, laser plot, 2D plot,
    refresh interval, processing options.
  - `doc/source/user_guide/lif/data_storage.rst` — LIF data files
    on disk and how to interpret them.

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
    alongside Blackchirp from the same package).
  - How to launch it and how it relates to the main program.
  - The View Experiment workflow inside the main program vs.
    using the standalone viewer.
  - Limitations relative to the main program (read-only, no
    hardware, no acquisition).

## Out of scope

- LIF source-code internals (developer guide).
- The Log tab's relationship to `LogHandler` API (bundle 13f).

## Sources

- Source: `src/modules/lif/` — for LIF acquisition pipeline.
- Source: `src/gui/lif/` — for the LIF Display tab UI.
- Source: `src/data/lif/lifconfig.{h,cpp}` — confirm
  user-relevant configuration fields.
- Source: `src/data/storage/applicationconfigmanager.{h,cpp}` —
  confirm runtime-debug toggle key.
- Source: `src/main/blackchirpviewer/` (or the executable's main)
  — confirm the viewer's feature set.
- `dev-docs/devel-roadmap.md` — note that LIF was historically
  marked experimental; review whether that caveat still applies.

## Sphinx file deltas

**Replaced/modified:**
- `doc/source/user_guide/lif.rst` — converted from stub to
  intro + toctree.
- `doc/source/user_guide/rolling-aux-data.rst` — light refresh.

**Created:**
- `doc/source/user_guide/lif/configuration.rst`
- `doc/source/user_guide/lif/acquisition.rst`
- `doc/source/user_guide/lif/visualization.rst`
- `doc/source/user_guide/lif/data_storage.rst`
- `doc/source/user_guide/log_tab.rst`
- `doc/source/user_guide/viewer.rst`

## Toctree delta

In `user_guide.rst`:

```
   user_guide/lif
   user_guide/log_tab
   user_guide/viewer
```

In `lif.rst` (new toctree):

```rst
.. toctree::
   :hidden:

   lif/configuration
   lif/acquisition
   lif/visualization
   lif/data_storage
```

## Screenshots

- `_static/user_guide/lif/display_tab.png` — full LIF Display tab
  with time trace, delay plot, laser plot, 2D plot.
- `_static/user_guide/lif/configuration.png` — LIF channel
  configuration UI.
- `_static/user_guide/lif/acquisition_setup.png` — LIF acquisition
  page from the experiment wizard.
- `_static/user_guide/log_tab/log_tab.png` — Log tab with a mix of
  severities visible.
- `_static/user_guide/log_tab/debug_toggle.png` — Application
  Configuration showing the debug-logging toggle.
- `_static/user_guide/viewer/main_window.png` — Blackchirp-viewer
  main window viewing an experiment.

## Acceptance criteria

- The LIF chapter is no longer a single paragraph; it covers
  configuration, acquisition, visualization, and storage.
- The Log tab page documents the runtime debug toggle and the
  on-disk log file layout.
- The viewer chapter clarifies the relationship between
  `blackchirp` and `blackchirp-viewer` and notes that the viewer
  ships in the same binary package.
- Rolling/Aux page no longer has stale references.
