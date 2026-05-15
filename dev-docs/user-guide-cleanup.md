# User Guide cleanup — continuation notes

Ephemeral scratchpad. The committed style rules live in
`doc/source/AGENTS.md`; this file is just the running checklist for
the pre-2.0.0 user-guide pass and the cleanup principles to apply on
each remaining page.

The pass through `installation.rst`, `first_run.rst`,
`hardware_config.rst`, the new `hardware_config/library_status.rst`,
and `application_config.rst` (commit `021bdc83`) established the
pattern. Continue across the rest of the user guide with the same
principles.

## What to look for on each page

The user-guide pages were drafted by Sonnet and tend to share a small
set of recurring problems. On each page:

- **Strip source-evolution language.** "compile-time vs runtime",
  "previously", "now uses", "added in v1.x", "Phase N", "linked at
  compile time" all need to go. Runtime program execution markers
  ("after the experiment completes", "before the FID arrives") are
  fine and often necessary. See `AGENTS.md` for the timeless-prose
  rule.
- **Strip apologia and marketing.** Sentences that sell a feature,
  apologize for it, or justify why it exists ("This isolation
  prevents stale or incompatible settings from silently affecting
  behavior...") rarely help a user; they're notes from the author to
  themselves. Cut or trim.
- **Don't duplicate content from dedicated pages.** If a page
  introduces a dialog or workflow that has its own reference page,
  give a one-paragraph orientation and link. Sonnet drafts tend to
  re-explain everything in place — the result is a documentation
  tree where every reader reads every fact twice.
- **Match the current UI.** Walk through the dialog as a user, not
  as a previous version of the code. Counts of tabs, panels, and
  steps in the prose must match the running app. `first_run.rst`
  had a four-step sequence for what is now two dialogs; flag and
  fix anything similar.
- **Condense wide tables to definition lists** when the columns are
  mostly prose. Tables wider than ~70 chars at any column rarely
  render well; `:doc:` and `:ref:` links inside a wide cell wrap
  poorly. Reserve tables for short tabular data (status values,
  format mappings).
- **Preserve labels (`.. _foo:`) that other pages reference.**
  Before renaming or removing an anchor, `grep -rn ':ref:`<name>`'
  doc/source/` and either keep the anchor or retarget every caller
  in the same commit.
- **Update cross-references when moving content.** Same grep on
  `:doc:` paths. Moving a page without retargeting links produces
  broken sidebar links that Sphinx will not warn about if the new
  path also resolves.
- **American English, present tense, impersonal voice** per the
  committed style rules.

## Navigation structure

The user guide was restructured so the sidebar reflects task-based
chapters. The toctrees live in `doc/source/index.rst` (captioned and
`:hidden:`); `user_guide.rst` no longer exists. The chapter captions
become sidebar section headers in `sphinx_rtd_theme`.

```
Getting Started
  installation, first_run, application_config, ui_overview

Hardware Setup
  hardware_config (+ profiles, loadouts, ftmw_presets, library_status)
  python_hardware (+ overview, selecting, writing_a_driver,
                   hot_reload, per_type_capabilities)
  hardware_menu (+ hwdialog as a sub-page)
  hardware_details (globs hw/*.rst)

Running Experiments
  experiment_setup (+ experiment/* sub-pages + lif/experiment_setup)
  ftmw_configuration (+ ftmw_configuration/rf_configuration,
                       chirp_setup, digitizer_setup)
  lif/configuration

Inspecting Data
  cp-ftmw, lif/lif_tab, plot_controls, overlays,
  rolling-aux-data, log_tab, python

Data Format and Diagnostics
  data_storage, lif/data_storage, crash_reports

Blackchirp Viewer
  viewer

Project Reference
  migration, changelog, developer_guide, classes
```

Reorganization commit: see the `Restructure user guide…` commit on
master.

## Pages still to review

Roughly in the order a new user reads them. The reorg commit moved
files but did not rewrite content; the per-page cleanup pass below
applies the principles above to the prose.

Getting Started:

- [x] `installation.rst`
- [x] `first_run.rst`
- [x] `application_config.rst`
- [x] `ui_overview.rst`

Hardware Setup:

- [x] `hardware_config.rst` and `hardware_config/library_status.rst`
- [ ] `hardware_config/profiles.rst`
- [ ] `hardware_config/loadouts.rst`
- [ ] `hardware_config/ftmw_presets.rst`
- [ ] `python_hardware.rst` and `python_hardware/` sub-pages
- [ ] `hardware_menu.rst`
- [ ] `hwdialog.rst` (now a sub-page of `hardware_menu`)
- [ ] `hardware_details.rst`
- [ ] `hw/*.rst` — per-device pages; light pass already touched
  `ftmwdigitizer.rst`, `lifdigitizer.rst`, `ioboard.rst` to strip
  "compile time" language. The remaining device pages likely have
  similar phrasing.

Running Experiments:

- [ ] `experiment_setup.rst` and `experiment/` sub-pages
- [ ] `ftmw_configuration.rst`
- [ ] `ftmw_configuration/rf_configuration.rst` (moved from the
  user-guide root)
- [ ] `ftmw_configuration/chirp_setup.rst`
- [ ] `ftmw_configuration/digitizer_setup.rst`
- [ ] `lif/configuration.rst` (LIF Configuration dialog, sibling of
  `ftmw_configuration`)
- [ ] `lif/experiment_setup.rst` (folded into `experiment_setup`'s
  toctree)

Inspecting Data:

- [ ] `cp-ftmw.rst`
- [ ] `lif/lif_tab.rst` (promoted to top-level sibling of `cp-ftmw`)
- [ ] `plot_controls.rst`
- [ ] `overlays.rst`
- [ ] `rolling-aux-data.rst`
- [ ] `log_tab.rst`

Data Format and Diagnostics:

- [ ] `data_storage.rst` — split into four pages: general experiment
  files (header.csv, hardware.csv, etc.), CP-FTMW data
  (clocks.csv, markers.csv, chirp.csv, fid/), other data files
  (rollingdata, log files, debug_log), with `lif/data_storage.rst`
  becoming the LIF page.
- [ ] `lif/data_storage.rst`
- [ ] `crash_reports.rst`

Blackchirp Viewer:

- [ ] `viewer.rst`

## Screenshots to refresh

The UI changes in commits `7dfd9c18` through `c8f4f6c4` invalidated
several screenshots:

- `7dfd9c18` ExperimentSetupDialog reflow (Summary nav page,
  smaller min size, status text + Validate moved under nav tree);
  LifProcessingWidget compacted (2x2 gate grid, checkable
  Savitzky-Golay); DigitizerConfigWidget replaced stacked
  groupboxes with QTableWidgets and a 3-column bottom row.
- `1a528eb0` LIF Configuration dialog tightened; channel-name
  column folded into DigitizerConfigWidget; in-canvas legend on
  LifTracePlot.
- `a7173f1c` TemperatureControlWidget moved to QTableWidget layout.
- `f935b2ad` GasControlWidget moved to QTableWidget layout; channel
  enable decoupled from setpoint; GasFlowDisplayBox visibility
  predicate switched to the explicit enable flag.
- `a910c9f3` PulseConfigWidget split into Standard/Advanced
  QTableWidgets, per-row Cfg popup gone, Enable became a themed
  power toggle; PulseStatusBox moved to 2-channel-per-row with
  elided names; ExperimentSetupDialog lost its hard-coded minimum
  size; HWDialog::sizeHint now honors a larger control widget.
- `c8f4f6c4` PulseConfigWidget tab row heights aligned.

Screenshots that need a refresh as the cleanup pass reaches each
page:

- [ ] `ui_overview-window.png` — main UI; shows the updated
  PulseStatusBox.
- [ ] `ftmw_configuration-digitizer.png` — Digitizer Config tab;
  layout change is substantial.
- [ ] `hwdialog-control_tab.png` — Pulse Generator Control tab;
  channel grid was rewritten.
- [ ] `lif-lif_config.png` — LIF Configuration dialog.
- [ ] `lif-lif_tab.png` — LIF Display tab (processing panel and
  in-canvas legend).
- [ ] `python_hardware-hwdialog_python.png` — PythonFlowController
  HwDialog; embeds the new GasControlWidget.

Check on the next pass — refresh only if the affected widget is
actually visible at the capture frame:

- [ ] `cp-ftmw-overview.png` — CP-FTMW tab; PulseStatusBox is in the
  status sidebar.
- [ ] `cp-ftmw-peakfind.png` — peak-find panel; status sidebar may
  be in frame.
- [ ] `experiment-startpage.png`, `experiment-loscan.png`,
  `experiment-drscan.png`, `experiment-validation.png`,
  `experiment-quickexpt_1.png`, `experiment-quickexpt_2.png`,
  `experiment-sequence.png`, `lif-lif_exp_setup.png` — all live
  inside ExperimentSetupDialog; dialog chrome (nav tree, status
  area) changed even if the page-specific contents did not.
- [ ] `hwdialog-settings_tab.png` — Pulse Generator Settings tab;
  the device-level settings registry did not change, but the dialog
  size and frame did.
- [ ] No new TemperatureControlWidget screenshot exists in the doc
  set; flag if a user-guide page is added that surfaces it.

## Reference

- `doc/AGENTS.md` — style rules, screenshot sizing, index entries,
  cross-reference conventions.
- `doc/source/developer_guide/conventions.rst` — the API-style /
  prose-vs-API-page contract; relevant when a user-guide page is
  tempted to recap a class's behavior instead of linking the API
  page.
- Reference commit: `021bdc83` (the initial pass).
