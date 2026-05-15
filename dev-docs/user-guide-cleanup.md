# User Guide cleanup — continuation notes

Ephemeral scratchpad. The committed style rules live in
`doc/source/AGENTS.md`; this file is just the running checklist for
the pre-2.0.0 user-guide pass and the cleanup principles to apply on
each remaining page.

**Cleanup-time note:** when this pass finishes and the doc is removed
from `dev-docs/`, also delete
`.claude/commands/user-guide-cleanup.md` — the slash command exists
only to re-enter this workflow across sessions and has no purpose
once the pass is done.

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

## Heuristics learned in earlier passes

These came out of cleaning the Getting Started, Hardware Setup, and
parts of Running Experiments chapters. They build on the per-page
checklist above and apply to the remaining pages.

- **Chapter intro page = canonical home for the whole-dialog
  screenshot and dialog chrome.** Sub-pages should defer to it for
  the full-dialog view and use focused captures (or no screenshot) on
  the panels they specifically document. Reuse one screenshot across
  multiple pages when they show the same surface; rename the file so
  its page-prefix matches the canonical home. The same page also owns
  the explanation of the dialog's chrome — navigation tree / page
  list, status / warning / error area, and action buttons (Validate,
  Start, Cancel, etc.) — since the sub-pages otherwise have no place
  to describe controls that aren't on any individual sub-panel.
- **Parallel halves of a dialog = sibling sub-pages.** When a dialog
  has two co-equal halves (e.g., the Experiment Setup dialog's FTMW
  group and LIF group), give each its own sub-page and treat them as
  siblings under the chapter intro rather than nesting one as an
  addendum to the other. The chapter intro frames them as parallel
  options and hosts shared controls (e.g., the Common Settings group).
  Applied to Experiment Setup: ``experiment/acquisition_types.rst``
  retitled "FTMW Experiment Setup" parallel to
  ``lif/experiment_setup.rst`` "LIF Experiment Setup", with
  ``Common Settings`` moved up to the chapter intro.
- **Suppress one specific sidebar entry by folding the heading into
  the chapter intro flow.** Section headings inside a page surface in
  the sidebar as ``toctree-l2`` children. Where a wrapping heading
  (e.g., "Experiment Setup Dialog" inside the Experiment Setup chapter
  intro) duplicates the page title, drop the heading and let the
  content flow under H1 directly, keeping H2s only for substantive
  thematic blocks like ``Common Settings``.
- **Cross-page duplications to flag.** Section headings in
  `hardware_menu.rst` are the canonical reference for the menu's
  submenus (Loadout, FTMW Preset, per-device). Sub-pages that
  describe the same submenus should defer with ``:ref:`` rather
  than restate. When subordinate pages disagree with the canonical
  page on details like state-gating, defer to the canonical page
  and drop the conflicting prose.
- **Place documentation where the user interacts.** FTMW presets
  belong under FTMW Configuration even though loadouts own them,
  because users interact with presets in the FTMW Configuration
  dialog, not the Hardware Configuration dialog. Hardware-area
  mentions of presets are limited to the drift-detection prompt in
  `loadouts.rst`. Apply the same principle if other pages surface
  features whose primary UI is on a different page.
- **Anchor renames cascade.** Renaming ``_foo-bar:`` to
  ``_baz-qux:`` requires updating the sub-anchors
  (``_foo-bar-current:``, etc.) and every ``:ref:`` user across the
  whole docs tree. Use ``grep -rn`` and a single sed sweep; verify
  with a follow-up grep for the old name.
- **Cross-reference retargeting touches a lot of trees.** Moving a
  user-guide page typically updates ~15 cross-references across
  ``classes/``, ``developer_guide/``, ``migration/``, ``changelog/``,
  ``python/``, and the user guide itself. The bulk sed sweep
  pattern from earlier commits works; just verify with a follow-up
  grep that nothing stale remains.
- **Sphinx CMake glob caches.** After ``git mv``-ing or deleting
  ``.rst`` files, re-run ``cmake . -B build`` once before the doc
  build, otherwise ninja errors on the stale file list. After
  ``.rst``-content-only edits, ``touch doc/source/index.rst &&
  cmake --build build --target docs`` is sufficient.
- **Sidebar behavior in sphinx_rtd_theme.** Section headings inside
  a page surface in the sidebar as ``toctree-l2``/``l3`` children of
  the current page. To suppress one specific entry, restructure the
  page so the section becomes part of the chapter intro (no
  ``---``/``~~~`` heading). To suppress globally would require
  ``html_theme_options = {'titles_only': True}`` in ``conf.py``;
  not currently set because the in-page section anchors are useful
  on long pages.
- **Second-person voice.** AGENTS.md asks for impersonal voice, but
  the earlier-cleaned pages keep some "you" usage where it reads
  naturally (especially in walk-throughs and instructions). Strip
  awkward second-person ("clicking it allows you to reclaim screen
  space..."); leave the unawkward kind.
- **Section-marker styles.** Pages use different combinations of
  ``===``/``---``/``~~~``/``....`` for headings. Preserve the
  existing style on each page rather than imposing one across the
  user guide.
- **Repeated boilerplate per variant.** Lists of variants that each
  repeat "appears when X is present" (or similar) compress well to
  one umbrella sentence above the list. Watched this pattern on
  ``ui_overview.rst`` (status box variants).

## Navigation structure

The user guide was restructured so the sidebar reflects task-based
chapters. The toctrees live in `doc/source/index.rst` (captioned and
`:hidden:`); `user_guide.rst` no longer exists. The chapter captions
become sidebar section headers in `sphinx_rtd_theme`.

```
Getting Started
  installation, first_run, application_config, ui_overview

Hardware Setup
  hardware_config (+ profiles, loadouts, library_status)
  python_hardware (+ overview, selecting, writing_a_driver,
                   hot_reload, per_type_capabilities)
  hardware_menu (+ hwdialog as a sub-page)
  hardware_details (globs hw/*.rst)

Running Experiments
  experiment_setup (+ experiment/* sub-pages + lif/experiment_setup)
  ftmw_configuration (+ ftmw_configuration/rf_configuration,
                       chirp_setup, digitizer_setup, presets)
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
- [x] `hardware_config/profiles.rst`
- [x] `hardware_config/loadouts.rst`
- [x] `hardware_config/ftmw_presets.rst` (moved to
  `ftmw_configuration/presets.rst` under Running Experiments)
- [x] `python_hardware.rst` and `python_hardware/` sub-pages
- [x] `hardware_menu.rst`
- [x] `hwdialog.rst` (now a sub-page of `hardware_menu`)
- [x] `hardware_details.rst`
- [x] `hw/*.rst` — per-device pages; full pass stripped
  source-evolution language ("compile-time", "data-path refactor",
  "set up for the MIT group"), trimmed feature-request solicitations
  and forward-looking notes, and rephrased the AD9914, M8190, Sirah
  Cobra, Opolette, Lakeshore, and PressureController commentary.

Running Experiments:

- [x] `experiment_setup.rst` and `experiment/` sub-pages
- [ ] `ftmw_configuration.rst` — **initial pass landed; awaiting
  UI cleanup + screenshot refresh before final pass.** First pass
  reframed the intro so the same controls appear as a page inside
  the Experiment Setup dialog (primary path) and also open standalone
  via Hardware → FTMW Configuration; standalone-only state-gating
  (Idle, read-only when disconnected) is scoped to the standalone
  bullet. Trimmed the duplicate preset definition that restated
  content from `presets.rst`; dropped the stale `seealso` block
  (one entry referenced the Hardware Configuration dialog, where
  presets are not created); dropped the redundant closing
  "documented on the following pages" line. Removed the unused
  ``_rf-configuration-preset-bar:`` anchor; kept ``_ftmw-preset-bar:``
  (still referenced by ``migration/v1_to_v2.rst``). `ftmw_configuration-dialog.png`
  needs a fresh capture once the planned UI cleanup lands.
- [ ] `ftmw_configuration/rf_configuration.rst` — **initial pass
  landed; awaiting UI cleanup + screenshot refresh.** Dropped the
  single "RF Config Tab" wrapping H2; promoted the five sub-sections
  (Clock Role Table, Common LO, Frequency Multiplication, Sideband
  Selection, Copy from Other FTMW Preset) to H2 siblings. Renamed
  "Copying RF Settings from Another Preset" → "Copy from Other FTMW
  Preset" so the section title reflects the actual cross-tab scope;
  rewrote the body to describe the combo's per-tab behavior in one
  paragraph. Trimmed a "useful for…" explainer on the Apply Clock
  Settings Now button. All ``_rf-configuration-*`` anchors preserved.
  `ftmw_configuration-rf_configuration.png` needs refresh after the UI
  pass.
- [ ] `ftmw_configuration/chirp_setup.rst` — **initial pass landed;
  awaiting UI cleanup + screenshot refresh.** Promoted the two orphan
  H3s (Multiple Chirps, Protection Marker Validation) to H2 siblings
  under the page title. Stripped "versatile interface for defining
  chirps and chirp sequences" marketing phrase from the intro.
  Stripped a "matching the previous default" source-evolution phrase
  from the marker-defaults paragraph. Converted the 6-row marker
  `list-table` to a definition list — the 15/85 width split put prose
  in the wide column and forced horizontal overflow in rendered HTML;
  the definition list reads identically and reflows cleanly. Removed
  the trailing "Each enabled marker is drawn as a labeled rectangle…"
  paragraph that duplicated the figure caption and the Role/Enabled
  row descriptions. External anchors ``chirp-setup-markers`` and
  ``chirp-setup-marker-validation`` are preserved (cross-referenced
  from ``data_storage.rst``, ``changelog/2.0.0.rst``,
  ``migration/v1_to_v2.rst``). `ftmw_configuration-segments.png` and
  `ftmw_configuration-markers.png` need refresh after the UI pass.
- [x] `ftmw_configuration/digitizer_setup.rst` — UI cleanup landed
  (Data Transfer, Trigger, and Acquisition Setup group boxes
  converted from QFormLayout/QGridLayout to embedded QTableWidgets;
  Acquisition Setup is a 2x2 table with vertical headers
  "Block Average"/"Multiple Records" and horizontal headers
  "On"/"Count"; Sample Rate's editable+validator fallback removed
  since the registry-driven picker is the only path). Screenshot
  refreshed and prose rewritten to match: figure caption describes
  Analog Channels on top with the three side-by-side groups below;
  Analog Channels prose references the **Enable**/**Full Scale**/
  **Offset** column headers; Data Transfer prose uses "Bytes Per
  Point" capitalization matching the row header and notes the
  Sample Rate picker is a fixed list (no free-text entry); Trigger
  prose lists rows by their row-header labels (Source/Slope/Delay/
  Level) and explains the Aux special value; Acquisition Setup
  prose references the On and Count columns instead of "**# Averages**"
  / "**# Records**" labels that no longer exist. Initial pass:
  Heading structure (five flat H2s using ``...``) already satisfied
  the no-orphan check. Removed the second intro paragraph that
  re-stated the "tab also accessible in the Experiment Setup
  dialog" access-pattern claim already made on the main page.
  Trimmed the preset-storage explainer in the first paragraph that
  duplicated `presets.rst`. Replaced "Blackchirp sets a sensible
  default for each supported digitizer" with a neutral statement
  about per-device defaults. Rewrote the "Maximizing Transfer
  Efficiency" opening paragraph to drop the "Blackchirp is designed
  to receive independent records … not the primary intended mode of
  operation" author-side rationale, keeping the factual co-averaging
  behavior and the math-waveform caveat.
- [x] `ftmw_configuration/presets.rst` (moved from
  `hardware_config/ftmw_presets.rst`; cleanup landed in same pass)
- [ ] `lif/configuration.rst` (LIF Configuration dialog, sibling of
  `ftmw_configuration`)
- [x] `lif/experiment_setup.rst` (folded into `experiment_setup`'s
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
- [x] `ftmw_configuration-digitizer.png` — refreshed alongside the
  Data Transfer / Trigger / Acquisition Setup QTableWidget refactor.
- [ ] `lif-lif_config.png` — LIF Configuration dialog.
- [ ] `lif-lif_tab.png` — LIF Display tab (processing panel and
  in-canvas legend).

Check on the next pass — refresh only if the affected widget is
actually visible at the capture frame:

- [ ] `cp-ftmw-overview.png` — CP-FTMW tab; PulseStatusBox is in the
  status sidebar.
- [ ] `cp-ftmw-peakfind.png` — peak-find panel; status sidebar may
  be in frame.
- [ ] `experiment-quickexpt_1.png`, `experiment-quickexpt_2.png`,
  `experiment-sequence.png` — all live inside ExperimentSetupDialog;
  dialog chrome (nav tree, status area) changed even if the
  page-specific contents did not.
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
