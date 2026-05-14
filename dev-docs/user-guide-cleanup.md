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

## Pages still to review

Roughly in the order a new user reads them.

Getting Started:

- [x] `installation.rst`
- [x] `first_run.rst`
- [x] `application_config.rst`
- [ ] `ui_overview.rst`

Hardware Setup:

- [x] `hardware_config.rst` and `hardware_config/library_status.rst`
- [ ] `hardware_config/profiles.rst`
- [ ] `hardware_config/loadouts.rst`
- [ ] `hardware_config/ftmw_presets.rst`
- [ ] `python_hardware.rst` and `python_hardware/` sub-pages
- [ ] `hardware_menu.rst`
- [ ] `hwdialog.rst`
- [ ] `hardware_details.rst`
- [ ] `hw/*.rst` — per-device pages; light pass already touched
  `ftmwdigitizer.rst`, `lifdigitizer.rst`, `ioboard.rst` to strip
  "compile time" language. The remaining device pages likely have
  similar phrasing.

Running Experiments:

- [ ] `ftmw_configuration.rst`
- [ ] `experiment_setup.rst`
- [ ] `rf_configuration.rst`

Inspecting Data:

- [ ] `cp-ftmw.rst`
- [ ] `data_storage.rst`
- [ ] `overlays.rst`
- [ ] `plot_controls.rst`
- [ ] `rolling-aux-data.rst`
- [ ] `log_tab.rst`
- [ ] `crash_reports.rst`
- [ ] `viewer.rst`

Modules:

- [ ] `lif.rst`

## Reference

- `doc/AGENTS.md` — style rules, screenshot sizing, index entries,
  cross-reference conventions.
- `doc/source/developer_guide/conventions.rst` — the API-style /
  prose-vs-API-page contract; relevant when a user-guide page is
  tempted to recap a class's behavior instead of linking the API
  page.
- Reference commit: `021bdc83` (the initial pass).
