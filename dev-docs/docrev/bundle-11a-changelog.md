# Bundle 11a — Changelog: 2.0.0 release notes (user perspective)

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Changelog and Migration Guide chapter. Populates
the 2.0.0 release notes.

## Scope

Two RST files:

1. `doc/source/changelog.rst` — already converted from a
   placeholder to a chapter intro by bundle 11; 11a only edits the
   intro if a change is needed for the per-release-page convention
   to read correctly.
2. `doc/source/changelog/2.0.0.rst` — the substantive page. New
   file. The chapter toctree on `changelog.rst` (lands in bundle
   11) already references it.

The page should answer: **from a v1.0 user's perspective, what is
new in 2.0.0?** It is *not* a per-commit log. The page focuses on
major user-visible features and behavior changes. Bug fixes are
out of scope unless they restore a previously-broken documented
feature; routine fixes and internal refactors are skipped.

**The 2.0.0 page is user-audience-only by exception.** The 2.0.0
change set is large enough that adding a developer-oriented
section on the same page would drown out the user-visible
content. Per the convention documented in the chapter intro
(`changelog.rst`), every release page **after** 2.0.0 will
address developers as a secondary audience — backend / non-user-
visible changes (refactors, internal API rearrangements,
threading or storage rework that does not surface in the UI) get
their own short section on each post-2.0.0 release page. The
2.0.0 page does not carry such a section.

### Page content

Per-release page structure:

1. **H1: Blackchirp 2.0.0**.
2. **Release date** if known; otherwise omit (the page should
   read correctly without one).
3. **One-paragraph summary** framing 2.0.0 as the major release
   that introduced binary distribution, runtime hardware
   selection, Python hardware drivers, and a number of
   acquisition and visualization improvements. Cross-link to
   the migration guide (`:doc:`/migration/v1_to_v2``) as the
   place to learn how to upgrade.
4. **Categorized bullets**. Group the entries by category; one
   short bullet per logical feature, each cross-linked to the
   user-guide page that documents it. Suggested categories
   (the orchestrator may merge or split as the actual change
   set warrants):
   - Build & distribution
   - Hardware configuration model (profiles, loadouts, FTMW
     presets, settings registry, `HwDialog`)
   - Python hardware drivers
   - Communication (runtime protocol selection)
   - AWG & chirp (marker system)
   - Digitizer data flow (WaveformBuffer, parallel parse)
   - Overlays (catalog, generic XY, BCExperiment, unified
     dialog, persistence)
   - LIF (runtime toggle, randomized delay points)
   - Logging (`bcLog`/`hwLog`, runtime debug toggle, log
     highlighting)
   - UI (collapsible status boxes, ScientificSpinBox, Heroicons
     icons, theme-aware colors, Help/About menu)
   - Blackchirp-viewer
   - QSettings versioning
   - File-format / data-storage changes (markers.csv, version
     bump in version.csv, etc.)

### Carry-forward to 11b

If 11a discovers user-visible changes that the 11b migration-
guide spec does not already enumerate (and that a v1.x user
would need an upgrade action for), append a
**`### Carry-forward to 11b`** section to
`bundle-11b-migration-guide.md` listing each item with one or
two lines of context. Stage that change as part of 11a's
content commit. The 11b spec lists the migration topics it
already covers; anything outside that list that requires user
action belongs in the carry-forward.

If the items are too speculative or design-decisions are needed
before 11b can act on them, list them in 11a's status-log entry
instead and flag for the user in the handoff.

## Out of scope

- The migration guide itself (bundle 11b).
- Per-feature deep dives (those live in their respective user-
  guide chapters; 11a cross-links to them).
- Internal refactors (developer guide).
- Per-commit attribution. The 2.0.0 page summarizes features,
  not the commit history that produced them.
- Bug fixes that do not change documented behavior.

## Sources

### Related source files

- `git log 8bc115ae..HEAD --oneline` — the primary input.
  Recommended approach: dispatch a research agent to pull the
  log and pre-categorize entries (build/install, hardware
  config, hardware runtime, UI, data, etc.). The orchestrator
  judges which entries are user-visible and synthesizes the
  bullet text.
- `doc/source/changelog.rst` — chapter landing (created by
  bundle 11; do not re-edit unless necessary).

### Related dev-docs

Read for context only; do not link into `dev-docs/` from the
rendered page.

- `dev-docs/awg-marker-system.md` — for the marker-system
  bullet.
- `dev-docs/loadout-system.md` — for the hardware-configuration
  bullet.
- `dev-docs/python-hardware.md` — for the Python-hardware
  bullet.
- Any other `dev-docs/*.md` that documents a 2.0-era feature
  the orchestrator decides to include.

### Related user-guide pages

Every changelog bullet should `:doc:`-link to the user-guide
page that documents the feature. Inventory:

- `doc/source/user_guide/installation.rst`
- `doc/source/user_guide/first_run.rst`
- `doc/source/user_guide/application_config.rst`
- `doc/source/user_guide/hardware_config.rst` (and subpages)
- `doc/source/user_guide/hardware_menu.rst`
- `doc/source/user_guide/hardware_details.rst`
- `doc/source/user_guide/python_hardware.rst` (and subpages)
- `doc/source/user_guide/ftmw_configuration.rst` (and
  subpages)
- `doc/source/user_guide/experiment_setup.rst` (and subpages)
- `doc/source/user_guide/cp-ftmw.rst`
- `doc/source/user_guide/data_storage.rst`
- `doc/source/user_guide/overlays.rst`
- `doc/source/user_guide/lif.rst`
- `doc/source/user_guide/log_tab.rst`
- `doc/source/user_guide/rolling-aux-data.rst`
- `doc/source/user_guide/viewer.rst`
- `doc/source/user_guide/plot_controls.rst`

The orchestrator does not need to read all of these in full
during the session; the inventory is here so the right page can
be looked up when authoring each bullet.

### Related API reference pages

None directly. Changelog entries cite user-guide pages, not API
pages.

## Sphinx file deltas

**Created:**

- `doc/source/changelog/2.0.0.rst`.

**Modified:**

- `doc/source/changelog.rst` — only if the chapter intro from
  bundle 11 needs adjustment to read correctly with the 2.0.0
  page in place.

If the orchestrator chooses the *preferred* carry-forward
location (append to 11b's sub-bundle file), then also:

- `dev-docs/docrev/bundle-11b-migration-guide.md` — append a
  `### Carry-forward to 11b` section under *Scope*.

## Page structure

Suggested H2 organization for `changelog/2.0.0.rst`:

- *Highlights* — 3–5 bullets surfacing the largest changes.
- *Build, distribution, and configuration* — CMake build,
  binary packages, runtime hardware configuration model.
- *Hardware* — Python hardware, communication-protocol
  selection, marker system, per-type behavior changes.
- *Acquisition and data flow* — WaveformBuffer, parallel
  processing, digitizer changes.
- *User interface* — status boxes, plot controls, overlays,
  icons, theme.
- *LIF*
- *Logging and diagnostics*
- *File formats and data storage* — version-bumped files,
  new CSVs.
- *Tooling* — Blackchirp-viewer.

The orchestrator may rearrange or merge sections to match the
actual feature set surfaced from the commit log.

## Acceptance criteria

- `changelog/2.0.0.rst` is organized into clearly-labeled
  sections covering every category named in the *Page content*
  section above (or a justified merge thereof).
- Every bullet describes a user-visible change and cross-links
  to the user-guide page that documents it.
- Bug fixes that do not restore a documented behavior are
  excluded.
- The 2.0.0 page does not carry a developer-audience section
  (per the chapter convention, that is a post-2.0.0 thing). The
  chapter intro on `changelog.rst` (landed by bundle 11) already
  states the convention; 11a does not need to restate it on the
  2.0.0 page.
- A research agent (or the orchestrator) has walked
  `git log 8bc115ae..HEAD` and the entries actually present in
  the page reflect that survey, not a pre-baked feature list.
- The carry-forward to 11b is recorded — either as an appended
  section in `bundle-11b-migration-guide.md` (preferred) or in
  this sub-bundle's status-log entry with a note to the user.
- No rendered link points into `dev-docs/`.
- Build is clean: `touch doc/source/index.rst && conda run -n
  breathe cmake --build build --target docs` produces no new
  warnings on the new page.
