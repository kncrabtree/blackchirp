# Bundle 11b — Migration Guide: v1.x → 2.0.0

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Changelog and Migration Guide chapter. Populates
the v1.x → 2.0 migration page.

## Scope

Two RST files:

1. `doc/source/migration.rst` — already converted from a
   placeholder to a chapter intro by bundle 11; 11b only edits
   the intro if a change is needed for the per-version-page
   convention to read correctly.
2. `doc/source/migration/v1_to_v2.rst` — the substantive page.
   New file. The chapter toctree on `migration.rst` (lands in
   bundle 11) already references it.

The page is a checklist for v1.x users upgrading to 2.0. Each
section names the v1.x starting condition, the 2.0 end state,
and the steps to get from one to the other. The audience is a
v1.x user with a working installation and possibly a body of
acquired data; the page does not assume they have read the
changelog, but cross-links to it for the authoritative summary
of changes.

### Topic checklist

The 11b spec covers the following migration topics. 11a's
*Carry-forward to 11b* section (below, if any) extends this list
with anything 11a discovered that this checklist does not
already enumerate.

- **Build flow.** Replace `config.pri` editing with binary
  install (preferred) or CMake source build. Cross-link to
  `:doc:`/user_guide/installation``.
- **Hardware selection.** Compile-time `HARDWARE=…` lines are
  gone; hardware is selected at runtime through profile
  creation in `RuntimeHardwareConfigDialog`. Cross-link to
  `:doc:`/user_guide/hardware_config`` and the profiles
  subpage.
- **QSettings location.** v1.x and v2.x configurations are
  isolated by the new QSettings versioning. Users do not need
  to do anything but should not expect their v1.x settings to
  carry over.
- **Marker timing.** The four protection/gate spinboxes have
  been replaced with the marker table. Describe how to recreate
  the v1.x default configuration in 2.0 and cross-link to the
  marker-table user-guide page.
- **Hardware identification.** Implementation key strings have
  changed in some cases; show a `hardware.csv` example with
  the new label-based form so a v1.x user can compare their
  pre-upgrade file against the new format.
- **Quick experiment compatibility.** Now requires the same
  loadout, not just the same compile-time hardware list.
  Describe what a v1.x user should do if they want to repeat
  pre-upgrade experiments.
- **LIF.** Now a runtime toggle in addition to a build option.
  Describe how to enable it on a 2.0 install where it was
  compiled in.
- **GPIB controllers.** Runtime protocol selection means
  GPIB-LAN and GPIB-RS232 can coexist; describe how a v1.x
  user with one type of GPIB hardware migrates the
  configuration.
- **Data storage.** Minor format additions
  (`markers.csv`, version bump in `version.csv`). Old data
  remains readable; new acquisitions write the additional
  files. Cross-link to `:doc:`/user_guide/data_storage``.

### Carry-forward to 11b

These items surfaced in 11a's commit-log survey and are
user-visible upgrade actions that the existing checklist above
does not enumerate. 11b should fold them into the appropriate
section (likely "Recreating your v1.x configuration" or a new
"User-interface relabels" section).

- **Application Configuration dialog supplants v1.x's separate
  configuration menus.** v1.x users opened independent dialogs
  for font, save path, and other app-wide settings; in 2.0
  these (plus the new LIF and debug-logging toggles)
  consolidate into a single Application Configuration dialog
  reachable from the experiment-info panel. Note where v1.x
  users should look for each former entry. Cross-link to
  `:doc:`/user_guide/application_config``.
- **FTMW Configuration menu reorganization.** v1.x had an
  *RF Configuration* dialog as the primary FTMW-config entry
  point; 2.0 routes those controls through a consolidated
  *FTMW Configuration* menu/dialog reachable from the same
  area. Tell users navigating by menu name where the v1.x
  controls landed. Cross-link to
  `:doc:`/user_guide/ftmw_configuration``.
- **HwDialog field hiding.** Fields that v1.x users edited
  per-device in HwDialog (notably ``commType`` and ``model``)
  are now managed at the registry level via the runtime
  hardware configuration dialog rather than the per-device
  dialog. A v1.x user looking for those fields in HwDialog
  will not find them. Cross-link to
  `:doc:`/user_guide/hwdialog`` and
  `:doc:`/user_guide/hardware_config``.
- **Overlay files in the experiment directory.** The Overlays
  subsystem is new in 2.0; a v1.x experiment opened in 2.0
  has no overlays attached, but new acquisitions and any
  overlays the user creates in the viewer will write per-
  experiment overlay state under the experiment directory.
  No migration action is required for legacy data; mention
  the new files exist so users are not surprised. Cross-link
  to `:doc:`/user_guide/overlays``.

### Page structure

Suggested layout:

1. **H1: Migrating from Blackchirp 1.x to 2.0**.
2. **One-paragraph intro** framing the page as a migration
   checklist; cite `:doc:`/changelog/2.0.0`` as the
   authoritative summary of what changed.
3. **Pre-upgrade checklist.** Things to do before the upgrade:
   note the QSettings isolation; suggest exporting any settings
   the user wants as a reference; back up data directories
   (pure good-practice, not strictly necessary).
4. **Installation.** Walks the user through choosing binary or
   source install (cross-link to the installation user-guide
   page).
5. **First-time setup on 2.0.** Walks the user through the
   first-run flow, profile creation, library status, etc.
   (cross-link to first-run, hardware-config user-guide pages).
6. **Recreating your v1.x configuration.** Per-area sections
   for marker timing, hardware identification labels, quick-
   experiment compatibility, LIF runtime toggle, GPIB protocol
   selection. Each section: v1.x state → 2.0 state → steps.
7. **Working with v1.x data.** Cross-link to data-storage
   page; note format-additions and that the v1.x data remains
   readable.
8. **Where to go next.** Pointer to the changelog for the full
   list of new features; pointer to the user guide for the new
   capabilities a v1.x user would not have encountered.

The orchestrator may merge or split sections as the actual
checklist length warrants; the structure above is a starting
point.

## Out of scope

- The 2.0.0 changelog itself (bundle 11a).
- Per-feature deep dives (those live in the user-guide
  chapters; this page cross-links to them).
- Internal refactors (developer guide).
- Migration from any version other than v1.x. The page is
  titled `v1_to_v2`; future major-version migrations get their
  own page.

## Sources

### Related source files

- `doc/source/migration.rst` — chapter landing (created by
  bundle 11; do not re-edit unless necessary).

### Related dev-docs

Read for context only; do not link into `dev-docs/` from the
rendered page.

- `dev-docs/loadout-system.md` — for the hardware-selection
  migration paragraph.
- `dev-docs/awg-marker-system.md` — for the marker-timing
  migration paragraph (especially the "v1.x defaults
  expressed in the new model" detail).
- `dev-docs/python-hardware.md` — only as cross-reference for
  the LIF / GPIB protocol notes if they overlap.
- `dev-docs/qsettings-versioning.md` (or whichever dev-doc
  describes the v1/v2 QSettings isolation, if one exists) — for
  the QSettings-isolation paragraph. If no such dev-doc
  exists, walk the relevant header keys directly.

### Related user-guide pages

The migration page cross-links into these:

- `doc/source/user_guide/installation.rst`
- `doc/source/user_guide/first_run.rst`
- `doc/source/user_guide/application_config.rst`
- `doc/source/user_guide/hardware_config.rst` (and the
  profiles subpage)
- `doc/source/user_guide/hardware_menu.rst`
- `doc/source/user_guide/ftmw_configuration.rst` (chirp setup
  / marker table)
- `doc/source/user_guide/data_storage.rst`
- `doc/source/user_guide/lif.rst` (when LIF is in the user's
  configuration)
- Any user-guide page added by 11a's carry-forward list.

### Related API reference pages

None. The migration page is user-facing; it cites user-guide
pages, not API pages.

### Related sub-bundle output

- `dev-docs/docrev/bundle-11a-changelog.md` — read for
  context, especially any *Carry-forward to 11b* section it
  added (see below).
- `doc/source/changelog/2.0.0.rst` — the changelog page from
  11a. Read it; the migration guide's section ordering should
  loosely mirror the changelog's category ordering so a reader
  can pivot between the two.

## Sphinx file deltas

**Created:**

- `doc/source/migration/v1_to_v2.rst`.

**Modified:**

- `doc/source/migration.rst` — only if the chapter intro from
  bundle 11 needs adjustment to read correctly with the
  v1_to_v2 page in place.

## Acceptance criteria

- A v1.x user lands on `migration/v1_to_v2.rst` and can
  complete every action needed to use 2.0: building or
  installing, creating hardware profiles, recreating the v1.x
  marker timing, understanding the new data-storage layout,
  enabling LIF if applicable.
- The page is structured as a checklist (steps, not prose
  monologue); each section names the v1.x state, the 2.0 state,
  and the steps.
- Every cross-reference to a user-guide page uses `:doc:`;
  every cross-reference to a changelog bullet uses `:doc:` or
  `:ref:`.
- Every item from the 11a carry-forward list (if any) is
  addressed in the page.
- The page cross-references the 2.0.0 changelog at least once
  in the intro.
- No rendered link points into `dev-docs/`.
- Build is clean: `touch doc/source/index.rst && conda run -n
  breathe cmake --build build --target docs` produces no new
  warnings on the new page.
