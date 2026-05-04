# Bundle 11a — Changelog: 1.1.0 and 2.0.0 release notes

**Status:** complete

<!--
Status log:
- 2026-05-03 — not started → complete (content commit 1100c515).
  Mid-session scope revision (recorded on bundle 11's status log)
  expanded 11a from one page to two: changelog/1.1.0.rst (devel-
  only, 8bc115ae..eec074ae) and changelog/2.0.0.rst (cmakemigration-
  only, eec074ae..2bda56a7). Bug fixes are now in scope on every
  release page; per-page format is Highlights → subsystem sections
  → Bug fixes (no user/dev split). Bug-fix entries link to the
  implementing commit via a new :commit: extlink role added to
  conf.py (sphinx.ext.extlinks). Re-edited changelog.rst intro to
  drop the H2-then-toctree pattern in favor of plain paragraphs
  followed directly by the toctree, listing 2.0.0 then 1.1.0.
  Removed the bundle-00 placeholder changelog/v2_0_0_alpha.rst.
  Carry-forward to 11b appended in
  bundle-11b-migration-guide.md (four migration topics outside
  the 11b checklist: Application Configuration dialog, FTMW
  Configuration menu, HwDialog field hiding, overlay files in
  experiment directory). Two transient build warnings remain
  until 11b lands: changelog/2.0.0.rst:8 and migration.rst:30,
  both pointing at /migration/v1_to_v2.
-->

Sub-page of the Changelog and Migration Guide chapter. Populates
the 1.1.0 and 2.0.0 release-notes pages, and adjusts the chapter
intro on `changelog.rst` if the per-release-page convention
shifts during drafting.

## Scope

Three RST files:

1. `doc/source/changelog.rst` — already converted from a
   placeholder to a chapter intro by bundle 11; 11a edits the
   intro if a change is needed for the per-release-page
   convention to read correctly (and updates the toctree to list
   both 1.1.0 and 2.0.0).
2. `doc/source/changelog/1.1.0.rst` — release notes for the
   1.0.0 → 1.1.0 range (the devel-only commits,
   `8bc115ae..eec074ae`). New file. 1.1.0 will be tagged on the
   `devel` branch before cmakemigration is merged into master,
   so the page exists in the rendered docs once cmakemigration
   lands.
3. `doc/source/changelog/2.0.0.rst` — release notes for the
   1.1.0 → 2.0.0 range (the cmakemigration-only commits,
   `eec074ae..<cmakemigration tip>`). New file.

The pages summarize what changed in each release for a Blackchirp
user. They are not per-commit logs; condense related commits into
one bullet per logical feature or fix. Bug fixes that change
observable behavior, restore a previously-broken feature, or fix
instability the user would notice are in scope and live in a
**Bug fixes** section at the bottom of each page; pure refactors
and internal-only fixes are out of scope.

### Page structure

Each release page follows the same shape (per the umbrella's
*Per-page structure* section):

1. **H1: Blackchirp <version>**.
2. **Release date** if known; otherwise omit.
3. **Short summary paragraph** framing the release. For 2.0.0,
   cross-link to the migration guide (`:doc:`/migration/v1_to_v2``).
4. **Highlights** — 3–5 bullets surfacing the largest changes.
5. **Component-level sections** grouped by subsystem; only the
   subsystems the release actually touched. Standard buckets the
   orchestrator can draw from:
   - Build & distribution
   - Hardware configuration (profiles, loadouts, FTMW presets,
     settings registry, `HwDialog`)
   - Hardware drivers (Python hardware, communication, AWG/chirp,
     GPIB, per-driver behavior)
   - Acquisition and data flow
   - User interface
   - Overlays
   - LIF
   - Logging and diagnostics
   - File formats and data storage
   - Tooling (`blackchirp-viewer`)
6. **Bug fixes**, also grouped by the same subsystem labels,
   listing user-noticeable fixes (crashes, hangs, races,
   restored-behavior fixes, observable misbehaviors).

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
- Per-feature deep dives (those live in their respective
  user-guide chapters; 11a cross-links to them).
- Internal refactors and pure-bookkeeping commits.
- Per-commit attribution. Each release page summarizes features
  and bug fixes, not the commit history that produced them.

## Sources

### Related source files

- `git log 8bc115ae..eec074ae --oneline` — primary input for the
  1.1.0 page (devel-only commits).
- `git log eec074ae..<cmakemigration tip> --oneline` — primary
  input for the 2.0.0 page (cmakemigration-only commits).
  Recommended approach: dispatch a research agent to pull both
  ranges and pre-categorize entries (build/install, hardware
  config, hardware runtime, UI, data, bug fixes, etc.) split by
  release. The orchestrator judges which entries are
  user-visible and synthesizes the bullet text.
- `doc/source/changelog.rst` — chapter landing (created by
  bundle 11; 11a updates the toctree to list both 1.1.0 and
  2.0.0, and adjusts the intro if the per-release-page convention
  shifts).

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

- `doc/source/changelog/1.1.0.rst`.
- `doc/source/changelog/2.0.0.rst`.

**Modified:**

- `doc/source/changelog.rst` — adjust the chapter intro and the
  toctree to list both 1.1.0 and 2.0.0.

**Deleted:**

- `doc/source/changelog/v2_0_0_alpha.rst` — bundle-00 placeholder
  superseded by `2.0.0.rst`.

If the orchestrator chooses the *preferred* carry-forward
location (append to 11b's sub-bundle file), then also:

- `dev-docs/docrev/bundle-11b-migration-guide.md` — append a
  `### Carry-forward to 11b` section under *Scope*.

## Acceptance criteria

- Both `changelog/1.1.0.rst` and `changelog/2.0.0.rst` follow the
  shared per-page structure above (Highlights → component-level
  sections by subsystem → Bug fixes), listing only the
  subsystems each release actually touched.
- Every feature and bug-fix entry cross-links to the user-guide
  page that documents the affected feature when one exists; a
  bug-fix entry without a stable user-guide target may stand on
  its own.
- A research agent (or the orchestrator) has walked the two
  commit ranges and the entries actually present on each page
  reflect that survey, not a pre-baked feature list.
- The carry-forward to 11b is recorded — either as an appended
  section in `bundle-11b-migration-guide.md` (preferred) or in
  this sub-bundle's status-log entry with a note to the user.
- No rendered link points into `dev-docs/`.
- Build is clean: `touch doc/source/index.rst && conda run -n
  breathe cmake --build build --target docs` produces no new
  warnings on the new pages.
