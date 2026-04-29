# Bundle 11 — Migration Guide v1.x → 2.0.0 + Changelog

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Populates the changelog and migration-guide scaffolds created by
bundle 00. The changelog is a forward-looking document that will be
maintained for every release; the migration guide is a one-shot
reference for users coming from v1.x.

## Scope

### Changelog

- Populate `doc/source/changelog.rst` with an introductory
  paragraph and a link/structure that lets future releases be
  added without restructuring.
- Add a per-release page at
  `doc/source/changelog/2.0.0.rst` listing the major changes that
  ship in 2.0.0. Group by category:
  - Build & distribution (CMake migration, binary packages)
  - Hardware configuration (runtime profiles, loadouts, FTMW
    presets, settings registry, `HwSettingsWidget`/`HwDialog`)
  - Python hardware
  - Communication (runtime protocol selection)
  - AWG & chirp (generalised marker system)
  - Digitizer data flow (WaveformBuffer, parallel parse)
  - Overlays (Catalog SPCAT/XIAM, GenericXY, BCExperiment, Unified
    dialog, persistence)
  - LIF (runtime toggle, randomized delay points)
  - Logging (`bcLog`/`hwLog`, runtime debug toggle, log
    highlighting)
  - UI (collapsible status boxes, ScientificSpinBox, Heroicons
    SVG icons, theme-aware colours, Help menu/About)
  - Blackchirp-viewer
  - QSettings versioning
- Establish a "Best practices for keeping the changelog updated"
  one-paragraph note: every PR that adds user-visible behaviour
  should append an entry to the in-development release page.

### Migration guide

- Populate `doc/source/migration.rst` with a chapter intro plus an
  authoritative `1.x → 2.0` migration page at
  `doc/source/migration/v1_to_v2.rst`. Topics:
  - Build flow: replace `config.pri` editing with binary
    install or CMake source build.
  - Hardware selection: from compile-time `HARDWARE=…` lines to
    runtime profile creation in `RuntimeHardwareConfigDialog`.
  - QSettings location: v1.x and v2.x configurations are
    isolated; users do not need to do anything but should not
    expect their v1.x settings to carry over.
  - Marker timing: the four protection/gate spinboxes have been
    replaced with the marker table; describe how to recreate the
    v1.x default in 2.0.
  - Hardware identification: implementation key strings have
    changed in some cases; the `hardware.csv` example shows the
    new label-based form.
  - Quick experiment compatibility: now requires the same
    loadout, not just the same compile-time hardware.
  - LIF: now a runtime toggle in addition to a build option.
  - GPIB controllers: runtime protocol selection means GPIB-LAN
    and GPIB-RS232 can coexist.
  - Data storage: minor format additions (markers.csv, version
    bump in version.csv).

## Out of scope

- Per-feature deep dives (those live in their respective user-
  guide chapters; this bundle cross-links to them).
- Internal refactors (developer guide).

## Sources

- This is a synthesis bundle; the inputs are every other bundle's
  output plus the commit history. Run
  `git log --oneline 8bc115ae..HEAD` and walk it for entries that
  introduce user-visible behaviour.
- `dev-docs/awg-marker-system.md` — for the marker timing
  migration paragraph.
- `dev-docs/loadout-system.md` — for the hardware-selection
  paragraph.
- `dev-docs/python-hardware.md` — for the Python hardware
  introduction line in the changelog.

## Sphinx file deltas

**Modified:**
- `doc/source/changelog.rst` — populated.
- `doc/source/migration.rst` — populated.

**Created:**
- `doc/source/changelog/2.0.0.rst`
- `doc/source/migration/v1_to_v2.rst`

## Toctree delta

In `changelog.rst`:

```rst
.. toctree::
   :maxdepth: 1

   changelog/2.0.0
```

In `migration.rst`:

```rst
.. toctree::
   :maxdepth: 1

   migration/v1_to_v2
```

## Screenshots

None.

## Acceptance criteria

- A v1.x user lands on `migration/v1_to_v2.rst` and can complete
  every action needed to use 2.0 — building/installing, adding
  hardware as profiles, recreating the v1.x marker timing,
  understanding the new data-storage layout.
- The 2.0.0 changelog entry covers every category listed above
  with at least one bullet apiece, each cross-linked to the
  user-guide chapter that documents it.
- The "best practices" note is concrete enough that a contributor
  with no project history can follow it on their next PR.
- The changelog and migration guide cross-reference each other:
  the migration guide points to specific changelog bullets where
  appropriate, and the changelog's release page points to the
  migration guide as the place to learn how to upgrade.
