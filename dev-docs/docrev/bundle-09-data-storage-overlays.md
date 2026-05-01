# Bundle 09 — FTMW Data Viewing, Overlays, Data Storage Refresh

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-04-30: drafted → complete. User reviewed, requested overlays.rst voice/style alignment with the rest of the user guide (orchestrator rewrote it to match cp-ftmw.rst/data_storage.rst conventions: terse prose, ``setting``-format bullet lists, no marketing language) and three corrections (overlay type is user-selected not auto-detected; "sample backings" → "sample backing pressures"; removed a note about a non-existent Preview button). Content committed in 444babf7.
- 2026-04-30: in progress → drafted. Drafter (Sonnet) produced edits to all three target RSTs; verifier flagged 2 load-bearing items (markers.csv cross-reference pointed at cp-ftmw.rst which has no marker prose; clocks.csv LO-scan excerpt still used indexed Clock.0 keys) and 2 minor (auxdata.csv paragraph still had HTML cross-refs after a typo edit; orphan Markers index entry on cp-ftmw.rst). Orchestrator fixed all four directly. Awaiting user commit.
- 2026-04-30: not started → in progress. Drafter dispatch pending; orchestrator confirmed scope-currency against experiment data/experiments/0/0/49 (hardware.csv now has a third hardwareType column — drafter brief includes that addition).
-->

Refreshes the FTMW tab page, syncs the overlays content, and updates
the data storage page for new files (markers.csv) and 2.0 metadata.

## Scope

- Refresh `doc/source/user_guide/cp-ftmw.rst`:
  - Verify the description of `FtmwViewWidget` (Live, FT1, FT2,
    Main FT) is still accurate.
  - The Overlays toolbar entry point goes via the squares-plus
    icon — confirm the screenshot reference and forward-link to
    `overlays.rst`.
  - Note the FT autoscale curve attribute (added in commit
    `d34be87b`) and the autoscale curve toggle in the context menu.
  - Light prose refresh; no structural changes.
- Refresh `doc/source/user_guide/overlays.rst` if anything has
  shifted since it was added — review against the current
  `UnifiedOverlayDialog` implementation. The page is largely
  current; primarily verify the keyboard-shortcut list, the
  overlay-table column descriptions, and the catalog/Generic XY
  troubleshooting sections still match the implementation.
- Refresh `doc/source/user_guide/data_storage.rst`:
  - Add a `markers.csv` subsection (new file in 2.0; see
    `BC::CSV::markersFile`). Document its columns (Channel, Name,
    Role, TimingMode, StartUs, EndUs, Enabled).
  - Update the `version.csv` example to BCMajorVersion 2 and an
    appropriate alpha/beta release string.
  - Update the `header.csv` excerpt to reflect a 2.x experiment
    (label-based hardware keys like `FtmwDigitizer.Default`,
    new keys from the settings registry).
  - Verify the overlays subsection still matches the current on-
    disk layout (`overlays.csv`, `[label].settings.csv`,
    `[label].data.csv`).
  - Drop the `processing.csv` `Window Function` table if it can
    be cross-referenced from `cp-ftmw.rst` to avoid duplication;
    otherwise leave it.

## Out of scope

- Adding entirely new chapters about overlay sub-features (the
  existing `overlays.rst` is already comprehensive).
- LIF data storage details (bundle 10).

## Sources

- `dev-docs/awg-marker-system.md` — for the markers.csv schema.
- Source: `src/data/storage/blackchirpcsv.{h,cpp}` — confirm
  filenames and separator conventions.
- Source: `src/data/experiment/chirpconfig.cpp` —
  `readMarkersFile` / `writeMarkersFile` for the column layout.
- Source: `src/data/experiment/overlaybase.{h,cpp}`,
  `src/gui/overlay/unifiedoverlaydialog.{h,cpp}` — for any
  overlay UI changes since the existing page was written.
- Source: `src/data/storage/headerstorage.{h,cpp}` — confirm the
  header.csv format is unchanged.

## Sphinx file deltas

**Modified:**
- `doc/source/user_guide/cp-ftmw.rst`
- `doc/source/user_guide/overlays.rst`
- `doc/source/user_guide/data_storage.rst`

## Toctree delta

None.

## Screenshots

- `_static/user_guide/ui_overview/cp_ftmw.png` — refresh if the
  toolbar layout has changed (icons are now SVG / theme-aware).
- `_static/user_guide/overlays/*` — verify the existing screenshots
  match the current dialog; refresh as needed.

## Acceptance criteria

- `data_storage.rst` includes a `markers.csv` subsection with the
  full column list and an example block.
- The `version.csv` example shows `BCMajorVersion;2` and the
  current release string.
- The `header.csv` excerpt uses label-based hardware keys
  consistent with the post-refactor format.
- `cp-ftmw.rst` documents the autoscale-curve toggle.
- `overlays.rst` does not contain stale references to dialog
  classes or workflows that no longer exist.
