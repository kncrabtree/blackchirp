# Bundle 07 — RF Configuration, Chirp Setup, FTMW Digitizer

Updates the FTMW configuration pages to reflect the new
`FtmwConfigDialog`/preset-bar entry point and rewrites the chirp
setup page for the generalised marker system.

## Scope

- Move the Rf Configuration discussion out of `hardware_menu.rst`
  (which is bundle 04's territory) and into a dedicated page at
  `doc/source/user_guide/rf_configuration.rst`. Keep the existing
  upconversion/downconversion diagram and the per-clock-role
  explanations, but reframe the entry point as Hardware → FTMW
  Configuration (which opens `FtmwConfigDialog`) and document the
  preset bar.
- Rewrite `doc/source/user_guide/experiment/chirp_setup.rst`:
  - Replace the four protection/gate spinbox section with the
    new Markers tab description (uses `MarkerTableModel`).
  - Document marker channel index, role (Protection, Gate,
    Trigger, Custom), name, start time / end time relative to
    chirp start/end, enabled checkbox.
  - Document the `markerCount` capability that hides the Markers
    tab when the active AWG reports zero markers.
  - Document the safety validation warning when a Protection
    marker is missing or does not enclose the chirp/gate.
  - Keep the existing chirp-segment table description and the
    multi-chirp/Apply-to-All explanation.
  - Include a note that absolute timing and per-chirp marker
    overrides are not currently exposed (Phase 2 of the marker
    plan is deferred).
- Refresh `doc/source/user_guide/experiment/digitizer_setup.rst`:
  - Note the digitizer configuration is now part of an FTMW
    preset and is loaded/saved with the preset.
  - Cross-link the WaveformBuffer pre-accumulation behaviour
    sentence added in bundle 05 (one-line user-visible summary).
  - Light prose refresh; no structural changes.

## Out of scope

- The `MarkerChannel` / `MarkerTableModel` C++ API (bundle 13e).
- Loadout/preset CRUD details (bundle 03).
- WaveformBuffer internals (bundle 12).

## Sources

- `dev-docs/awg-marker-system.md` — primary for the chirp setup
  rewrite. Use the user-facing Marker Data Model section and the
  Phase 1 UI description.
- `dev-docs/loadout-system.md` — preset-bar semantics referenced
  on the RF Configuration page.
- Source: `src/gui/dialog/ftmwconfigdialog.{h,cpp}`,
  `src/gui/widget/ftmwconfigwidget.{h,cpp}`,
  `src/data/model/markertablemodel.{h,cpp}`,
  `src/gui/widget/chirpconfigwidget.{h,cpp}` — confirm the UI
  layout the prose describes.
- Source: `src/gui/expsetup/experimentchirpconfigpage.{h,cpp}` —
  confirm the validation warnings.

## Sphinx file deltas

**Created:**
- `doc/source/user_guide/rf_configuration.rst`

**Modified:**
- `doc/source/user_guide/experiment/chirp_setup.rst` — Markers
  tab rewrite.
- `doc/source/user_guide/experiment/digitizer_setup.rst` — light
  refresh.
- `doc/source/user_guide/hardware_menu.rst` — remove the Rf
  Configuration subsection (already moved by this bundle); add a
  pointer to the new page.

## Toctree delta

In `user_guide.rst` (after `hardware_config`):

```
   user_guide/rf_configuration
```

## Screenshots

- `_static/user_guide/rf_configuration/ftmwconfigdialog.png` — the
  new dialog showing the preset bar at the top and the RF
  configuration content below.
- `_static/user_guide/rf_configuration/preset_bar_actions.png` —
  preset-bar context (combo + Apply / Save / Save As / Delete
  buttons).
- `_static/user_guide/experiment/chirp_setup_markers.png` —
  Markers tab populated with Protection on channel 0 and Gate on
  channel 1, plus a custom Trigger row.
- `_static/user_guide/experiment/chirp_setup_segments.png` —
  refresh of the existing segments tab.
- `_static/user_guide/experiment/chirpconfigplot.png` — the chirp
  preview plot with multiple marker curves.

## Acceptance criteria

- The RF Configuration page documents the new Hardware → FTMW
  Configuration menu entry as the canonical entry point and
  describes the preset bar.
- The chirp setup page has zero references to "Pre-Chirp
  Protection", "Pre-Chirp Delay", "Post-Chirp Delay",
  "Post-Chirp Protection" spinboxes; those have been replaced by
  the marker table description.
- The chirp setup page documents the protection-marker safety
  validation warning users may encounter.
- The digitizer setup page mentions that its settings are stored
  in an FTMW preset, not as standalone QSettings keys.
- Cross-references from `acquisition_types.rst` (bundle 08) to
  the new `rf_configuration.rst` page are reachable.
