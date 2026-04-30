# Bundle 07 — RF Configuration, Chirp Setup, FTMW Digitizer

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-04-30: drafted → complete. Landed as commit 6fbcc60d ("Add
  FTMW Configuration chapter; refresh chirp and digitizer setup").
  All six bundle screenshots present (`ftmw_configuration.png`,
  `rf_configuration.png`, `clocks.svg`, `segments.png`,
  `markers.png`, `digitizer.png`). Cross-bundle edits to bundle 03's
  `hardware_config/` pages noted in `bundle-03-hardware-config.md`.
- 2026-04-30: drafted (digitizer screenshot captured). User added
  `digitizer.png` to `_static/user_guide/ftmw_configuration/`. Inserted
  a `.. figure::` block at the top of `digitizer_setup.rst` (between
  the opening framing paragraphs and the Analog Channels section) with
  alt text and caption describing the tab layout. Bundle Screenshots
  list updated to include the new asset.
- 2026-04-30: drafted (chirp screenshots captured). User captured
  `segments.png` and `markers.png` and dropped them in
  `_static/user_guide/ftmw_configuration/`. The chirp preview plot is
  visible inside both screenshots, so the standalone
  `chirpconfigplot.png` figure was removed from `chirp_setup.rst` and
  the captions on the segments and markers figures were extended to
  call out the preview plot. File names also reconciled: rst was
  expecting `chirp_setup_segments.png` / `chirp_setup_markers.png`;
  updated the figure paths to match the captured `segments.png` /
  `markers.png`. All bundle-07 figures now have real assets behind
  them — no `TODO: capture` markers remain.
- 2026-04-30: drafted (asset consolidation pass). All bundle-07
  screenshots consolidated under
  `_static/user_guide/ftmw_configuration/`: `clocks.svg` `git mv`-ed
  in from `_static/user_guide/hardware_menu/`; user added
  `rf_configuration.png` and dropped it into the same folder; the
  three chirp-tab figure paths updated in `chirp_setup.rst` to
  reference the consolidated folder. `rf_configuration.rst` now
  carries two figures — the RF Config tab screenshot
  (`rf_configuration.png`, file present, no TODO) and the signal
  chain diagram (`clocks.svg`, file present) — both inserted ahead
  of the Clock Role Table section. Bundle Screenshots section
  rewritten to reflect the new file layout and which assets are
  present vs. TODO. Three TODO captures remain (chirp segments,
  chirp markers, chirpconfigplot).
- 2026-04-30: drafted (revision pass 2). User-requested follow-ups
  applied across bundle 07 and (with explicit user approval) bundle
  03's `hardware_config/` territory: (a) renamed and moved
  `_static/user_guide/hardware_config/preset_bar.png` →
  `_static/user_guide/ftmw_configuration/ftmw_configuration.png` via
  `git mv`; this image now serves as the single dialog screenshot on
  `ftmw_configuration.rst` and on `ftmw_presets.rst`. (b) Removed the
  two `TODO: capture` figure blocks on `ftmw_configuration.rst`
  (`ftmwconfigdialog.png` replaced; `preset_bar_actions.png` deleted —
  the preset bar is visible in the main dialog screenshot). (c)
  De-duplicated `ftmw_presets.rst`: collapsed "The Preset Bar" body to
  a one-paragraph cross-reference + figure, removed "Accepting the
  FTMW Configuration Dialog" entirely, condensed "Deleting a Preset"
  to the conceptual constraint with a `:doc:` link. (d) Stripped all
  user-facing references to the internal `__LastUsed__` sentinel and
  related class names (`currentFtmwPreset`, `FtmwConfigDialog`)
  across `ftmw_configuration.rst`, `ftmw_presets.rst`,
  `hardware_config.rst`, `loadouts.rst` (13 occurrences total);
  reframed each in user-facing language ("Blackchirp remembers your
  configuration…"). The `_hardware-config-ftmw-presets-current`
  anchor is preserved (no external refs broken); the
  `_hardware-config-ftmw-presets-accept` anchor was removed alongside
  its section (no external refs found). Three remaining bundle-07
  `TODO: capture` markers: `chirp_setup_segments.png`,
  `chirp_setup_markers.png`, `chirpconfigplot.png`.
- 2026-04-30: in progress → drafted. Verifier punch list resolved.
  Revision pass restructured the docs into a top-level "FTMW
  Configuration" parent (`doc/source/user_guide/ftmw_configuration.rst`,
  120 lines, label `_ftmw-configuration:`) with three sibling detail
  pages — `rf_configuration.rst` (RF Config tab content only),
  `experiment/chirp_setup.rst`, `experiment/digitizer_setup.rst` — per
  user feedback. The parent owns the dialog overview and Preset Bar;
  the RF tab page was trimmed to clock/Common LO/multiplication/
  sideband/copy-from-preset content. `user_guide.rst` toctree updated
  (rf_configuration → ftmw_configuration). Cross-refs updated in
  `hardware_menu.rst`, `hardware_config/ftmw_presets.rst` (incl.
  removal of pending-bundle TODO), and the opening lines of
  `chirp_setup.rst` and `digitizer_setup.rst`. Temporal "currently"
  removed from chirp_setup.rst marker note. Five screenshots remain
  as `TODO: capture` markers per bundle scope. Awaiting user review +
  commit.
- 2026-04-30: not started → in progress. Drafter dispatched (Sonnet,
  isolated worktree). Pre-dispatch scope check fixed the cited
  validation source: `experimentchirpconfigpage.{h,cpp}` does not
  exist; the warnings are emitted by
  `ExperimentFtmwConfigPage::validate` in
  `src/gui/expsetup/experimentftmwconfigpage.cpp` (lines 64-184).
  Bundle Sources section updated accordingly.
-->

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
- Source: `src/gui/expsetup/experimentftmwconfigpage.{h,cpp}` —
  `ExperimentFtmwConfigPage::validate` emits the protection-marker
  warnings (no protection configured, disabled, starts at/after
  chirp, ends at/before chirp, starts after gate, ends before gate).

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

All bundle-07 screenshots live in
`_static/user_guide/ftmw_configuration/` (consolidated to keep all
FTMW Configuration parent-and-children assets co-located).

- `ftmw_configuration.png` *(file present — repurposed from the
  original `_static/user_guide/hardware_config/preset_bar.png`)* —
  the FTMW Configuration dialog showing the preset bar at the top
  and the three tabs below it.
- `rf_configuration.png` *(file present)* — the RF Config tab
  showing the clock role table, Common LO checkbox, multiplication
  factors, and sideband selectors.
- `clocks.svg` *(file present — moved from
  `_static/user_guide/hardware_menu/`)* — Blackchirp's signal-chain
  diagram (AWG → multiplication → upconversion → sample →
  downconversion → digitizer).
- `segments.png` *(file present)* — the Chirp Segments sub-tab with
  the segment table, the multi-chirp controls, and the chirp preview
  plot below the tab.
- `markers.png` *(file present)* — the Markers sub-tab with the
  marker channel table and the chirp preview plot showing the marker
  pulses overlaid on the waveform.
- `digitizer.png` *(file present)* — the Digitizer Config tab with
  the analog channel list, data-transfer settings, trigger group, and
  acquisition-mode selector.

The chirp preview plot is visible in both `segments.png` and
`markers.png`, so a separate `chirpconfigplot.png` is not needed.

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
