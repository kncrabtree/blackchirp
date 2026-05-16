# Overlay dialog cleanup — status & next task

Ephemeral planning doc; purged before release.

## Implementation summary (done)

The overlay creation/configure dialog was flattened off its nested
double-`QGroupBox` scaffold onto a shared table widget. Landed commits
(`ccce8c1a..HEAD` on `ftmw-view-dock-refactor`):

- **`SettingsTable`** (`gui/widget/settingstable.{h,cpp}`) — a
  `QTableWidget` subclass: borderless, headerless, non-selectable,
  size-to-content, no vertical scrollbar. API: `addSettingRow(label,
  value[, second], tip)`, `addSectionRow(title)` (bold, centered,
  `AlternateBase`-shaded span), `addCheckableSectionRow(title, checked,
  &box)`, `bindSectionRows(section, rows)` (bound rows collapse via
  `setRowHidden`; on expand the enclosing window is grown by the
  revealed rows' exact pixel height, grow-only). Section rows can also
  be retitled (`setSectionTitle`), switched between checkbox and plain
  centered heading (`setSectionCheckable`, swapping the displayed cell
  while keeping the `QCheckBox` alive so connections survive),
  enable/disabled without hiding (`setBoundRowsEnabled`), reconciled
  without window growth (`applySectionVisibility`), and re-grabbed
  (`sectionCheckBox`). Registered in both `BlackchirpGui.cmake` and
  `BlackchirpViewerGui.cmake`.
- **`HwSettingsWidget`** refactored onto `SettingsTable` (Part 1);
  `addArrayTableRow` reimplemented on the paired-widget row API.
- **`OverlayBaseOptionsWidget`** (2a), **`CurveAppearanceWidget`**
  (2b), **`OverlayTypeSpecificWidget` + the three subclasses** (2c)
  flattened.
- **Checkable-container conversion** (final step). The last two flat
  `QGroupBox`es kept only for their state machines are gone:
  - `OverlayTypeSpecificWidget::createSourceFileConfigUI` now takes a
    `SettingsTable*`. The base adds the checkable "Source File
    Configuration" section row, binds the subclass's
    file/path/status rows to it, and drives the Creation/Settings
    state machine through `isSourceConfigEnabled` /
    `setSourceConfigChecked` (signal-blocked, no grow) /
    `setSourceConfigCheckable` / `setSourceConfigTitle` plus a
    `sourceConfigToggled` relay. A `refreshSourceFileConfigState()`
    hook lets a subclass re-assert dynamic row visibility after the
    base applies context state. BCExp/Catalog/GenericXY build their
    config rows as table rows; catalog's parsed-file detail rows live
    in the config table (shown only when a catalog is parsed **and**
    the section is expanded), replacing the object-named details
    frame.
  - `CatalogOverlayWidget`'s convolution box is a checkable
    "Convolution Enabled" section with the Line Shape / Frequency
    Range sub-headings and the `Convolve` button as bound rows that
    collapse with it. Every `p_convolutionGroupBox` call site routes
    through `isConvolutionEnabled` / `setConvolutionEnabled` and a
    `convolutionEnabledChanged` relay.
- **Dialog shell** (Part 3): dropped the `Expanding` column spacers and
  the fixed 800×400 size (`adjustSize()`, still resizable); the
  validity message + progress bar moved into a bottom strip beside the
  button box. Changelog: one **User interface** entry in `2.0.0.rst`.
- **Review polish**: headerless tables, centered section rows, 380px
  min width on the base-options table, wider progress bar, Width/Size
  inlined beside Style/Marker in the appearance table, icon-only
  preset Save/Delete, deterministic grow-on-expand.
- Raw `setStyleSheet("color:…")` hint strings replaced with
  `ThemeColors` palette/`getCSSColor` roles. Dropped cosmetic
  `font-size:11px`/padding (acceptable per scope).

Decision of note: `SettingsTable` section-row shading uses
`QPalette::AlternateBase` (band) + `ThemeColors::EmphasisText` (text);
`ThemeColors` has no surface role.

## Not yet tested (verify on a running GUI)

Build is clean for `blackchirp` + `blackchirp-viewer` and docs build;
no runtime testing was possible. Test:

1. **Source File Configuration section (highest risk)**: Settings
   context — section is a centered checkbox titled "… (Optional)",
   starts unchecked with the file/path/status rows hidden; checking
   reveals them (window grows) and `Source File Settings` becomes
   visible/enabled; unchecking hides them again. Creation context —
   section renders as a plain centered heading (no checkbox), file
   picker rows always visible **and enabled** (you can always pick a
   file), `Source File Settings` enables once the file validates.
   Switch overlay type with the section toggled both ways.
2. **Catalog convolution end-to-end**: load `.cat`/`.xo`, toggle the
   "Convolution Enabled" section (collapse/expand of the Line Shape /
   Frequency Range sub-rows **and the Convolve button** + window
   grow), change shape / FWHM / range / points, Convolve runs/cancels
   with progress bar; parsed file-detail rows show/hide on load.
   Opening Settings on an overlay that had convolution enabled expands
   the section on open (grows the dialog once — grow-only, expected).
3. **GenericXY**: preview table grows with the dialog; Format
   Detection / Column Mapping / Data Filtering stay content-sized;
   the pre-existing filtering load quirk (section collapsed but inner
   checkbox checked) is unchanged, not fixed.
4. **BCExp `Configure FT…`** still opens its sub-dialog; FT status
   text updates.
5. **`CurveAppearanceWidget` in its three `QMenu` popups**
   (main-plot right-click, overlay-manager per-row, Peak Find toolbar
   button): table sizes the popup sensibly, no internal scrollbar.
6. **`HwSettingsWidget`** per-device dialog: Important/Advanced
   tables + array `Edit…` rows behave as before.
7. **Frequency Limits grow**: first check enlarges the dialog by the
   two rows; uncheck leaves it grown (grow-only, expected).
8. Section-row shading + status-label colors in light **and** dark.
9. Screenshots `overlays-overlay_creation_dialog`,
   `overlays-catalog_convolution_settings`,
   `overlays-generic_xy_preview` are stale — recapture from the GUI.
10. **Catalog detail-row edge case**: in Settings with an invalid /
    empty catalog, expanding the source-file section briefly grows the
    dialog for the detail rows before `refreshSourceFileConfigState()`
    re-hides them (grow-only over-grow; cosmetic, not a regression).
    Confirm detail rows are hidden whenever the section is collapsed
    and shown only with a parsed catalog while expanded.
11. **BCExp / GenericXY config rows as table rows**: experiment-number
    / custom-path / browse / status (BCExp) and file / status
    (GenericXY) now render as `SettingsTable` rows under the section;
    confirm layout, the "OR custom path" cell, and the path/browse
    pair behave as before, in both Creation and Settings.

## Status — planned coding complete

The checkable-container conversion (the former "next task") is done
and landed; no further coding is planned for this effort. What remains
is GUI verification of the items above and screenshot recapture.

### Behavior-preservation note (decision of record)

The spec's Part C wording said the Creation-context source rows should
be "*enabled* only when the file is valid (`setBoundRowsEnabled`, not
hide)". The original state machine instead left `p_sourceFileConfigBox`
unconditionally enabled in Creation (`setEnabled(true)`) — the file
picker must stay usable to *select* a file; only `p_sourceFileSettings`
gated on validity. To honor the binding "byte-for-byte unchanged except
the intended visual flatten" constraint, Creation keeps the config rows
always enabled (`setBoundRowsEnabled(section, true)`); the
validity-gating still lives on the separate `Source File Settings`
region, exactly as before. `setBoundRowsEnabled` is implemented and
available; it is simply driven with the value that reproduces the
original behavior.

### Implementation choices worth knowing

- The base auto-binds **every** row a subclass adds in
  `createSourceFileConfigUI` to the section (uniform collapse). Catalog
  therefore manages its detail-row visibility separately via
  `applyDetailRowVisibility()` / the `refreshSourceFileConfigState()`
  hook, which the base calls at the end of `updateSourceFileControls()`
  so dynamic rows are reconciled after context state is applied.
- Source-file-config programmatic state during setup uses the
  signal-blocked `setSourceConfigChecked()` + `applySectionVisibility()`
  (no relay, no grow) — matching the original blocked-`setChecked`.
  Catalog convolution `setConvolutionEnabled()` is **not** blocked, so
  it fires `convolutionEnabledChanged` and the collapse/grow exactly as
  the old `QGroupBox::toggled` did; the only new effect is grow-on-open
  when Settings restores an overlay with convolution enabled (item 2).
- `setSectionCheckable(false)` expands all bound rows (a plain heading
  never collapses); the subclass refresh hook then re-hides any
  dynamic rows. `setSectionCheckable` early-returns when the mode is
  unchanged, so repeated `updateSourceFileControls()` calls are safe.
