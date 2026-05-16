# Overlay creation/configure dialog — UI cleanup

Ephemeral planning doc. The Overlay **Manager** table
(`OverlayManagerWidget`) is already clean and is **out of scope**.
This doc covers only the unified overlay **creation/configure**
dialog. Purged before release.

## Problem

`UnifiedOverlayDialog` → `UnifiedOverlayWidget` composes three
columns in a `QHBoxLayout`, each with stretch 1 and trailing
`QSpacerItem(Expanding)`:

- **Left:** `OverlayTypeSpecificWidget` (base scaffold) with three
  stacked `QGroupBox`es — `Source File Configuration`,
  `Source File Settings`, `Type-Specific Settings` — populated by
  the per-type subclasses `BCExpOverlayWidget`,
  `CatalogOverlayWidget`, `GenericXYOverlayWidget`, each of which
  hand-rolls *further* nested `QGroupBox`es (`Format Detection`,
  `Column Mapping`, `Data Filtering`, `Line Shape`,
  `Frequency Range & Resolution`, …) with `QGridLayout`/`QFormLayout`.
- **Center:** `OverlayBaseOptionsWidget` inside a `Base Options`
  box, itself holding flat `QGroupBox`es `Overlay Identity`
  (`QFormLayout`), `Scale & Position` (hand-rolled `QHBoxLayout`
  label+widget rows), `Frequency Limits` (checkable, `QGridLayout`).
- **Right:** `CurveAppearanceWidget` inside a `Curve Appearance`
  box, holding flat `QGroupBox`es `Presets` and `Appearance`
  (four `QGridLayout` rows).

Symptoms (all visible in `doc/source/_static/user_guide/
overlays-overlay_creation_dialog.png`, `…-catalog_convolution_
settings.png`, `…-generic_xy_preview.png`):

1. Content fills ~45% of a ~1160×650 window; the validity line and
   the progress bar float orphaned in the dead space because of the
   `Expanding` spacers.
2. Group boxes nested two deep; the theme centers box titles, so
   stacked centered bold banners and doubled frames dominate.
3. Every section uses a different layout primitive; sibling label
   columns do not align.
4. Hint/subtle text uses raw `setStyleSheet("color:…;font-style:
   italic;font-size:11px")` instead of `ThemeColors` roles; cosmetic
   defects such as the `"Scale  Position"` double space.

These dialogs are **code-built (no `.ui`)**, so this is a
layout-logic refactor, like the `PeakListExportDialog` de-UI.

## Decisions (locked)

- **Layout strategy:** keep the three-column arrangement (Type-
  specific | Overlay | Appearance) but flatten it. Remove the
  nested `QGroupBox`es; each logical section becomes one 2-column
  settings table. Drop the `Expanding` spacers; size the dialog to
  content. Move the validity message + progress bar into a real
  bottom bar adjacent to the button box.
- **Shared helper:** extract the settings-table builder out of
  `hwsettingswidget.cpp` into a shared gui util and have both
  `HwSettingsWidget` and the overlay widgets use it.
- **Scope:** layout **plus small UX fixes** — merge obviously
  redundant controls, fix label/units inconsistencies, replace raw
  stylesheet strings with `ThemeColors` roles. **No** changes to
  settings keys, validation logic, the `Configure FT…` sub-dialog,
  catalog convolution threading, preview behavior, or the on-disk
  format. Behavior is otherwise byte-for-byte unchanged.

## Part 1 — Shared settings-table helper

`makeSettingsTable()` is currently a file-static in
`gui/widget/hwsettingswidget.cpp`; `addTableRow()` is a
`HwSettingsWidget` member. Promote to `gui/widget/settingstable.h`
(+ `.cpp`), free functions in a small namespace:

```
QTableWidget *makeSettingsTable(QWidget *parent);
int addSettingRow(QTableWidget *t, const QString &label,
                  QWidget *valueWidget, const QString &tooltip = {});
int addSectionRow(QTableWidget *t, const QString &title);   // spanned
                  // bold separator row — replaces nested box titles
```

Carry over the exact `makeSettingsTable` body (col 0
`ResizeToContents`, last-section stretch, vertical header hidden,
no selection, no edit triggers, `AdjustToContents`, scrollbar off)
and the `addTableRow` row-height logic. `addSectionRow` spans both
columns, bold, non-selectable — used wherever a nested `QGroupBox`
title currently acts as a sub-heading.

Refactor `HwSettingsWidget` to call the shared functions (delete its
private copies; keep its `addArrayTableRow` which is registry-
specific). Add `settingstable.{h,cpp}` to `cmake/BlackchirpGui.cmake`
**and** `cmake/BlackchirpViewerGui.cmake` (overlay widgets are in
both binaries).

Commit: refactor only, no behavior change, no changelog.

## Part 2 — Flatten the three composing widgets

Each widget's `setupUI()` drops its outer + nested `QGroupBox`es and
builds one `makeSettingsTable` per logical group, with
`addSectionRow` for the headings that used to be box titles. The
outer per-column `QGroupBox` in `UnifiedOverlayWidget`
(`p_overlayBaseOptionsBox`, `p_curveAppearanceBox`) is kept as the
single column frame (one level only).

**2a. `OverlayBaseOptionsWidget`** — one table:

| Row | Control(s) |
|-----|------------|
| `addSectionRow("Identity")` | — |
| Label | `p_labelLineEdit` |
| Storage Name | `p_sanitizedLabelDisplay` (ThemeColors SubtleText, no stylesheet string) |
| Comment | `p_commentLineEdit` |
| Plot ID | `p_plotIdComboBox` |
| `addSectionRow("Scale & Position")` | — |
| Y Scale | h-box: `p_yScaleInputWidget` + `Invert` |
| Autoscale | h-box: `p_autoscalePercentageSpinBox` + `Autoscale` |
| Y Offset | `p_yOffsetSpinBox` |
| X Offset | `p_xOffsetSpinBox` (own row; drop the side-by-side cramming) |
| `addSectionRow("Frequency Limits")` | — |
| Min Frequency | h-box: `p_minFreqCheckBox` + `p_minFreqSpinBox` |
| Max Frequency | h-box: `p_maxFreqCheckBox` + `p_maxFreqSpinBox` |

Keep the checkable-collapse behavior by enabling/disabling the two
freq rows from a single checkbox (a `Frequency Limits` checkbox in
its section row), rather than a checkable nested box.

**2b. `CurveAppearanceWidget`** — Presets bar + one table:

- Presets: a compact top row (`Preset` combo + `Save` + `Delete`),
  not a nested box.
- Table rows: Color, Type (`p_curveStyleBox`), Width, Style
  (`p_lineStyleBox`), Marker, Size, Y Axis; `Visible` and
  `Autoscale` as one trailing h-box row (or two checkable rows).
- **Shared-widget caveat:** `CurveAppearanceWidget` is also used by
  `zoompanplot` and `curveappearancepresetmanager` (the main-plot
  curve-styling path), not only overlays. Flattening improves all
  consumers but the main-plot curve-appearance editor **must be
  regression-checked**, not just the overlay dialog.

**2c. `OverlayTypeSpecificWidget` + the three subclasses** — replace
the three-box scaffold with: a fixed source-file row
(path/experiment + `Browse`/status), then one `makeSettingsTable`
per logical group the subclass needs:

- `BCExpOverlayWidget`: source row + `Configure FT…` action; FT
  controls unchanged (do not touch the sub-dialog).
- `CatalogOverlayWidget`: source row (+ parsed-file detail rows),
  then a `Convolution` section table (`Enabled`, `Shape`, `FWHM`,
  `Range` from/to, `Points`, `Spacing` readout) + `Convolve`
  action. Convolution threading/caching untouched.
- `GenericXYOverlayWidget`: source row, `Format Detection`
  (`Delimiter`, `Header Lines`, `Re-detect`), `Column Mapping`
  (`X`/`Y` combos, `Parse File`), `Data Filtering` section, then the
  data-preview `QTableView` as the one widget that keeps stretch 1
  (it is the legitimate table that should grow).

Keep the three subclass virtuals (`createSourceFileConfigUI` etc.)
but have them fill tables instead of nested boxes; the base scaffold
provides the column frame and stretch policy.

Commit slicing (each builds + is independently reviewable):
3. `OverlayBaseOptionsWidget` flatten.
4. `CurveAppearanceWidget` flatten (note shared consumers in body).
5. `OverlayTypeSpecificWidget` base + the three subclasses.

## Part 3 — Dialog shell

`UnifiedOverlayWidget::setupUI()` / `UnifiedOverlayDialog::setupUI()`:

- Drop trailing `QSpacerItem(Expanding)` from the three column
  layouts. Column stretch stays balanced but the dialog sizes to
  content (`adjustSize()`); resizable.
- Relocate the validity line + `QProgressBar` into a bottom strip
  next to the OK/Cancel (`Create Overlay`/`Apply Changes`) box,
  removing the orphaned mid-dialog band.
- Verify the per-type title (`Overlay BC Experiment Overlay` etc.)
  and the create-vs-configure button labels are unchanged.

Commit 6: dialog shell + spacer/progress relocation. Changelog: a
single top-level **User interface** entry — "Streamlined the overlay
creation dialog layout" — referencing `:doc:/user_guide/overlays`.

## Part 4 — Docs

Refresh the three creation-dialog screenshots in
`doc/source/_static/user_guide/` (`overlays-overlay_creation_dialog`,
`overlays-catalog_convolution_settings`, `overlays-generic_xy_
preview`) after the visual change; the manager screenshot is
unchanged. Light copy pass on `overlays.rst` "Creating an Overlay"
if control groupings change names. Build docs via the `breathe`
env (see `AGENTS.local.md`).

## Open items to confirm during implementation

- `addSectionRow` styling: bold spanned cell vs. a themed separator
  row — pick one and use it everywhere for consistency with
  `HwSettingsWidget`'s look.
- Whether the `Frequency Limits` collapse should hide the rows
  (table `setRowHidden`) or just disable them; hiding is closer to
  the current checkable-box behavior.
- Exact `2.0.0.rst` heading for the Part 3 changelog entry — confirm
  against the file as in prior feature work (top-level
  **User interface**, not Bug fixes).
- Confirm no other consumer of `CurveAppearanceWidget` relies on its
  current fixed nested-box geometry before flattening (grep:
  `zoompanplot`, `curveappearancepresetmanager`,
  `overlaymanagerwidget`).
