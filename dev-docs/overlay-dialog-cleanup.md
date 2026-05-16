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
- **Shared widget:** extract the settings-table idiom out of
  `hwsettingswidget.cpp` into a reusable `SettingsTable`
  (`QTableWidget` subclass) and have both `HwSettingsWidget` and the
  overlay widgets use it. The API is designed to also absorb the
  other hand-rolled copies (`ExperimentSetupPage`,
  `DigitizerConfigWidget`, `LifProcessingWidget`, the
  `FtmwViewWidget` toolbar panels) in later backport work — those
  backports are **out of scope here**.
- **Scope:** layout **plus small UX fixes** — merge obviously
  redundant controls, fix label/units inconsistencies, replace raw
  stylesheet strings with `ThemeColors` roles. **No** changes to
  settings keys, validation logic, the `Configure FT…` sub-dialog,
  catalog convolution threading, preview behavior, or the on-disk
  format. Behavior is otherwise byte-for-byte unchanged.

## Part 1 — `SettingsTable` widget

`makeSettingsTable()` is currently a file-static in
`gui/widget/hwsettingswidget.cpp`; `addTableRow()` is a
`HwSettingsWidget` member. The same compact 2-column-table idiom has
since been hand-rolled in several other places (`ExperimentSetupPage`,
`DigitizerConfigWidget`, `LifProcessingWidget`, the `FtmwViewWidget`
toolbar panels). Rather than a bag of free functions, introduce a
real `QTableWidget` subclass that owns the configuration and the
row-building API — this gives a clean place for checkable section
rows and multi-widget value cells, and a single class the other
consumers can be backported onto later.

`gui/widget/settingstable.{h,cpp}`:

```
class SettingsTable : public QTableWidget {
    Q_OBJECT
public:
    explicit SettingsTable(QWidget *parent = nullptr);

    // Label + single value widget. Returns the new row index.
    int addSettingRow(const QString &label, QWidget *value,
                       const QString &tooltip = {});

    // Label + two widgets side by side in the value cell
    // (checkbox+spinbox, input+invert, …). Returns the row index.
    int addSettingRow(const QString &label, QWidget *first,
                       QWidget *second, const QString &tooltip = {});

    // Bold, spanned, themed separator/heading row. Returns row index.
    int addSectionRow(const QString &title);

    // Section row with a leading checkbox; rows bound via
    // bindSectionRows() are shown/hidden (setRowHidden) on toggle.
    int addCheckableSectionRow(const QString &title, bool checked,
                               QCheckBox **outBox = nullptr);
    void bindSectionRows(int sectionRow, const QList<int> &rows);
};
```

The constructor carries over the exact `makeSettingsTable` body (col 0
`ResizeToContents`, `setStretchLastSection`, vertical header hidden,
`NoSelection`, `NoEditTriggers`, `AdjustToContents`, vertical
scrollbar off). `addSettingRow` carries over the `addTableRow`
row-height logic (`sizeHint().height() + 4`); the two-widget overload
wraps the pair in a `QWidget` + zero-margin `QHBoxLayout`.
`addSectionRow` spans both columns (`setSpan(row,0,1,2)`), bold,
non-selectable, with a background fill from a `ThemeColors` role (the
existing subtle-surface/header role) — **not** a raw stylesheet
string. The checkable variant drives `setRowHidden` on its bound rows,
giving the `Frequency Limits` collapse for free and matching the old
checkable-`QGroupBox` behavior.

Refactor `HwSettingsWidget` to use `SettingsTable` (delete its private
`makeSettingsTable`/`addTableRow`; keep `addArrayTableRow`, which is
registry-specific, reimplemented against the new class). Add
`settingstable.{h,cpp}` to `cmake/BlackchirpGui.cmake` **and**
`cmake/BlackchirpViewerGui.cmake` (overlay widgets are in both
binaries).

`ExperimentSetupPage`, `DigitizerConfigWidget`, `LifProcessingWidget`,
and the `FtmwViewWidget` toolbar panels are **backport candidates,
out of scope here** — the API is shaped to absorb them later, but this
work migrates only `HwSettingsWidget` and the overlay widgets. Do not
touch the others.

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

Build `Frequency Limits` with `addCheckableSectionRow` and
`bindSectionRows` the two freq value rows to it: unchecking hides
those rows (`setRowHidden`) so the table shrinks via
`AdjustToContents`, matching the old checkable-`QGroupBox` collapse
rather than merely disabling the rows in place. The section row
itself stays visible so the user can re-expand.

**2b. `CurveAppearanceWidget`** — Presets bar + one table:

- Presets: a compact top row (`Preset` combo + `Save` + `Delete`),
  not a nested box.
- Table rows: Color, Type (`p_curveStyleBox`), Width, Style
  (`p_lineStyleBox`), Marker, Size, Y Axis; `Visible` and
  `Autoscale` as one trailing h-box row (or two checkable rows).
- **Shared-widget caveat:** `CurveAppearanceWidget` is embedded in a
  popup `QMenu` (via `QWidgetAction`) by three other consumers, none
  of which pins its geometry — the menu sizes itself to the widget's
  `sizeHint`:
  - `zoompanplot` — main-plot right-click curve styling.
  - `overlaymanagerwidget` — the manager's per-overlay appearance
    popup. The manager table itself stays out of scope, but it shares
    this widget so the popup is in the blast radius.
  - the Peak Find panel — appearance popup from its toolbar button;
    same function, same embedding.
  Dropping a `QTableWidget` into a `QMenu` is the real risk (popup
  resize, an unexpected internal scrollbar). `SettingsTable`'s
  `AdjustToContents` + scrollbar-off config should prevent the
  scrollbar, but **all three popup consumers must be visually
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

Commit 6: dialog shell + spacer/progress relocation. Changelog: one
bullet appended to the end of the top-level **User interface** list in
`doc/source/changelog/2.0.0.rst` (after the experiment-setup
minimum-size entry), **no** `:commit:` tag (that section does not use
them), phrased like its neighbors:

> - Streamlined the overlay creation and configuration dialog:
>   flattened the nested group boxes into aligned settings tables and
>   moved the validity and progress indicators into the dialog button
>   bar. See :doc:`/user_guide/overlays`.

## Part 4 — Docs

Refresh the three creation-dialog screenshots in
`doc/source/_static/user_guide/` (`overlays-overlay_creation_dialog`,
`overlays-catalog_convolution_settings`, `overlays-generic_xy_
preview`) after the visual change; the manager screenshot is
unchanged. Light copy pass on `overlays.rst` "Creating an Overlay"
if control groupings change names. Build docs via the `breathe`
env (see `AGENTS.local.md`).

## Decisions resolved (do not re-litigate)

These were the open questions; all are now locked into the parts
above:

- **Section-row styling:** bold spanned cell with a `ThemeColors`
  background role, defined once in `SettingsTable::addSectionRow`. No
  raw stylesheet strings anywhere. (`HwSettingsWidget` has no existing
  in-table section style to match — this is the new canonical one.)
- **`Frequency Limits` collapse:** hide the value rows
  (`setRowHidden`) via `addCheckableSectionRow` / `bindSectionRows`,
  not disable-in-place.
- **Changelog:** one bullet at the end of the **User interface**
  section of `2.0.0.rst`, no `:commit:` tag — wording in Part 3.
- **`CurveAppearanceWidget` consumers:** flatten proceeds; the three
  `QMenu`-popup consumers (`zoompanplot`, `overlaymanagerwidget`, the
  Peak Find panel) are the regression surface — see the Part 2b
  caveat. (`curveappearancepresetmanager` only constructs
  `CurveAppearance` value structs; it has no geometry stake.)
