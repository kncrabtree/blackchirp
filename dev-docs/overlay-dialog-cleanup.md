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

## Next task — type-specific panel structure refactor

GUI evaluation (BCExp Creation, screenshot
`dev-docs/Screenshot_20260516_131134.png`) showed the left
(type-specific) column is visually disjoint from the center "Base
Options" / right "Curve Appearance" columns: no wrapping titled box,
two heading mechanisms stacked, mismatched column widths, truncated
text.

### Root cause

The virtual API conflates *container type* with *content*.
`createSourceFileConfigUI` was migrated to take a `SettingsTable*`, but
`createSourceFileSettingsUI` / `createTypeSpecificSettingsUI` still
hand subclasses a `QGroupBox*`, and each subclass creates *its own*
`SettingsTable` inside that box with its own `addSectionRow` headings.
The left column therefore renders three independent `SettingsTable`s
in three containers, with the flat-`QGroupBox` title (e.g. "Source
File Settings") stacked directly above an inner-table section band
(e.g. "FT Configuration"), independent column widths, and — because
`UnifiedOverlayWidget::setupUI` adds `p_typeSpecificWidget` bare to
`leftVLayout` while center/right go through
`createOverlayBaseOptionsBox`/`createCurveAppearanceBox` — no outer
titled frame.

### Target

Left panel becomes structurally identical to center/right: one titled
`QGroupBox` → one `SettingsTable`, with **Source File Configuration**,
**Source File Settings**, **Type-Specific Settings** as
`addSectionRow`/`addCheckableSectionRow` rows in that single table.
The base owns the table and the three section anchors; subclasses only
append rows.

### Steps

1. **Dialog shell** (`unifiedoverlaywidget.cpp:243-264`): wrap
   `p_typeSpecificWidget` in a titled `QGroupBox` the way
   `createOverlayBaseOptionsBox`/`createCurveAppearanceBox` do (title =
   overlay type name); equalize the three column stretches.
2. **Base virtual contract** (`overlaytypespecificwidget.h`): replace
   the three `create*UI` pure virtuals with row-population hooks
   against the base-owned table —
   `populateSourceFileConfigRows(SettingsTable*)`,
   `populateSourceFileSettingsRows(SettingsTable*)`,
   `populateTypeSpecificRows(SettingsTable*)`. Base `setupUI()` builds
   one `SettingsTable`, adds each section anchor, calls the matching
   hook, binds the appended rows, and records the three section
   indices as members.
3. **State machine** (`updateSourceFileControls()` +
   `validateSourceFile()`): repoint every
   `p_sourceFileSettingsBox->setVisible/setEnabled` and
   `p_overlaySettingsBox->setVisible` to section-scoped table ops —
   region visibility via a **new** additive
   `SettingsTable::setSectionVisible(section, bool)` (hides section row
   + bound rows regardless of checkable state; no equivalent today),
   region enable via existing `setBoundRowsEnabled`, type-specific
   presence driven purely by `hasTypeSpecificSettings()`. Preserve
   verbatim the Creation-keeps-config-rows-enabled decision (below),
   the deferred validation triggers, and context ordering.
4. **Per-subclass migration** (order: BCExp → GenericXY → Catalog):
   - **BCExp**: FT Configuration rows move into
     `populateSourceFileSettingsRows`; override
     `hasTypeSpecificSettings()` → `false` (today it `parent->hide()`s
     while the base `setVisible(true)` overrides that — a latent bug
     this resolves).
   - **GenericXY**: Format Detection / Column Mapping / Data Filtering
     become rows. The inline preview `QTableWidget` is **removed** in
     favor of a modal popup: `populateTypeSpecificRows` adds a
     "Columns" row (count from `d_parsedData.columnNames().size()`,
     names in tooltip), a "Statistics" row (the existing
     `p_dataStatsLabel` content), and a "Preview Data…" button that
     opens a modal `QDialog` hosting a `QTableWidget` built on demand
     by the existing `updatePreview()` logic (made lazy; the X/Y
     column-change and parse-complete paths just mark the cached
     preview stale instead of repopulating an embedded table).
   - **Catalog** (highest risk): `populateSourceFileSettingsRows` =
     Filtering section; `populateTypeSpecificRows` = the nested
     checkable "Convolution Enabled" section + sub-rows + Convolve
     button. `parent->setTitle("Catalog Settings")` becomes
     `setSectionTitle` on the type-specific section. The base must
     expose the type-specific section index (or a
     `setSectionBoundRowsEnabled` helper) so
     `updateFileInfo`/`updateConvolutionControls` can keep gating the
     convolution region (currently
     `p_overlaySettingsBox->setEnabled(...)`). Detail-row /
     `refreshSourceFileConfigState` machinery is unchanged.
5. **Cleanup**: delete `p_sourceFileSettingsBox`,
   `p_overlaySettingsBox`, `configureGroupBoxAppearance`, and the
   per-subclass inner-table+layout scaffolding; set a sensible
   `setMinimumWidth` on the shared table (mirror OverlayBaseOptions'
   380) to remove the column truncation; grep for stray references.

### Side-effects to watch

- `SettingsTable::setSectionVisible` is a genuine new primitive, not a
  repoint — additive but must hide section row + bound rows for both
  plain and checkable sections.
- Base must expose per-section enable to subclasses (catalog
  convolution gating).
- Catalog's "Convolution Enabled" becomes a checkable section nested
  among other sections in the shared table. `SettingsTable` already
  supports multiple sections (OverlayBaseOptions uses plain +
  checkable together) and `growEnclosingWindow` walks to the top-level
  window, so the grow-on-expand behavior is preserved — but this
  multi-section interaction is the item most worth runtime-testing
  (alongside items 1, 2, 10, 11 above).

### Section-shading consistency (folded into this refactor)

A non-checkable `addSectionRow` heading rendered on the window
background while a checkable `addCheckableSectionRow` rendered on the
`AlternateBase` band (visible in Base Options: "Identity" / "Scale &
Position" vs "Frequency Limits"). Root cause: the two paths used
different mechanisms — `QTableWidgetItem::setBackground()` vs an
`autoFillBackground` cell widget. Fix: `addSectionRow` now uses the
same centered cell-widget path as the checkable variant (a `QLabel`
in place of the `QCheckBox`), and is tracked as a `Section` (with a
null checkbox) so it is also retitleable / bindable / show-hideable.
Both heading kinds now paint the identical band.

## Status — implemented, build-clean, runtime-unverified

Landed: `SettingsTable` generalized (every section a tracked
`Section` + centered cell widget; new collapse-aware
`setSectionVisible`; `bindSectionRows`/`setBoundRowsEnabled`
generalized for a null checkbox; section-shading unified). Base
rewritten onto one table with three `populate*Rows` hooks; QGroupBox
members + `configureGroupBoxAppearance` deleted. `UnifiedOverlayWidget`
wraps the tier in a titled box. BCExp/Catalog/GenericXY migrated;
GenericXY preview is now a "Columns"/"Statistics"/"Preview Data…"
modal. `blackchirp` + `blackchirp-viewer` build clean; no runtime
testing was possible. All verification items above still apply and the
GenericXY preview item is now the modal, not an inline table.

### GUI-evaluation follow-ups (landed, build-clean, runtime-unverified)

From a first GUI pass:

1. **Expanding value widgets stretch**: the 2-widget `addSettingRow`
   now lets a horizontally-expanding widget (e.g. the path `QLineEdit`)
   fill the value cell; the trailing spacer is only added when nothing
   expands (preserves the left-aligned input+button rows). BCExp's
   experiment spinbox is given an Expanding policy + stretch in its
   cell.
2. **Glyph-free status**: dropped the `✓`/`✗`/"FT Configured ✓"
   characters (they could render as tofu); status now conveys state
   purely through `ThemeColors` success/error/info roles. GenericXY's
   two previously-unstyled error states ("file not found", "no parser")
   now set `StatusError`.
3. **Themed browse icon**: BCExp/Catalog/GenericXY browse buttons use
   `:/icons/folder-open.svg` via `ThemeColors::createThemedIcon`
   (`IconSecondary`) instead of the `📁` unicode literal.
4. **Grow-once on expand**: `SettingsTable` sections now grow the
   enclosing window on the *first* expand only, and only when the
   section started collapsed (per-section one-shot `growPending`).
   Repeated collapse/expand no longer grows without bound; a
   section that starts expanded never grows.
5. **GenericXY parser gate** (separate subsystem, opportunistic):
   `getParser()` fetched the `GenericXYParser` via
   `findParserOfType()`, whose `canParse()`/`analyzeFile()` heuristic
   over-eagerly rejected valid tabular files ("No GenericXY parser
   available"). It now fetches the parser by type from
   `getAllParsers()` without the discovery gate; `parseWithSettings()`
   still reports a real error for genuinely non-tabular input. **Needs
   targeted runtime check with the file(s) that previously failed.**

### GUI-evaluation follow-ups, round 2 (landed, build-clean, runtime-unverified)

1. **Section heading background** (corrected in round 3): the
   *intended* look is the original plain `addSectionRow` rendering —
   an `AlternateBase` band painted by the row's `QTableWidgetItem`,
   contrasting with the table's `Base`. Only the checkable variant was
   wrong (it painted via a cell-widget palette that didn't match).
   Fix: both kinds now carry the band on a styled `QTableWidgetItem`;
   a checkable row keeps a **transparent** centering cell widget on
   top of that item (no `autoFillBackground`/`Window`), so the
   item-painted band shows through and the two render identically.
   Plain headings are item-based again (no cell widget) and still
   tracked as sections (title via the item, visibility via row hide,
   enable via item flags).
2/3. **Unified heading text**: the band/centre/bold come from the
   shared item styling; the checkable row's checkbox text is
   emphasized to match.
4. **Dropped redundant "FT Configuration" heading** — the base
   already emits the "Source File Settings" tier heading above those
   rows.
1(status). BCExp Path/FT status labels: `wordWrap(false)` +
   vertically-centered + tooltip mirrors the text (kills the spurious
   extra line); colour continues to come from `ThemeColors`
   success/info/error roles, now also tooltip-synced on every update.
5. **Catalog Creation filtering**: `onFilePathChanged()` now calls the
   base `validateSourceFile()` so the Source File Settings tier
   (filtering) enables as soon as a valid catalog is chosen in
   Creation, not only in Settings/edit mode.
6. **Reverted** the single-column `indexAsX` parser work — the failing
   file was a single-column FID chosen by mistake, not a parser bug.
   Instead, the GenericXY parse-failure path now probes column count
   and reports *"Cannot parse: file must contain at least 2 columns
   (X and Y); only 1 column was found."* (falling back to the parser's
   own error message, then a generic one) instead of the opaque
   "Failed to parse file".

### GUI-evaluation follow-ups, round 3

- **Section band restored correctly** (item 1 above) — band is the
  item-painted `AlternateBase`, checkable rows transparent on top.
- **Status-label colour**: the per-widget `styleStatusLabel` /
  `styleSubtleLabel` helpers (BCExp/Catalog/GenericXY) now set a
  widget-scoped **stylesheet** colour via `ThemeColors::getCSSColor`
  instead of a `QPalette` `WindowText` override — a `QLabel` hosted in
  a `QTableWidget` cell does not reliably honour the palette override
  but always honours a stylesheet colour (the pattern the dialog's own
  status strip already uses). Colour is still theme-derived.
- **GenericXY source-file validity is now existence-based**
  (mirrors the catalog Creation fix but stronger): a readable file is
  a valid *source*; whether it parses is a parsing-settings concern
  (still reported by `validateSettings`, still gating acceptance via
  `isDataValid`). `setSourceFilePath()` calls the base
  `validateSourceFile()` so the Format Detection / Column Mapping /
  Data Filtering tier enables the moment a file exists — the user
  needs those controls active to coax a non-default file into
  parsing.

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
