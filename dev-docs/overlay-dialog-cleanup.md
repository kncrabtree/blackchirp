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
  revealed rows' exact pixel height, grow-only). Registered in both
  `BlackchirpGui.cmake` and `BlackchirpViewerGui.cmake`.
- **`HwSettingsWidget`** refactored onto `SettingsTable` (Part 1);
  `addArrayTableRow` reimplemented on the paired-widget row API.
- **`OverlayBaseOptionsWidget`** (2a), **`CurveAppearanceWidget`**
  (2b), **`OverlayTypeSpecificWidget` + the three subclasses** (2c)
  flattened. The three subclass containers and the catalog
  `Convolution` box were kept as flat `QGroupBox`es to preserve their
  check/enable/visible state machines verbatim — see next task.
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

1. **Settings-context "Source File Configuration (Optional)"**:
   checkable, starts unchecked (settings region hidden), toggling
   shows/enables it; Creation context non-checkable, settings enable
   only when the source file is valid. (Highest risk — see next task.)
2. **Catalog convolution end-to-end**: load `.cat`/`.xo`, toggle
   Convolution (collapse/expand + window grow), change shape / FWHM /
   range / points, Convolve runs/cancels with progress bar; parsed
   file-detail rows show/hide on load.
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

## Next task — finish the checkable-container conversions

**Goal:** remove the last flat `QGroupBox`es kept only for their
check/enable state machines: the catalog **Convolution** box and the
shared base **Source File Configuration** box (the latter across
`OverlayTypeSpecificWidget` + `BCExpOverlayWidget`,
`CatalogOverlayWidget`, `GenericXYOverlayWidget`). Replace them with
`SettingsTable` checkable section rows.

**Sequencing (decided): one combined commit.** Within it, still do the
decouple-then-swap *internally* so the diff is reviewable: introduce
accessor/relay helpers, repoint all call sites, then point the helpers
at the new control — but land it as a single commit. Build both
binaries; behavior must be byte-for-byte unchanged except the
intended visual flatten.

### Part A — `SettingsTable` API additions

The base box is not a clean checkable (catalog convolution is). Add:

- `void setSectionTitle(int row, const QString &title)` — retitle a
  section row in place (Creation "Source File Configuration" vs
  Settings "Source File Configuration (Optional)").
- `void setSectionCheckable(int row, bool checkable)` — a section row
  that can drop its checkbox and render as a plain centered heading
  (Creation context: non-checkable; Settings: checkable). Implement by
  building the row with a checkbox always present and swapping the
  cell widget (checkbox ⟷ plain shaded label) so the row index and
  bound-row wiring survive the mode change.
- `void setBoundRowsEnabled(int section, bool)` — in addition to the
  existing hide-on-uncheck, support **enable/disable** of bound rows
  without hiding them (Creation context keeps the source-file rows
  visible but disabled until the file is valid; only Settings context
  hides). Keep `bindSectionRows` (hide semantics) and add an
  enable-only variant or a mode flag.
- Expose the section checkbox after creation
  (`QCheckBox *sectionCheckBox(int row) const`) so the relay/connect
  can be repointed.

Keep the grow-on-expand behavior; it already benefits these.

### Part B — decoupling helpers (catalog)

Add to `CatalogOverlayWidget`:

- `bool isConvolutionEnabled() const;`
- `void setConvolutionEnabled(bool);`
- `signals: void convolutionEnabledChanged(bool);` — emitted from one
  internal slot connected to *whichever* control backs it.

Replace every `p_convolutionGroupBox->isChecked()` (~`setupForSettings`,
`applyToOverlay`, `getSettingsHash`, `createOverlay`, `loadSettings`,
`saveSettings`, `updateConvolutionControls`,
`getCurrentConvolutionState`) with `isConvolutionEnabled()`, every
`->setChecked(x)` with `setConvolutionEnabled(x)`, and both
`connect(p_convolutionGroupBox, &QGroupBox::toggled, …)` (the
`onConvolutionEnabledToggled` one and the content-visibility one) with
connections to `convolutionEnabledChanged`. The `Convolve` button (not
a table row) collapses with the section: drive its visibility from the
relay too, or make it a trailing settings row bound to the section.

Then swap: build the convolution config as one `SettingsTable` with
`addCheckableSectionRow("Convolution Enabled", …, &box)`, the existing
`addSectionRow("Line Shape")` / `"Frequency Range & Resolution"`
sub-headings, and `bindSectionRows` covering all convolution rows;
point the three helpers + relay at `box`. Delete `p_convolutionGroupBox`.

### Part C — decoupling helpers (base source-file-config)

`OverlayTypeSpecificWidget` drives `p_sourceFileConfigBox` via
`setupUI`, `updateSourceFileControls`, `onSourceFileConfigToggled`,
`configureForContext`. Introduce a small abstraction the state machine
talks to instead of the box directly:

- `bool isSourceConfigEnabled() const;`
- `void setSourceConfigChecked(bool);` (signal-blocked variant
  preserved — `updateSourceFileControls` blocks signals around the
  Settings-context `setChecked`).
- `void setSourceConfigCheckable(bool);`
- `void setSourceConfigTitle(const QString &);`
- `signals: void sourceConfigToggled(bool);` relayed to
  `onSourceFileConfigToggled`.

The subclass `createSourceFileConfigUI(QGroupBox*)` virtuals must now
fill a `SettingsTable` instead of a box. Either change the virtual's
parameter to `SettingsTable*` (update all three subclasses + the base
caller consistently) or keep the box as a thin frameless host of the
table. Recommended: change the signature to `SettingsTable*` and add
a checkable "Source File Configuration" section row at the top, with
the subclass's browse/path/status rows bound to it; the base state
machine repoints onto the Part A APIs (`setSectionTitle`,
`setSectionCheckable`, `setBoundRowsEnabled`, `sectionCheckBox`).

**Hard constraints (unchanged behavior):**

- Creation: section non-checkable, titled "Source File Configuration",
  source rows always visible, *enabled* only when the file is valid
  (`setBoundRowsEnabled`, not hide).
- Settings: section checkable, titled "Source File Configuration
  (Optional)", `setChecked(d_sourceFileEnabled)` under blocked
  signals, source rows hidden when unchecked (existing hide
  semantics), `onSourceFileConfigToggled` fired identically.
- `p_sourceFileSettingsBox` / `p_overlaySettingsBox` enable/visible
  logic and `hasTypeSpecificSettings()` gating untouched (those are
  separate regions; only the *config* box is being converted).
- No changes to settings keys, persistence, validation, preview,
  `Configure FT…`, convolution threading/caching, or on-disk format.

### Risks / watch list

- The base box conversion is the riskiest change in the whole effort
  (the Creation/Settings state machine). Regression-test item 1 above
  before and after.
- `setSectionCheckable` swapping the cell widget must not orphan
  `bindSectionRows` connections or the relay; re-grab `sectionCheckBox`
  after a mode change.
- Window grow-on-expand now fires for the source-file/convolution
  sections too — confirm no unwanted growth during context setup
  (initial state is applied without resize by design; keep that).
- Catalog `Convolve` button visibility must track the section exactly
  as the old content-widget `setVisible` did.
