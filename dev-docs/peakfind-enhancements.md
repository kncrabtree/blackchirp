# Peak Find panel — enhancement implementation plan

Ephemeral planning doc for the post-side-dock-refit enhancements to
`PeakFindWidget`. Purged before release. The shipped behavior is
documented in the user guide / changelog, not here.

Baseline: `PeakFindWidget` is now code-built (no `.ui`), a `QToolBar`
+ `QTableView` over `PeakListModel` → `QSortFilterProxyModel`, reaching
its host via `findFtmwView()`. Public API is signal/slot only; keep it
that way — `FtmwViewWidget` does the cross-widget wiring.

## Locked design decisions

- Filtering ships **both** static filters and a live "in main-plot
  view" toggle (requires a new visible-x-range signal on
  `ZoomPanPlot`).
- Peak navigation uses an **explicit y framing**, not autoscale, so
  the selected peak is clearly visible: `ymax = 1.25·Int`; if the
  plot's current y-range dips below zero, frame symmetrically
  `[-1.25·|Int|, +1.25·|Int|]`, otherwise `[0, 1.25·Int]`.
- Appearance popup appears **next to the Peak Find toolbar button**
  (menu `exec`'d at the button's global position), reusing
  `MainFtPlot`/`ZoomPanPlot`'s existing curve-apply path — no
  duplicated `setCurve*` logic.
- Layout: one top `QToolBar`; a filter strip directly below it, above
  the table, hidden until a checkable "Filter" toolbar action is on;
  navigation lives in the table's row context menu. The existing
  `adjustToolbarStyle()` icon-only fallback absorbs the extra actions.

## Feature A — "Appearance" button for the FTPeaks curve

The peaks are a single `BlackchirpPlotCurve` (`MainFtPlot::p_peakData`,
key `BC::Key::peakCurve` = `"FTPeaks"`), already styleable via the
main-plot right-click menu. This is a discoverability shortcut.

Steps:

1. **Factor out the curve-appearance menu builder.** Extract the
   ~70-line `CurveAppearanceWidget` block in
   `ZoomPanPlot::buildContextMenu` (`zoompanplot.cpp:1348–1421`) into
   a reusable protected helper:
   `QMenu *ZoomPanPlot::buildCurveAppearanceMenu(BlackchirpPlotCurveBase *curve, QWidget *parent)`.
   Have `buildContextMenu` call it. No behavior change — this is a
   pure refactor commit (no changelog entry).
2. **`MainFtPlot::showPeakAppearanceMenu(const QPoint &globalPos)`** —
   builds the helper menu for `p_peakData.get()` and `exec`s it at
   `globalPos`.
3. **`PeakFindWidget`** — add `p_appearanceAction` (icon
   `:/icons/palette.svg` or `swatch.svg`; tooltip "Edit the peak
   marker appearance"). On trigger, emit
   `editPeakAppearanceRequested(QPoint globalPos)`, where `globalPos`
   is the action's button rect mapped to global
   (`p_toolBar->widgetForAction(p_appearanceAction)->mapToGlobal(...)`)
   so the menu opens against the button.
4. **`FtmwViewWidget::showPeakFinder`** — connect
   `p_pfw::editPeakAppearanceRequested` →
   `p_mainFtPlot->showPeakAppearanceMenu(pos)`.

Changelog: "User interface" improvement entry, own commit.

## Feature B — peak filtering

### B1. Static filters (freq range, min intensity)

1. New `PeakListFilterProxyModel : public QSortFilterProxyModel`
   (`src/data/model/peaklistfilterproxymodel.{h,cpp}`, register in
   `cmake/BlackchirpData.cmake` + viewer list). State: optional
   `[minFreq,maxFreq]`, optional `minIntensity`, plus the live-view
   fields (B2). Override `filterAcceptsRow` reading source col 0
   (freq) and col 1 (intensity) via `Qt::EditRole` (doubles).
   Setters call `invalidateFilter()`.
2. Swap `PeakFindWidget`'s plain `QSortFilterProxyModel` for it; keep
   `setSortRole(Qt::EditRole)`.
3. Persist filter state under the `peakFind` group: new
   `BC::Key` entries `pfFilterMinFreq`, `pfFilterMaxFreq`,
   `pfFilterMinInt`, `pfFilterEnabled`, `pfViewSync` declared in
   `peakfindwidget.h`.

These are **display** filters, distinct from the Options dialog's
min/max which bound the *search*. Tooltips must say so; do not reuse
the same keys.

### B2. "Only peaks in main-plot view" toggle

1. **`ZoomPanPlot` visible-range signal.** Add
   `void visibleXRangeChanged(double min, double max)`. Emit at the
   end of `replot()` after the xBottom scale div is finalized
   (post-`rescaleAxes`/autoscale path around `zoompanplot.cpp:100–152`),
   guarded by a "changed since last emit" check to avoid thrash
   during pan/zoom. Confirm one emit per settled view (debounce if
   the rescale path runs multiple times per interaction).
2. **`FtmwViewWidget::showPeakFinder`** — connect
   `p_mainFtPlot::visibleXRangeChanged` → `p_pfw::setMainPlotXRange`.
   On toggle-on, seed the current range (add
   `FtmwViewWidget::mainPlotXRange()` or have the plot emit once).
3. **`PeakFindWidget`** — checkable "In view" control in the filter
   strip; when on, `setMainPlotXRange` feeds the proxy's view bounds
   and `invalidateFilter()`.

### B3. Filter strip UI

Checkable "Filter" action in the top toolbar shows/hides a slim strip
(its own `QWidget`+`QHBoxLayout`, or a second compact `QToolBar` using
`addWidget`) placed between toolbar and table: min/max freq
`QDoubleSpinBox` (MHz, `specialValueText` = unbounded), a min-intensity
`ScientificSpinBox`, and the "In view" checkbox. Hidden by default to
preserve vertical space in a tall narrow dock.

Changelog: one "User interface" improvement entry (filtering), own
commit; the `ZoomPanPlot` signal is part of that commit (it has no
standalone user-visible effect).

## Feature C — center a plot on a selected peak

1. **`PeakFindWidget` context menu.** `setContextMenuPolicy(CustomContextMenu)`
   on `p_peakListView`; build a menu with a "Center on ▸" submenu
   populated from `findFtmwView()->getPlotNames()`. Double-click a row
   = center the main plot (fast path). On choice, emit
   `centerPlotRequested(QString plotName, double freq, double intensity)`
   (freq/intensity from the source-mapped model row).
2. **Navigation half-width** — remembered setting
   `BC::Key::pfNavHalfWidth` (default 2.0 MHz). Surface it as a new
   row in the existing Options dialog (cheap, discoverable) rather
   than another inline control.
3. **`FtmwViewWidget`** — slot resolves `plotName` via `d_plotMap`
   (`std::map<QString,FtPlot*>`), calls
   `FtPlot::zoomToPeak(freq, halfWidth, intensity)`.
4. **`ZoomPanPlot::zoomToPeak(double xCenter, double xHalfWidth, double intensity)`**:
   - x: build a `QwtScaleDiv` for `[xCenter-xHalfWidth, xCenter+xHalfWidth]`
     and call `setXRanges`. **Verify FtPlot top/bottom axis coupling**
     — `setXRanges` takes both bottom and top divs; FtPlot's top axis
     may be a transformed (LO-relative) scale. Confirm the correct
     top div before finalizing (likely mirror bottom or recompute
     from the plot's existing x transform).
   - y: read the current `yLeft` interval; pick `[0, 1.25·Int]` or
     symmetric `[-1.25·|Int|, +1.25·|Int|]` per the locked rule. Apply
     via a new `void ZoomPanPlot::setYRangeOverride(double min, double max)`
     using the existing `setAxisOverride` + `overrideRect` mechanism
     (pattern at `zoompanplot.cpp:334–335`); do not poke `d_config`
     from outside.

Changelog: "User interface" improvement entry, own commit.

## Commit / changelog sequence

1. Refactor: extract `buildCurveAppearanceMenu` (no changelog).
2. Feature A (Appearance button) + changelog.
3. Feature B (filter proxy + strip + `visibleXRangeChanged`) +
   changelog.
4. Feature C (navigation + `zoomToPeak`/`setYRangeOverride`) +
   changelog.

Enhancement entries go under the changelog's **User interface**
improvements list (these are features, not the Bug-fixes section used
for the earlier two fixes). Confirm the exact 2.0.0.rst subsection
heading before writing.

## Open items to verify during implementation

- FtPlot x top-axis transform when calling `setXRanges` from
  `zoomToPeak` (Feature C step 4).
- Debounce semantics of `visibleXRangeChanged` — one settled emit per
  pan/zoom, not per intermediate `replot`.
- Toolbar width with the added actions: confirm `adjustToolbarStyle()`
  still collapses cleanly at ~250 px; all new actions need tooltips.
- Interaction of the live-view filter with `Live Update`: re-find +
  view-filter should not fight (filter runs on already-found peaks;
  fine, but sanity-check the busy/waiting path).
