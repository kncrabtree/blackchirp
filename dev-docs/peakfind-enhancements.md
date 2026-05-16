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
- Layout: a **two-row toolbar** at the top (row 1: Find, Live,
  Appearance, Filter — the "what to show" actions; row 2: Options,
  Export, Remove, Show Parent — the "manage the list" actions); a
  filter strip directly below it, above the table, hidden until the
  checkable "Filter" action is on; navigation lives in the table's row
  context menu. Two rows are chosen over relying on `QToolBar`'s
  overflow-extension (`>>`) menu: with the added Appearance + Filter
  actions the single-row icon-only fallback lands right at the dock's
  ~250 px boundary, and a hidden-behind-`>>` action is poor
  discoverability for first-class controls. `adjustToolbarStyle()`
  still applies its text-beside-icon → icon-only fallback per row.

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
   `void visibleXRangeChanged(double min, double max)`. A bare
   "changed since last emit" guard is **insufficient**: `pan()` calls
   `replot()` on every mouse-move (each frame genuinely changes the
   range), and `replot()` splits across threads — the synchronous path
   sets the xBottom scale then kicks off the async filter worker
   (`zoompanplot.cpp:422–431`), and the worker-finished lambda issues a
   *second* `QwtPlot::replot()` (`zoompanplot.cpp:103–110`). There is
   no single "settled replot" to hook. Instead, debounce: add a
   single-shot `QTimer` (~75–100 ms) and `d_lastEmitXMin/Max` members.
   Every `replot()` exit (both branches) and the worker-finished
   lambda call `timer->start()` (cheap restart). The timer slot reads
   `axisInterval(QwtPlot::xBottom)` — reliable because the scale is
   always set *before* the async kickoff, independent of the
   replot/busy split — and emits only if it moved beyond an epsilon
   vs. the last emitted pair. One emit per settled view.
2. **`FtmwViewWidget::showPeakFinder`** — connect
   `p_mainFtPlot::visibleXRangeChanged` → `p_pfw::setMainPlotXRange`
   **unconditionally**; no busy-path gating. The proxy filter is
   purely downstream of the peak-find pass (`newPeakList` replaces the
   model; the view-range change only calls `invalidateFilter()`), so
   an in-flight `findPeaks()` and a view-filter invalidation cannot
   contend. On toggle-on, seed the current range (add
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
     and call `setXRanges`, passing the **same div** for both bottom
     and top. `FtPlot` never configures `xTop` (constructor sets only
     `xBottom`/`yLeft` titles, no custom `QwtScaleDraw` or LO-relative
     transform); nothing is attached to `xTop`, so `replot()` leaves it
     disabled and the top div is display-irrelevant. Mirroring bottom
     is correct and matches `AuxDataViewWidget::pushXAxis`, the only
     other `setXRanges` caller. `setXRanges` also pins `autoScale=false`
     on both x axes until the user autoscales — desired here.
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

## Resolved verification items

Verified against the source before implementation; resolutions folded
into the feature sections above.

- **FtPlot x top-axis transform (Feature C step 4)** — *resolved.* No
  transform exists; `FtPlot` never configures `xTop`. Pass the same
  div for bottom and top in `setXRanges`. Caveat struck.
- **Debounce of `visibleXRangeChanged` (Feature B2 step 1)** —
  *resolved.* `replot()` fires per mouse-move and splits across a
  sync/async pair, so a dirty-check alone thrashes. Use a single-shot
  ~75–100 ms `QTimer` sampling `axisInterval(xBottom)`; one emit per
  settled view. Detail in B2 step 1.
- **Toolbar width with the added actions** — *resolved into a design
  change.* Single-row icon-only lands at the ~250 px boundary with the
  Appearance + Filter additions; rather than depend on the `>>`
  overflow menu, the toolbar becomes **two rows** (see Locked design
  decisions). All new actions still need tooltips (icon-only is the
  steady state in a narrow dock).
- **Live-view filter vs. `Live Update`** — *resolved.* No contention:
  the proxy filter is downstream of the find pass and never touches
  the `d_busy`/`d_waiting` path. Wire B2 connection unconditionally
  (Feature B2 step 2).
