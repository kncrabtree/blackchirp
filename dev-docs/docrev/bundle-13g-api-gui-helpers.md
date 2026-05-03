# Bundle 13g — API Reference: GUI Helper Classes

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-05-02: drafted → complete. Content commit feb21753. User review of the rendered docs surfaced one nitpick (`blackchirpplotcurve.rst` "Downsampling filter" rendered `2×*w*` literally because the asterisks abutted the multiplication sign — fixed by switching to `:math:`2w``). User also asked for the source bugs the Group A drafter had flagged to be fixed in the same content commit since they were tightly coupled to the documentation work. Fixes landed: `BlackchirpPlotCurve::appendPoint` bounding-rect typo (`setBottom`→`setTop`), `BlackchirpFIDCurve` destructor mutex leak, `ZoomPanPlot` box-zoom stuck-mode bug (dead `QEvent::Leave` handler), `ZoomPanPlot::panH`/`panV` synced with `pan()` to honor `override` and `spectrogramMode`, redundant `zoomXLock`/`zoomYLock` duplicate-check removed (`CustomZoomer::lockX/lockY` is authoritative), `ZoomPanPlot::resetPlot()` calls `waitForFilterComplete()` before `detachItems()` to close the most common item-list race. Doxyfile change: `MACRO_EXPANSION = YES`, `EXPAND_ONLY_PREDEF = YES`, `PREDEFINED = Q_PROPERTY(x)=` so Breathe does not crash on `Q_PROPERTY` declarations (`ScientificSpinBox` is the only currently-documented Q_PROPERTY-bearing class). Architectural follow-up — full thread-safe item-list snapshot for `filterData()` — recorded in `dev-docs/devel-roadmap.md` under the Medium tier.
- 2026-05-02: in progress → drafted. Three parallel Sonnet drafter/verifier pairs landed nine new RST pages and nine refreshed headers (the 8 cited headers plus `hardwareregistry.h` for the `HwSettingPriority` enumerator comments needed by Group C's directive). Punch lists resolved: Group A — Meta and Alt modifier-key descriptions in `zoompanplot.rst` were factually wrong (verifier read the source and confirmed Meta locks horizontal axes AND yRight while Alt locks horizontal AND yLeft); fixed. Group B — `\c key`/`\c overlay` Doxygen escapes in `curvefactory.rst` rendered as literal text; fixed via direct Edit. Class-level header `\brief` blocks in `curvefactory.h`, `curveappearancewidget.h` (the `CurveAppearance` struct), and `themecolors.h` trimmed to single sentences per the post-13f convention. Group C — RST page used "Optional / Advanced" for the third priority tier where the enum value is just `Optional` (Advanced is the QGroupBox label); fixed, and `\param storageKey` in `hwsettingswidget.h` left as-is (verifier confirmed it is consistent with the default `{}`). Coherence reviewer caught: heading-style inconsistency (the 9 new pages used over- and underlined `#`/`*`/`***` while every other class page uses `=` underline-only — normalized to `=`); raw `\a color` / `\a background` / `\a minContrastRatio` / `\a colorRole` Doxygen tokens in `themecolors.rst` prose (would render as literal backslash-a sequences) — fixed. Doc-build pass surfaced two more issue families: Q_PROPERTY KeyError in Sphinx (Doxyfile fix above) and duplicate-declaration warnings from explicit `doxygenstruct`/`doxygenenum` directives that were also auto-included by the parent class's `:members:` (`AxisConfig`/`PlotConfig` on `zoompanplot.rst`, `CurveAppearance` on `curveappearance.rst`, `ColorRole` on `themecolors.rst`, and `HwSettingPriority` cross-file with `hardwareregistry.rst`) — explicit nested directives dropped, prose `:cpp:enum:` cross-link added on `hwsettingswidget.rst` to point at the `HardwareRegistry` page for the enum.
- 2026-05-02: not started → in progress. Scope adjustment: dropped `markertablemodel.rst` and `chirptablemodel.rst` (used only by `FtmwConfigWidget`; not part of the broader reusable GUI surface). AC4 (MarkerTableModel columns) and the `dev-docs/awg-marker-system.md` source citation removed accordingly. Corrected `hwsettingswidget` source path: actual location is `src/gui/widget/hwsettingswidget.h`. Three parallel Sonnet drafter/verifier pairs split the nine remaining pages into Group A (zoompanplot / blackchirpplotcurve — plot core, heaviest), Group B (curveappearance / curvefactory / themecolors — plot appearance & construction), and Group C (scientificspinbox / enumcombobox / experimentconfigpage / hwsettingswidget — form widgets & config-page contract). A final coherence reviewer runs after all group revisions are accepted.
-->

Adds API reference pages for the GUI helper classes that
contributors are most likely to reuse.

## Scope

New pages under `doc/source/classes/`:

- `zoompanplot.rst` ← `src/gui/plot/zoompanplot.h` (`ZoomPanPlot`).
  Documents the zoom/pan keyboard and mouse contract that
  Blackchirp plot widgets inherit.
- `blackchirpplotcurve.rst` ←
  `src/gui/plot/blackchirpplotcurve.h`.
- `curveappearance.rst` ←
  `src/gui/plot/curveappearancewidget.h` (`CurveAppearance`,
  `CurveAppearanceWidget`).
- `curvefactory.rst` ← `src/gui/plot/curvefactory.h`
  (`CurveFactory`).
- `themecolors.rst` ← `src/gui/style/themecolors.h`.
- `scientificspinbox.rst` ←
  `src/gui/widget/scientificspinbox.h`.
- `enumcombobox.rst` ← `src/gui/widget/enumcombobox.h`.
- `experimentconfigpage.rst` ←
  `src/gui/expsetup/experimentconfigpage.h`.
- `hwsettingswidget.rst` ←
  `src/gui/widget/hwsettingswidget.h`.

## Out of scope

- Top-level windows/dialogs (`MainWindow`, `HwDialog`,
  `RuntimeHardwareConfigDialog`, `FtmwConfigDialog`,
  `UnifiedOverlayDialog`). They are concrete UIs whose primary
  documentation is the user guide; their API surface is not
  generally reused.
- `OverlayManagerWidget`, `BCExpOverlayWidget`,
  `CatalogOverlayWidget`, `GenericXYOverlayWidget`,
  `UnifiedOverlayWidget`. Not commonly reused outside the overlay
  feature itself.
- `MarkerTableModel` and `ChirpTableModel`. Used only by
  `FtmwConfigWidget`; not part of the broader reusable GUI
  surface.

## Sources

- The header files listed above.

## Sphinx file deltas

**Created:** one per page above.

**Possibly modified (Doxygen comment refresh):**
- All headers listed above.

## Acceptance criteria

- `ZoomPanPlot` page documents the modifier-key contract
  (Ctrl/Shift/Meta/Alt) so reusers don't have to read the
  source.
- `ScientificSpinBox` page documents the precision and step
  modes (Adaptive vs. Fixed).
- `CurveAppearance` page documents the appearance fields
  (color, line style, thickness, marker, etc.) reused by
  overlays.
