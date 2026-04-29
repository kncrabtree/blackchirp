# Bundle 13g — API Reference: GUI Helper Classes

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
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
  `src/gui/widget/hwsettingswidget.h` (or
  `src/gui/dialog/hwsettingswidget.*` depending on actual
  location).
- `markertablemodel.rst` ←
  `src/data/model/markertablemodel.h`.
- `chirptablemodel.rst` ←
  `src/data/model/chirptablemodel.h`.

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

## Sources

- The header files listed above.
- `dev-docs/awg-marker-system.md` — for `MarkerTableModel`.

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
- `MarkerTableModel` page enumerates the columns and links to the
  user-guide chirp setup page (bundle 07).
