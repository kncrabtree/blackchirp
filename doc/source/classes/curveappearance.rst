.. index::
   single: CurveAppearance
   single: CurveAppearanceWidget
   single: curve; appearance fields
   single: curve; preset management
   single: CurveAppearancePresetManager
   single: plot; curve configuration

CurveAppearance and CurveAppearanceWidget
=========================================

``CurveAppearanceWidget`` is a reusable Qt widget that presents a full set
of curve-appearance controls to the user. It edits a ``CurveAppearance``
struct — a plain-data aggregate that bundles every visual property a plot
curve carries. The same struct is the currency exchanged between the widget,
:cpp:class:`BlackchirpPlotCurveBase` (which stores and applies the values at
draw time), and :cpp:class:`OverlayBase` (which serializes a copy in its
metadata blob so overlay curves restore their appearance across sessions).

The widget is used in the plot context-menu's per-curve configuration panel.
The user-guide description of the resulting controls is in
:doc:`/user_guide/plot_controls`.

Appearance fields
-----------------

``CurveAppearance`` collects the following properties:

- **color** — pen color (``QColor``).
- **curveStyle** — rendering mode: ``QwtPlotCurve::Lines``, ``Sticks``,
  ``Steps``, ``Dots``, or ``NoCurve``.
- **lineThickness** — pen width in pixels (``double``).
- **lineStyle** — dash pattern: ``Qt::SolidLine``, ``Qt::DashLine``, etc.
  Pass ``Qt::NoPen`` to suppress the line completely.
- **markerStyle** — symbol drawn at each data point
  (``QwtSymbol::Style``); ``QwtSymbol::NoSymbol`` suppresses markers.
- **markerSize** — symbol size in pixels (``int``).
- **visible** — whether the curve is drawn at all (``bool``).
- **autoscale** — whether the curve is included when axis limits are
  computed during an autoscale operation (``bool``).
- **yAxis** — which Y axis the curve is plotted against
  (``QwtAxisId``; maps to the left or right axis).

``CurveAppearance`` is registered with ``Q_DECLARE_METATYPE`` so it can be
carried in ``QVariant`` containers and emitted through queued signals.

Init/apply paths
----------------

The widget provides two parallel init/apply paths depending on whether the
caller owns a curve object or an overlay object:

**Curve path** — used by the main plot context menu:

- ``initializeFromCurve(BlackchirpPlotCurveBase*)`` — reads the current
  appearance settings from the curve's storage and populates the controls.
- ``applyToCurve(BlackchirpPlotCurveBase*)`` — writes the current control
  state back to the curve, triggering an immediate visual update.

**Overlay path** — used when editing an overlay curve:

- ``initializeFromOverlay(std::shared_ptr<OverlayBase>)`` — reads the
  serialized appearance from the overlay's metadata blob.
- ``applyToOverlay(std::shared_ptr<OverlayBase>)`` — writes the current
  control state into the overlay's metadata.

Both paths ultimately operate on the same ``CurveAppearance`` struct, which
is also accessible directly via ``getCurrentAppearance()`` and
``setCurrentAppearance()``. Whenever any control changes, the widget emits
``curveAppearanceChanged(CurveAppearance)`` so callers can react
immediately without polling.

Preset management
-----------------

When a ``CurveAppearancePresetManager`` is attached via
``setPresetManager()``, a preset combo box and Save/Delete buttons become
active. The ``CurveAppearancePresetManager`` singleton (stored in
QSettings) ships with nine default presets — three curve, three stem, and
three scatter variants — and persists any user-defined
presets across sessions. The widget provides the following preset surface:

- ``applyPreset(name)`` — loads the named preset and emits
  ``curveAppearanceChanged()``.
- ``saveCurrentAsPreset(name)`` — saves the current control state as a
  preset; overwrites an existing entry if the name matches.
- ``deletePreset(name)`` — removes a user-defined preset (default presets
  cannot be deleted).
- ``refreshPresetList()`` — repopulates the combo box from the manager,
  e.g., after an external preset change.

The widget does not open dialogs itself; it delegates user-input steps
(name entry, delete confirmation) to callers via the signals
``presetSaveRequested(suggestedName)`` and
``presetDeleteRequested(presetName)``.

The user-facing description of the preset workflow — including the nine
default presets and the **Save Curve Appearance Preset** dialog — is in
:doc:`/user_guide/plot_controls`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: CurveAppearanceWidget
   :members:
   :protected-members:
   :undoc-members:
