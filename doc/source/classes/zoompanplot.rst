.. index::
   single: ZoomPanPlot
   single: plot; zoom and pan
   single: plot; modifier-key contract
   single: plot; autoscale model
   single: plot; settings keys
   single: SettingsStorage; ZoomPanPlot

ZoomPanPlot
===========

``ZoomPanPlot`` is the interactive plot base class used throughout Blackchirp.
It subclasses both ``QwtPlot`` and :cpp:class:`SettingsStorage`, combining Qwt's
rendering engine with persistent per-plot configuration (zoom factors, tracker
options, grid style) stored under the plot's unique name. Every plot widget in
the application — FT view, FID view, tracking plots, LIF plots — inherits from
``ZoomPanPlot`` and obtains consistent zoom, pan, and curve-management behavior
without reimplementing it.

Subclasses override the protected hook layer (``filterData``, ``pan``, ``zoom``,
``getLimitRect``, ``buildContextMenu``, ``contextMenu``) to specialize behavior
for their data type. The :cpp:class:`BlackchirpPlotCurveBase` family provides the
curve objects attached to the plot; curve creation is handled by
:cpp:class:`CurveFactory`.

Modifier-key contract
---------------------

Mouse-wheel zoom
^^^^^^^^^^^^^^^^

Mouse-wheel events on the plot canvas zoom all applicable axes simultaneously.
Holding a modifier key locks one or more axes:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Modifier
     - Effect
   * - (none)
     - Zoom all axes around the mouse pointer.
   * - **Ctrl**
     - Lock both y axes; zoom x axes only.
   * - **Shift**
     - Lock both x axes; zoom y axes only.
   * - **Meta**
     - Lock x axes (xBottom, xTop) and yRight; zoom yLeft only.
   * - **Alt**
     - Lock x axes (xBottom, xTop) and yLeft; zoom yRight only (Qt may remap
       Alt+wheel to a horizontal scroll event; the implementation falls back
       to the x-angle delta in that case).

Each wheel notch zooms by the per-axis ``zoomFactor`` fraction (default 10 %).
The zoom is anchored at the mouse-pointer position, so the data point under
the cursor stays fixed.

Keyboard navigation
^^^^^^^^^^^^^^^^^^^

Keyboard events are processed when the plot canvas has focus:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Effect
   * - **Home**
     - Autoscale all axes.
   * - **Left / Right**
     - Pan x axes by 50 % of the current range.
   * - **Up / Down**
     - Pan y axes by 50 % of the current range.
   * - **Ctrl + Left / Right**
     - Zoom out / in x axes by a factor of 1.5.
   * - **Ctrl + Up / Down**
     - Zoom in / out y axes by a factor of 1.5.
   * - **Shift + Left / Right**
     - Zoom out / in x axes by a factor of 1.1.
   * - **Shift + Up / Down**
     - Zoom in / out y axes by a factor of 1.1.
   * - **Alt + Left / Right**
     - Pan x axes by 10 % of the current range.
   * - **Alt + Up / Down**
     - Pan y axes by 10 % of the current range.

While **Ctrl** is held, the rubber-band zoomer locks the y axis (zoom-box
changes x only). While **Shift** is held, it locks the x axis.

Mouse gestures
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Gesture
     - Effect
   * - Left-button drag
     - Rubber-band zoom (via ``CustomZoomer``).
   * - Middle-button drag
     - Drag-pan all non-override axes.
   * - Double-click (any button)
     - Autoscale all axes.
   * - Right-click
     - Context menu (see ``contextMenu()``).

Autoscale model
---------------

Each of the four Qwt axes (``xBottom``, ``xTop``, ``yLeft``, ``yRight``) has an
independent ``AxisConfig`` record. Per-axis state is held in ``AxisConfig``; see
the rendered struct below for individual flag semantics.

Settings keys
-------------

The following keys are stored in ``SettingsStorage`` under the plot's name.
Per-axis values live in a sub-array keyed by ``BC::Key::axes``; each element
of the array corresponds to one axis in the order ``yLeft``, ``yRight``,
``xBottom``, ``xTop``.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Key
     - Type
     - Description
   * - ``BC::Key::axes`` (array)
     - array
     - Per-axis sub-maps; each entry holds ``zoomFactor``,
       ``trackerDecimals``, and ``trackerScientific``.
   * - ``BC::Key::zoomFactor``
     - double
     - Fractional zoom step per wheel notch for one axis (default 0.1).
   * - ``BC::Key::trackerDecimals``
     - int
     - Decimal places shown in the cursor tracker for one axis (default 4).
   * - ``BC::Key::trackerScientific``
     - bool
     - Cursor tracker uses scientific notation for one axis (default false).
   * - ``BC::Key::trackerEn``
     - bool
     - Whether the cursor-position tracker overlay is shown (default false).
   * - ``BC::Key::kzCenter``
     - bool
     - Keyboard Y-zoom anchors to canvas center when true (default false).
   * - ``BC::Key::majorGridColor``
     - QColor
     - Color of the major grid lines.
   * - ``BC::Key::majorGridStyle``
     - Qt::PenStyle
     - Dash pattern of the major grid lines (default ``Qt::NoPen``).
   * - ``BC::Key::minorGridColor``
     - QColor
     - Color of the minor grid lines.
   * - ``BC::Key::minorGridStyle``
     - Qt::PenStyle
     - Dash pattern of the minor grid lines (default ``Qt::NoPen``).

Curve-management slot family
-----------------------------

``ZoomPanPlot`` provides a set of slots that apply an appearance change to a
:cpp:class:`BlackchirpPlotCurveBase` and then emit ``curveMetadataChanged()``
and trigger a replot.  These slots are called by the per-curve appearance
widget embedded in the context menu and may also be called directly by
subclasses.

The ``setCurveColor()`` slot opens a ``QColorDialog``; all other slots in the
family accept a value directly:

``setCurveColor``, ``setCurveStyle``, ``setCurveLineThickness``,
``setCurveLineStyle``, ``setCurveMarker``, ``setCurveMarkerSize``,
``setCurveVisible``, ``setCurveAutoscale``, ``setCurveAxisY``.

The context menu also contains an **Export XY** action for each curve that
calls ``exportCurve()``, opening a save-file dialog and writing the curve data
as a CSV file.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: ZoomPanPlot
   :members:
   :protected-members:
   :undoc-members:
