.. index::
   single: BlackchirpPlotCurveBase
   single: BlackchirpPlotCurve
   single: BCEvenSpacedCurveBase
   single: BlackchirpFTCurve
   single: BlackchirpFIDCurve
   single: plot curve; hierarchy
   single: plot curve; storage backend
   single: plot curve; downsampling filter
   single: plot curve; attach and detach
   single: CurveFactory
   single: CurveStorageInterface

Blackchirp Plot Curve Classes
=============================

This page documents the five curve classes that form the plot-curve hierarchy
used by :cpp:class:`ZoomPanPlot` and its subclasses.  All five are defined in
``src/gui/plot/blackchirpplotcurve.h``.

- **BlackchirpPlotCurveBase** — abstract root of the hierarchy.
- **BlackchirpPlotCurve** — concrete, general-purpose point-cloud curve.
- **BCEvenSpacedCurveBase** — abstract base for uniformly-spaced data.
- **BlackchirpFTCurve** — concrete curve for Fourier-transform spectra.
- **BlackchirpFIDCurve** — concrete curve for free-induction decay waveforms.

Curve objects are created exclusively through :cpp:class:`CurveFactory`, which
injects the appropriate storage backend.  The appearance fields of every curve
(color, line style, thickness, marker, visibility, autoscale, y-axis) are
bundled into :cpp:struct:`CurveAppearanceWidget::CurveAppearance` and surfaced
for editing via :cpp:class:`CurveAppearanceWidget` embedded in the
:cpp:class:`ZoomPanPlot` context menu.

Curve hierarchy
---------------

.. code-block:: text

   QwtPlotCurve
   └── BlackchirpPlotCurveBase       (abstract)
       ├── BlackchirpPlotCurve       (concrete — arbitrary x spacing)
       └── BCEvenSpacedCurveBase     (abstract — uniform x spacing)
           ├── BlackchirpFTCurve     (concrete — Ft spectrum)
           └── BlackchirpFIDCurve    (concrete — FID waveform)

Storage backend abstraction
----------------------------

``BlackchirpPlotCurveBase`` does not inherit from ``SettingsStorage`` directly.
Instead, it owns a ``std::unique_ptr<CurveStorageInterface>`` that is injected
at construction by :cpp:class:`CurveFactory`.  Two concrete backends are
provided in ``curvefactory.h``:

- **SettingsStorageWrapper** — wraps :cpp:class:`SettingsStorage`
  (``QSettings``-backed) so that appearance settings persist across sessions
  under the curve's key.  Used for standard curves that live on persistent
  plots (FT view, tracking plots, etc.).
- **OverlayMetadataStorage** — stores appearance settings as metadata inside
  an ``OverlayBase`` object.  Used for overlay curves; settings travel with the
  overlay rather than being written to the global settings file.

The ``StorageType`` enum (``Settings`` or ``OverlayMetadata``) is detected
automatically at construction by dynamic-casting the injected storage pointer.
Callers that need to retrieve the associated overlay can call
``getOverlay()``, which returns a ``std::shared_ptr<OverlayBase>`` or
``nullptr`` for ``Settings``-backend curves.

This indirection keeps the curve classes free of knowledge about *where* their
settings live; only :cpp:class:`CurveFactory` and the overlay system need to
reason about which backend to use.

Settings keys
-------------

The following keys are stored in the curve's active ``CurveStorageInterface``
backend.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Key
     - Type
     - Description
   * - ``BC::Key::bcCurveColor``
     - QColor
     - Pen and symbol color.
   * - ``BC::Key::bcCurveCurveStyle``
     - int (QwtPlotCurve::CurveStyle)
     - Drawing mode (Lines, Dots, Sticks, etc.).
   * - ``BC::Key::bcCurveLineStyle``
     - int (Qt::PenStyle)
     - Pen dash pattern.
   * - ``BC::Key::bcCurveThickness``
     - double
     - Pen width in pixels (default 1.0).
   * - ``BC::Key::bcCurveMarker``
     - int (QwtSymbol::Style)
     - Point-marker shape.
   * - ``BC::Key::bcCurveMarkerSize``
     - int
     - Marker diameter in pixels (default 5).
   * - ``BC::Key::bcCurveAxisX``
     - int (QwtPlot::Axis)
     - X axis assignment (default xBottom).
   * - ``BC::Key::bcCurveAxisY``
     - int (QwtPlot::Axis)
     - Y axis assignment (default yLeft).
   * - ``BC::Key::bcCurveVisible``
     - bool
     - Curve visibility (default true).
   * - ``BC::Key::bcCurveAutoscale``
     - bool
     - Whether the curve participates in autoscale (default true).
   * - ``BC::Key::bcCurvePlotIndex``
     - int
     - Index of the ZoomPanPlot panel that owns this curve (default -1).

Setter family and visibility persistence
-----------------------------------------

``BlackchirpPlotCurveBase`` provides two overlapping ways to control visibility.
The ``setCurve*`` family (``setCurveVisible``, ``setCurveAutoscale``,
``setCurveAxisX``, ``setCurveAxisY``, ``setCurvePlotIndex``) writes to the
``CurveStorageInterface`` backend so the curve's state is restored across
sessions, whereas the inherited ``QwtPlotItem::setVisible()`` and the
lower-level appearance setters (``setColor``, etc.) are for transient changes
that do not need to survive a restart. See the ``setCurveVisible`` note in the
rendered API below for details.

Attaching and detaching curves
------------------------------

``QwtPlotItem::attach`` and ``QwtPlotItem::detach`` are made ``private``
inside ``BlackchirpPlotCurveBase`` via ``using``-declarations.  Calling
``curve->attach(plot)`` or ``curve->detach()`` from anywhere outside
:cpp:class:`ZoomPanPlot` (the only friend) is a hard compile error.

The supported entry points are ``ZoomPanPlot::attachCurve`` and
``ZoomPanPlot::detachCurve``, which drain the asynchronous filter
worker before mutating the plot's curve registry.  See the *Curve registry
and thread safety* section of :doc:`zoompanplot` for the full rationale.

Curves held in ``std::unique_ptr`` do not need an explicit
``detachCurve`` before destruction.  ``~BlackchirpPlotCurveBase`` calls
``ZoomPanPlot::_unregisterCurve`` (a private helper friended to this
class) which drains the worker and removes the pointer from the registry
before ``~QwtPlotItem`` performs its own detach.  This makes container
clearing patterns such as ``d_overlayCurves.clear()`` and
``unique_ptr::reset()`` safe with respect to the worker.

Downsampling filter
-------------------

``BlackchirpPlotCurveBase::filter()`` is called from a ``QtConcurrent``
worker dispatched by ``ZoomPanPlot`` whenever the x range changes.  The
worker iterates an immutable snapshot of the plot's curve registry that
was built under the plot's mutex; it never touches the live registry, the
QwtPlot item list, or any widget state.  ``filter()`` delegates to the
protected pure-virtual ``_filter(int w, const QwtScaleMap map)`` and
stores the result in the Qwt sample buffer under a mutex so that the
paint thread always reads a consistent snapshot.

Each implementation compresses the visible portion of the data to at most
:math:`2w` points using **min/max compression per pixel column**: for each pixel
column, if more than one data point maps to that column, both the minimum and
maximum y values are emitted so that vertical features (peaks, noise spikes)
are preserved at all zoom levels.

``BlackchirpPlotCurve::_filter`` iterates over the ``QVector<QPointF>``
data.  ``BCEvenSpacedCurveBase::_filter`` (sealed ``final``) computes pixel
boundaries analytically from ``xFirst()``, ``spacing()``, and the scale map,
avoiding a linear search.

``curveData()`` and ``boundingRect()`` are the two pure-virtual methods that
every concrete class must implement:

- ``curveData()`` returns the full (non-downsampled) data as
  ``QVector<QPointF>``; used for CSV export and context-menu Export XY.
- ``boundingRect()`` returns the data extent in plot coordinates; used by
  ``ZoomPanPlot::replot()`` for autoscale.  Return an invalid rect (width < 0
  or height < 0) when no data is available.

BCEvenSpacedCurveBase extension hooks
---------------------------------------

Subclasses of ``BCEvenSpacedCurveBase`` implement four protected methods to
describe their data:

- ``xFirst()`` — x coordinate of the first data point.
- ``spacing()`` — uniform x increment between consecutive points.
- ``numPoints()`` — total number of data points.
- ``yData()`` — y values as ``QVector<double>``; called once per filter pass
  and should return a detached copy so the filter loop can read without holding
  a lock.

The ``_filter`` implementation in ``BCEvenSpacedCurveBase`` is declared
``final`` to prevent accidental re-overriding in leaf classes.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: BlackchirpPlotCurveBase
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: BlackchirpPlotCurve
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: BCEvenSpacedCurveBase
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: BlackchirpFTCurve
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: BlackchirpFIDCurve
   :members:
   :protected-members:
   :undoc-members:
