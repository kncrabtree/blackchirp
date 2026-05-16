#ifndef ZOOMPANPLOT_H
#define ZOOMPANPLOT_H

#include <QFutureWatcher>
#include <QList>

#include <qwt6/qwt_plot.h>
#include <qwt6/qwt_symbol.h>
#include <qwt6/qwt_plot_grid.h>
#include <qwt6/qwt_scale_engine.h>

#include <data/storage/settingsstorage.h>
#include <gui/plot/customzoomer.h>
#include <gui/plot/blackchirpplotcurve.h>

class QMenu;
class CustomTracker;
class QMutex;


/// \brief Settings keys used by ZoomPanPlot and its per-axis configuration.
///
/// These keys are stored under the plot's SettingsStorage name so each plot
/// instance maintains independent persistent settings.
namespace BC::Key {
inline constexpr QLatin1StringView axes{"axes"};           ///< Array key holding per-axis sub-maps.
inline constexpr QLatin1StringView bottom{"Bottom"};       ///< Name token for the xBottom axis.
inline constexpr QLatin1StringView top{"Top"};             ///< Name token for the xTop axis.
inline constexpr QLatin1StringView left{"Left"};           ///< Name token for the yLeft axis.
inline constexpr QLatin1StringView right{"Right"};         ///< Name token for the yRight axis.
inline constexpr QLatin1StringView kzCenter{"keyZoomYCenter"};         ///< Keyboard Y-zoom anchors to plot center when true.
inline constexpr QLatin1StringView zoomFactor{"zoomFactor"};           ///< Fractional zoom step per wheel notch (per axis).
inline constexpr QLatin1StringView trackerDecimals{"trackerDecimals"}; ///< Decimal places shown in the cursor tracker (per axis).
inline constexpr QLatin1StringView trackerScientific{"trackerScientific"}; ///< Cursor tracker uses scientific notation when true (per axis).
inline constexpr QLatin1StringView trackerEn{"trackerEnabled"};        ///< Whether the cursor tracker is displayed.
inline constexpr QLatin1StringView majorGridColor{"majorGridColor"};   ///< Color of major grid lines.
inline constexpr QLatin1StringView majorGridStyle{"majorGridStyle"};   ///< Qt::PenStyle of major grid lines.
inline constexpr QLatin1StringView minorGridColor{"minorGridColor"};   ///< Color of minor grid lines.
inline constexpr QLatin1StringView minorGridStyle{"minorGridStyle"};   ///< Qt::PenStyle of minor grid lines.
}

/// \brief QwtPlot subclass providing interactive zoom, pan, and curve management.
///
/// ZoomPanPlot wraps QwtPlot with mouse and keyboard navigation, autoscaling,
/// per-axis configuration, and a curve-appearance context menu. It also
/// inherits SettingsStorage so all interactive preferences (zoom factors,
/// tracker options, grid style) survive across sessions. Subclasses override
/// the protected hook layer to add data filtering, custom context-menu entries,
/// and axis-limit policies appropriate for their data type.
class ZoomPanPlot : public QwtPlot, public SettingsStorage
{
    Q_OBJECT
public:
    /// \brief Constructs a ZoomPanPlot with the given settings storage name.
    /// \param name Unique name used as the SettingsStorage key; appears in
    ///        context menus and exported file names.
    /// \param parent Optional Qt parent widget.
    explicit ZoomPanPlot(const QString &name, QWidget *parent = nullptr);

    /// \brief Destructor; waits for any in-flight data-filter operation to finish.
    virtual ~ZoomPanPlot();

    /// \brief Returns \c true when every axis has autoscale enabled.
    bool isAutoScale();

    /// \brief Detaches all plot items and resets all axes to autoscale.
    ///
    /// Items are detached without being deleted (\c autoDelete=false), so
    /// subclasses that own their items via \c std::unique_ptr (or any other
    /// external owner) are safe to call this. Curves are also removed from
    /// the internal registry so the next filter pass sees an empty set.
    void resetPlot();

    /// \brief Switches the plot into spectrogram mode.
    ///
    /// In spectrogram mode the yRight and xTop axes are excluded from
    /// mouse-wheel zoom and drag-pan interactions so the color-axis range
    /// is managed separately.
    /// \param b \c true to enable spectrogram mode (default), \c false to disable.
    void setSpectrogramMode(bool b = true);

    /// \brief Sets the x-axis scale divisions directly and disables autoscale on both x axes.
    /// \param bottom Scale division for the xBottom axis.
    /// \param top    Scale division for the xTop axis.
    void setXRanges(const QwtScaleDiv &bottom, const QwtScaleDiv &top);

    /// \brief Sets the maximum plot-index value used by the "Change plot" context-menu action.
    /// \param i Maximum zero-based plot index; set to 0 to hide the action.
    void setMaxIndex(int i){ d_maxIndex = i; }

    /// \brief Sets the plot title text.
    /// \param text Title string.
    void setPlotTitle(const QString &text);

    /// \brief Sets the title text for a single axis.
    /// \param a    Axis identifier (e.g. QwtPlot::xBottom).
    /// \param text Title string.
    void setPlotAxisTitle(QwtPlot::Axis a, const QString &text);

    /// \brief Attaches \p curve to this plot and registers it for the filter pass.
    ///
    /// This is the only supported way to attach a BlackchirpPlotCurveBase
    /// to a ZoomPanPlot. \c QwtPlotItem::attach() is hidden in
    /// BlackchirpPlotCurveBase to prevent direct attach/detach calls that
    /// would race with the asynchronous filter pass driven by replot().
    ///
    /// Blocks until any in-flight filter pass completes, then attaches the
    /// curve and adds it to the internal registry under \c p_mutex. Safe to
    /// call multiple times with the same curve (subsequent calls are no-ops
    /// after the registry insert is deduplicated).
    /// \param curve Curve to attach; must not be null.
    void attachCurve(BlackchirpPlotCurveBase *curve);

    /// \brief Detaches \p curve from this plot and removes it from the registry.
    ///
    /// Blocks until any in-flight filter pass completes before detaching, so
    /// the worker cannot dereference the curve after it is removed. Safe to
    /// call on a curve that is not currently attached.
    /// \param curve Curve to detach; must not be null.
    void detachCurve(BlackchirpPlotCurveBase *curve);

    const QString d_name; ///< Unique plot name used as the SettingsStorage key.

public slots:
    /// \brief Enables autoscale on all axes and triggers a replot.
    void autoScale();

    /// \brief Pins the autoscale range of an axis to [min, max] without disabling autoscale.
    ///
    /// When \c overrideAutoScaleRange is set, autoscale still operates but
    /// cannot expand beyond the specified range. This is useful for axes
    /// whose data range is constrained by the experiment configuration.
    /// \param a   Axis to constrain.
    /// \param min Lower bound of the override range.
    /// \param max Upper bound of the override range.
    void overrideAxisAutoScaleRange(QwtPlot::Axis a, double min, double max);

    /// \brief Clears an axis autoscale-range override, allowing the data bounding rect to drive scaling.
    /// \param a Axis whose override should be removed.
    void clearAxisAutoScaleOverride(QwtPlot::Axis a);

    /// \brief Recomputes axis scales and triggers a Qwt replot; starts an async data-filter pass when the x range changes.
    virtual void replot();

    /// \brief Stores a new wheel-zoom step factor for the given axis.
    /// \param a Axis identifier.
    /// \param v Fractional zoom step per wheel notch (e.g. 0.1 = 10 % per step).
    void setZoomFactor(QwtPlot::Axis a, double v);

    /// \brief Controls whether keyboard Y-zoom is anchored to the plot center.
    ///
    /// When disabled (default), vertical keyboard zoom is symmetric about y = 0.
    /// When enabled, it is symmetric about the current plot-canvas center.
    /// \param en \c true to anchor at center.
    void setKeyZoomYCenter(bool en);

    /// \brief Shows or hides the cursor-position tracker overlay.
    /// \param en \c true to enable the tracker.
    void setTrackerEnabled(bool en);

    /// \brief Sets the number of decimal places shown by the cursor tracker for one axis.
    /// \param a   Axis identifier.
    /// \param dec Number of decimal places (0–9).
    void setTrackerDecimals(QwtPlot::Axis a, int dec);

    /// \brief Enables or disables scientific notation in the cursor tracker for one axis.
    /// \param a   Axis identifier.
    /// \param sci \c true to use scientific notation.
    void setTrackerScientific(QwtPlot::Axis a, bool sci);

    /// \brief Opens a save-file dialog and exports the curve's XY data as CSV.
    /// \param curve Curve whose data is exported; the curve's name is used as
    ///        the default filename.
    void exportCurve(BlackchirpPlotCurveBase *curve);

    /// \brief Opens a color-picker dialog and applies the chosen color to \p curve.
    ///
    /// Emits curveMetadataChanged() and triggers a replot on success.
    /// \param curve Curve to recolor.
    void setCurveColor(BlackchirpPlotCurveBase* curve);

    /// \brief Sets the Qwt curve drawing style for \p curve.
    /// \param curve Target curve.
    /// \param s     New curve style (e.g. \c QwtPlotCurve::Lines).
    void setCurveStyle(BlackchirpPlotCurveBase* curve, QwtPlotCurve::CurveStyle s);

    /// \brief Sets the pen line thickness for \p curve.
    /// \param curve Target curve.
    /// \param t     Line width in pixels.
    void setCurveLineThickness(BlackchirpPlotCurveBase* curve, double t);

    /// \brief Sets the pen dash pattern for \p curve.
    /// \param curve Target curve.
    /// \param s     Qt::PenStyle dash pattern.
    void setCurveLineStyle(BlackchirpPlotCurveBase* curve, Qt::PenStyle s);

    /// \brief Sets the point-marker symbol style for \p curve.
    /// \param curve Target curve.
    /// \param s     QwtSymbol::Style marker type.
    void setCurveMarker(BlackchirpPlotCurveBase* curve, QwtSymbol::Style s);

    /// \brief Sets the pixel size of the point-marker symbol for \p curve.
    /// \param curve Target curve.
    /// \param s     Marker diameter in pixels.
    void setCurveMarkerSize(BlackchirpPlotCurveBase* curve, int s);

    /// \brief Sets visibility for \p curve, persisting the choice to storage.
    /// \param curve Target curve.
    /// \param v     \c true to show the curve.
    void setCurveVisible(BlackchirpPlotCurveBase* curve, bool v);

    /// \brief Enables or disables autoscale participation for \p curve.
    ///
    /// A curve with autoscale disabled is still drawn but its bounding rect
    /// does not influence axis range calculations.
    /// \param curve   Target curve.
    /// \param enabled \c true to include in autoscale.
    void setCurveAutoscale(BlackchirpPlotCurveBase* curve, bool enabled);

    /// \brief Moves \p curve to a different y axis.
    /// \param curve Target curve.
    /// \param a     Destination y axis (QwtPlot::yLeft or QwtPlot::yRight).
    void setCurveAxisY(BlackchirpPlotCurveBase* curve, QwtPlot::Axis a);

    /// \brief Reads the major-grid color and style from settings and applies them to the grid pen.
    void configureGridMajorPen();

    /// \brief Reads the minor-grid color and style from settings and applies them to the grid pen.
    void configureGridMinorPen();

signals:
    /// \brief Emitted when the user begins dragging with the middle mouse button.
    void panningStarted();

    /// \brief Emitted when the user releases the middle mouse button after panning.
    void panningFinished();

    /// \brief Emitted on a right-click on the plot canvas before the context menu appears.
    /// \param ev The originating mouse event.
    void plotRightClicked(QMouseEvent *ev);

    /// \brief Emitted when the user selects "Change plot" for a BlackchirpPlotCurve.
    /// \param curve Curve to move.
    /// \param index Zero-based target plot index.
    void curveMoveRequested(BlackchirpPlotCurve *curve, int index);

    /// \brief Emitted after any slot in the curve-management family modifies a curve's appearance or settings.
    /// \param curve The modified curve.
    void curveMetadataChanged(BlackchirpPlotCurveBase* curve);

protected:
    int d_maxIndex;          ///< Maximum valid plot index; drives the "Change plot" submenu entry count.
    QwtPlotGrid *p_grid;     ///< Grid item attached to the plot.
    CustomTracker *p_tracker;///< Cursor-position tracker overlay.
    CustomZoomer *p_zoomerLB;///< Rubber-band zoomer for the xBottom/yLeft axis pair.
    CustomZoomer *p_zoomerRT;///< Rubber-band zoomer for the xTop/yRight axis pair.

    /// \brief Per-axis persistent configuration.
    ///
    /// One AxisConfig is held for each of the four QwtPlot axes. The
    /// \c autoScale flag drives whether the axis follows the data bounding
    /// rect on each replot. \c override suppresses the automatic
    /// enable/disable of an axis based on whether any items are attached
    /// to it. \c overrideAutoScaleRange, when true, constrains autoscaling
    /// to the rectangle stored in \c overrideRect rather than the data rect.
    struct AxisConfig {
        int index{-1};                          ///< Position index into the BC::Key::axes settings array.
        bool autoScale {true};                  ///< When true, the axis range tracks attached item bounding rects.
        bool override {false};                  ///< When true, the axis visible state is not controlled automatically.
        bool overrideAutoScaleRange {false};    ///< When true, autoscale is clamped to \c overrideRect.
        QRectF overrideRect{1.0,1.0,-2.0,-2.0};///< Range constraint applied when \c overrideAutoScaleRange is true.
        QRectF boundingRect{1.0,1.0,-2.0,-2.0};///< Union of all attached items' bounding rects (updated each replot).
        double zoomFactor {0.1};                ///< Fractional zoom step per mouse-wheel notch.
        QString name;                           ///< Human-readable axis name used in context-menu labels.

        explicit AxisConfig(int i, const QString n) : index(i), name(n) {}
        AxisConfig() {}
    };

    /// \brief Top-level plot interaction state.
    struct PlotConfig {
        std::map<Axis,AxisConfig> axisMap;      ///< Per-axis configuration keyed by QwtPlot::Axis.
        bool xDirty{false};                     ///< True when the x range changed and a new data-filter pass is needed.
        bool panning{false};                    ///< True while the user is drag-panning.
        bool spectrogramMode{false};            ///< When true, yRight and xTop are excluded from zoom and pan.
        QPoint panClickPos;                     ///< Canvas position where the current pan gesture started.
        bool keyZoomYCenter{false};             ///< When true, keyboard Y-zoom centers on the canvas midpoint.
    };

    /// \brief Locks the named axis into a fixed visible state, overriding automatic show/hide.
    /// \param axis     Axis to lock.
    /// \param override \c true to lock (default), \c false to restore automatic control.
    void setAxisOverride(QwtPlot::Axis axis, bool override = true);

    /// \brief Blocks until any concurrent data-filter operation completes.
    void waitForFilterComplete();

    /// \brief Handles canvas resize events; updates axis fonts and marks x as dirty.
    virtual void resizeEvent(QResizeEvent *ev);

    /// \brief Routes canvas events to the zoom, pan, and keyboard handlers.
    ///
    /// Subclasses that override this method should call the base implementation
    /// for events they do not consume, or navigation behavior will be lost.
    virtual bool eventFilter(QObject *obj, QEvent *ev);

    /// \brief Translates the axis scales based on the delta from the last pan position.
    /// \param me Mouse move event carrying the current cursor position.
    virtual void pan(QMouseEvent *me);

    /// \brief Pans both x axes by a fraction of their current range.
    /// \param factor Signed fraction of the current range to shift
    ///        (positive = right/up, negative = left/down).
    virtual void panH(double factor);

    /// \brief Pans both y axes by a fraction of their current range.
    /// \param factor Signed fraction of the current range to shift.
    virtual void panV(double factor);

    /// \brief Zooms all applicable axes in response to a wheel event.
    ///
    /// The modifier-key contract for wheel zoom is:
    /// - **No modifier** — zoom all axes around the mouse pointer.
    /// - **Ctrl** — lock both y axes (zoom x only).
    /// - **Shift** — lock horizontal axes (zoom y only, symmetric about center).
    /// - **Meta** — lock yRight (zoom xBottom, xTop, and yLeft).
    /// - **Alt** — lock both y axes and horizontal axes simultaneously
    ///   (effectively locks all; useful when Qt remaps Alt+wheel to x scroll).
    ///
    /// Each wheel notch zooms by the per-axis \c zoomFactor fraction.
    /// \param we Wheel event from the canvas.
    virtual void zoom(QWheelEvent *we);

    /// \brief Zooms to a rectangle selected by the rubber-band zoomer.
    /// \param rect  Target rectangle in plot coordinates.
    /// \param xAx   X axis of the zoomer pair.
    /// \param yAx   Y axis of the zoomer pair.
    virtual void zoom(const QRectF &rect, Axis xAx, Axis yAx);

    /// \brief Zooms a single axis by a scale factor around the current center.
    ///
    /// For y axes, when \c keyZoomYCenter is false the zoom is symmetric
    /// about y = 0 (the default, suited to FID/FT plots); when true it
    /// is symmetric about the current canvas center.
    /// \param ax     Axis to zoom.
    /// \param factor Values > 1 zoom out; values < 1 zoom in.
    virtual void zoom(Axis ax, double factor);

    /// \brief Returns the maximum allowed axis extent for a pair of axes.
    ///
    /// The limit rect is the union of the overrideRect (if active) or the
    /// data bounding rect for each of the two axes. Zoom and pan operations
    /// use this to prevent the view from moving outside the data range.
    /// \param xAx X axis of the pair.
    /// \param yAx Y axis of the pair.
    virtual QRectF getLimitRect(Axis xAx, Axis yAx) const;

    /// \brief Creates and displays the context menu at the position of \p me.
    ///
    /// The default implementation delegates to contextMenu() and calls
    /// \c QMenu::popup(). Subclasses that want to intercept right-click
    /// without replacing the menu may connect to plotRightClicked() instead.
    /// \param me Right-click mouse event.
    virtual void buildContextMenu(QMouseEvent *me);

    /// \brief Constructs and returns the right-click context menu.
    ///
    /// The returned menu includes: Autoscale, Zoom Settings (per-axis zoom
    /// factors and keyboard Y-center toggle), Tracker (enable/disable,
    /// decimals, scientific notation), Grid (major/minor pen color and style),
    /// and Curves (per-curve appearance widget, Export XY, and Change Plot).
    /// Subclasses may override to extend or replace the menu; the returned
    /// QMenu is owned by the caller (has \c Qt::WA_DeleteOnClose set).
    virtual QMenu *contextMenu();

    /// \brief Builds a menu containing the full curve-appearance editor for \p curve.
    ///
    /// The returned QMenu holds a single QWidgetAction wrapping a
    /// CurveAppearanceWidget wired to the live curve-update slots
    /// (color, style, thickness, marker, visibility, autoscale, y axis)
    /// and to the appearance-preset save/delete dialogs. It is the
    /// shared builder used both by the per-curve right-click submenu and
    /// by discoverability shortcuts that pop the editor elsewhere
    /// (e.g. a toolbar button). The caller owns the returned menu and
    /// decides its lifetime (parent ownership for an embedded submenu,
    /// or \c Qt::WA_DeleteOnClose for a standalone popup).
    /// \param curve  Curve whose appearance the widget edits; must not be null.
    /// \param parent Parent object for the returned menu.
    QMenu *buildCurveAppearanceMenu(BlackchirpPlotCurveBase *curve, QWidget *parent);

private:
    PlotConfig d_config;
    QFutureWatcher<void> *p_watcher;
    bool d_busy{false};
    QMutex *p_mutex;
    QList<BlackchirpPlotCurveBase*> d_curveRegistry; ///< Curves the worker is allowed to filter; guarded by \c p_mutex.

    /// \brief Removes \p curve from the registry; called by ~BlackchirpPlotCurveBase.
    ///
    /// Drains any in-flight filter pass first so the worker cannot hold a
    /// stale snapshot pointer to a curve that is mid-destruction.
    /// \param curve Curve being destroyed; may or may not be in the registry.
    void _unregisterCurve(BlackchirpPlotCurveBase *curve);
    friend class BlackchirpPlotCurveBase;

    /// \brief Snapshots the registry and launches the asynchronous filter worker.
    ///
    /// Builds a list of (curve, x-axis canvasMap) pairs under \c p_mutex,
    /// captures the canvas width, and dispatches the worker via
    /// QtConcurrent::run. The lambda owns its snapshot by value, so the
    /// worker never touches the live curve registry, the QwtPlot item list,
    /// or any widget state. Sets \c d_busy = true.
    void _kickoffFilterPass();

    /// \brief Recomputes per-axis bounding rects from the current registry.
    ///
    /// Runs on the UI thread (called from the QFutureWatcher::finished slot).
    /// Mirrors the bounding-rect union logic in replot(), but limited to the
    /// curve registry — markers and other items are handled by replot().
    /// Caller must hold \c p_mutex.
    void _recomputeBoundingRects();

    // QWidget interface
public:
    virtual QSize sizeHint() const;
    virtual QSize minimumSizeHint() const;

    // QWidget interface
protected:
    virtual void showEvent(QShowEvent *event);
};

#endif // ZOOMPANPLOT_H
