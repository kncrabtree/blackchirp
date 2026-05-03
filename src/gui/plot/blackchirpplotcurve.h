#ifndef BLACKCHIRPPLOTCURVE_H
#define BLACKCHIRPPLOTCURVE_H

#include <qwt6/qwt_plot.h>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_symbol.h>
#include <qwt6/qwt_text.h>
#include <qwt6/qwt_scale_map.h>
#include <memory>

class QMutex;
class OverlayBase;
class ZoomPanPlot;

// Meta-type declarations moved to blackchirpplotcurve.cpp to avoid redefinition issues

#include <data/storage/settingsstorage.h>

// Forward declarations for new storage system
class CurveStorageInterface;
class OverlayMetadataStorage;

/// \brief Settings keys used by BlackchirpPlotCurveBase and its subclasses.
///
/// Each key is stored in the curve's CurveStorageInterface backend.
/// The top-level \c bcCurve key groups the per-curve sub-map in
/// SettingsStorage-backed curves.
namespace BC::Key {
inline constexpr QLatin1StringView bcCurve{"Curve"};                     ///< Top-level group key for curve settings.
inline constexpr QLatin1StringView bcCurveColor{"color"};                ///< QColor: pen and symbol color.
inline constexpr QLatin1StringView bcCurveCurveStyle{"curveStyle"};      ///< int (QwtPlotCurve::CurveStyle): drawing mode.
inline constexpr QLatin1StringView bcCurveLineStyle{"lineStyle"};        ///< int (Qt::PenStyle): dash pattern.
inline constexpr QLatin1StringView bcCurveThickness{"thickness"};        ///< double: pen width in pixels.
inline constexpr QLatin1StringView bcCurveMarker{"marker"};              ///< int (QwtSymbol::Style): point-marker shape.
inline constexpr QLatin1StringView bcCurveMarkerSize{"markerSize"};      ///< int: marker diameter in pixels.
inline constexpr QLatin1StringView bcCurveAxisX{"xAxis"};                ///< int (QwtPlot::Axis): x axis assignment.
inline constexpr QLatin1StringView bcCurveAxisY{"yAxis"};                ///< int (QwtPlot::Axis): y axis assignment.
inline constexpr QLatin1StringView bcCurveVisible{"visible"};            ///< bool: curve visibility.
inline constexpr QLatin1StringView bcCurveAutoscale{"autoscale"};        ///< bool: whether the curve participates in autoscale.
inline constexpr QLatin1StringView bcCurvePlotIndex{"plotIndex"};        ///< int: index of the ZoomPanPlot panel that owns this curve.
}

/// \brief Abstract base class for all Blackchirp plot curves.
///
/// BlackchirpPlotCurveBase extends QwtPlotCurve with a storage-backend
/// abstraction (CurveStorageInterface) that persists appearance settings
/// to either QSettings (via SettingsStorageWrapper) or overlay metadata
/// (via OverlayMetadataStorage). The concrete storage object is injected
/// at construction by CurveFactory, keeping the curve classes independent
/// of the persistence mechanism.
///
/// Subclasses must implement curveData(), boundingRect(), and the
/// protected _filter() template method. The base class calls _filter()
/// asynchronously (from ZoomPanPlot's filter pass) and marshals the
/// result into the Qwt sample buffer under a mutex.
class BlackchirpPlotCurveBase : public QwtPlotCurve
{
public:
    /// \brief Selects which storage backend the curve uses.
    enum class StorageType {
        Settings,        ///< Appearance settings are stored in QSettings via SettingsStorageWrapper.
        OverlayMetadata  ///< Appearance settings are stored as metadata in an OverlayBase object.
    };

    /// \brief Constructs the curve with an injected storage backend.
    ///
    /// If \p title is empty, the curve's Qwt title is set to \p key.
    /// Default appearance values are written to storage only when no
    /// existing value is found for that key.
    /// \param storage          Heap-allocated storage backend; ownership is transferred.
    /// \param key              Unique identifier for this curve within its storage namespace.
    /// \param title            Display title (shown in legends and menus).
    /// \param defaultLineStyle Initial Qt::PenStyle if not already in storage.
    /// \param defaultMarker    Initial QwtSymbol::Style if not already in storage.
    /// \param defaultStyle     Initial QwtPlotCurve::CurveStyle if not already in storage.
    BlackchirpPlotCurveBase(std::unique_ptr<CurveStorageInterface> storage,
                           const QString key,
                           const QString title=QString(""),
                           Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                           QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol,
                           QwtPlotCurve::CurveStyle defaultStyle = QwtPlotCurve::Lines);
    virtual ~BlackchirpPlotCurveBase();

    /// \brief Sets the curve color and updates both the pen and the symbol brush.
    /// \param c New color; persisted to storage.
    void setColor(const QColor c);

    /// \brief Sets the Qwt drawing style (Lines, Dots, Sticks, etc.) and persists it.
    /// \param s New QwtPlotCurve::CurveStyle value.
    void setCurveStyle(CurveStyle s);

    /// \brief Sets the pen line width and persists it.
    /// \param t Line width in pixels.
    void setLineThickness(double t);

    /// \brief Sets the pen dash pattern and persists it.
    /// \param s Qt::PenStyle value.
    void setLineStyle(Qt::PenStyle s);

    /// \brief Sets the point-marker symbol shape and persists it.
    /// \param s QwtSymbol::Style value.
    void setMarkerStyle(QwtSymbol::Style s);

    /// \brief Sets the point-marker diameter and persists it.
    /// \param s Diameter in pixels.
    void setMarkerSize(int s);

    /// \brief Sets the curve's display title (shown in legends and context menus).
    /// \param t New title string.
    void setName(const QString &t);

    /// \brief Returns the curve's display title.
    QString name() const { return title().text(); }

    /// \brief Returns the storage key passed at construction.
    QString key() const { return d_key; }

    /// \brief Returns which storage backend this curve uses.
    StorageType getStorageType() const { return d_storageType; }

    /// \brief Returns the associated OverlayBase when the storage type is OverlayMetadata, or nullptr.
    std::shared_ptr<OverlayBase> getOverlay() const;

    /// \brief Returns the full curve data as a point vector.
    ///
    /// The returned vector reflects the canonical data set before any
    /// downsampling; it is used for CSV export and bounding-rect calculations.
    virtual QVector<QPointF> curveData() const =0;

    /// \brief Sets curve visibility and persists the state to storage.
    ///
    /// \note Use QwtPlotItem::setVisible() directly if persistence is not
    /// desired (for example when temporarily hiding a curve in a view-only
    /// context). This method is intended for user-driven visibility changes
    /// on tracking and logging plots where the setting should survive a
    /// session restart.
    /// \param v \c true to show the curve.
    void setCurveVisible(bool v);

    /// \brief Enables or disables autoscale participation and persists the choice.
    ///
    /// A curve with autoscale disabled contributes to the legend but its
    /// bounding rect is excluded from ZoomPanPlot's axis range calculations.
    /// \param enabled \c true to include in autoscale.
    void setCurveAutoscale(bool enabled);

    /// \brief Sets the x axis and persists the assignment.
    /// \param a QwtPlot::Axis value (xBottom or xTop).
    void setCurveAxisX(QwtPlot::Axis a);

    /// \brief Sets the y axis and persists the assignment.
    /// \param a QwtPlot::Axis value (yLeft or yRight).
    void setCurveAxisY(QwtPlot::Axis a);

    /// \brief Sets the owning plot-panel index and persists it.
    ///
    /// The plot index identifies which ZoomPanPlot panel this curve belongs
    /// to when a view holds multiple panels (e.g. the rolling-data view).
    /// \param i Zero-based panel index; -1 means unassigned.
    void setCurvePlotIndex(int i);

    /// \brief Returns the stored plot-panel index, or -1 if not set.
    int plotIndex() const;

    /// \brief Re-applies all appearance settings from storage to the Qwt curve.
    ///
    /// Call after a storage backend is refreshed externally (e.g. when an
    /// overlay's metadata is reloaded from disk).
    void updateFromSettings();

    /// \brief Computes a downsampled sample set and stores it in the Qwt sample buffer.
    ///
    /// Called by ZoomPanPlot's concurrent filter pass.  The canvas width \p w
    /// and scale map \p map are used to limit the rendered sample count to
    /// approximately 2×\p w points via min/max compression per pixel column.
    /// \param w   Canvas width in pixels.
    /// \param map Current x-axis scale map.
    void filter(int w, const QwtScaleMap map);

private:
    /// \name Hidden QwtPlotItem attach/detach
    ///
    /// These are made private (and only ZoomPanPlot is friended below) so
    /// that callers cannot accidentally call \c curve->attach(plot) or
    /// \c curve->detach() — those bypass ZoomPanPlot's curve registry and
    /// race with the asynchronous filter pass. Use
    /// \c ZoomPanPlot::attachCurve() and \c ZoomPanPlot::detachCurve()
    /// instead.
    /// @{
    using QwtPlotItem::attach;
    using QwtPlotItem::detach;
    /// @}
    friend class ZoomPanPlot;

    std::unique_ptr<CurveStorageInterface> d_storage;
    const QString d_key;
    QMutex *p_samplesMutex;
    StorageType d_storageType;
    OverlayMetadataStorage* p_overlayMetadataStorage; // Raw pointer for type checking

    void configurePen();
    void configureSymbol();
    void configureCurveStyle();
    void setSamples(const QVector<QPointF> d);

protected:
    /// \brief Produces a downsampled point vector for the current view.
    ///
    /// Called by filter(). Implementations should return at most 2×\p w points,
    /// using min/max compression to preserve peak fidelity when multiple data
    /// points map to the same pixel column.
    /// \param w   Canvas width in pixels.
    /// \param map Current x-axis scale map used to determine pixel boundaries.
    /// \return    Downsampled point vector to pass to the Qwt sample buffer.
    virtual QVector<QPointF> _filter(int w, const QwtScaleMap map) =0;

    template<typename It>
    void increment(It i) { ++i; }

public:
    /// \brief Returns the data bounding rect in plot coordinates.
    ///
    /// Must reflect the full (non-downsampled) data range so that ZoomPanPlot
    /// can compute correct autoscale limits. Return an invalid rect
    /// (width < 0 or height < 0) when no data is available.
    virtual QRectF boundingRect() const override =0;

    // QwtPlotItem interface
public:
    void draw(QPainter *painter, const QwtScaleMap &xMap, const QwtScaleMap &yMap, const QRectF &canvasRect) const override;
};

/// \brief Concrete BlackchirpPlotCurveBase for arbitrary (x, y) point-cloud data.
///
/// BlackchirpPlotCurve stores its data as a QVector<QPointF> and supports
/// both bulk replacement (setCurveData()) and incremental appending
/// (appendPoint()). It is the general-purpose curve type used for tracking
/// plots, aux-data traces, and any other non-uniform x-spacing data.
class BlackchirpPlotCurve : public BlackchirpPlotCurveBase
{
public:
    /// \brief Constructs the curve with an injected storage backend.
    /// \sa BlackchirpPlotCurveBase::BlackchirpPlotCurveBase
    BlackchirpPlotCurve(std::unique_ptr<CurveStorageInterface> storage,
                        const QString key,
                        const QString title=QString(""),
                        Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                        QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol,
                        QwtPlotCurve::CurveStyle defaultStyle = QwtPlotCurve::Lines);
    ~BlackchirpPlotCurve();

    /// \brief Replaces the stored data and recomputes the bounding rect.
    /// \param d New point vector; the bounding rect height is computed from all y values.
    void setCurveData(const QVector<QPointF> d);

    /// \brief Replaces the stored data with a pre-computed bounding rect height.
    ///
    /// Use this overload when the caller already knows the y-range to avoid
    /// a second pass over the data.
    /// \param d   New point vector.
    /// \param min Minimum y value (bounding rect top).
    /// \param max Maximum y value (bounding rect bottom).
    void setCurveData(const QVector<QPointF> d, double min, double max);

    /// \brief Appends a single point and extends the bounding rect.
    /// \param p Point to append.
    void appendPoint(const QPointF p);

private:
    void calcBoundingRectHeight();

    QRectF d_boundingRect;
    QVector<QPointF> d_curveData;
    QMutex *p_dataMutex;

    // QwtPlotItem interface
public:
    QRectF boundingRect() const override;

    // BlackchirpPlotCurveBase interface
public:
    QVector<QPointF> curveData() const override;

protected:
    QVector<QPointF> _filter(int w, const QwtScaleMap map) override;
};

/// \brief Abstract base for evenly spaced curves (constant x increment).
///
/// BCEvenSpacedCurveBase provides a closed-form _filter() implementation
/// that avoids iterating over all data points to find pixel boundaries —
/// it computes array indices directly from \c xFirst(), \c spacing(), and
/// the scale map. Subclasses supply the three geometric hooks and the y-data
/// vector; the filter logic is sealed (\c final) and cannot be further
/// overridden.
class BCEvenSpacedCurveBase : public BlackchirpPlotCurveBase
{
public:
    /// \brief Constructs the curve with an injected storage backend.
    /// \sa BlackchirpPlotCurveBase::BlackchirpPlotCurveBase
    BCEvenSpacedCurveBase(std::unique_ptr<CurveStorageInterface> storage,
                          const QString key,
                          const QString title=QString(""),
                          Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                          QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol,
                          QwtPlotCurve::CurveStyle defaultStyle = QwtPlotCurve::Lines);
    virtual ~BCEvenSpacedCurveBase();

    /// \brief Returns the x coordinate of data point \p i.
    /// \param i Zero-based data index.
    double xVal(int i) const;

    /// \brief Returns the index of the last data point with x < \p xVal.
    /// \param xVal X coordinate to search for.
    int indexBefore(double xVal) const;

protected:
    /// \brief Returns the x coordinate of the first data point.
    virtual double xFirst() const =0;

    /// \brief Returns the uniform x spacing between consecutive data points.
    virtual double spacing() const =0;

    /// \brief Returns the total number of data points.
    virtual int numPoints() const =0;

    /// \brief Returns the y-data vector; called once per filter pass.
    ///
    /// Implementations should return a detached (copy-on-write) vector so
    /// the filter loop can read without holding a lock.
    virtual QVector<double> yData() =0;

    /// \brief Sealed min/max-compression filter exploiting uniform x spacing.
    QVector<QPointF> _filter(int w, const QwtScaleMap map) override final;
};

#include <data/analysis/ft.h>

/// \brief Curve for displaying a Fourier transform (Ft) spectrum.
///
/// BlackchirpFTCurve wraps an Ft object and exposes its frequency axis
/// (minFreqMHz / xSpacing) and y-data through the BCEvenSpacedCurveBase
/// hooks. The Ft is set atomically under a mutex so the filter thread
/// always reads a consistent snapshot.
class BlackchirpFTCurve : public BCEvenSpacedCurveBase
{
public:
    /// \brief Constructs the curve with an injected storage backend.
    /// \sa BlackchirpPlotCurveBase::BlackchirpPlotCurveBase
    BlackchirpFTCurve(std::unique_ptr<CurveStorageInterface> storage,
                      const QString key,
                      const QString title=QString(""),
                      Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                      QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol,
                      QwtPlotCurve::CurveStyle defaultStyle = QwtPlotCurve::Lines);
    ~BlackchirpFTCurve();

    /// \brief Replaces the displayed Ft spectrum.
    ///
    /// Thread-safe: the Ft is stored under the internal mutex and the next
    /// filter pass picks up the new data.
    /// \param f New Ft object to display.
    void setCurrentFt(const Ft f);

private:
    QMutex *p_mutex;
    Ft d_currentFt;

    // QwtPlotItem interface
public:
    QRectF boundingRect() const override;

    // BlackchirpPlotCurveBase interface
public:
    QVector<QPointF> curveData() const override;

    // BCEvenSpacedCurveBase interface
private:
    double xFirst() const override;
    double spacing() const override;
    int numPoints() const override;
    QVector<double> yData() override;
};

/// \brief Curve for displaying a free-induction decay (FID) waveform.
///
/// BlackchirpFIDCurve stores raw FID samples as a QVector<double> with a
/// uniform time spacing. The x axis starts at 0 and extends to
/// \c spacing * numPoints. Y bounds are supplied explicitly when
/// setCurrentFid() is called to avoid a full-vector scan.
class BlackchirpFIDCurve : public BCEvenSpacedCurveBase
{
public:
    /// \brief Constructs the curve with an injected storage backend.
    /// \sa BlackchirpPlotCurveBase::BlackchirpPlotCurveBase
    BlackchirpFIDCurve(std::unique_ptr<CurveStorageInterface> storage,
                       const QString key,
                       const QString title=QString(""),
                       Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                       QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol,
                       QwtPlotCurve::CurveStyle defaultStyle = QwtPlotCurve::Lines);
    ~BlackchirpFIDCurve();

    /// \brief Replaces the displayed FID waveform.
    ///
    /// Thread-safe: all fields are updated under the internal mutex.
    /// \param d       Y-value samples.
    /// \param spacing Uniform time step between samples (in the plot's x-axis unit).
    /// \param min     Minimum y value; used directly as the bounding rect top.
    /// \param max     Maximum y value; used directly as the bounding rect bottom.
    void setCurrentFid(const QVector<double> d, double spacing=1.0, double min=0.0, double max=0.0);

private:
    QMutex *p_mutex;
    QVector<double> d_fidData;
    double d_spacing{1.0}, d_min{0.0}, d_max{1.0};

    // QwtPlotItem interface
public:
    QRectF boundingRect() const override;

    // BlackchirpPlotCurveBase interface
public:
    QVector<QPointF> curveData() const override;

    // BCEvenSpacedCurveBase interface
protected:
    double xFirst() const override;
    double spacing() const override;
    int numPoints() const override;
    QVector<double> yData() override;
};

#endif // BLACKCHIRPPLOTCURVE_H
