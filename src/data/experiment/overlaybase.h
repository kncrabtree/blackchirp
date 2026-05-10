#ifndef OVERLAYBASE_H
#define OVERLAYBASE_H

#include <map>
#include <QString>
#include <QPointF>
#include <QVariant>
#include <data/presentation/curveappearance.h>

class OverlayMetadataStorage;

/// \brief Storage keys used by OverlayBase and OverlayStorage for per-overlay settings files.
namespace BC::Key::Overlay {
inline constexpr QLatin1StringView oLabel{"label"};                      ///< User-visible overlay label.
inline constexpr QLatin1StringView oSourceFile{"sourceFile"};            ///< Path to the original source data file.
inline constexpr QLatin1StringView oDestFile{"destFile"};                ///< Path to the overlay's storage destination file.
inline constexpr QLatin1StringView oPlotId{"plotId"};                    ///< Identifier of the plot panel this overlay belongs to.
inline constexpr QLatin1StringView oYScale{"yScale"};                    ///< Multiplicative Y-axis scaling factor.
inline constexpr QLatin1StringView oYOffset{"yOffset"};                  ///< Additive Y-axis offset.
inline constexpr QLatin1StringView oXOffset{"xOffset"};                  ///< Additive X-axis offset applied before frequency filtering.
inline constexpr QLatin1StringView oMinFreqEnabled{"minFreqEnabled"};    ///< Whether the low-frequency clip is active.
inline constexpr QLatin1StringView oMinFreqValue{"minFreqValue"};        ///< Low-frequency clip value in MHz.
inline constexpr QLatin1StringView oMaxFreqEnabled{"maxFreqEnabled"};    ///< Whether the high-frequency clip is active.
inline constexpr QLatin1StringView oMaxFreqValue{"maxFreqValue"};        ///< High-frequency clip value in MHz.
inline constexpr QLatin1StringView oEnabled{"enabled"};                  ///< Whether the overlay is visible on the plot.
inline constexpr QLatin1StringView oComment{"comment"};                  ///< Free-text annotation.
inline const QString overlaySettingsFile{"%1.settings.csv"};             ///< Settings filename template; %1 is the sanitized label.
inline const QString overlayDataFile{"%1.data.csv"};                     ///< Data filename template; %1 is the sanitized label.
}

/*!
 * \brief Abstract base class for all plot overlays.
 *
 * OverlayBase defines the common data model shared by every overlay type:
 * transformations (X/Y offset, Y scale), frequency-range clipping, visibility,
 * plot assignment, and curve appearance metadata.  Concrete subclasses
 * implement the data source (a Blackchirp experiment FT, a spectroscopic
 * catalog, or a generic XY file) by overriding the pure-virtual members.
 *
 * The type discriminator OverlayType identifies each concrete kind and drives
 * the factory in OverlayStorage.  The three values are:
 * - \c BCExperiment — an FT spectrum taken from a Blackchirp experiment file.
 * - \c Catalog — a spectroscopic line-catalog from SPCAT, XIAM, or similar.
 * - \c GenericXY — an arbitrary two-column data file (CSV, TSV, etc.).
 *
 * xyData() applies X/Y offsets and frequency-range filtering on the raw data
 * returned by the pure-virtual _xyData() and caches the result; the cache is
 * invalidated whenever an offset or frequency-limit setter is called.
 *
 * The \p preview flag suppresses disk I/O, allowing the UI to create and
 * display temporary overlays before the user commits them.
 *
 * \sa OverlayStorage, BCExpOverlay, CatalogOverlay, GenericXYOverlay
 */
class OverlayBase
{
    Q_GADGET
    friend class OverlayStorage;
    friend class OverlayMetadataStorage;
    friend class UnifiedOverlayWidget;

public:
    /*!
     * \brief Discriminator tag identifying the concrete overlay subclass.
     */
    enum OverlayType {
        BCExperiment, ///< FT spectrum from a Blackchirp experiment file.
        Catalog,      ///< Spectroscopic line catalog (SPCAT, XIAM, etc.).
        GenericXY     ///< Arbitrary two-column XY data file.
    };
    Q_ENUM(OverlayType)

    /*!
     * \brief Construct an overlay of the given type.
     * \param type The OverlayType discriminator for this instance.
     */
    OverlayBase(OverlayType type);

    /*!
     * \brief Virtual destructor.
     *
     * Required because OverlayBase is an abstract polymorphic base
     * destroyed through base-class pointers (see the std::shared_ptr
     * usage in OverlayTableModel and friends — shared_ptr's type
     * erasure happens to dispatch correctly today, but a future
     * unique_ptr<OverlayBase> or a raw `delete bp;` would silently
     * skip the derived destructor without this).
     */
    virtual ~OverlayBase() = default;

    /*!
     * \brief Return the transformed and frequency-filtered XY data, using a cached copy when valid.
     *
     * Applies d_xOffset, d_yScale, d_yOffset, and the optional frequency clip
     * limits to the raw data from _xyData().  The result is cached until any
     * transformation parameter changes.
     */
    QVector<QPointF> xyData() const;

    /// \brief Return the user-visible overlay label.
    QString getLabel() const;
    /// \brief Return the path to the original source data file.
    QString getSourceFile() const;
    /// \brief Return the path to the overlay's on-disk destination file.
    QString getDestFile() const;
    /// \brief Return the plot panel identifier this overlay is assigned to.
    QString getPlotId() const;
    /// \brief Return the free-text annotation for this overlay.
    QString getComment() const;
    /// \brief Return the multiplicative Y-axis scaling factor.
    double getYScale() const;
    /// \brief Return the additive Y-axis offset.
    double getYOffset() const;
    /// \brief Return the additive X-axis offset.
    double getXOffset() const;
    /// \brief Return \c true if the low-frequency clip is active.
    bool getMinFreqEnabled() const;
    /// \brief Return the low-frequency clip value in MHz.
    double getMinFreqValue() const;
    /// \brief Return \c true if the high-frequency clip is active.
    bool getMaxFreqEnabled() const;
    /// \brief Return the high-frequency clip value in MHz.
    double getMaxFreqValue() const;
    /// \brief Return \c true if the overlay is visible on the plot.
    bool getEnabled() const;

    /*!
     * \brief Return the maximum absolute raw Y value (before scale/offset are applied).
     *
     * Triggers a cache refresh if the cache is stale.
     */
    double yMax() const;

    /// \brief Return the OverlayType discriminator for this instance.
    OverlayType type() const { return d_type; }

    /// \brief Return the most recent error description, or an empty string if none.
    QString errorString() const { return d_errorString; }

    /// \brief Return \c true if any property has been changed since the last save().
    bool isModified() const { return d_modified; }

    /// \brief Return \c true if this overlay is in preview (non-persistent) mode.
    bool isPreview() const { return d_preview; }

    /*!
     * \brief Return the curve metadata value for \p key, or an invalid QVariant if absent.
     * \param key Metadata key (e.g. a CurveKey constant).
     */
    QVariant getCurveMetadata(const QString &key) const;

    /*!
     * \brief Set a curve metadata value and mark the overlay modified.
     * \param key   Metadata key.
     * \param value Value to store.
     */
    void setCurveMetadata(const QString &key, const QVariant &value);

    /// \brief Set the user-visible label and mark the overlay modified.
    void setLabel(const QString &newlabel);
    /// \brief Set the source data file path and mark the overlay modified.
    void setSourceFile(const QString &newsourceFile);
    /// \brief Set the on-disk destination file path and mark the overlay modified.
    void setDestFile(const QString &newdestFile);
    /// \brief Set the plot panel identifier and mark the overlay modified.
    void setPlotId(const QString &newplotId);
    /// \brief Set the free-text annotation and mark the overlay modified.
    void setComment(const QString &newcomment);
    /// \brief Set the multiplicative Y-axis scale and mark the overlay modified.
    void setYScale(double newyScale);
    /// \brief Set the additive Y-axis offset and mark the overlay modified.
    void setYOffset(double newyOffset);

    /*!
     * \brief Set the additive X-axis offset, invalidate the cache, and mark modified.
     * \param newxOffset New X offset value.
     */
    void setXOffset(double newxOffset);

    /*!
     * \brief Set the low-frequency clip, invalidate the cache, and mark modified.
     * \param enabled \c true to activate the clip.
     * \param value   Clip value in MHz.
     */
    void setMinFreqLimit(bool enabled, double value);

    /*!
     * \brief Set the high-frequency clip, invalidate the cache, and mark modified.
     * \param enabled \c true to activate the clip.
     * \param value   Clip value in MHz.
     */
    void setMaxFreqLimit(bool enabled, double value);

    /*!
     * \brief Set overlay visibility, update curve metadata, and mark modified.
     * \param enabled \c true to make the overlay visible on the plot.
     */
    void setEnabled(bool enabled);

    /*!
     * \brief Set preview mode; marks the overlay modified when transitioning out of preview.
     *
     * In preview mode, save() is a no-op and no data is written to disk.
     * Clearing the preview flag (setting it to \c false) marks the overlay
     * as modified so that the next save() writes it to disk.
     * \param preview \c true to enable preview mode.
     */
    void setPreview(bool preview);

    /*!
     * \brief Write the overlay to its destination file and clear the modified flag.
     *
     * A no-op when preview mode is active.
     */
    void save();



protected:
    /*!
     * \brief Set or clear the modified flag.
     * \param modified New modified state; defaults to \c true.
     */
    void setModified(bool modified = true) { d_modified = modified; }

    /*!
     * \brief Load data from the overlay's destination file.
     *
     * Called by OverlayStorage when reading an existing overlay from disk.
     * Implementations should populate the internal data structures from
     * the file at d_destFile.
     */
    virtual void readFromDest() =0;

    /*!
     * \brief Write data to the overlay's destination file.
     *
     * Called by save() when the overlay is not in preview mode.
     * Implementations should serialize their data to d_destFile.
     */
    virtual void writeToDest() =0;

    /*!
     * \brief Store subclass-specific metadata into the settings map.
     *
     * Called during save()/storeMetadata() to allow each subclass to add its
     * own key-value pairs to the shared settings map before it is written to
     * the settings CSV.
     * \param m The metadata map to populate.
     */
    virtual void _storeMetadata(std::map<QString,QVariant,std::less<>> &m) =0;

    /*!
     * \brief Restore subclass-specific metadata from the settings map.
     *
     * Called during OverlayStorage load to allow each subclass to extract its
     * own key-value pairs from the shared settings map.
     * \param m The metadata map read from the settings CSV.
     */
    virtual void _retrieveMetadata(const std::map<QString,QVariant,std::less<>> &m) =0;

    /*!
     * \brief Invalidate the filtered XY data cache.
     *
     * Must be called by any setter that changes a parameter affecting
     * the output of xyData() (offsets, frequency limits).
     */
    void invalidateCache();

    QString d_errorString; ///< Most recent error description; empty when no error.

private:
    /*!
     * \brief Return the raw XY data for this overlay (before offset/filter transforms).
     *
     * Must be implemented by each concrete subclass to return the underlying
     * data points.  The base-class xyData() applies transformations on top
     * of this result.
     */
    virtual QVector<QPointF> _xyData() const = 0;

    OverlayType d_type;
    QString d_label{""}, d_sourceFile{""}, d_destFile{""}, d_plotId{""}, d_comment{""};
    double d_yScale{1.0}, d_yOffset{0.0}, d_xOffset{0.0};

    // Frequency range filtering
    bool d_minFreqEnabled{false}, d_maxFreqEnabled{false};
    double d_minFreqValue{0.0}, d_maxFreqValue{1000.0};

    // Overlay visibility control
    bool d_enabled{true};

    // Preview mode control (prevents disk writing)
    bool d_preview{false};

    bool d_modified{false};

    // Curve metadata storage for direct access by OverlayMetadataStorage
    std::map<QString, QVariant, std::less<>> d_curveMetadata;

    // Caching for filtered xyData
    mutable QVector<QPointF> d_cachedFilteredData;
    mutable bool d_cacheValid{false};
    mutable double d_cachedYMax{0.0};


    void storeMetadata(std::map<QString,QVariant,std::less<>> &m);
    void retrieveMetadata(const std::map<QString,QVariant,std::less<>> &m);
};

#endif // OVERLAYBASE_H
