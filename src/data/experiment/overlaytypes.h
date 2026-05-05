#ifndef OVERLAYTYPES_H
#define OVERLAYTYPES_H

#include <data/experiment/overlaybase.h>
#include <data/analysis/ft.h>
#include <data/experiment/catalogdata.h>
#include <functional>

/// \brief Additional metadata keys used by the concrete OverlayBase subclasses.
namespace BC::Key::Overlay {
// FT metadata keys
inline constexpr QLatin1StringView ftYMin{"ftYMin"};           ///< Minimum FT Y value (pre-scale).
inline constexpr QLatin1StringView ftYMax{"ftYMax"};           ///< Maximum FT Y value (pre-scale).
inline constexpr QLatin1StringView ftX0MHz{"ftX0MHz"};         ///< FT frequency axis origin in MHz.
inline constexpr QLatin1StringView ftSpacingMHz{"ftSpacingMHz"};///< FT frequency bin spacing in MHz.
inline constexpr QLatin1StringView ftLoFreqMHz{"ftLoFreqMHz"}; ///< LO frequency associated with the FT in MHz.
inline constexpr QLatin1StringView ftShots{"ftShots"};         ///< Number of shots accumulated in the FT.

/// \brief Sub-namespace for metadata keys specific to CatalogOverlay.
namespace Catalog {
    inline constexpr QLatin1StringView sourceProgram{"catalogSourceProgram"};             ///< Name of the catalog source program (SPCAT, XIAM, etc.).
    inline constexpr QLatin1StringView moleculeName{"catalogMoleculeName"};               ///< Molecule or species name.
    inline constexpr QLatin1StringView convolutionEnabled{"catalogConvolutionEnabled"};   ///< Whether lineshape convolution is active.
    inline constexpr QLatin1StringView lineshapeType{"catalogLineshapeType"};             ///< Lineshape type (Lorentzian or Gaussian).
    inline constexpr QLatin1StringView linewidthKHz{"catalogLinewidthKHz"};               ///< Convolution linewidth FWHM in kHz.
    inline constexpr QLatin1StringView convolutionMinFreq{"catalogConvolutionMinFreq"};   ///< Lower bound of the convolution frequency range in MHz.
    inline constexpr QLatin1StringView convolutionMaxFreq{"catalogConvolutionMaxFreq"};   ///< Upper bound of the convolution frequency range in MHz.
    inline constexpr QLatin1StringView numConvolutionPoints{"catalogNumConvolutionPoints"}; ///< Number of points in the convolution grid.
    inline constexpr QLatin1StringView transitionCount{"catalogTransitionCount"};         ///< Number of transitions loaded from the catalog.
    inline constexpr QLatin1StringView frequencyRange{"catalogFrequencyRange"};           ///< Frequency range of the catalog data in MHz.
    inline constexpr QLatin1StringView filterMinFreq{"catalogFilterMinFreq"};             ///< Lower bound of the display filter range in MHz.
    inline constexpr QLatin1StringView filterMaxFreq{"catalogFilterMaxFreq"};             ///< Upper bound of the display filter range in MHz.
}

/// \brief Sub-namespace for metadata keys specific to GenericXYOverlay.
namespace GenericXY {
    inline constexpr QLatin1StringView delimiter{"genericXYDelimiter"};       ///< Column delimiter character or type string.
    inline constexpr QLatin1StringView headerLines{"genericXYHeaderLines"};   ///< Number of header lines to skip when parsing.
    inline constexpr QLatin1StringView xColumn{"genericXYXColumn"};           ///< Zero-based index of the X data column.
    inline constexpr QLatin1StringView yColumn{"genericXYYColumn"};           ///< Zero-based index of the Y data column.
    inline constexpr QLatin1StringView columnNames{"genericXYColumnNames"};   ///< Comma-separated column name list.
    inline constexpr QLatin1StringView dataPoints{"genericXYDataPoints"};     ///< Number of data points loaded.
    inline constexpr QLatin1StringView xMin{"genericXYXMin"};                 ///< Minimum X value in the loaded data.
    inline constexpr QLatin1StringView xMax{"genericXYXMax"};                 ///< Maximum X value in the loaded data.
    inline constexpr QLatin1StringView yMin{"genericXYYMin"};                 ///< Minimum Y value in the loaded data.
    inline constexpr QLatin1StringView yMax{"genericXYYMax"};                 ///< Maximum Y value in the loaded data.
    inline constexpr QLatin1StringView filterMinX{"genericXYFilterMinX"};     ///< Lower bound of the display filter range.
    inline constexpr QLatin1StringView filterMaxX{"genericXYFilterMaxX"};     ///< Upper bound of the display filter range.
}
}

/*!
 * \brief OverlayBase subclass for FT spectra from Blackchirp experiment files.
 *
 * BCExpOverlay stores an Ft object that was produced by FtWorker and
 * serializes it to the overlay destination file.  The raw XY data is
 * derived from the frequency axis and amplitude vector of the stored Ft.
 *
 * \sa OverlayBase, Ft
 */
class BCExpOverlay : public OverlayBase
{
public:
    /// \brief Construct a BCExperiment-type overlay.
    BCExpOverlay();

    // OverlayBase interface
private:
    /// \brief Return XY data derived from the stored Ft object.
    QVector<QPointF> _xyData() const override;

public:
    /*!
     * \brief Set the Ft data for this overlay.
     * \param ftData Ft object to store.
     */
    void setFtData(const Ft &ftData);

    /*!
     * \brief Return the stored Ft object.
     */
    Ft getFtData() const;

protected:
    /// \brief Load Ft data from the destination file.
    void readFromDest() override;
    /// \brief Write Ft data to the destination file.
    void writeToDest() override;
    /// \brief Store FT-specific metadata fields into the settings map.
    void _storeMetadata(std::map<QString, QVariant, std::less<>> &m) override;
    /// \brief Restore FT-specific metadata fields from the settings map.
    void _retrieveMetadata(const std::map<QString, QVariant, std::less<>> &m) override;


private:
    Ft d_ft; ///< The stored FT spectrum.

};

/*!
 * \brief OverlayBase subclass for spectroscopic line catalogs with optional lineshape convolution.
 *
 * CatalogOverlay loads transition data from a catalog file produced by SPCAT,
 * XIAM, or a compatible program.  When convolution is enabled the raw stick
 * spectrum is convolved with a Lorentzian or Gaussian lineshape on a
 * user-defined frequency grid to produce a simulated absorption profile that
 * can be compared directly with a measured FT spectrum.
 *
 * Convolution is computationally intensive; the class provides a
 * background-operation cache (Invalid → Pending → Valid) and a
 * ProgressCallback mechanism so callers can monitor and cancel long runs.
 *
 * \sa OverlayBase, CatalogData
 */
class CatalogOverlay : public OverlayBase
{
    Q_GADGET
public:
    /*!
     * \brief Lineshape function used when convolving the stick spectrum.
     */
    enum LineshapeType {
        Lorentzian, ///< Lorentzian (Cauchy) profile.
        Gaussian    ///< Gaussian profile.
    };
    Q_ENUM(LineshapeType)

    /// \brief Construct a Catalog-type overlay.
    CatalogOverlay();

    /*!
     * \brief Return the loaded catalog data.
     */
    CatalogData catalogData() const;

    /*!
     * \brief Replace the catalog data and mark the overlay modified.
     * \param data New catalog data.
     */
    void setCatalogData(const CatalogData &data);

    /*!
     * \brief Return \c true if lineshape convolution is active.
     */
    bool convolutionEnabled() const;

    /*!
     * \brief Enable or disable lineshape convolution and mark modified.
     * \param enabled \c true to enable convolution.
     */
    void setConvolutionEnabled(bool enabled);

    /*!
     * \brief Return the active lineshape type.
     */
    LineshapeType lineshapeType() const;

    /*!
     * \brief Set the lineshape type and mark modified.
     * \param type Lorentzian or Gaussian.
     */
    void setLineshapeType(LineshapeType type);

    /*!
     * \brief Return the convolution linewidth FWHM in kHz.
     */
    double linewidth() const;

    /*!
     * \brief Set the convolution linewidth FWHM in kHz and mark modified.
     * \param width FWHM in kHz.
     */
    void setLinewidth(double width);

    /// \brief Return the lower bound of the convolution frequency range in MHz.
    double convolutionMinFreq() const;
    /// \brief Return the upper bound of the convolution frequency range in MHz.
    double convolutionMaxFreq() const;

    /*!
     * \brief Set the convolution frequency range and mark modified.
     * \param minFreq Lower bound in MHz.
     * \param maxFreq Upper bound in MHz.
     */
    void setConvolutionFreqRange(double minFreq, double maxFreq);

    /*!
     * \brief Return the number of points in the convolution frequency grid.
     */
    int numConvolutionPoints() const;

    /*!
     * \brief Set the number of convolution grid points and mark modified.
     * \param numPoints Desired grid size.
     */
    void setNumConvolutionPoints(int numPoints);

    /*!
     * \brief Return the spacing between convolution grid points in MHz.
     */
    double calculatePointSpacing() const;

    /// \brief Return the lower bound of the display filter range in MHz.
    double filterMinFreq() const;
    /// \brief Return the upper bound of the display filter range in MHz.
    double filterMaxFreq() const;

    /*!
     * \brief Set the display filter frequency range and mark modified.
     * \param minFreq Lower bound in MHz.
     * \param maxFreq Upper bound in MHz.
     */
    void setFilterRange(double minFreq, double maxFreq);

    /*!
     * \brief Set all convolution parameters in a single call and mark modified.
     * \param enabled    Enable convolution.
     * \param lineshape  Lineshape type.
     * \param linewidth  FWHM in kHz.
     * \param minFreq    Lower bound of convolution range in MHz.
     * \param maxFreq    Upper bound of convolution range in MHz.
     * \param numPoints  Number of convolution grid points.
     */
    void setConvolutionSettings(bool enabled, LineshapeType lineshape,
                               double linewidth, double minFreq, double maxFreq,
                               int numPoints);

    /*!
     * \brief Callback type for progress reporting during chunked convolution.
     *
     * The callback receives a percentage (0–100) and a status message string.
     * Return \c true to continue processing or \c false to cancel.
     */
    using ProgressCallback = std::function<bool(int percentage, const QString& message)>;

    /*!
     * \brief Generate the convolved spectrum from the loaded catalog data.
     * \return Vector of (frequency MHz, intensity) points.
     */
    QVector<QPointF> generateConvolvedSpectrum() const;

    /*!
     * \brief Generate the convolved spectrum with progress reporting.
     * \param progressCallback Callback invoked after each chunk; return \c false to cancel.
     * \return Vector of (frequency MHz, intensity) points, or empty if cancelled.
     */
    QVector<QPointF> generateConvolvedSpectrum(ProgressCallback progressCallback) const;

    /// \brief Invalidate the convolution cache, forcing recomputation on next access.
    void invalidateConvolutionCache();
    /// \brief Mark the convolution cache as pending (background operation in progress).
    void setCachePending();

    /*!
     * \brief Mark the convolution cache as valid and store the result.
     * \param convolvedData Computed convolution result to cache.
     */
    void setCacheValid(const QVector<QPointF> &convolvedData);

    /// \brief Return \c true if the convolution cache holds a valid result.
    bool isCacheValid() const;
    /// \brief Return \c true if the cache is in the Valid state and contains data.
    bool hasConvolvedData() const;

protected:
    /// \brief Load catalog data from the destination file.
    void readFromDest() override;
    /// \brief Write catalog data to the destination file.
    void writeToDest() override;
    /// \brief Store catalog-specific metadata fields into the settings map.
    void _storeMetadata(std::map<QString, QVariant, std::less<>> &m) override;
    /// \brief Restore catalog-specific metadata fields from the settings map.
    void _retrieveMetadata(const std::map<QString, QVariant, std::less<>> &m) override;

private:
    /// \brief Return the convolved spectrum (or raw sticks if convolution is disabled).
    QVector<QPointF> _xyData() const override;


    // Lineshape functions (x and x0 in MHz, width in kHz)
    double lorentzianProfile(double x, double x0, double fwhmKHz) const;
    double gaussianProfile(double x, double x0, double fwhmKHz) const;

    // Chunked processing support
    int calculateChunkSize(int numConvolutionPoints, int numTransitions) const;

    // Data members
    CatalogData d_catalogData;

    // Convolution settings
    bool d_convolutionEnabled{false};
    LineshapeType d_lineshapeType{Lorentzian};
    double d_linewidth{100.0};            // FWHM in kHz
    double d_convolutionMinFreq{0.0};     // MHz
    double d_convolutionMaxFreq{1000.0};  // MHz
    int d_numConvolutionPoints{1000};     // Number of points in convolution grid

    // Filtering range settings
    double d_filterMinFreq{0.0};          // MHz
    double d_filterMaxFreq{1000.0};       // MHz

    // Cache state management
    enum class CacheState {
        Invalid,     // Cache is invalid/empty
        Pending,     // Background operation is in progress
        Valid        // Cache contains valid data
    };

    // Cached convolved data
    mutable QVector<QPointF> d_convolvedCache;
    mutable CacheState d_cacheState{CacheState::Invalid};
};

/*!
 * \brief OverlayBase subclass for arbitrary two-column XY data files.
 *
 * GenericXYOverlay parses delimited text files (CSV, TSV, space-separated,
 * etc.) with configurable delimiter, header line count, and column indices.
 * A separate display filter range allows restricting the visible X range
 * independently of the frequency clipping applied by the base class.
 *
 * \sa OverlayBase
 */
class GenericXYOverlay : public OverlayBase
{
    Q_GADGET
public:
    /*!
     * \brief Delimiter type used when parsing the data file.
     *
     * Stored as an enum to avoid embedding literal delimiter characters in
     * the settings CSV where commas and semicolons have structural meaning.
     */
    enum DelimiterType {
        Comma,      ///< Comma-separated values.
        Tab,        ///< Tab-separated values.
        Space,      ///< Single-space-separated values.
        Semicolon,  ///< Semicolon-separated values.
        Whitespace  ///< Any run of whitespace (greedy split).
    };
    Q_ENUM(DelimiterType)

    /// \brief Construct a GenericXY-type overlay.
    GenericXYOverlay();

    /*!
     * \brief Return the raw parsed data points before base-class transformations.
     */
    QVector<QPointF> rawData() const;

    /*!
     * \brief Replace the raw data and update cached statistics.
     * \param data New XY data points.
     */
    void setRawData(const QVector<QPointF> &data);

    /*!
     * \brief Return the delimiter string used for parsing.
     */
    QString delimiter() const;

    /*!
     * \brief Set the delimiter string and mark modified.
     * \param delim Delimiter string (e.g. ",", "\t").
     */
    void setDelimiter(const QString &delim);

    /*!
     * \brief Return the number of header lines skipped when parsing.
     */
    int headerLines() const;

    /*!
     * \brief Set the number of header lines and mark modified.
     * \param lines Number of lines to skip.
     */
    void setHeaderLines(int lines);

    /// \brief Return the zero-based index of the X data column.
    int xColumn() const;
    /// \brief Return the zero-based index of the Y data column.
    int yColumn() const;

    /*!
     * \brief Set the X and Y column indices and mark modified.
     * \param xCol Zero-based X column index.
     * \param yCol Zero-based Y column index.
     */
    void setDataColumns(int xCol, int yCol);

    /*!
     * \brief Return the column names parsed from the header, if any.
     */
    QStringList columnNames() const;

    /*!
     * \brief Set the column name list and mark modified.
     * \param names Column names in order.
     */
    void setColumnNames(const QStringList &names);

    /// \brief Return the number of data points loaded.
    int dataPointCount() const;
    /// \brief Return the minimum X value in the loaded data.
    double xMin() const;
    /// \brief Return the maximum X value in the loaded data.
    double xMax() const;
    /// \brief Return the minimum Y value in the loaded data.
    double yMin() const;
    /// \brief Return the maximum Y value in the loaded data.
    double yMax() const;

    /*!
     * \brief Return the (min, max) X range of the loaded data.
     */
    QPair<double, double> xRange() const;

    /*!
     * \brief Return the (min, max) Y range of the loaded data.
     */
    QPair<double, double> yRange() const;

    /// \brief Return the lower bound of the display filter range.
    double filterMinX() const;
    /// \brief Return the upper bound of the display filter range.
    double filterMaxX() const;

    /*!
     * \brief Set the display filter range and mark modified.
     * \param minX Lower filter bound.
     * \param maxX Upper filter bound.
     */
    void setFilterRange(double minX, double maxX);

protected:
    /// \brief Load XY data from the destination file.
    void readFromDest() override;
    /// \brief Write XY data to the destination file.
    void writeToDest() override;
    /// \brief Store GenericXY-specific metadata fields into the settings map.
    void _storeMetadata(std::map<QString, QVariant, std::less<>> &m) override;
    /// \brief Restore GenericXY-specific metadata fields from the settings map.
    void _retrieveMetadata(const std::map<QString, QVariant, std::less<>> &m) override;

private:
    /// \brief Return the raw XY data (no transformation applied).
    QVector<QPointF> _xyData() const override;
    void updateStatistics();

    // Delimiter conversion helpers for BlackchirpCSV compatibility
    DelimiterType stringToDelimiterType(const QString &delimiter) const;
    QString delimiterTypeToString(DelimiterType type) const;

    // Data storage
    QVector<QPointF> d_rawData;

    // Parsing settings
    QString d_delimiter{","};
    int d_headerLines{0};
    int d_xColumn{0};
    int d_yColumn{1};
    QStringList d_columnNames;

    // Cached statistics
    int d_dataPoints{0};
    double d_xMin{0.0};
    double d_xMax{0.0};
    double d_yMin{0.0};
    double d_yMax{0.0};

    // Filtering range settings
    double d_filterMinX{0.0};
    double d_filterMaxX{1000.0};
};

#endif // OVERLAYTYPES_H
