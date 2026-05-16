#ifndef FIDSTORAGEBASE_H
#define FIDSTORAGEBASE_H

#include <queue>
#include <QDateTime>
#include <QLatin1StringView>

#include <data/storage/datastoragebase.h>
#include <data/experiment/fid.h>
#include <data/analysis/ftworker.h>
#include <data/analysis/peakfindsettings.h>

class BlackchirpCSV;

/// \brief CSV key constants for the FID processing-settings file.
namespace BC::Key::FidStorage {
inline constexpr QLatin1StringView fidStart{"FidStartUs"};          ///< Start of the processing time window in microseconds.
inline constexpr QLatin1StringView fidEnd{"FidEndUs"};              ///< End of the processing time window in microseconds.
inline constexpr QLatin1StringView fidExp{"FidExpfUs"};             ///< Exponential apodization time constant in microseconds (0 = disabled).
inline constexpr QLatin1StringView zpf{"FidZeroPadFactor"};         ///< Zero-padding multiplier applied before the FFT.
inline constexpr QLatin1StringView rdc{"FidRemoveDC"};              ///< When \c true, subtract the mean before the FFT (DC removal).
inline constexpr QLatin1StringView units{"FtUnits"};                ///< Output magnitude units; maps to \c FtWorker::FtUnits.
inline constexpr QLatin1StringView autoscaleIgnore{"AutoscaleIgnoreMHz"}; ///< Half-width around the LO (MHz) excluded from autoscale tracking.
inline constexpr QLatin1StringView winf{"FidWindowFunction"};       ///< Apodization window function; maps to \c FtWorker::FtWindowFunction.
}

/// \brief CSV key constants for the per-experiment peak-finder settings file.
namespace BC::Key::PeakStorage {
inline constexpr QLatin1StringView pkMinFreq{"PeakMinFreqMHz"};       ///< Lower bound of the peak-search frequency range (MHz).
inline constexpr QLatin1StringView pkMaxFreq{"PeakMaxFreqMHz"};       ///< Upper bound of the peak-search frequency range (MHz).
inline constexpr QLatin1StringView pkSnr{"PeakSnr"};                  ///< Signal-to-noise ratio threshold for detection.
inline constexpr QLatin1StringView pkWinSize{"PeakWindowSize"};       ///< Savitzky-Golay smoothing window size.
inline constexpr QLatin1StringView pkOrder{"PeakPolyOrder"};          ///< Savitzky-Golay smoothing polynomial order.
inline constexpr QLatin1StringView pkNavHalfWidth{"PeakNavHalfWidthMHz"}; ///< Plot-centering window half-width (MHz).
}

/*!
 \brief Abstract base class for FID data storage in FTMW acquisitions.

 Extends \c DataStorageBase with the storage model for free-induction decay
 (FID) waveforms.  Holds a fixed number of FID records per segment
 (\c d_numRecords) and implements the lifecycle methods: \c start() and
 \c finish() toggle an internal acquiring flag, \c save() writes the current
 FID list to disk (skipped for peak-up instances where \c d_number < 1),
 and \c advance() calls \c save() then delegates to the virtual
 \c _advance() hook for subclass-specific segment transitions.

 Three concrete subclasses cover the standard acquisition modes:
 \c FidSingleStorage (single-segment), \c FidMultiStorage (multi-segment /
 LO-scan), and \c FidPeakUpStorage (peak-up / rolling-average mode).

 \sa DataStorageBase, Fid, FtWorker
*/
class FidStorageBase : public DataStorageBase
{

public:
    /*!
     \brief Construct the storage manager.
     \param numRecords Number of FID records in each segment.
     \param number     Experiment number; pass \c -1 for a transient (peak-up) instance.
     \param path       Base path of the experiment data directory.
    */
    FidStorageBase(int numRecords, int number = -1, QString path = "");

    /// \brief Destructor.
    virtual ~FidStorageBase();

    const int d_numRecords; ///< Number of FID records per segment (set at construction).

    /*!
     \brief Flush the current segment and invoke the subclass transition hook.

     Calls \c save() then \c _advance() so subclasses can update their segment
     index or perform other per-segment bookkeeping.
    */
    void advance() override;

    /*!
     \brief Write the current FID list to disk.

     Skips silently if \c d_number < 1 (peak-up / dummy experiment).
    */
    void save() override;

    /// \brief Mark the start of acquisition.
    void start() override;

    /// \brief Mark the end of acquisition.
    void finish() override;

    /*!
     \brief Load the FID list for segment \a i from cache or disk.
     \param i Segment index.
     \return The stored \c FidList, or an empty list if \a i is out of range.
    */
    FidList loadFidList(int i);

    /*!
     \brief Load the background-subtracted FID list for segment \a i.
     \param i Segment index.
     \return Differential \c FidList; implementation is subclass-specific.
    */
    virtual FidList loadDifferentialFidList(int i) =0;

    /*!
     \brief Return the shot count accumulated in the current segment.
     \return Shot count of the first FID in the current list, or 0 if empty.
    */
    virtual quint64 currentSegmentShots();

    /*!
     \brief Co-average \a other into the current segment.
     \param other FID list to accumulate.
     \param shift Optional time-domain sample shift applied to \a other.
     \return \c false if the sizes are incompatible.
    */
    virtual bool addFids(const FidList other, int shift =0);

    /*!
     \brief Replace the current FID list with \a other.
     \param other Replacement FID list.
     \return \c false if \a other has a different size from the existing list.
    */
    virtual bool setFidsData(const FidList other);

    /*!
     \brief Return a thread-safe copy of the current in-progress FID list.
     \return Copy of \c d_currentFidList under the mutex.
    */
    virtual FidList getCurrentFidList();

    /// \brief Save a backup of the current FID list (no-op in the base; overridden by \c FidSingleStorage).
    virtual void backup() { return; }

    /*!
     \brief Return the number of backups available (0 in the base; overridden by \c FidSingleStorage).
     \return Backup count.
    */
    virtual int numBackups() { return 0; }

    /*!
     \brief Return the index of the segment being accumulated.
     \return Current segment index; must be implemented by each concrete subclass.
    */
    virtual int getCurrentIndex() =0;

    /*!
     \brief Serialize a \c FtWorker::FidProcessingSettings struct to \c fid/processing.csv.
     \param c Processing settings to write.
    */
    void writeProcessingSettings(const FtWorker::FidProcessingSettings &c);

    /*!
     \brief Deserialize a \c FtWorker::FidProcessingSettings struct from \c fid/processing.csv.
     \param out Processing settings populated on success.
     \return \c true if the file was found and parsed successfully.
    */
    bool readProcessingSettings(FtWorker::FidProcessingSettings &out);

    /*!
     \brief Serialize a \c PeakFindSettings struct to \c fid/peakfind.csv.
     \param c Peak-finder settings to write.
    */
    void writePeakFindSettings(const PeakFindSettings &c);

    /*!
     \brief Deserialize a \c PeakFindSettings struct from \c fid/peakfind.csv.
     \param out Peak-finder settings populated on success.
     \return \c true if the file was found and parsed successfully.
    */
    bool readPeakFindSettings(PeakFindSettings &out);

    /*!
     \brief Return the [min, max] probe-frequency range across all stored FID segments.
     \return Pair of (minimum, maximum) probe frequency in MHz, or (-1, -1) if no data.
    */
    std::pair<double,double> getLORange();

protected:
    FidList d_currentFidList; ///< In-progress co-averaged FID list for the current segment.

    /// \brief Subclass hook called by \c advance() after \c save(); default implementation is a no-op.
    virtual void _advance() {}

    /*!
     \brief Write \a l to disk as segment \a i and update the cache.
     \param l FID list to persist.
     \param i Segment index.
    */
    void saveFidList(const FidList l, int i);

private:
    bool d_acquiring{false};
    int d_currentSegment{0};
    std::size_t d_maxCacheSize{1 << 28}; ///< Maximum cache size in bytes (~256 MB).
    QVector<Fid> d_templateList;
    std::unique_ptr<QMutex> pu_baseMutex;
    std::queue<int> d_cacheKeys;         ///< Insertion-ordered queue of cached segment indices.
    std::map<int,FidList> d_cache;       ///< In-memory cache mapping segment index to FID list.

    /*!
     \brief Insert or update \a fl in the cache, evicting the oldest entry when full.
     \param fl FID list to cache.
     \param i  Segment index.
    */
    void updateCache(const FidList fl, int i);

};

#endif // FIDSTORAGEBASE_H
