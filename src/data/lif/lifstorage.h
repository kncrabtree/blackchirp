#ifndef LIFSTORAGE_H
#define LIFSTORAGE_H

#include <QDateTime>
#include <QMutex>

#include <data/storage/datastoragebase.h>
#include <data/lif/liftrace.h>

class BlackchirpCSV;

/// \brief CSV key constants for the LIF processing-settings file.
namespace BC::Key::LifStorage {
inline constexpr QLatin1StringView lifGateStart("LifGateStartPoint"); ///< Start sample index of the LIF integration gate.
inline constexpr QLatin1StringView lifGateEnd("LifGateEndPoint");     ///< End sample index of the LIF integration gate.
inline constexpr QLatin1StringView refGateStart("RefGateStartPoint"); ///< Start sample index of the reference integration gate.
inline constexpr QLatin1StringView refGateEnd("RefGateEndPoint");     ///< End sample index of the reference integration gate.
inline constexpr QLatin1StringView lowPassAlpha("LowPassAlpha");      ///< Smoothing factor for the exponential low-pass filter (0 = disabled).
inline constexpr QLatin1StringView savGol{"SavGolEnabled"};           ///< When \c true, apply a Savitzky-Golay smoothing filter.
inline constexpr QLatin1StringView sgWin{"SavGolWindow"};             ///< Savitzky-Golay window length (must be odd and > \c sgPoly).
inline constexpr QLatin1StringView sgPoly{"SavGolPoly"};              ///< Polynomial order for the Savitzky-Golay filter.
}

/*!
 \brief Storage manager for LIF trace data in a two-dimensional (delay, laser) grid.

 Extends \c DataStorageBase to manage \c LifTrace waveforms for
 Laser-Induced Fluorescence (LIF) acquisitions.  The grid dimensions are
 fixed at construction: \c d_delayPoints delay points and \c d_laserPoints
 laser positions.  Each cell \c (di, li) maps to a flat index via
 \c di * d_laserPoints + li, which is used as the filename stem for the
 per-cell CSV files under the \c lif/ subdirectory.

 \sa DataStorageBase, LifTrace, LifConfig
*/
class LifStorage : public DataStorageBase
{
public:
    /*!
     \brief Construct the storage manager for a LIF acquisition grid.
     \param dp   Number of delay points in the scan grid.
     \param lp   Number of laser positions in the scan grid.
     \param num  Experiment number; pass \c -1 for a transient (peak-up) instance.
     \param path Base path of the experiment data directory.
    */
    LifStorage(int dp, int lp, int num, QString path="");

    /// \brief Destructor.
    ~LifStorage();

    const int d_delayPoints;  ///< Number of delay points in the scan grid.
    const int d_laserPoints;  ///< Number of laser positions in the scan grid.

    /*!
     \brief Flush the current cell to disk and arm the storage for the next cell.

     Sets \c d_nextNew so that the next \c addTrace() starts a fresh accumulation,
     then calls \c save().
    */
    void advance() override;

    /*!
     \brief Write the current \c LifTrace and the grid parameter index to disk.
    */
    void save() override;

    /// \brief Arm the storage for incoming trace data.
    void start() override;

    /// \brief Disarm the storage after acquisition ends.
    void finish() override;

    /*!
     \brief Return the shot count of the trace being accumulated in the current cell.
     \return Shot count of \c d_currentTrace.
    */
    int currentTraceShots() const;

    /*!
     \brief Return the total number of shots accumulated across all completed cells.

     During active acquisition, the in-progress cell is included only if it has
     already been started (i.e., \c d_nextNew is \c false).
     \return Total shot count.
    */
    int completedShots() const;

    /*!
     \brief Return the trace for grid cell (\a di, \a li).

     Returns the in-memory trace if the cell is the one being accumulated.
     Falls back to the completed-cell map, then to a disk read if the
     experiment is no longer acquiring.
     \param di Delay index (0-based).
     \param li Laser index (0-based).
     \return The corresponding \c LifTrace, or a default-constructed trace if not found.
    */
    LifTrace getLifTrace(int di, int li);

    /*!
     \brief Return the trace being accumulated in the current cell (no locking).
     \return Copy of \c d_currentTrace.
    */
    LifTrace currentLifTrace() const { return d_currentTrace; }

    /*!
     \brief Read the trace for cell (\a di, \a li) from disk.
     \param di Delay index (0-based).
     \param li Laser index (0-based).
     \return The loaded \c LifTrace, or a default-constructed trace on failure.
    */
    LifTrace loadLifTrace(int di, int li);

    /*!
     \brief Write a single trace to disk (used for direct writes outside the
            normal acquisition flow, e.g., post-processing updates).
     \param t Trace to write.
    */
    void writeLifTrace(const LifTrace t);

    /*!
     \brief Accumulate \a t into the current cell.

     On the first call after \c advance() (or \c start()), seeds \c d_currentTrace
     from the completed-cell map if the cell already has data, then adds \a t.
     On subsequent calls, adds \a t directly to \c d_currentTrace.
     \param t Incoming \c LifTrace to accumulate.
    */
    void addTrace(const LifTrace t);

    /*!
     \brief Serialize a \c LifTrace::LifProcSettings struct to \c lif/processing.csv.
     \param c Processing settings to write.
    */
    void writeProcessingSettings(const LifTrace::LifProcSettings &c);

    /*!
     \brief Deserialize a \c LifTrace::LifProcSettings struct from \c lif/processing.csv.
     \param out Processing settings populated on success.
     \return \c true if the file was found and parsed successfully.
    */
    bool readProcessingSettings(LifTrace::LifProcSettings &out);


private:
    bool d_acquiring{false}, d_nextNew{true};
    std::map<int,LifTrace> d_data;   ///< Completed-cell map keyed by flat grid index.
    LifTrace d_currentTrace;         ///< Trace being accumulated for the current cell.

    /*!
     \brief Convert a (delay, laser) pair to a flat grid index.
     \param dp Delay index.
     \param lp Laser index.
     \return Flat index: \c dp * d_laserPoints + lp.
    */
    int index(int dp, int lp) const;

};

#endif // LIFSTORAGE_H
