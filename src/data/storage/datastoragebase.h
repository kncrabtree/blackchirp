#ifndef DATASTORAGEBASE_H
#define DATASTORAGEBASE_H

#include <memory>
#include <QString>
#include <QMutex>
#include <QVariant>
#include <QLatin1StringView>

class BlackchirpCSV;

/// \brief CSV key constants used by DataStorageBase helpers.
namespace BC::Key::DS {
inline constexpr QLatin1StringView proc{"processing.csv"}; ///< Filename of the per-subdirectory processing-settings CSV.
inline constexpr QLatin1StringView peakFind{"peakfind.csv"}; ///< Filename of the per-experiment peak-finder settings CSV.
}

/*!
 \brief Abstract base class for all experiment data storage objects.

 Every node in the experiment data-storage tree derives from \c DataStorageBase.
 The four pure-virtual methods (\c start, \c advance, \c save, \c finish) form
 the acquisition lifecycle interface that subclasses implement.
*/
class DataStorageBase
{
public:
    /*!
     \brief Construct a storage object for the given experiment.
     \param number Experiment number; pass \c -1 for a transient (peak-up) instance.
     \param path   Base path under which the experiment directory is located.
    */
    DataStorageBase(int number = -1, const QString &path = {});

    /// \brief Destructor.
    virtual ~DataStorageBase();

    const int d_number;    ///< Experiment number; \c -1 indicates a transient instance with no disk I/O.
    const QString d_path;  ///< Base path of the experiment data directory.

    /// \brief Called at each segment boundary; flush the current segment and prepare for the next.
    virtual void advance() =0;

    /// \brief Persist the current in-memory state to disk.
    virtual void save() =0;

    /// \brief Called when acquisition begins; implementations set the acquiring flag and initialize state.
    virtual void start() =0;

    /// \brief Called when acquisition ends; implementations clear the acquiring flag.
    virtual void finish() =0;

protected:
    std::unique_ptr<QMutex> pu_mutex; ///< Mutex guarding mutable state shared across threads.
    std::unique_ptr<BlackchirpCSV> pu_csv; ///< CSV helper scoped to this experiment's directory.

    /*!
     \brief Write a key-value map to a named CSV file within the experiment directory.
     \param file Filename (relative to the experiment directory, or to \a dir if given).
     \param dat  Map of string keys to QVariant values to serialize.
     \param dir  Optional subdirectory relative to the experiment directory.
    */
    void writeMetadata(const QString &file, const std::map<QString,QVariant,std::less<>> &dat, const QString &dir = {});

    /*!
     \brief Read a previously written metadata CSV back into a key-value map.
     \param file Filename (relative to the experiment directory, or to \a dir if given).
     \param out  Map to populate with the deserialized key-value pairs.
     \param dir  Optional subdirectory relative to the experiment directory.
    */
    void readMetadata(const QString &file, std::map<QString,QVariant,std::less<>> &out, const QString &dir = {});
};

#endif // DATASTORAGEBASE_H
