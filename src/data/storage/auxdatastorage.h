#ifndef AUXDATASTORAGE_H
#define AUXDATASTORAGE_H

#include <set>
#include <map>
#include <vector>

#include <QString>
#include <QVariant>
#include <QDateTime>
#include <QStringView>

class BlackchirpCSV;

/// \brief Namespace for auxiliary data storage constants.
namespace  BC::Aux {
/// \brief Format string used by \c AuxDataStorage::makeKey to compose a compound key.
///
/// The two positional arguments are the object key and the value key,
/// producing keys of the form \c "ObjKey.ValueKey".
inline const QString keyTemplate{"%1.%2"};
}

/*!
 * \brief Collects auxiliary time-series data during an experiment.
 *
 * Owned directly by \c Experiment; not a \c DataStorageBase subclass.
 * Hardware objects register their keys via \c registerKey, push readings
 * via \c addDataPoints, and the acquisition system seals each point via
 * \c startNewPoint, which appends a row to \c BC::CSV::auxFile.
 */
class AuxDataStorage
{
public:
    /// \brief Map from compound key to scalar value, keyed transparently on \c QStringView.
    using AuxDataMap = std::map<QString,QVariant,std::less<>>;

    /// \brief A single sealed time point consisting of a timestamp and a map of values.
    struct TimePointData {
        QDateTime dateTime; ///< Timestamp at which this point was sealed.
        AuxDataMap map;     ///< Values contributed to this point, keyed by compound key.
    };

    /*!
     * \brief Composes a compound key from an object key and a value key.
     * \param s1 Object key (typically the hardware object's settings key).
     * \param s2 Value key (the individual reading name within that object).
     * \return Compound key in the form \c "s1.s2", as defined by \c BC::Aux::keyTemplate.
     */
    static inline QString makeKey(const QString &s1, const QString &s2) {
        return BC::Aux::keyTemplate.arg(s1).arg(s2);
    }

    int d_number{-1};    ///< Experiment number; \c -1 for a detached (unsaved) instance.
    QString d_path{""};  ///< Base data path used to locate the experiment directory.

    /// \brief Constructs a default instance with no experiment association.
    ///
    /// Used when constructing an \c Experiment that has not yet been assigned a
    /// number. No file I/O is performed until \c startNewPoint is called on a
    /// properly initialized instance.
    AuxDataStorage() {}

    /*!
     * \brief Constructs an instance by loading saved auxiliary data from disk.
     * \param csv Pointer to the experiment's \c BlackchirpCSV instance, used to
     *        tokenize lines with the correct delimiter.
     * \param number Experiment number.
     * \param path Optional base path override. When empty, \c SettingsStorage
     *        provides the application data path.
     *
     * Reads \c BC::CSV::auxFile from the experiment directory and populates
     * \c d_savedData with the full time series. The column headers are used to
     * reconstruct the key ordering.
     */
    AuxDataStorage(BlackchirpCSV *csv, int number, const QString &path = {});

    /*!
     * \brief Registers a key as a valid receiver of auxiliary data.
     * \param objKey Object key identifying the hardware source.
     * \param key Value key identifying the individual reading.
     *
     * Only keys registered before acquisition begins appear as columns in
     * \c BC::CSV::auxFile. Calls to \c addDataPoints that supply unregistered
     * keys are silently ignored.
     */
    void registerKey(const QString &objKey, const QString &key);

    /*!
     * \brief Merges a map of new readings into the current time point.
     * \param m Map of compound keys to new values. Keys absent from the
     *        registered set are ignored.
     *
     * Called by hardware objects whenever fresh readings are available.
     * Multiple calls between successive \c startNewPoint calls accumulate
     * additively into the current point.
     */
    void addDataPoints(AuxDataMap &m);

    /*!
     * \brief Seals the current time point and appends it to the output file.
     *
     * On the very first call, writes the column-header row to
     * \c BC::CSV::auxFile. On subsequent calls, serializes the accumulated
     * readings for the point that was started by the previous call, then
     * advances the current point timestamp to \c QDateTime::currentDateTime.
     * If \c d_number is negative or the allowed-key set is empty, no I/O is
     * performed.
     */
    void startNewPoint();

    /*!
     * \brief Returns the timestamp of the current (unsealed) time point.
     * \return \c QDateTime of the point in progress, or a null \c QDateTime
     *         if no point has been started.
     */
    QDateTime currentPointTime() const { return d_currentPoint.dateTime; }

    /*!
     * \brief Returns the full list of sealed time points.
     * \return Vector of (timestamp, value map) pairs in chronological order.
     *
     * When constructed from a saved experiment, this vector is populated from
     * the file; during live acquisition it grows as each point is sealed.
     */
    std::vector<std::pair<QDateTime,AuxDataMap>> savedData() const;

private:
    std::set<QString> d_allowedKeys;
    TimePointData d_currentPoint;
    QDateTime d_startTime;

    std::vector<std::pair<QDateTime,AuxDataMap>> d_savedData;


};

Q_DECLARE_METATYPE(AuxDataStorage::AuxDataMap)

#endif // AUXDATASTORAGE_H
