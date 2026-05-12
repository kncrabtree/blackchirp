#ifndef BLACKCHIRPCSV_H
#define BLACKCHIRPCSV_H

#define _STR(x) #x
#define STRINGIFY(x) _STR(x)

#include <QIODevice>
#include <QTextStream>
#include <QVector>
#include <QPointF>
#include <QDir>
#include <QLatin1StringView>

#include <data/experiment/fid.h>
#include <data/bcglobals.h>
#include <data/storage/enumcsvconvert.h>

/// \brief Namespace for canonical CSV constants used across the storage subsystem.
namespace BC::CSV {

// Delimiter and formatting constants
inline constexpr QLatin1StringView del{";"};      ///< Primary column delimiter used in all CSV files.
inline constexpr QLatin1StringView altDel{"|"};   ///< Alternative delimiter for QStringList serialization.
inline constexpr QLatin1StringView nl{"\n"};      ///< Line separator written by all CSV write helpers.
inline constexpr QLatin1StringView x{"x"};        ///< Default column header for the X axis in XY files.
inline constexpr QLatin1StringView y{"y"};        ///< Default column header for the Y axis in XY files.
inline constexpr QLatin1StringView sep{"_"};      ///< Separator inserted between a prefix and an axis label.

// Header CSV column-name constants
inline constexpr QLatin1StringView ok{"ObjKey"};     ///< Column name for the object key in header CSV files.
inline constexpr QLatin1StringView ak{"ArrayKey"};   ///< Column name for the array key in header CSV files.
inline constexpr QLatin1StringView ai{"ArrayIndex"};  ///< Column name for the array index in header CSV files.
inline constexpr QLatin1StringView vk{"ValueKey"};   ///< Column name for the value key in header CSV files.
inline constexpr QLatin1StringView vv{"Value"};      ///< Column name for the value in header CSV files.
inline constexpr QLatin1StringView vu{"Units"};      ///< Column name for the units in header CSV files.

// Experiment-root canonical filenames
inline constexpr QLatin1StringView versionFile{"version.csv"};      ///< Stores the Blackchirp version metadata for the experiment.
inline constexpr QLatin1StringView validationFile{"validation.csv"}; ///< Stores experiment validation results.
inline constexpr QLatin1StringView objectivesFile{"objectives.csv"}; ///< Records the active acquisition objectives.
inline constexpr QLatin1StringView hwFile{"hardware.csv"};           ///< Records hardware loadout configuration.
inline constexpr QLatin1StringView headerFile{"header.csv"};         ///< Full experiment header with all configuration parameters.
inline constexpr QLatin1StringView chirpFile{"chirps.csv"};          ///< Chirp waveform data for the experiment.
inline constexpr QLatin1StringView markersFile{"markers.csv"};       ///< User-defined markers placed on FID data.
inline constexpr QLatin1StringView clockFile{"clocks.csv"};          ///< Clock frequency configuration snapshot.
inline constexpr QLatin1StringView auxFile{"auxdata.csv"};           ///< Auxiliary time-series data (pressure, flow, temperature, etc.).

// Version-key constants written to versionFile
inline constexpr QLatin1StringView majver{"BCMajorVersion"};    ///< Key for the Blackchirp major version number.
inline constexpr QLatin1StringView minver{"BCMinorVersion"};    ///< Key for the Blackchirp minor version number.
inline constexpr QLatin1StringView patchver{"BCPatchVersion"};  ///< Key for the Blackchirp patch version number.
inline constexpr QLatin1StringView relver{"BCReleaseVersion"};  ///< Key for the Blackchirp release label (e.g., "alpha", "rc1").
inline constexpr QLatin1StringView buildver{"BCBuildVersion"};  ///< Key for the Blackchirp build identifier.

// FID artifact filenames and directory
inline constexpr QLatin1StringView fidparams{"fidparams.csv"}; ///< FID acquisition parameters file written alongside FID data.
inline constexpr QLatin1StringView fidDir{"fid"};              ///< Subdirectory under the experiment root that contains FID data files.

// LIF artifact filenames and directory
inline constexpr QLatin1StringView lifparams{"lifparams.csv"}; ///< LIF acquisition parameters file written alongside LIF data.
inline constexpr QLatin1StringView lifDir{"lif"};              ///< Subdirectory under the experiment root that contains LIF data files.

}

/*!
 * \brief Workhorse persistence class for experiment CSV I/O.
 *
 * \c BlackchirpCSV owns two complementary roles. In its static form it
 * provides the write helpers, directory helpers, and format utilities that
 * every storage class calls directly — none of those functions require an
 * instance. In its instance form it holds a snapshot of the version metadata
 * read from a saved experiment's \c version.csv, which the loader passes to
 * DataStorageBase subclasses via \c readLine and \c readFidLine.
 *
 * The on-disk layout that these helpers enforce is described by the canonical
 * filename constants in the \c BC::CSV namespace: experiment-root files
 * (\c headerFile, \c chirpFile, \c clockFile, etc.), FID artifacts in the
 * \c fidDir subdirectory, and LIF artifacts in the \c lifDir subdirectory.
 * All CSV files use the semicolon delimiter (\c BC::CSV::del); the
 * alternative pipe delimiter (\c BC::CSV::altDel) is reserved for
 * QStringList fields embedded within a cell.
 *
 * See \c DataStorageBase and its subclasses for the classes that build on
 * this foundation.
 */
class BlackchirpCSV
{
public:
    /// \brief Constructs a default instance with no version metadata loaded.
    BlackchirpCSV();

    /*!
     * \brief Constructs an instance by reading version metadata from a saved experiment.
     * \param num Experiment number. Combined with \a path to locate the experiment directory.
     * \param path Base data path override. When empty, the path stored in
     *        \c SettingsStorage is used.
     *
     * Opens \c version.csv in the experiment directory and populates the
     * internal configuration map. The delimiter is detected from the first line
     * of that file so that experiments saved with older delimiter conventions
     * are read correctly.
     */
    BlackchirpCSV(const int num, const QString path);

    /*!
     * \brief Writes a single XY data set to \a device as a two-column CSV.
     * \param device Output device, which must not already be open.
     * \param d Vector of points to write.
     * \param prefix Optional column-name prefix; when non-empty, column headers
     *        become \c prefix_x and \c prefix_y instead of \c x and \c y.
     * \return \c true on success.
     */
    static bool writeXY(QIODevice &device, const QVector<QPointF> d, const QString prefix = "");

    /*!
     * \brief Writes multiple XY data sets side by side as a multi-column CSV.
     * \param device Output device, which must not already be open.
     * \param l Vector of point-vectors, one per data set.
     * \param n Optional column-name prefixes, one per data set. Sets without a
     *        matching entry receive auto-generated names (\c x0/y0, \c x1/y1, …).
     * \return \c true on success.
     */
    static bool writeMultiple(QIODevice &device, const std::vector<QVector<QPointF>> &l, const std::vector<QString> &n = {});

    /*!
     * \brief Writes a single-column Y-only CSV with an optional title row.
     * \tparam T Element type; must be convertible via \c QVariant::toString.
     * \param device Output device, which must not already be open.
     * \param d Values to write, one per row.
     * \param title Column header text; defaults to \c "y".
     * \return \c true on success.
     */
    template<typename T>
    static bool writeY(QIODevice &device, const QVector<T> d, QString title="")
    {
        using namespace BC::CSV;
        if(!device.open(QIODevice::WriteOnly | QIODevice::Text))
            return false;

        QTextStream t(&device);

        if(title.isEmpty())
            t << y;
        else
            t << title;

        for(auto it = d.constBegin(); it != d.constEnd(); it++)
            t << nl << QVariant{*it}.toString();

        device.close();
        return true;
    }

    /*!
     * \brief Writes multiple single-column Y-only data sets side by side.
     * \tparam T Element type; must be convertible via \c QVariant::toString.
     * \param device Output device, which must not already be open.
     * \param titles Column header labels, one per data set.
     * \param l Data vectors, one per data set. Must match the size of \a titles.
     * \return \c true on success; \c false if \a titles and \a l have different sizes
     *         or the device cannot be opened.
     */
    template<typename T>
    static bool writeYMultiple(QIODevice &device, std::initializer_list<QString> titles, std::initializer_list<QVector<T>> l)
    {
        using namespace BC::CSV;
        if(titles.size() != l.size())
            return false;

        if(!device.open(QIODevice::WriteOnly | QIODevice::Text))
            return false;

        QTextStream t(&device);
        QVector<QVector<T>> list{l};

        auto it = titles.begin();
        int max = 0;
        for(int i = 0; it != titles.end(); ++it, ++i)
        {
            if(it != titles.begin())
                t << del;
            t << *it;
            max = qMax(max,list.at(i).size());
        }

        for(int i=0; i<max; ++i)
        {
            t << nl;
            for(int j=0; j<list.size(); j++)
            {
                if(j>0)
                    t << del;
                if(i < list.at(j).size())
                    t << QVariant{list.at(j).at(i)}.toString();
            }
        }

        device.close();
        return true;

    }

    /*!
     * \brief Writes a full experiment header map as a six-column CSV.
     * \param device Output device, which must not already be open.
     * \param header Multimap from object key to a tuple of
     *        (array key, array index, value key, value, units).
     * \return \c true on success.
     *
     * Column order matches the \c BC::CSV constants: \c ok, \c ak, \c ai,
     * \c vk, \c vv, \c vu.
     */
    static bool writeHeader(QIODevice &device, const std::multimap<QString,std::tuple<QString,QString,QString,QString,QString>> header);

    /*!
     * \brief Writes a single delimited row to an already-open \c QTextStream.
     * \param t Destination stream.
     * \param l Values to write; each is converted via \c QVariant::toString.
     *
     * Appends a newline (\c BC::CSV::nl) after the last field.
     */
    static void writeLine(QTextStream &t, const std::vector<QVariant> l);

    /*!
     * \brief Formats a 64-bit integer as a base-36 string.
     * \param n Value to format.
     * \return Base-36 string representation, with a leading \c "-" for negative values.
     *
     * FID raw sample values are stored in base 36 to keep file sizes compact.
     */
    static QString formatInt64(qint64 n);

    /*!
     * \brief Writes a list of FID objects as a multi-column base-36 CSV.
     * \param device Output device, which must not already be open.
     * \param l FID list to serialize; each FID becomes one column.
     *
     * The header row names columns \c fid0, \c fid1, etc. Sample values are
     * encoded with \c formatInt64.
     */
    static void writeFidList(QIODevice &device, const FidList l);

    /*!
     * \brief Writes the \c version.csv file for the given experiment.
     * \param num Experiment number.
     * \return \c true on success.
     *
     * Records the Blackchirp version constants (\c majver, \c minver,
     * \c patchver, \c relver, \c buildver) so that future reads can detect the
     * format version.
     */
    static bool writeVersionFile(int num);

    /*!
     * \brief Tests whether the directory for the given experiment exists.
     * \param num Experiment number.
     * \return \c true if the experiment directory exists on disk.
     */
    static bool exptDirExists(int num);

    /*!
     * \brief Scans the data path for the highest experiment number on disk.
     * \param basePath Optional base path. When empty, the active
     *        \c savePath from settings is used.
     * \return Highest experiment number under \c \<basePath\>/experiments,
     *         or 0 if no numeric subdirectory was found.
     *
     * Walks the rightmost branch of the
     * \c \<basePath\>/experiments/mil/th/num hierarchy. Used to keep the
     * stored \c exptNum reconciled with what is actually on disk so that
     * users who switch acquisition app versions, or modify the data tree
     * outside the program, do not allocate a duplicate experiment number.
     */
    static int scanMaxExptNumOnDisk(const QString &basePath = QString());

    /*!
     * \brief Mirrors the next-experiment counter into the v1.x settings store.
     * \param num Experiment number that v2.x has just allocated.
     *
     * Temporary cross-version coupling for the v2.x pre-release window:
     * users who hit a regression may need to fall back to the v1.x
     * acquisition app, which predates the per-major-version
     * \c applicationName convention and so reads the unsuffixed
     * \c Blackchirp.conf rather than v2.x's \c Blackchirp2.conf.
     * Mirroring the counter prevents v1.x from allocating a number
     * that already exists on disk.
     *
     * The mirror is a no-op when v1.x's \c savePath differs from
     * v2.x's, since \c exptNum is per-tree — writing v2.x's counter
     * into v1.x's settings would corrupt v1.x's own counter for its
     * own data tree.
     *
     * \note Safe to remove once v1.x is no longer a supported fallback.
     */
    static void mirrorExptNumToV1Settings(int num);

    /*!
     * \brief Creates the directory hierarchy for the given experiment.
     * \param num Experiment number.
     * \return \c true if the directory was created or already existed.
     *
     * The hierarchy follows the pattern \c savePath/experiments/mil/th/num,
     * where \c mil and \c th are the million- and thousand-buckets of \a num.
     */
    static bool createExptDir(int num);

    /*!
     * \brief Returns a \c QDir pointing to the given experiment's root directory.
     * \param num Experiment number.
     * \param path Optional base path override. When empty, \c SettingsStorage
     *        provides the application data path.
     * \return \c QDir for the experiment root (may not exist).
     */
    static QDir exptDir(int num, QString path="");

    /*!
     * \brief Returns a \c QDir pointing to the application log directory.
     * \return \c QDir for the log directory (may not exist).
     */
    static QDir logDir();

    /*!
     * \brief Returns a \c QDir pointing to the text-export output directory.
     * \return \c QDir for the text-export directory (may not exist).
     */
    static QDir textExportDir();

    /*!
     * \brief Returns a \c QDir pointing to the hardware-tracking data directory.
     * \return \c QDir for the tracking directory (may not exist).
     */
    static QDir trackingDir();

    /*!
     * \brief Reads and tokenizes the next line from \a device using the loaded delimiter.
     * \param device Open, readable device positioned at the start of a line.
     * \return List of field values, one \c QVariant per column; empty if the line is blank.
     *
     * The delimiter is the one detected when this instance was constructed from
     * a saved experiment. On the default-constructed instance it is \c BC::CSV::del.
     */
    QVariantList readLine(QIODevice &device);

    /*!
     * \brief Reads and decodes a FID data line from \a device.
     * \param device Open, readable device positioned at the start of a FID data line.
     * \return Vector of raw 64-bit sample values decoded from base-36 encoding.
     */
    QVector<qint64> readFidLine(QIODevice &device);

    /*!
     * \brief Returns the major version number from the loaded version metadata.
     * \return Major version, or \c -1 if no metadata was loaded.
     */
    int majorVersion() const;

    /*!
     * \brief Returns the minor version number from the loaded version metadata.
     * \return Minor version, or \c -1 if no metadata was loaded.
     */
    int minorVersion() const;

    /*!
     * \brief Returns the patch version number from the loaded version metadata.
     * \return Patch version, or \c -1 if no metadata was loaded.
     */
    int patchVersion() const;

    /*!
     * \brief Returns the release label from the loaded version metadata.
     * \return Release label string (e.g., \c "alpha"), or an empty string if not present.
     */
    QString releaseVersion() const;

    /*!
     * \brief Returns the build identifier from the loaded version metadata.
     * \return Build identifier string, or an empty string if not present.
     */
    QString buildVersion() const;

private:
    std::map<QString,QVariant,std::less<>> d_configMap;
    QString d_delimiter;

};

#endif // BLACKCHIRPCSV_H
