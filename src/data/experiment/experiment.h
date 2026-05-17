#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#define _STR(x) #x
#define STRINGIFY(x) _STR(x)

#include <memory>

#include <QPair>
#include <QDateTime>
#include <QMetaType>
#include <QSet>

#include <data/storage/headerstorage.h>
#include <data/storage/auxdatastorage.h>
#include <data/storage/overlaystorage.h>
#include <data/experiment/experimentvalidator.h>
#include <data/experiment/ftmwconfig.h>
#include <data/experiment/hardwaredatacontainer.h>
#include <data/loghandler.h>
#include <data/experiment/hardware/optional/pulsegenerator/pulsegenconfig.h>
#include <data/experiment/hardware/optional/flowcontroller/flowconfig.h>
#include <data/experiment/hardware/optional/ioboard/ioboardconfig.h>
#include <data/experiment/hardware/optional/pressurecontroller/pressurecontrollerconfig.h>
#include <data/experiment/hardware/optional/tempcontroller/temperaturecontrollerconfig.h>

//these are included because they define datatypes needed by qRegisterMetaType in main.cpp
#include <data/analysis/ft.h>
#include <data/analysis/ftworker.h>

#include <data/lif/lifconfig.h>

/// \brief Storage keys for the top-level experiment header section.
namespace BC::Store::Exp {
inline constexpr QLatin1StringView key{"Experiment"};      ///< Header section key.
inline constexpr QLatin1StringView num{"Number"};          ///< Experiment number.
inline constexpr QLatin1StringView timeData{"TimeDataInterval"}; ///< Auxiliary data sampling interval in seconds.
inline constexpr QLatin1StringView backupInterval{"BackupInterval"}; ///< Periodic backup interval in minutes.
inline constexpr QLatin1StringView ftmwEn{"FtmwEnabled"};  ///< Whether FTMW acquisition is active.
inline constexpr QLatin1StringView ftmwType{"FtmwType"};   ///< FTMW acquisition mode string.
inline constexpr QLatin1StringView majver{"BCMajorVersion"};   ///< Blackchirp major version at acquisition time.
inline constexpr QLatin1StringView minver{"BCMinorVersion"};   ///< Blackchirp minor version at acquisition time.
inline constexpr QLatin1StringView patchver{"BCPatchVersion"}; ///< Blackchirp patch version at acquisition time.
inline constexpr QLatin1StringView relver{"BCReleaseVersion"}; ///< Blackchirp release label at acquisition time.
inline constexpr QLatin1StringView buildver{"BCBuildVersion"}; ///< Blackchirp build version string at acquisition time.
}

/// \brief Top-level experiment record, owning all acquisition sub-configurations.
///
/// \c Experiment is the root of the \c HeaderStorage tree. It aggregates an
/// optional \c FtmwConfig, an optional \c LifConfig, auxiliary hardware
/// configurations (pulse generator, flow controller, IO board, pressure
/// controller, temperature sensor), auxiliary time-series data, and overlay
/// storage. All child objects contribute their fields to the shared experiment
/// header file.
///
/// Two construction paths exist: the default constructor creates an empty
/// experiment ready for setup, and the disk-loading constructor — which
/// takes the experiment number, an optional base path, and a header-only
/// flag — reads a saved experiment from disk by experiment number.
///
/// An experiment is identified as a *dummy* when it runs in Peak Up mode
/// without LIF enabled; dummy experiments are not assigned a number and are
/// not saved to disk.
///
/// \sa FtmwConfig, LifConfig, HeaderStorage, AuxDataStorage, OverlayStorage
class Experiment : private HeaderStorage
{
    Q_GADGET
public:
    /// \brief Constructs a new, empty experiment ready for configuration.
    Experiment();

    /// \brief Constructs a copy of \a other.
    Experiment(const Experiment &other) = default;

    /// \brief Constructs an experiment by reading saved data from disk.
    ///
    /// Loads the header, hardware map, objective list, chirp file, clock file,
    /// FID data (unless \a headerOnly is true), LIF data, auxiliary data, and
    /// overlays for experiment number \a num under \a exptPath (or the default
    /// data directory when \a exptPath is empty).
    /// \param num      Experiment number to load.
    /// \param exptPath Optional override for the root data directory.
    /// \param headerOnly When true, FID data, auxiliary data, and overlays are
    ///                   not loaded; only the header and configuration are read.
    Experiment(const int num, QString exptPath = QString(""), bool headerOnly = false);

    ~Experiment();

    bool d_hardwareSuccess{false};    ///< Set to true after the hardware map loads successfully.
    bool d_initSuccess{false};        ///< Set to true after initialize() succeeds.
    BC::Data::HardwareDataContainer d_hardwareData; ///< Hardware type-to-key map recorded at acquisition time.
    int d_number{0};                  ///< Experiment number; -1 for dummy (Peak Up) experiments.
    QDateTime d_startTime;            ///< Wall-clock time when the experiment started.
    QDateTime d_lastBackupTime;       ///< Wall-clock time of the most recent backup (periodic or manual); used to schedule the next periodic backup.
    int d_timeDataInterval{300};      ///< Auxiliary data sampling period in seconds.
    int d_backupIntervalMinutes{0};   ///< Periodic backup interval in minutes; 0 disables backups.
    QString d_errorString;            ///< Human-readable description of the last error, if any.
    QString d_startLogMessage;        ///< Message written to the log when the experiment starts.
    QString d_endLogMessage;          ///< Message written to the log when the experiment ends.
    LogHandler::MessageCode d_endLogMessageCode{LogHandler::Highlight}; ///< Severity of the end-of-experiment log message.
    QString d_majorVersion{STRINGIFY(BC_MAJOR_VERSION)};   ///< Blackchirp major version string.
    QString d_minorVersion{STRINGIFY(BC_MINOR_VERSION)};   ///< Blackchirp minor version string.
    QString d_patchVersion{STRINGIFY(BC_PATCH_VERSION)};   ///< Blackchirp patch version string.
    QString d_releaseVersion{STRINGIFY(BC_RELEASE_VERSION)}; ///< Blackchirp release label string.
    QString d_buildVersion{STRINGIFY(BC_BUILD_VERSION)};   ///< Blackchirp build version string.
    QSet<ExperimentObjective*> d_objectives; ///< Non-owning pointers to all active experiment objectives (FTMW, LIF).

    /// \brief Returns true if the experiment has been aborted.
    inline bool isAborted()  const { return d_isAborted; }

    /// \brief Returns true if this is a dummy (Peak Up, no-save) experiment.
    inline bool isDummy() const { return d_isDummy; }

    /// \brief The storage path override ("" follows the configured save path).
    inline QString path() const { return d_path; }

    /// \brief Returns true when all active experiment objectives have completed.
    bool isComplete() const;

    /// \brief Returns a flat map of all header key-value pairs suitable for display.
    HeaderStrings getSummary();

    /// \brief Returns true if FTMW acquisition is configured.
    inline bool ftmwEnabled() const { return ps_ftmwConfig.get() != nullptr; }

    /// \brief Creates and registers an \c FtmwConfig of the given \a type, replacing any existing one.
    /// \return Non-owning pointer to the newly created \c FtmwConfig.
    FtmwConfig* enableFtmw(FtmwConfig::FtmwType type);

    /// \brief Removes and destroys the active \c FtmwConfig.
    void disableFtmw();

    /// \brief Returns a non-owning pointer to the active \c FtmwConfig, or \c nullptr if FTMW is disabled.
    inline FtmwConfig* ftmwConfig() const {return ps_ftmwConfig.get(); }

    /// \brief Returns a non-owning pointer to the auxiliary time-series data storage.
    inline AuxDataStorage *auxData() const { return ps_auxData.get(); }

    /// \brief Returns a shared pointer to the overlay storage for this experiment.
    inline std::shared_ptr<OverlayStorage> overlayStorage() const { return ps_overlayStorage; }

    /// \brief Returns a weak pointer to the optional hardware configuration associated with \a key.
    ///
    /// Returns an empty weak pointer if no configuration for \a key is registered.
    template<typename T> std::weak_ptr<T> getOptHwConfig(const QString key) const {
        auto it = d_optHwData.find(key);
        return it == d_optHwData.end() ? std::weak_ptr<T>() : std::dynamic_pointer_cast<T>(it->second);
    }

    /// \brief Registers a copy of optional hardware configuration \a c, keyed by its header key.
    template<typename T> void addOptHwConfig(const T &c) {
        QString key = c.headerKey();
        d_optHwData[key] = std::make_shared<T>(c);
    }

    /// \brief Removes the optional hardware configuration registered under \a key.
    void removeOptHwConfig(const QString key) {
        d_optHwData.erase(key);
    }

    /// \brief Populates optional hardware configurations from the loaded hardware data map.
    ///
    /// Called during the disk-load constructor after the hardware map has been read.
    void initOptHwFromData();

    /// \brief Records a batch of auxiliary data points; returns false if any value fails validation.
    bool addAuxData(AuxDataStorage::AuxDataMap m);

    /// \brief Installs a validation map used to range-check incoming auxiliary data values.
    void setValidationMap(const ExperimentValidator::ValidationMap &m);

    /// \brief Validates a single key-value pair against the installed validation map.
    ///
    /// Sets \c d_errorString and returns false if the value is out of range.
    bool validateItem(const QString key, const QVariant val);

    /// \brief Returns true if LIF acquisition is configured.
    inline bool lifEnabled() const { return ps_lifCfg.get() != nullptr; }

    /// \brief Returns a non-owning pointer to the active \c LifConfig, or \c nullptr if LIF is disabled.
    inline LifConfig* lifConfig() const { return ps_lifCfg.get(); }

    /// \brief Creates and registers a \c LifConfig, replacing any existing one.
    /// \return Non-owning pointer to the newly created \c LifConfig.
    LifConfig *enableLif();

    /// \brief Removes and destroys the active \c LifConfig.
    void disableLif();

    /// \brief Assigns an experiment number, creates the on-disk directory, and writes initial files.
    ///
    /// Must be called before acquisition begins. Returns false and sets \c d_errorString on
    /// failure. For Peak Up experiments without LIF, the experiment is marked as a dummy and
    /// no files are written.
    bool initialize();

    /// \brief Marks the experiment as aborted and propagates the abort to all objectives.
    void abort();

    /// \brief Returns true if a periodic backup is due.
    bool canBackup();

    /// \brief Writes an incremental backup of FTMW FID data to disk.
    void backup();

    /// \brief Calls each objective's cleanup-and-save routine and writes the overlay storage.
    void finalSave();

    /// \brief Writes the objectives list CSV file for this experiment.
    bool saveObjectives();

    /// \brief Writes the hardware map CSV file for this experiment.
    bool saveHardware();

    /// \brief Writes the experiment header CSV file.
    bool saveHeader();

    /// \brief Writes the chirp segment CSV file via the active \c FtmwConfig's chirp config.
    bool saveChirpFile() const;

    /// \brief Writes the marker channel CSV file via the active \c FtmwConfig's chirp config.
    bool saveMarkersFile() const;

    /// \brief Writes the clock steps CSV file via the active \c FtmwConfig's RF config.
    bool saveClockFile() const;

private:
    bool d_isAborted{false};
    bool d_isDummy{false};

    //core ftmw data
    std::shared_ptr<FtmwConfig> ps_ftmwConfig;
    std::shared_ptr<AuxDataStorage> ps_auxData;
    std::shared_ptr<ExperimentValidator> ps_validator;
    std::shared_ptr<OverlayStorage> ps_overlayStorage;

    //optional hardware data
    std::map<QString,std::shared_ptr<HeaderStorage>,std::less<>> d_optHwData;


    QString d_path;

    std::shared_ptr<LifConfig> ps_lifCfg;

    // HeaderStorage interface
protected:
    /// \brief Writes experiment-level scalar fields to the header storage tree.
    void storeValues() override;

    /// \brief Reads experiment-level scalar fields back from the header storage tree.
    void retrieveValues() override;

    /// \brief Registers all child \c HeaderStorage nodes (FtmwConfig, validator, optional hardware, LIF).
    void prepareChildren() override;
};

Q_DECLARE_METATYPE(Experiment)
Q_DECLARE_METATYPE(std::shared_ptr<Experiment>)

#endif // EXPERIMENT_H
