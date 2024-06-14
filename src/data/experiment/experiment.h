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
#include <data/experiment/experimentvalidator.h>
#include <data/experiment/ftmwconfig.h>
#include <data/loghandler.h>
#include <hardware/optional/pulsegenerator/pulsegenconfig.h>
#include <hardware/optional/flowcontroller/flowconfig.h>
#include <hardware/optional/ioboard/ioboardconfig.h>
#include <hardware/optional/pressurecontroller/pressurecontrollerconfig.h>
#include <hardware/optional/tempcontroller/temperaturecontrollerconfig.h>

//these are included because they define datatypes needed by qRegisterMetaType in main.cpp
#include <data/analysis/ft.h>
#include <data/analysis/ftworker.h>

#ifdef BC_LIF
#include <modules/lif/data/lifconfig.h>
#endif

namespace BC::Store::Exp {
static const QString key{"Experiment"};
static const QString num{"Number"};
static const QString timeData{"TimeDataInterval"};
static const QString backupInterval{"BackupInterval"};
static const QString ftmwEn{"FtmwEnabled"};
static const QString ftmwType{"FtmwType"};
static const QString majver{"BCMajorVersion"};
static const QString minver{"BCMinorVersion"};
static const QString patchver{"BCPatchVersion"};
static const QString relver{"BCReleaseVersion"};
static const QString buildver{"BCBuildVersion"};
}

class Experiment : private HeaderStorage
{
    Q_GADGET
public:
    Experiment();
    Experiment(const Experiment &other) = default;
    Experiment(const int num, QString exptPath = QString(""), bool headerOnly = false);
    ~Experiment();

    bool d_hardwareSuccess{false};
    bool d_initSuccess{false};
    std::map<QString,QString> d_hardware;
    int d_number{0};
    QDateTime d_startTime;
    QDateTime d_lastBackupTime;
    int d_timeDataInterval{300};
    int d_backupIntervalMinutes{0};
    QString d_errorString;
    QString d_startLogMessage;
    QString d_endLogMessage;
    LogHandler::MessageCode d_endLogMessageCode{LogHandler::Highlight};
    QString d_majorVersion{STRINGIFY(BC_MAJOR_VERSION)};
    QString d_minorVersion{STRINGIFY(BC_MINOR_VERSION)};
    QString d_patchVersion{STRINGIFY(BC_PATCH_VERSION)};
    QString d_releaseVersion{STRINGIFY(BC_RELEASE_VERSION)};
    QString d_buildVersion{STRINGIFY(BC_BUILD_VERSION)};
    QSet<ExperimentObjective*> d_objectives;


    inline bool isAborted()  const { return d_isAborted; }
    inline bool isDummy() const { return d_isDummy; }
    bool isComplete() const;
    HeaderStrings getSummary();


    inline bool ftmwEnabled() const { return ps_ftmwConfig.get() != nullptr; }
    FtmwConfig* enableFtmw(FtmwConfig::FtmwType type);
    void disableFtmw();
    inline FtmwConfig* ftmwConfig() const {return ps_ftmwConfig.get(); }

    inline AuxDataStorage *auxData() const { return ps_auxData.get(); }

    template<typename T> std::weak_ptr<T> getOptHwConfig(const QString key) const {
        auto it = d_optHwData.find(key);
        return it == d_optHwData.end() ? std::weak_ptr<T>() : std::dynamic_pointer_cast<T>(it->second);
    }

    template<typename T> void addOptHwConfig(const T &c) {
        QString key = c.headerKey();
        d_optHwData[key] = std::make_shared<T>(c);
    }

    void removeOptHwConfig(const QString key) {
        d_optHwData.erase(key);
    }

    bool addAuxData(AuxDataStorage::AuxDataMap m);
    void setValidationMap(const ExperimentValidator::ValidationMap &m);
    bool validateItem(const QString key, const QVariant val);

#ifdef BC_LIF
    inline bool lifEnabled() const { return ps_lifCfg.get() != nullptr; }
    inline LifConfig* lifConfig() const { return ps_lifCfg.get(); }
    LifConfig *enableLif();
    void disableLif();
#endif

    bool initialize();
    void abort();
    bool canBackup();
    void backup();
    void finalSave();

    bool saveObjectives();
    bool saveHardware();
    bool saveHeader();
    bool saveChirpFile() const;
    bool saveClockFile() const;

private:
    bool d_isAborted{false};
    bool d_isDummy{false};

    //core ftmw data
    std::shared_ptr<FtmwConfig> ps_ftmwConfig;
    std::shared_ptr<AuxDataStorage> ps_auxData;
    std::shared_ptr<ExperimentValidator> ps_validator;

    //optional hardware data
    std::map<QString,std::shared_ptr<HeaderStorage>> d_optHwData;


    QString d_path;

#ifdef BC_LIF
    std::shared_ptr<LifConfig> ps_lifCfg;
#endif

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
    void prepareChildren() override;
};

Q_DECLARE_METATYPE(Experiment)
Q_DECLARE_METATYPE(std::shared_ptr<Experiment>)

#endif // EXPERIMENT_H
