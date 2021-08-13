#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <memory>

#include <QPair>
#include <QDateTime>
#include <QMetaType>

#include <data/storage/headerstorage.h>
#include <data/storage/auxdatastorage.h>
#include <data/experiment/experimentvalidator.h>
#include <data/experiment/ftmwconfig.h>
#include <data/datastructs.h>
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

#ifdef BC_MOTOR
#include <modules/motor/data/motorscan.h>
#endif

namespace BC::Store::Exp {
static const QString key("Experiment");
static const QString num("Number");
static const QString timeData("TimeDataInterval");
static const QString autoSave("AutoSaveShotsInterval");
static const QString ftmwEn("FtmwEnabled");
static const QString ftmwType("FtmwType");
}

namespace BC::Config::Exp {
static const QString ftmwType{"FtmwType"};
}

class Experiment : private HeaderStorage
{
    Q_GADGET
public:
    Experiment();
    Experiment(const Experiment &other);
    Experiment(const int num, QString exptPath = QString(""), bool headerOnly = false);
    ~Experiment();

    bool d_hardwareSuccess{false};
    std::map<QString,QString> d_hardware;
    int d_number{0};
    QDateTime d_startTime;
    QDateTime d_lastAutosaveTime;
    int d_timeDataInterval{300};
    int d_autoSaveIntervalHours{0};
    QString d_errorString;
    QString d_startLogMessage;
    QString d_endLogMessage;
    BlackChirp::LogMessageCode d_endLogMessageCode{BlackChirp::LogHighlight};

    inline bool isAborted()  const { return d_isAborted; }
    inline bool isDummy() const { return d_isDummy; }
    bool isComplete() const;


    inline bool ftmwEnabled() const { return pu_ftmwConfig.get() != nullptr; }
    FtmwConfig* enableFtmw(FtmwConfig::FtmwType type);
    inline FtmwConfig* ftmwConfig() const {return pu_ftmwConfig.get(); }
    bool incrementFtmw();
    void setFtmwClocksReady();

    inline AuxDataStorage *auxData() const { return pu_auxData.get(); }

    inline IOBoardConfig *iobConfig() const { return pu_iobCfg.get(); }
    inline PulseGenConfig *pGenConfig() const { return pu_pGenCfg.get(); }
    inline FlowConfig *flowConfig() const { return pu_flowCfg.get(); }
    inline PressureControllerConfig *pcConfig() const { return pu_pcConfig.get(); }
    inline TemperatureControllerConfig *tcConfig() const { return pu_tcConfig.get(); }

    void setIOBoardConfig(const IOBoardConfig &cfg);
    void setPulseGenConfig(const PulseGenConfig &c);
    void setFlowConfig(const FlowConfig &c);
    void setPressureControllerConfig(const PressureControllerConfig &c);
    void setTempControllerConfig(const TemperatureControllerConfig &c);

    bool addAuxData(AuxDataStorage::AuxDataMap m);
    void setValidationMap(const ExperimentValidator::ValidationMap &m);
    bool validateItem(const QString key, const QVariant val);
//    void finalizeFtmwSnapshots(const FtmwConfig final);

#ifdef BC_LIF
    bool isLifWaiting() const;
    LifConfig lifConfig() const;
    void setLifEnabled(bool en = true);
    void setLifWaiting(bool wait);
    void setLifConfig(const LifConfig cfg);
    bool addLifWaveform(const LifTrace t);
#endif

#ifdef BC_MOTOR
    MotorScan motorScan() const;
    void setMotorEnabled(bool en = true);
    void setMotorScan(const MotorScan s);
    bool addMotorTrace(const QVector<double> d);
#endif

    bool initialize();
    void abort();
    bool canAutosave();
    QFuture<void> autosave();
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
    std::unique_ptr<FtmwConfig> pu_ftmwConfig;
    std::unique_ptr<AuxDataStorage> pu_auxData;
    std::unique_ptr<ExperimentValidator> pu_validator;

    //optional hardware data
    std::unique_ptr<IOBoardConfig> pu_iobCfg;
    std::unique_ptr<PulseGenConfig> pu_pGenCfg;
    std::unique_ptr<FlowConfig> pu_flowCfg;
    std::unique_ptr<PressureControllerConfig> pu_pcConfig;
    std::unique_ptr<TemperatureControllerConfig> pu_tcConfig;

    QString d_path;

#ifdef BC_LIF
    LifConfig d_lifCfg;
    bool d_waitForLifSet{false};
#endif

#ifdef BC_MOTOR
    MotorScan d_motorScan;
#endif

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

Q_DECLARE_METATYPE(Experiment)
Q_DECLARE_METATYPE(std::shared_ptr<Experiment>)

#endif // EXPERIMENT_H
