#ifndef HARDWAREMANAGER_H
#define HARDWAREMANAGER_H

#include <QObject>
#include <memory>
#include <functional>
#include <data/loghandler.h>
#include <data/storage/auxdatastorage.h>
#include <data/storage/settingsstorage.h>
#include <data/experiment/rfconfig.h>

#include <hardware/optional/flowcontroller/flowconfig.h>
#include <hardware/optional/pulsegenerator/pulsegenconfig.h>
#include <hardware/optional/pressurecontroller/pressurecontrollerconfig.h>
#include <hardware/optional/tempcontroller/temperaturecontrollerconfig.h>

#ifdef BC_LIF
#include <modules/lif/hardware/lifdigitizer/lifdigitizerconfig.h>
#endif

class HardwareObject;
class ClockManager;
class Experiment;

namespace BC::Key {
static const QString hw{"hardware"};
static const QString allHw{"instruments"};
}

class HardwareManager : public QObject, public SettingsStorage
{
    Q_OBJECT
public:
    explicit HardwareManager(QObject *parent = 0);
    ~HardwareManager();

    QString getHwName(const QString key);
    const std::set<QString> d_optHwTypes;

signals:
    void logMessage(QString,LogHandler::MessageCode = LogHandler::Normal);
    void statusMessage(QString,int=0);
    void hwInitializationComplete();

    void allHardwareConnected(bool);
    /*!
     * \brief Emitted when a connection is being tested from the communication dialog
     * \param QString The HardwareObject key
     * \param bool Whether connection was successful
     * \param QString Error message
     */
    void testComplete(QString,bool,QString);
    void beginAcquisition();
    void abortAcquisition();
    void experimentInitialized(std::shared_ptr<Experiment>);
    void endAcquisition();
    void auxData(AuxDataStorage::AuxDataMap);
    void validationData(AuxDataStorage::AuxDataMap);
    void rollingData(AuxDataStorage::AuxDataMap,QDateTime);

    void ftmwScopeShotAcquired(QByteArray);

    void clockFrequencyUpdate(RfConfig::ClockType, double);
    void allClocksReady(QHash<RfConfig::ClockType,RfConfig::ClockFreq>);

    void pGenSettingUpdate(QString,int,PulseGenConfig::Setting,QVariant);
    void pGenConfigUpdate(QString,PulseGenConfig);

    void flowUpdate(QString,int,double);
    void flowSetpointUpdate(QString,int,double);
    void gasPressureUpdate(QString,double);
    void gasPressureSetpointUpdate(QString,double);
    void gasPressureControlMode(QString,bool);

    void pressureControlReadOnly(bool);
    void pressureUpdate(double);
    void pressureSetpointUpdate(double);
    void pressureControlMode(bool);

    void temperatureUpdate(int,double);
    void temperatureEnableUpdate(int,bool);

#ifdef BC_LIF
    void lifScopeShotAcquired(QVector<qint8>);
    void lifSettingsComplete(bool success = true);
    void lifLaserPosUpdate(double);
    void lifConfigAcqStarted(LifDigitizerConfig);
    void lifLaserFlashlampUpdate(bool);
#endif

public slots:
    void initialize();

    /*!
     * \brief Records whether hardware connection was successful
     * \param obj A HardwareObject that was tested
     * \param success Whether communication was sucessful
     * \param msg Error message
     */
    void connectionResult(HardwareObject *obj, bool success, QString msg);

    /*!
     * \brief Sets hardware status in d_status to false, disables program
     * \param obj The object that failed.
     *
     * TODO: Consider generating an abort signal here
     */
    void hardwareFailure();

    void sleep(bool b);

    void initializeExperiment(std::shared_ptr<Experiment> exp);
    void experimentComplete();

    void testAll();
    void testObjectConnection(const QString type, const QString key);
    void updateObjectSettings(const QString key);
    QStringList getForbiddenKeys(const QString key) const;

    void getAuxData();

    QHash<RfConfig::ClockType,RfConfig::ClockFreq> getClocks();
    void configureClocks(QHash<RfConfig::ClockType,RfConfig::ClockFreq> clocks);
    void setClocks(QHash<RfConfig::ClockType,RfConfig::ClockFreq> clocks);

    void setPGenSetting(const QString key, int index, PulseGenConfig::Setting s, QVariant val);
    void setPGenConfig(const QString key, const PulseGenConfig &c);
    PulseGenConfig getPGenConfig(const QString key);

    void setFlowSetpoint(const QString key, int index, double val);
    void setFlowChannelName(const QString key, int index, QString name);
    void setGasPressureSetpoint(const QString key, double val);
    void setGasPressureControlMode(const QString key, bool en);
    FlowConfig getFlowConfig(const QString key);

    void setPressureSetpoint(double val);
    void setPressureControlMode(bool en);
    void openGateValve();
    void closeGateValve();
    PressureControllerConfig getPressureControllerConfig();

    void setTemperatureChannelEnabled(int ch, bool en);
    void setTemperatureChannelName(int ch, const QString name);
    TemperatureControllerConfig getTemperatureControllerConfig();

    void storeAllOptHw(Experiment *exp, std::map<QString,bool> hw = {});

#ifdef BC_LIF
    void setLifParameters(double delay, double pos);
    bool setPGenLifDelay(double d);
    void setLifLaserPos(double pos);
    void lifLaserSetComplete(double pos);
    void startLifConfigAcq(const LifDigitizerConfig &c);
    void stopLifConfigAcq();
    double lifLaserPos();
    bool lifLaserFlashlampEnabled();
    void setLifLaserFlashlampEnabled(bool en);
#endif

public:
    std::map<QString,QStringList> validationKeys() const;
    std::map<QString,QString> currentHardware() const;

private:
    std::size_t d_responseCount{0};
    void checkStatus();

    std::map<QString,HardwareObject*> d_hardwareMap;
    std::unique_ptr<ClockManager> pu_clockManager;

    template<class T>
    T* findHardware(const QString key) const {
        auto it = d_hardwareMap.find(key);
        return it == d_hardwareMap.end() ? nullptr : static_cast<T*>(it->second);
    }

};

#endif // HARDWAREMANAGER_H
