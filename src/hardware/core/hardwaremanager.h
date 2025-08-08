#ifndef HARDWAREMANAGER_H
#define HARDWAREMANAGER_H

#include <QObject>
#include <memory>
#include <functional>
#include <data/loghandler.h>
#include <data/storage/auxdatastorage.h>
#include <data/storage/settingsstorage.h>
#include <data/experiment/rfconfig.h>
#include <QMutex>

#include <data/experiment/hardware/optional/flowcontroller/flowconfig.h>
#include <data/experiment/hardware/optional/pulsegenerator/pulsegenconfig.h>
#include <data/experiment/hardware/optional/pressurecontroller/pressurecontrollerconfig.h>
#include <data/experiment/hardware/optional/tempcontroller/temperaturecontrollerconfig.h>
#include <data/experiment/hardware/optional/ioboard/ioboardconfig.h>

#include <data/lif/lifconfig.h>

class HardwareObject;
class ClockManager;
class Experiment;
class GpibController;

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
    
    // Static const access for thread-safe hardware resolution
    static const HardwareManager& constInstance();

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

    void pressureControlReadOnly(QString,bool);
    void pressureUpdate(QString,double);
    void pressureSetpointUpdate(QString,double);
    void pressureControlMode(QString,bool);

    void temperatureUpdate(QString,uint,double);
    void temperatureEnableUpdate(QString,uint,bool);

    void lifScopeShotAcquired(QVector<qint8>);
    void lifSettingsComplete(bool success = true);
    void lifLaserPosUpdate(double);
    void lifConfigAcqStarted();
    void lifLaserFlashlampUpdate(bool);

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
    void testObjectConnection(const QString hwKey);
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

    void setPressureSetpoint(const QString key, double val);
    void setPressureControlMode(const QString key, bool en);
    void openGateValve(const QString key);
    void closeGateValve(const QString key);
    PressureControllerConfig getPressureControllerConfig(const QString key);

    void setTemperatureChannelEnabled(const QString key, uint ch, bool en);
    void setTemperatureChannelName(const QString key, uint ch, const QString name);
    TemperatureControllerConfig getTemperatureControllerConfig(const QString key);

    IOBoardConfig getIOBoardConfig(const QString key);

    void storeAllOptHw(Experiment *exp, std::map<QString,bool> hw = {});

    void setLifParameters(double delay, double pos);
    bool setPGenLifDelay(double d);
    bool setLifLaserPos(double pos);
    void lifLaserSetComplete(double pos);
    void startLifConfigAcq(const LifConfig &c);
    void stopLifConfigAcq();
    double lifLaserPos();
    bool lifLaserFlashlampEnabled();
    void setLifLaserFlashlampEnabled(bool en);

public:
    std::map<QString,QStringList> validationKeys() const;
    std::map<QString,QString> currentHardware() const;
    
    // Thread-safe GPIB controller resolution with callback
    void resolveGpibController(const QString& controllerKey, std::function<void(GpibController*)> callback) const;

private:
    std::size_t d_responseCount{0};
    void checkStatus();
    
    // Phase 2.4.2: Constructor refactoring methods
    void createVirtualHardwareForCapabilityDiscovery();
    void setupHardwareObject(HardwareObject* obj);
    void finalizeInitialization();

    std::map<QString,HardwareObject*> d_hardwareMap;
    std::unique_ptr<ClockManager> pu_clockManager;
    
    // Static instance management for const access
    static HardwareManager* s_instance;
    
    // Mutex for thread-safe access to shared data
    mutable QMutex d_accessMutex;
    
    // Private helper for internal use
    template<class T>
    T* findHardware(const QString key) const {
        QMutexLocker locker(&d_accessMutex);
        auto it = d_hardwareMap.find(key);
        return it == d_hardwareMap.end() ? nullptr : static_cast<T*>(it->second);
    }

};

#endif // HARDWAREMANAGER_H
