#ifndef HARDWAREMANAGER_H
#define HARDWAREMANAGER_H

#include <QObject>
#include <memory>
#include <functional>
#include <atomic>
#include <data/loghandler.h>
#include <data/storage/auxdatastorage.h>
#include <data/storage/settingsstorage.h>
#include <data/experiment/rfconfig.h>
#include <QMutex>
#include <QReadWriteLock>

#include <data/experiment/hardware/optional/flowcontroller/flowconfig.h>
#include <data/experiment/hardware/optional/pulsegenerator/pulsegenconfig.h>
#include <data/experiment/hardware/optional/pressurecontroller/pressurecontrollerconfig.h>
#include <data/experiment/hardware/optional/tempcontroller/temperaturecontrollerconfig.h>
#include <data/experiment/hardware/optional/ioboard/ioboardconfig.h>

#include <data/lif/lifconfig.h>
#include <hardware/core/communication/communicationprotocol.h>

class HardwareObject;
class ClockManager;
class Experiment;
class GpibController;

namespace BC::Key {
inline constexpr QLatin1StringView hw{"hardware"};
inline constexpr QLatin1StringView allHw{"instruments"};
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
    std::set<QString> d_optHwTypes;

signals:
    void statusMessage(QString,int=0);
    void hwInitializationComplete();

    void allHardwareConnected(bool);
    /*!
     * \brief Unified signal for all connection status changes and test results
     * \param hwKey Hardware key (e.g., "FtmwScope.mainScope") 
     * \param success Whether connection was successful
     * \param msg Status or error message
     */
    void connectionResult(const QString& hwKey, bool success, const QString& msg);
    void profileDeleted(const QString& hwKey);
    
    // Task 3.3.8: Communication protocol management signals
    void hardwareCommunicationInfoReady(const QString& hwKey, CommunicationProtocol::CommType currentProtocol, 
                                       QVector<CommunicationProtocol::CommType> supportedProtocols, bool connected);
    void protocolSetResult(const QString& hwKey, bool success, const QString& msg);
    void gpibControllersAvailable(QStringList controllerKeys);
    
    void beginAcquisition();
    void abortAcquisition();
    void experimentInitialized(std::shared_ptr<Experiment>);
    void endAcquisition();
    void auxData(AuxDataStorage::AuxDataMap);
    void validationData(AuxDataStorage::AuxDataMap);
    void rollingData(AuxDataStorage::AuxDataMap,QDateTime);

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

    void pythonScriptReloadResult(const QString &hwKey, bool success, const QString &msg);

public slots:
    void initialize();

    /*!
     * \brief Records whether hardware connection was successful
     * \param hwKey Hardware key of the tested object
     * \param success Whether communication was successful
     * \param msg Error message
     */
    void handleConnectionResult(const QString& hwKey, bool success, const QString& msg);

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

    void storeAllOptHw(Experiment *exp, std::map<QString,bool,std::less<>> hw = {});

    void setLifParameters(double delay, double pos);
    bool setPGenLifDelay(double d);
    bool setLifLaserPos(double pos);
    void startLifConfigAcq(const LifConfig &c);
    void stopLifConfigAcq();
    double lifLaserPos();
    bool lifLaserFlashlampEnabled();
    void setLifLaserFlashlampEnabled(bool en);
    
    // Task 3.3.8: Communication protocol management API
    void getHardwareCommunicationInfo(const QString& hwKey);
    void setHardwareProtocol(const QString& hwKey, CommunicationProtocol::CommType protocol, const QString& gpibControllerKey = QString());
    void getActiveGpibControllers();
    
    // Hardware connection status queries
    bool allCriticalHardwareConnected() const;
    
    // Dynamic hardware synchronization
    void syncWithRuntimeConfig();
    
    // Phase 3.5.3: Library configuration integration
    bool applyVendorLibraryChanges();

    void reloadPythonScript(const QString &hwKey);

public:
    std::map<QString,QStringList,std::less<>> validationKeys() const;
    
    // Thread-safe GPIB controller resolution with callback
    void resolveGpibController(const QString& controllerKey, std::function<void(GpibController*)> callback) const;
    
    // Connection status tracking for UI
    bool connectionTestsInProgress() const { return d_connectionState.testsInProgress; }

private:
    // Connection test state management
    struct ConnectionTestState {
        std::atomic<size_t> responseCount{0};
        std::atomic<bool> testsInProgress{false};
        
        void reset() { responseCount = 0; testsInProgress = true; }
        void recordResponse() { responseCount++; }
        bool allResponded(size_t expected) const { return responseCount >= expected; }
        void markComplete() { testsInProgress = false; }
    };
    ConnectionTestState d_connectionState;

    void checkStatus();
    void initializeConnectionTesting();
    void resetConnectionTestState(); 
    void finalizeConnectionTesting();
    
    // Phase 2.4.2: Constructor refactoring methods
    void setupHardwareObject(HardwareObject* obj);

    // Phase 2.4.3: Runtime configuration integration
    HardwareObject* createSpecificHardware(const QString& type, const QString& implementation, const QString& label);
    
    // Phase 3.3: Dynamic hardware synchronization
    void removeHardwareInternal(const QString& hwKey);
    void addHardwareInternal(const QString& hwKey, const QString& implementation);
    void replaceHardwareInternal(const QString& hwKey, const QString& newImplementation);
    
    // Task 3.3.5: Synchronization orchestration
    std::vector<QString> findHardwareToRemove(const std::map<QString, QString, std::less<>>& targetHardware);
    std::vector<std::pair<QString, QString>> findHardwareToAdd(const std::map<QString, QString, std::less<>>& targetHardware);
    std::vector<std::pair<QString, QString>> findHardwareToReplace(const std::map<QString, QString, std::less<>>& targetHardware);
    void resolveGpibControllersForInstruments();
    void updateClockManager();
    
    // Phase 3.5.4: Library dependency tracking
    void addLibraryDependentHardwareToRecreation(const std::map<QString, QString, std::less<>>& targetHardware,
                                                std::vector<QString>& toRemove,
                                                std::vector<std::pair<QString, QString>>& toAdd,
                                                std::vector<std::pair<QString, QString>>& toReplace);
    
    // Connection tracking helpers for Task 3.3.2
    void storeConnection(const QString& hwKey, const QMetaObject::Connection& connection);
    void setupHardwareObjectWithTracking(HardwareObject* obj);
    void setupHardwareSpecificConnectionsWithTracking(HardwareObject* obj);
    void disconnectStoredConnections(const QString& hwKey);

    std::map<QString,HardwareObject*,std::less<>> d_hardwareMap;
    
    // Connection tracking infrastructure for Task 3.3.2
    std::map<QString, QVector<QMetaObject::Connection>, std::less<>> d_hardwareConnections;
    std::unique_ptr<ClockManager> pu_clockManager;
    
    // Static instance management for const access
    static HardwareManager* s_instance;
    
    // Multi-lock architecture for better concurrency and deadlock prevention
    mutable QReadWriteLock d_hardwareMapLock;  // Protects d_hardwareMap access
    mutable QMutex d_connectionStateLock;      // Protects d_connectionState access
    
    // Private helper for internal use
    template<class T>
    T* findHardware(const QString key) const {
        QReadLocker locker(&d_hardwareMapLock);
        auto it = d_hardwareMap.find(key);
        if (it == d_hardwareMap.end()) {
            return nullptr;
        }
        
        // Use safe casting instead of static_cast
        return qobject_cast<T*>(it->second);
    }

    // Utility function to find all hardware of a specific type using runtime configuration
    template<class T>
    QVector<T*> findHardwareByType() const {
        QReadLocker locker(&d_hardwareMapLock);
        
        QVector<T*> result;

        for (const auto& [key,hwObj] : d_hardwareMap) {
            QString hwType = BC::Key::parseKey(key).first;
            if (hwType == T::staticMetaObject.className()) {
                if (T* hw = qobject_cast<T*>(hwObj)) {  // Safe casting
                    result.append(hw);
                }
            }
        }
        
        return result;
    }

};

#endif // HARDWAREMANAGER_H
