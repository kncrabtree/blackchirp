#include <hardware/core/hardwaremanager.h>
#include <hardware/core/runtimehardwareconfig.h>
#include <hardware/core/hardwareregistry.h>
#include <hardware/core/hardwareprofilemanager.h>
#include <data/settings/hardwarekeys.h>
#include <data/bcglobals.h>

// Phase 3.5.3: Vendor library includes for library configuration integration
#include <hardware/library/spectrumlibrary.h>
#include <hardware/library/labjacklibrary.h>

#include <hardware/core/hardwareobject.h>
#include <hardware/python/pythonhardwarebase.h>
#include <hardware/core/clock/clockmanager.h>
#include <hardware/core/hw_h.h> // Generated at build time

#include <QThread>

#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#include <vector>

// Static instance for const access
HardwareManager* HardwareManager::s_instance = nullptr;

HardwareManager::HardwareManager(QObject *parent) : QObject(parent), SettingsStorage(BC::Key::hw),
    d_optHwTypes{QString(FlowController::staticMetaObject.className()),QString(IOBoard::staticMetaObject.className()),QString(PressureController::staticMetaObject.className()),QString(PulseGenerator::staticMetaObject.className()),QString(TemperatureController::staticMetaObject.className())}
{
    // Set static instance for const access
    s_instance = this;
    // Lock hardware map for entire initialization - no concurrency issues during startup
    QWriteLocker locker(&d_hardwareMapLock);
    
    // Initialize ClockManager
    pu_clockManager = std::make_unique<ClockManager>(this);
    connect(pu_clockManager.get(), &ClockManager::logMessage, this, &HardwareManager::logMessage);
    connect(pu_clockManager.get(), &ClockManager::clockFrequencyUpdate, this, &HardwareManager::clockFrequencyUpdate);

    // Phase 3.3.6: Clean constructor - all hardware creation now goes through dynamic system
    // HardwareManager starts with empty d_hardwareMap and will be populated via syncWithRuntimeConfig()
    emit logMessage("HardwareManager created. Hardware will be loaded from runtime configuration.", LogHandler::Normal);
}

HardwareManager::~HardwareManager()
{
    // Clear static instance
    s_instance = nullptr;
    
    //stop all threads
    for(auto &[key,obj] : d_hardwareMap)
    {
        Q_UNUSED(key)
        if(obj->d_threaded)
        {
            obj->thread()->quit();
            obj->thread()->wait();
        }
    }
}

QString HardwareManager::getHwName(const QString key)
{
    auto hw = findHardware<HardwareObject>(key);
    if(hw)
        return hw->d_key;

    return QString();
}

void HardwareManager::initialize()
{
    // Ensure system profiles exist and activate them for required types with no config
    HardwareProfileManager::instance().ensureSystemProfiles();
    RuntimeHardwareConfig::instance().activateMissingSystemProfiles();

    // Phase 3.3.6: Load hardware from runtime configuration before starting threads
    emit logMessage("Loading hardware configuration from runtime profiles...", LogHandler::Normal);
    syncWithRuntimeConfig();
    
    // Emit virtual instrument warnings. Hardware initialization and threading is
    // already handled by syncWithRuntimeConfig() via addHardwareInternal().
    for(auto it = d_hardwareMap.cbegin(); it != d_hardwareMap.cend(); ++it)
    {
        auto hw = it->second;
        if(hw->d_commType == CommunicationProtocol::Virtual)
            emit logMessage(QString("%1 is a virtual instrument. Be cautious about taking real measurements!")
                            .arg(hw->d_key),LogHandler::Warning);
    }

    emit hwInitializationComplete();
}

void HardwareManager::handleConnectionResult(const QString& hwKey, bool success, const QString& msg)
{
    // Find the hardware object for logging and connection management
    HardwareObject* obj = nullptr;
    {
        QReadLocker mapLocker(&d_hardwareMapLock);
        auto it = d_hardwareMap.find(hwKey);
        if (it != d_hardwareMap.end()) {
            obj = it->second;
        }
    }
    
    // Handle connection state separately with dedicated lock
    {
        QMutexLocker connLocker(&d_connectionStateLock);
        d_connectionState.recordResponse();
    }

    if (!obj) {
        // Hardware not found - emit signal anyway for UI feedback
        emit connectionResult(hwKey, false, QString("Hardware not found: %1").arg(hwKey));
        return;
    }

    if(success)
    {
        connect(obj,&HardwareObject::hardwareFailure,this,&HardwareManager::hardwareFailure,Qt::UniqueConnection);
        emit logMessage(obj->d_key + QString(": Connected successfully."));
    }
    else
    {
        disconnect(obj,&HardwareObject::hardwareFailure,this,&HardwareManager::hardwareFailure);
        LogHandler::MessageCode code = LogHandler::Error;
        if(!obj->d_critical)
            code = LogHandler::Warning;
        emit logMessage(obj->d_key + QString(": Connection failed!"),code);
        if(!msg.isEmpty())
            emit logMessage(msg,code);
    }

    // Emit unified connectionResult signal for both test results and connection changes
    emit connectionResult(hwKey, success, msg);
    checkStatus();
}

void HardwareManager::hardwareFailure()
{
    HardwareObject *obj = dynamic_cast<HardwareObject*>(sender());
    if(obj == nullptr)
        return;

    disconnect(obj,&HardwareObject::hardwareFailure,this,&HardwareManager::hardwareFailure);
    
    // Emit unified connectionResult signal (hardware failed = disconnected)
    emit connectionResult(obj->d_key, false, QString("Hardware failure"));

    if(obj->d_critical)
        emit abortAcquisition();

    checkStatus();
}

void HardwareManager::sleep(bool b)
{
    for(auto it = d_hardwareMap.cbegin(); it != d_hardwareMap.cend(); ++it)
    {
        auto obj = it->second;
        if(obj->isConnected())
            QMetaObject::invokeMethod(obj,[obj,b](){ obj->sleep(b); });
    }
}

void HardwareManager::initializeExperiment(std::shared_ptr<Experiment> exp)
{
    //do initialization
    bool success = pu_clockManager->prepareForExperiment(*exp);

    if(success) {
        for(auto it = d_hardwareMap.cbegin(); it != d_hardwareMap.cend(); ++it)
        {
            auto obj = it->second;
            if(obj->thread() != QThread::currentThread())
                QMetaObject::invokeMethod(obj,[obj,exp](){
                    return obj->hwPrepareForExperiment(*exp);
                },Qt::BlockingQueuedConnection,&success);
            else
                success = obj->hwPrepareForExperiment(*exp);

            if(!success)
            {
                emit logMessage(QString("Error initializing %1").arg(obj->d_key),LogHandler::Error);
                break;
            }
        }
    }

    exp->d_hardwareSuccess = success;

    if(exp->lifEnabled())
    {
        auto activeKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<LifLaser>();
        if (activeKeys.isEmpty()) {
            emit logMessage(QString("Could not perform LIF experiment because no LIF laser is configured."),LogHandler::Error);
            emit lifSettingsComplete(false);
            exp->d_hardwareSuccess = false;
        } else {
            auto ll = findHardware<LifLaser>(activeKeys.first());
            if(!ll)
            {
                emit logMessage(QString("Could not perform LIF experiment because no laser is available."),LogHandler::Error);
                emit lifSettingsComplete(false);
                exp->d_hardwareSuccess = false;
            }
        }
    }
    //any additional synchronous initialization can be performed here, before experimentInitialized() is emitted


    emit experimentInitialized(exp);

}

void HardwareManager::experimentComplete()
{
}

void HardwareManager::testAll()
{
    // Reset connection test state for current hardware set
    {
        QMutexLocker connLocker(&d_connectionStateLock);
        d_connectionState.reset();
    }
    
    QReadLocker mapLocker(&d_hardwareMapLock);
    for(auto it = d_hardwareMap.cbegin(); it != d_hardwareMap.cend(); ++it)
        QMetaObject::invokeMethod(it->second,&HardwareObject::bcTestConnection);

}

void HardwareManager::testObjectConnection(const QString hwKey)
{
    HardwareObject* obj = nullptr;
    {
        QReadLocker locker(&d_hardwareMapLock);
        auto it = d_hardwareMap.find(hwKey);
        if(it == d_hardwareMap.end()) {
            emit connectionResult(hwKey,false,QString("Device not found!"));
            return;
        }
        obj = it->second;
    }
    
    // Call outside the lock to avoid holding mutex during potentially long operation
    QMetaObject::invokeMethod(obj,&HardwareObject::bcTestConnection);
}

void HardwareManager::updateObjectSettings(const QString key)
{
    auto obj = findHardware<HardwareObject>(key);
    if(obj)
        QMetaObject::invokeMethod(obj,&HardwareObject::bcReadSettings);
}


void HardwareManager::getAuxData()
{
    for(auto it = d_hardwareMap.cbegin(); it != d_hardwareMap.cend(); ++it)
    {
        auto obj = it->second;
        QMetaObject::invokeMethod(obj,&HardwareObject::bcReadAuxData);
    }
}

QHash<RfConfig::ClockType, RfConfig::ClockFreq> HardwareManager::getClocks()
{
    return pu_clockManager->getCurrentClocks();
}

void HardwareManager::configureClocks(QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks)
{
    pu_clockManager->configureClocks(clocks);
}

void HardwareManager::setClocks(QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks)
{
    auto ftmwKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<FtmwScope>();
    auto fsc = ftmwKeys.isEmpty() ? nullptr : findHardware<FtmwScope>(ftmwKeys.first());

    // Gate the digitizer so no waveforms are emitted while clock frequencies change
    if(fsc)
    {
        if(fsc->thread() == QThread::currentThread())
            fsc->setAcquisitionGated(true);
        else
            QMetaObject::invokeMethod(fsc,[fsc](){ fsc->setAcquisitionGated(true); },
                                      Qt::BlockingQueuedConnection);
    }

    for(auto it = clocks.begin(); it != clocks.end(); ++it)
        it.value().desiredFreqMHz = pu_clockManager->setClockFrequency(it.key(),it.value().desiredFreqMHz);

    // Flush any scope-internal buffered waveform from the old frequency, then ungate
    if(fsc)
    {
        if(fsc->thread() == QThread::currentThread())
        {
            fsc->flushAcquisitionBuffer();
            fsc->setAcquisitionGated(false);
        }
        else
            QMetaObject::invokeMethod(fsc,[fsc](){
                fsc->flushAcquisitionBuffer();
                fsc->setAcquisitionGated(false);
            }, Qt::BlockingQueuedConnection);
    }

    emit allClocksReady(clocks);
}

void HardwareManager::setPGenSetting(const QString key, int index, PulseGenConfig::Setting s, QVariant val)
{
    auto pGen = findHardware<PulseGenerator>(key);
    if(pGen)
    {
        switch(s)
        {
        case PulseGenConfig::RepRateSetting:
            QMetaObject::invokeMethod(pGen,[pGen,val](){ pGen->setRepRate(val.toDouble());});
            break;
        case PulseGenConfig::PGenEnabledSetting:
            QMetaObject::invokeMethod(pGen,[pGen,val](){ pGen->setPulseEnabled(val.toBool());});
            break;
        case PulseGenConfig::PGenModeSetting:
            QMetaObject::invokeMethod(pGen,[pGen,val](){ pGen->setPulseMode(val.value<PulseGenConfig::PGenMode>());});
            break;
        default:
            QMetaObject::invokeMethod(pGen,[pGen,index,s,val](){ pGen->setPGenSetting(index,s,val);});
            break;
        }
    }

}

void HardwareManager::setPGenConfig(const QString key, const PulseGenConfig &c)
{
    auto pGen = findHardware<PulseGenerator>(key);
    if(pGen)
        QMetaObject::invokeMethod(pGen,[pGen,c](){ pGen->setAll(c); });
}

PulseGenConfig HardwareManager::getPGenConfig(const QString key)
{
    PulseGenConfig out(key);
    auto pg = findHardware<PulseGenerator>(key);
    if(pg)
    {
        if(pg->thread() != QThread::currentThread())
            QMetaObject::invokeMethod(pg,&PulseGenerator::config,Qt::BlockingQueuedConnection,&out);
        else
            out = pg->config();
    }

    return out;
}

void HardwareManager::setFlowSetpoint(const QString key, int index, double val)
{
    auto flow = findHardware<FlowController>(key);
    if(flow)
        QMetaObject::invokeMethod(flow,[flow,index,val](){flow->setFlowSetpoint(index,val);});
}

void HardwareManager::setFlowChannelName(const QString key, int index, QString name)
{
    auto flow = findHardware<FlowController>(key);
    if(flow)
        QMetaObject::invokeMethod(flow,[flow,index,name](){flow->setChannelName(index,name);});
}

void HardwareManager::setGasPressureSetpoint(const QString key, double val)
{
    auto flow = findHardware<FlowController>(key);
    if(flow)
        QMetaObject::invokeMethod(flow,[flow,val](){flow->setPressureSetpoint(val);});
}

void HardwareManager::setGasPressureControlMode(const QString key, bool en)
{
    auto flow = findHardware<FlowController>(key);
    if(flow)
        QMetaObject::invokeMethod(flow,[flow,en](){flow->setPressureControlMode(en);});
}

FlowConfig HardwareManager::getFlowConfig(const QString key)
{
    FlowConfig out(key);
    auto fc = findHardware<FlowController>(key);
    if(fc)
    {
        if(fc->thread() != QThread::currentThread())
            QMetaObject::invokeMethod(fc,&FlowController::config,Qt::BlockingQueuedConnection,&out);
        else
            out = fc->config();
    }

    return out;
}

std::map<QString, QStringList> HardwareManager::validationKeys() const
{
    QReadLocker locker(&d_hardwareMapLock);
    std::map<QString, QStringList> out;
    for(auto &[key,obj] : d_hardwareMap)
        out.insert_or_assign(key,obj->validationKeys());

    return out;
}



void HardwareManager::setPressureSetpoint(const QString key, double val)
{
    auto pc = findHardware<PressureController>(key);
    if(pc)
        QMetaObject::invokeMethod(pc,[pc,val](){pc->setPressureSetpoint(val);});
}

void HardwareManager::setPressureControlMode(const QString key, bool en)
{
    auto pc = findHardware<PressureController>(key);
    if(pc)
        QMetaObject::invokeMethod(pc,[pc,en](){pc->setPressureControlMode(en);});
}

void HardwareManager::openGateValve(const QString key)
{
    auto pc = findHardware<PressureController>(key);
    if(pc)
        QMetaObject::invokeMethod(pc,&PressureController::openGateValve);
}

void HardwareManager::closeGateValve(const QString key)
{
    auto pc = findHardware<PressureController>(key);
    if(pc)
        QMetaObject::invokeMethod(pc,&PressureController::closeGateValve);
}

PressureControllerConfig HardwareManager::getPressureControllerConfig(const QString key)
{
    PressureControllerConfig out(key);
    auto pc = findHardware<PressureController>(key);
    if(pc)
    {
        if(pc->thread() != QThread::currentThread())
            QMetaObject::invokeMethod(pc,&PressureController::getConfig,Qt::BlockingQueuedConnection,&out);
        else
            out = pc->getConfig();
    }

    return out;
}


void HardwareManager::setTemperatureChannelEnabled(const QString key, uint ch, bool en)
{
    auto tc = findHardware<TemperatureController>(key);
    if(tc)
        QMetaObject::invokeMethod(tc,[tc,ch,en](){ tc->setChannelEnabled(ch,en);});
}

void HardwareManager::setTemperatureChannelName(const QString key, uint ch, const QString name)
{
    auto tc = findHardware<TemperatureController>(key);
    if(tc)
        QMetaObject::invokeMethod(tc,[tc,ch,name](){ tc->setChannelName(ch,name);});
}

TemperatureControllerConfig HardwareManager::getTemperatureControllerConfig(const QString key)
{
    TemperatureControllerConfig out(key);
    auto tc = findHardware<TemperatureController>(key);
    if(tc)
    {
        if(tc->thread() != QThread::currentThread())
            QMetaObject::invokeMethod(tc,&TemperatureController::getConfig,Qt::BlockingQueuedConnection,&out);
        else
            out = tc->getConfig();
    }

    return out;
}

IOBoardConfig HardwareManager::getIOBoardConfig(const QString key)
{
    IOBoardConfig out(key);
    auto iob = findHardware<IOBoard>(key);
    if(iob)
    {
        if(iob->thread() != QThread::currentThread())
            QMetaObject::invokeMethod(iob,&IOBoard::getConfig,Qt::BlockingQueuedConnection,&out);
        else
            out = iob->getConfig();
    }

    return out;

}

void HardwareManager::storeAllOptHw(Experiment *exp, std::map<QString, bool> hw)
{
    // TODO: This nested if/else pattern could be improved - consider refactoring to use
    // a dispatch table or visitor pattern to reduce complexity and improve maintainability
    for(auto const &[hwKey,_] : d_hardwareMap)
    {
        auto [type, label] = BC::Key::parseKey(hwKey);

        if(d_optHwTypes.find(type) == d_optHwTypes.end() || label.isEmpty())
            continue;

        bool read = true;
        auto it = hw.find(hwKey);
        if(it != hw.end())
            read = it->second;

        if(read)
        {
            if(type == QString(PulseGenerator::staticMetaObject.className()))
                exp->addOptHwConfig(getPGenConfig(hwKey));
            else if(type == QString(FlowController::staticMetaObject.className()))
                exp->addOptHwConfig(getFlowConfig(hwKey));
            else if(type == QString(TemperatureController::staticMetaObject.className()))
                exp->addOptHwConfig(getTemperatureControllerConfig(hwKey));
            else if(type == QString(PressureController::staticMetaObject.className()))
                exp->addOptHwConfig(getPressureControllerConfig(hwKey));
            else if(type == QString(IOBoard::staticMetaObject.className()))
                exp->addOptHwConfig(getIOBoardConfig(hwKey));
        }
    }
}

void HardwareManager::checkStatus()
{
    // Connection state operations use dedicated lock
    size_t hardwareMapSize = 0;
    {
        QReadLocker mapLocker(&d_hardwareMapLock);
        hardwareMapSize = d_hardwareMap.size();
    }
    
    {
        QMutexLocker connLocker(&d_connectionStateLock);
        //gotta wait until all instruments have responded
        if(!d_connectionState.allResponded(hardwareMapSize))
            return;
        // All hardware has responded, tests are complete
        d_connectionState.markComplete();
    }
    
    bool success = true;
    {
        QReadLocker mapLocker(&d_hardwareMapLock);
        for(auto &[key,obj] : d_hardwareMap)
        {
            // Handle critical vs non-critical hardware distinction dynamically
            // using individual obj->isConnected() and obj->d_critical states  
            if(!obj->isConnected() && obj->d_critical)
                success = false;
        }
    }

    emit allHardwareConnected(success);
}

void HardwareManager::initializeConnectionTesting()
{
    QMutexLocker locker(&d_connectionStateLock);
    d_connectionState.reset();
}

void HardwareManager::resetConnectionTestState()
{
    QMutexLocker locker(&d_connectionStateLock);
    d_connectionState.responseCount = 0;
    d_connectionState.testsInProgress = false;
}

void HardwareManager::finalizeConnectionTesting()
{
    QMutexLocker locker(&d_connectionStateLock);
    d_connectionState.markComplete();
}

void HardwareManager::setLifParameters(double delay, double pos)
{
    auto activeKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<LifScope>();
    auto lsc = findHardware<LifScope>(activeKeys.first());

    // Gate the digitizer so no waveforms are emitted while hardware parameters change
    if(lsc)
    {
        if(lsc->thread() == QThread::currentThread())
            lsc->setAcquisitionGated(true);
        else
            QMetaObject::invokeMethod(lsc,[lsc](){ lsc->setAcquisitionGated(true); },
                                      Qt::BlockingQueuedConnection);
    }

    bool success = true;
    success &= setLifLaserPos(pos);
    if(success)
        success &= setPGenLifDelay(delay);

    // Flush any scope-internal buffered waveform from the old trigger, then ungate
    if(lsc)
    {
        if(lsc->thread() == QThread::currentThread())
        {
            lsc->flushAcquisitionBuffer();
            lsc->setAcquisitionGated(false);
        }
        else
            QMetaObject::invokeMethod(lsc,[lsc](){
                lsc->flushAcquisitionBuffer();
                lsc->setAcquisitionGated(false);
            }, Qt::BlockingQueuedConnection);
    }

    emit lifSettingsComplete(success);
}

bool HardwareManager::setPGenLifDelay(double d)
{
    // Check for pulse generator availability
    auto activeKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<PulseGenerator>();
    if (activeKeys.isEmpty()) {
        emit logMessage(QString("Could not set LIF delay because no pulse generator is available."), LogHandler::Error);
        return false;
    }

    bool out = true;
    for(const auto& key : activeKeys)
    {
        auto pGen = findHardware<PulseGenerator>(key);

        if(pGen->thread() == QThread::currentThread())
            out &= pGen->setLifDelay(d);
        else
            QMetaObject::invokeMethod(pGen,[pGen,d](){ return pGen->setLifDelay(d); },Qt::BlockingQueuedConnection,&out);
    }

    return out;
}

bool HardwareManager::setLifLaserPos(double pos)
{
    auto activeKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<LifLaser>();
    if (activeKeys.isEmpty()) {
        emit logMessage("Could not set LIF Laser position because no laser is configured.", LogHandler::Error);
        return false;
    }
    
    auto ll = findHardware<LifLaser>(activeKeys.first());
    if(!ll)
    {
        emit logMessage(QString("Could not set LIF Laser position because no laser is available."),LogHandler::Error);
        return false;
    }

    double newPos = -1.0;
    if(ll->thread() == QThread::currentThread())
        newPos = ll->setPosition(pos);
    else
        QMetaObject::invokeMethod(ll,[ll,pos](){ return ll->setPosition(pos); },Qt::BlockingQueuedConnection,&newPos);

    return newPos >= 0.0;
}

void HardwareManager::startLifConfigAcq(const LifConfig &c)
{
    auto activeKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<LifScope>();
    if (activeKeys.isEmpty()) {
        emit logMessage("Could not initialize LIF acquisition because no LIF digitizer is configured.", LogHandler::Error);
        return;
    }
    
    auto ld = findHardware<LifScope>(activeKeys.first());
    if(!ld)
    {
        emit logMessage("Could not initialize LIF acquisition because no digitizer was found.",LogHandler::Error);
        return;
    }

    if(ld->thread() == QThread::currentThread())
        ld->startConfigurationAcquisition(c);
    else
        QMetaObject::invokeMethod(ld,[ld,c](){ ld->startConfigurationAcquisition(c); });
}

void HardwareManager::stopLifConfigAcq()
{
    auto activeKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<LifScope>();
    if (activeKeys.isEmpty()) {
        emit logMessage("Could not stop LIF acquisition because no LIF digitizer is configured.", LogHandler::Error);
        return;
    }
    
    auto ld = findHardware<LifScope>(activeKeys.first());
    if(!ld)
    {
        emit logMessage("Could not stop LIF acquisition because no digitizer was found.",LogHandler::Error);
        return;
    }

    if(ld->thread() == QThread::currentThread())
        ld->endAcquisition();
    else
        QMetaObject::invokeMethod(ld,&LifScope::endAcquisition);
}

double HardwareManager::lifLaserPos()
{
    auto activeKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<LifLaser>();
    if (activeKeys.isEmpty()) {
        emit logMessage("Could not read LIF Laser position because no laser is configured.", LogHandler::Error);
        return -1.0;
    }
    
    auto ll = findHardware<LifLaser>(activeKeys.first());
    if(!ll)
    {
        emit logMessage(QString("Could not read LIF Laser position because no laser is available."),LogHandler::Error);
        return -1.0;
    }

    if(ll->thread() == QThread::currentThread())
        return ll->readPosition();

    double out;
    QMetaObject::invokeMethod(ll,[ll](){ return ll->readPosition(); },Qt::BlockingQueuedConnection,&out);
    return out;
}

bool HardwareManager::lifLaserFlashlampEnabled()
{
    auto activeKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<LifLaser>();
    if (activeKeys.isEmpty()) {
        emit logMessage("Could not read LIF Laser flashlamp status because no laser is configured.", LogHandler::Error);
        return false;
    }
    
    auto ll = findHardware<LifLaser>(activeKeys.first());
    if(!ll)
    {
        emit logMessage(QString("Could not read LIF Laser flashlamp status because no laser is available."),LogHandler::Error);
        return false;
    }

    if(ll->thread() == QThread::currentThread())
        return ll->readFlashLamp();

    bool out;
    QMetaObject::invokeMethod(ll,[ll](){ return ll->readFlashLamp(); },Qt::BlockingQueuedConnection,&out);
    return out;
}

void HardwareManager::setLifLaserFlashlampEnabled(bool en)
{
    auto activeKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<LifLaser>();
    if (activeKeys.isEmpty()) {
        emit logMessage("Could not set LIF Laser flashlamp status because no laser is configured.", LogHandler::Error);
        return;
    }
    
    auto ll = findHardware<LifLaser>(activeKeys.first());
    if(!ll)
    {
        emit logMessage(QString("Could not read LIF Laser flashlamp status because no laser is available."),LogHandler::Error);
        return;
    }

    if(ll->thread() == QThread::currentThread())
    {
        ll->setFlashLamp(en);
        return;
    }

    QMetaObject::invokeMethod(ll,[ll,en](){ ll->setFlashLamp(en); },Qt::BlockingQueuedConnection);

}

const HardwareManager& HardwareManager::constInstance()
{
    if (!s_instance) {
        throw std::runtime_error("HardwareManager instance not initialized");
    }
    return *s_instance;
}

void HardwareManager::resolveGpibController(const QString& controllerKey, std::function<void(GpibController*)> callback) const
{
    // Use read lock for safe hardware map access
    QReadLocker locker(&d_hardwareMapLock);
    
    GpibController* controller = nullptr;
    auto it = d_hardwareMap.find(controllerKey);
    if (it != d_hardwareMap.end()) {
        controller = static_cast<GpibController*>(it->second);
    }
    
    // Call the callback with the resolved controller (or nullptr if not found)
    callback(controller);
}


void HardwareManager::setupHardwareObject(HardwareObject* obj)
{
    // Common signal connections for all hardware objects
    connect(obj, &HardwareObject::logMessage, [this, obj](QString msg, LogHandler::MessageCode mc){
        emit logMessage(QString("%1: %2").arg(obj->d_key).arg(msg), mc);
    });
    connect(obj, &HardwareObject::connected, [obj, this](bool success, QString msg){
        handleConnectionResult(obj->d_key, success, msg);
    });
    connect(obj, &HardwareObject::auxDataRead, [obj, this](AuxDataStorage::AuxDataMap m){
        AuxDataStorage::AuxDataMap out;
        for(auto it = m.cbegin(); it != m.cend(); ++it)
            out.insert({AuxDataStorage::makeKey(obj->d_key, it->first), it->second});
        emit auxData(out);
    });
    connect(obj, &HardwareObject::auxDataRead, [obj, this](AuxDataStorage::AuxDataMap m){
        AuxDataStorage::AuxDataMap out;
        for(auto it = m.cbegin(); it != m.cend(); ++it)
            out.insert({AuxDataStorage::makeKey(obj->d_key, it->first), it->second});
        emit validationData(out);
    });
    connect(obj, &HardwareObject::rollingDataRead, [obj, this](AuxDataStorage::AuxDataMap m){
        AuxDataStorage::AuxDataMap out;
        for(auto it = m.cbegin(); it != m.cend(); ++it)
            out.insert({AuxDataStorage::makeKey(obj->d_key, it->first), it->second});
        emit rollingData(out, QDateTime::currentDateTime());
    });
    connect(this, &HardwareManager::beginAcquisition, obj, &HardwareObject::beginAcquisition);
    connect(this, &HardwareManager::endAcquisition, obj, &HardwareObject::endAcquisition);
}

// Phase 2.4.3: Runtime configuration integration methods

HardwareObject* HardwareManager::createSpecificHardware(const QString& type, const QString& implementation, const QString& label)
{
    // Use HardwareRegistry to create hardware dynamically
    HardwareObject* hwObj = HardwareRegistry::instance().createHardware(type, implementation, label);
    
    if (!hwObj) {
        emit logMessage(QString("Failed to create hardware: type=%1, implementation=%2, label=%3")
                       .arg(type, implementation, label), LogHandler::Error);
        return nullptr;
    }
    
    // Set up the hardware object with common signal connections
    setupHardwareObject(hwObj);
    
    emit logMessage(QString("Successfully created hardware: %1")
                   .arg(hwObj->d_key), LogHandler::Normal);
    
    return hwObj;
}

// Phase 3.3: Dynamic hardware synchronization methods

void HardwareManager::removeHardwareInternal(const QString& hwKey)
{
    HardwareObject* obj = nullptr;
    
    // Find and remove from map under write lock protection
    {
        QWriteLocker locker(&d_hardwareMapLock);
        auto it = d_hardwareMap.find(hwKey);
        if (it == d_hardwareMap.end()) {
            emit logMessage(QString("Hardware removal failed: Hardware key '%1' not found").arg(hwKey), LogHandler::Warning);
            return;
        }
        
        obj = it->second;
        d_hardwareMap.erase(it);
    }
    
    if (!obj) {
        return;
    }
    
    emit logMessage(QString("Removing hardware: %1").arg(obj->d_key), LogHandler::Normal);
    
    // Step 2b: Use stored connection information for robust disconnection
    disconnectStoredConnections(hwKey);
    
    // Thread cleanup for threaded hardware objects (based on destructor pattern lines 42-50)
    if (obj->d_threaded) {
        QThread* thread = obj->thread();
        if (thread && thread != this->thread()) {
            // Stop the thread gracefully
            thread->quit();
            if (!thread->wait(5000)) { // 5 second timeout
                emit logMessage(QString("Warning: Thread for hardware %1 did not stop gracefully, terminating").arg(obj->d_key), LogHandler::Warning);
                thread->terminate();
                thread->wait(1000); // Give it 1 more second after terminate
            }
            // Delete the thread
            thread->deleteLater();
        }
    }
    
    // Only purge settings if the profile was permanently deleted (not just deactivated).
    // purgeSettings() sets d_discard=true, preventing ~SettingsStorage() from re-writing
    // the cleared settings after deleteLater() is processed by the event loop.
    auto [hwType, hwLabel] = BC::Key::parseKey(hwKey);
    if (!HardwareProfileManager::instance().profileExists(hwType, hwLabel)) {
        obj->purgeSettings();
        SettingsStorage::purgeGroupsBySuffix(hwKey);
        emit profileDeleted(hwKey);
    }

    // Clean up the hardware object
    obj->deleteLater();
    
    // Emit unified connectionResult signal (hardware removed = disconnected)  
    emit connectionResult(hwKey, false, QString("Hardware removed"));
    
    emit logMessage(QString("Successfully removed hardware: %1").arg(hwKey), LogHandler::Normal);
}

// Phase 3.3.2: Connection tracking infrastructure implementation

void HardwareManager::storeConnection(const QString& hwKey, const QMetaObject::Connection& connection)
{
    d_hardwareConnections[hwKey].append(connection);
}

void HardwareManager::setupHardwareObjectWithTracking(HardwareObject* obj)
{
    QString hwKey = obj->d_key;
    
    // Clear any existing connections for this hardware key (safety measure)
    d_hardwareConnections[hwKey].clear();
    
    // Common signal connections for all hardware objects - store each connection
    storeConnection(hwKey, connect(obj, &HardwareObject::logMessage, [this, obj](QString msg, LogHandler::MessageCode mc){
        emit logMessage(QString("%1: %2").arg(obj->d_key).arg(msg), mc);
    }));
    
    storeConnection(hwKey, connect(obj, &HardwareObject::connected, [obj, this](bool success, QString msg){
        handleConnectionResult(obj->d_key, success, msg);
    }));
    
    storeConnection(hwKey, connect(obj, &HardwareObject::auxDataRead, [obj, this](AuxDataStorage::AuxDataMap m){
        AuxDataStorage::AuxDataMap out;
        for(auto it = m.cbegin(); it != m.cend(); ++it)
            out.insert({AuxDataStorage::makeKey(obj->d_key, it->first), it->second});
        emit auxData(out);
    }));
    
    storeConnection(hwKey, connect(obj, &HardwareObject::auxDataRead, [obj, this](AuxDataStorage::AuxDataMap m){
        AuxDataStorage::AuxDataMap out;
        for(auto it = m.cbegin(); it != m.cend(); ++it)
            out.insert({AuxDataStorage::makeKey(obj->d_key, it->first), it->second});
        emit validationData(out);
    }));
    
    storeConnection(hwKey, connect(obj, &HardwareObject::rollingDataRead, [obj, this](AuxDataStorage::AuxDataMap m){
        AuxDataStorage::AuxDataMap out;
        for(auto it = m.cbegin(); it != m.cend(); ++it)
            out.insert({AuxDataStorage::makeKey(obj->d_key, it->first), it->second});
        emit rollingData(out, QDateTime::currentDateTime());
    }));
    
    storeConnection(hwKey, connect(this, &HardwareManager::beginAcquisition, obj, &HardwareObject::beginAcquisition));
    storeConnection(hwKey, connect(this, &HardwareManager::endAcquisition, obj, &HardwareObject::endAcquisition));
}

void HardwareManager::setupHardwareSpecificConnectionsWithTracking(HardwareObject* obj)
{
    QString hwKey = obj->d_key;
    
    // Setup hardware-specific signal connections and store each connection
    if (auto pGen = qobject_cast<PulseGenerator*>(obj)) {
        QString k = pGen->d_key;
        storeConnection(hwKey, connect(pGen, &PulseGenerator::settingUpdate, [this, k](const int ch, const PulseGenConfig::Setting set, const QVariant val){
            emit pGenSettingUpdate(k, ch, set, val);
        }));
        storeConnection(hwKey, connect(pGen, &PulseGenerator::configUpdate, [this, k](const PulseGenConfig cfg){
            emit pGenConfigUpdate(k, cfg);
        }));
    }
    else if (auto flowController = qobject_cast<FlowController*>(obj)) {
        QString k = flowController->d_key;
        storeConnection(hwKey, connect(flowController, &FlowController::flowUpdate, [this, k](int i, double d){
            emit flowUpdate(k, i, d);
        }));
        storeConnection(hwKey, connect(flowController, &FlowController::flowSetpointUpdate, [this, k](int i, double d){
            emit flowSetpointUpdate(k, i, d);
        }));
        storeConnection(hwKey, connect(flowController, &FlowController::pressureUpdate, [this, k](double d){
            emit gasPressureUpdate(k, d);
        }));
        storeConnection(hwKey, connect(flowController, &FlowController::pressureSetpointUpdate, [this, k](double d){
           emit gasPressureSetpointUpdate(k, d);
        }));
        storeConnection(hwKey, connect(flowController, &FlowController::pressureControlMode, [this, k](bool b){
            emit gasPressureControlMode(k, b);
        }));
    }
    else if (auto pressureController = qobject_cast<PressureController*>(obj)) {
        QString k = pressureController->d_key;
        storeConnection(hwKey, connect(pressureController, &PressureController::pressureUpdate, this, [this, k](double d){
            emit pressureUpdate(k, d);
        }));
        storeConnection(hwKey, connect(pressureController, &PressureController::pressureSetpointUpdate, this, [this, k](double d){
            emit pressureSetpointUpdate(k, d);
        }));
        storeConnection(hwKey, connect(pressureController, &PressureController::pressureControlMode, this, [this, k](bool b){
            emit pressureControlMode(k, b);
        }));
    }
    else if (auto tempController = qobject_cast<TemperatureController*>(obj)) {
        QString k = tempController->d_key;
        storeConnection(hwKey, connect(tempController, &TemperatureController::channelEnableUpdate, this, [this, k](uint i, bool en){
            emit temperatureEnableUpdate(k, i, en);
        }));
        storeConnection(hwKey, connect(tempController, &TemperatureController::temperatureUpdate, this, [this, k](uint i, double t) {
            emit temperatureUpdate(k, i, t);
        }));
    }
    else if (auto lifScope = qobject_cast<LifScope*>(obj)) {
        storeConnection(hwKey, connect(lifScope, &LifScope::waveformRead, this, &HardwareManager::lifScopeShotAcquired));
        storeConnection(hwKey, connect(lifScope, &LifScope::configAcqComplete, this, &HardwareManager::lifConfigAcqStarted));
    }
    else if (auto lifLaser = qobject_cast<LifLaser*>(obj)) {
        storeConnection(hwKey, connect(lifLaser, &LifLaser::laserPosUpdate, this, &HardwareManager::lifLaserPosUpdate));
        storeConnection(hwKey, connect(lifLaser, &LifLaser::laserFlashlampUpdate, this, &HardwareManager::lifLaserFlashlampUpdate));
    }
}

void HardwareManager::disconnectStoredConnections(const QString& hwKey)
{
    auto it = d_hardwareConnections.find(hwKey);
    if (it != d_hardwareConnections.end()) {
        // Disconnect all stored connections for this hardware
        for (const auto& connection : it->second) {
            QObject::disconnect(connection);
        }
        // Remove the connection list for this hardware
        d_hardwareConnections.erase(it);
    }
}

void HardwareManager::addHardwareInternal(const QString& hwKey, const QString& implementation)
{
    QWriteLocker locker(&d_hardwareMapLock);
    
    // Parse the hardware key to get type and label
    auto [hardwareType, label] = BC::Key::parseKey(hwKey);
    
    emit logMessage(QString("Adding hardware: %1 (type=%2, implementation=%3, label=%4)")
                   .arg(hwKey, hardwareType, implementation, label), LogHandler::Normal);
    
    // Check if hardware already exists
    if (d_hardwareMap.find(hwKey) != d_hardwareMap.end()) {
        emit logMessage(QString("Hardware addition failed: Hardware key '%1' already exists").arg(hwKey), LogHandler::Warning);
        return;
    }
    
    // Reset connection test state for dynamic hardware changes
    locker.unlock();
    resetConnectionTestState();
    locker.relock();
    
    // Create hardware using HardwareRegistry
    HardwareObject* hwObj = HardwareRegistry::instance().createHardware(hardwareType, implementation, label);
    
    if (!hwObj) {
        QString errorMsg = QString("Hardware creation failed: Unable to create hardware (type=%1, implementation=%2, label=%3). Implementation may not be registered or factory failed.")
                          .arg(hardwareType, implementation, label);
        emit logMessage(errorMsg, LogHandler::Error);
        
        // Remove from RuntimeHardwareConfig on critical error
        auto& runtimeConfig = RuntimeHardwareConfig::instance();
        runtimeConfig.removeHardwareSelection(hardwareType, label);
        emit logMessage(QString("Removed failed hardware from configuration: %1.%2").arg(hardwareType, label), LogHandler::Warning);
        return;
    }
    
    // Validate the created hardware object
    if (hwObj->d_key != hwKey) {
        QString errorMsg = QString("Hardware creation failed: Key mismatch (expected=%1, actual=%2)").arg(hwKey, hwObj->d_key);
        emit logMessage(errorMsg, LogHandler::Error);
        
        // Clean up the created object and remove from config
        hwObj->deleteLater();
        auto& runtimeConfig = RuntimeHardwareConfig::instance();
        runtimeConfig.removeHardwareSelection(hardwareType, label);
        emit logMessage(QString("Removed failed hardware from configuration: %1.%2").arg(hardwareType, label), LogHandler::Warning);
        return;
    }
    
    try {
        // Add to hardware map early so signal connections can find it
        d_hardwareMap.emplace(hwKey, hwObj);
        
        // Set up all signal connections with tracking
        setupHardwareObjectWithTracking(hwObj);
        setupHardwareSpecificConnectionsWithTracking(hwObj);
        
        // Apply threading override from RuntimeHardwareConfig (if stored by user)
        // Falls back to the type-level default set in the intermediate class constructor
        auto threadedOverride = RuntimeHardwareConfig::constInstance().getThreaded(hwKey);
        if (threadedOverride.has_value())
            hwObj->d_threaded = *threadedOverride;

        // Handle threading setup
        if (hwObj->d_threaded) {
            auto thread = new QThread(this);
            thread->setObjectName(hwObj->d_key + "Thread");
            hwObj->moveToThread(thread);
            storeConnection(hwKey, connect(thread, &QThread::started, hwObj, &HardwareObject::bcInitInstrument));
            
            // Start the thread
            thread->start();
            
            emit logMessage(QString("Started thread for hardware: %1").arg(hwObj->d_key), LogHandler::Normal);
        } else {
            hwObj->setParent(this);
            // Initialize non-threaded hardware directly
            QMetaObject::invokeMethod(hwObj, &HardwareObject::bcInitInstrument);
        }
        
        // Note: Connection testing is deferred until all hardware changes are complete
        // to avoid GPIB controller resolution issues during dynamic creation
        emit logMessage(QString("Hardware created and initialized: %1 (connection testing deferred)").arg(hwObj->d_key), LogHandler::Normal);

        emit logMessage(QString("Successfully added hardware: %1").arg(hwKey), LogHandler::Normal);
        
    } catch (const std::exception& e) {
        QString errorMsg = QString("Hardware initialization failed with exception: %1").arg(e.what());
        emit logMessage(errorMsg, LogHandler::Error);
        
        // Clean up on failure
        disconnectStoredConnections(hwKey);
        d_hardwareMap.erase(hwKey);
        
        // Handle threaded cleanup
        if (hwObj->d_threaded) {
            QThread* thread = hwObj->thread();
            if (thread && thread != this->thread()) {
                thread->quit();
                if (!thread->wait(5000)) {
                    thread->terminate();
                    thread->wait(1000);
                }
                thread->deleteLater();
            }
        }
        
        hwObj->deleteLater();
        
        // Remove from RuntimeHardwareConfig on failure
        auto& runtimeConfig = RuntimeHardwareConfig::instance();
        runtimeConfig.removeHardwareSelection(hardwareType, label);
        emit logMessage(QString("Removed failed hardware from configuration: %1.%2").arg(hardwareType, label), LogHandler::Warning);
    } catch (...) {
        QString errorMsg = QString("Hardware initialization failed with unknown exception");
        emit logMessage(errorMsg, LogHandler::Error);
        
        // Clean up on failure - same as above
        disconnectStoredConnections(hwKey);
        d_hardwareMap.erase(hwKey);
        
        if (hwObj->d_threaded) {
            QThread* thread = hwObj->thread();
            if (thread && thread != this->thread()) {
                thread->quit();
                if (!thread->wait(5000)) {
                    thread->terminate();
                    thread->wait(1000);
                }
                thread->deleteLater();
            }
        }
        
        hwObj->deleteLater();
        
        auto& runtimeConfig = RuntimeHardwareConfig::instance();
        runtimeConfig.removeHardwareSelection(hardwareType, label);
        emit logMessage(QString("Removed failed hardware from configuration: %1.%2").arg(hardwareType, label), LogHandler::Warning);
    }
}

void HardwareManager::replaceHardwareInternal(const QString& hwKey, const QString& newImplementation)
{
    emit logMessage(QString("Replacing hardware: %1 with new implementation: %2")
                   .arg(hwKey, newImplementation), LogHandler::Normal);
    
    // Step 1: Remove old hardware completely (handles its own mutex protection)
    removeHardwareInternal(hwKey);
    
    // Step 2: Create new hardware from scratch (handles its own mutex protection)
    addHardwareInternal(hwKey, newImplementation);
    
    emit logMessage(QString("Successfully replaced hardware: %1 with implementation: %2")
                   .arg(hwKey, newImplementation), LogHandler::Normal);
}

// Task 3.3.5: Atomic Synchronization Orchestrator implementation

bool HardwareManager::applyVendorLibraryChanges()
{
    emit logMessage("Applying vendor library configuration changes", LogHandler::Normal);
    
    bool allSuccess = true;
    
    // Apply changes to SpectrumLibrary
    SpectrumLibrary& specLib = SpectrumLibrary::instance();
    if (specLib.hasUnstagedChanges()) {
        emit logMessage("Applying Spectrum library configuration changes", LogHandler::Normal);
        if (!specLib.applyChanges()) {
            emit logMessage(QString("Failed to apply Spectrum library changes: %1").arg(specLib.errorString()), LogHandler::Error);
            allSuccess = false;
        } else {
            emit logMessage("Spectrum library configuration applied successfully", LogHandler::Normal);
        }
    }
    
    // Apply changes to LabjackLibrary
    LabjackLibrary& ljLib = LabjackLibrary::instance();
    if (ljLib.hasUnstagedChanges()) {
        emit logMessage("Applying LabJack library configuration changes", LogHandler::Normal);
        if (!ljLib.applyChanges()) {
            emit logMessage(QString("Failed to apply LabJack library changes: %1").arg(ljLib.errorString()), LogHandler::Error);
            allSuccess = false;
        } else {
            emit logMessage("LabJack library configuration applied successfully", LogHandler::Normal);
        }
    }
    
    if (allSuccess) {
        emit logMessage("All vendor library configuration changes applied successfully", LogHandler::Normal);
    } else {
        emit logMessage("Some vendor library configuration changes failed to apply", LogHandler::Error);
    }
    
    return allSuccess;
}

void HardwareManager::reloadPythonScript(const QString &hwKey)
{
    HardwareObject *obj = nullptr;
    {
        QReadLocker locker(&d_hardwareMapLock);
        auto it = d_hardwareMap.find(hwKey);
        if (it == d_hardwareMap.end()) {
            emit pythonScriptReloadResult(hwKey, false, QStringLiteral("Hardware not found: %1").arg(hwKey));
            return;
        }
        obj = it->second;
    }

    auto *pyBase = dynamic_cast<PythonHardwareBase*>(obj);
    if (!pyBase) {
        emit pythonScriptReloadResult(hwKey, false, QStringLiteral("%1 is not a Python hardware implementation").arg(hwKey));
        return;
    }

    // Dispatch stop + test to the hardware object's thread.
    // stopProcess() kills the subprocess; bcTestConnection() re-reads settings
    // and calls testConnection() -> testPythonConnection() -> startPythonProcess()
    // automatically since the process is no longer running.
    QMetaObject::invokeMethod(obj, [this, obj, pyBase, hwKey]() {
        pyBase->stopProcess();
        obj->bcTestConnection();
        bool success = obj->isConnected();
        QString msg = success ? QStringLiteral("Script reloaded successfully")
                              : pyBase->pythonErrorString();
        emit pythonScriptReloadResult(hwKey, success, msg);
        if (success)
            emit logMessage(QStringLiteral("Python script reloaded for %1").arg(hwKey));
        else
            emit logMessage(QStringLiteral("Python script reload failed for %1: %2").arg(hwKey, msg), LogHandler::Error);
    });
}

void HardwareManager::syncWithRuntimeConfig()
{
    emit logMessage("Starting hardware synchronization with runtime configuration", LogHandler::Normal);
    
    const auto& config = RuntimeHardwareConfig::constInstance();
    auto targetHardware = config.getCurrentHardware();
    
    // Find differences between current and target states (with temporary read lock)
    std::vector<QString> toRemove;
    std::vector<std::pair<QString, QString>> toAdd;
    std::vector<std::pair<QString, QString>> toReplace;
    
    {
        QReadLocker locker(&d_hardwareMapLock);
        toRemove = findHardwareToRemove(targetHardware);
        toAdd = findHardwareToAdd(targetHardware);
        toReplace = findHardwareToReplace(targetHardware);
    }
    
    // Add hardware dependent on changed libraries to recreation lists
    addLibraryDependentHardwareToRecreation(targetHardware, toRemove, toAdd, toReplace);
    
    emit logMessage(QString("Hardware synchronization plan: %1 to remove, %2 to add, %3 to replace")
                   .arg(toRemove.size()).arg(toAdd.size()).arg(toReplace.size()), LogHandler::Normal);
    
    // Phase 1: Destroy hardware objects that need to be removed/replaced
    // This ensures hardware objects are destroyed using current/valid function pointers
    for(const auto& hwKey : toRemove) {
        removeHardwareInternal(hwKey);
    }
    
    for(const auto& [hwKey, impl] : toReplace) {
        // replaceHardwareInternal calls removeHardwareInternal first, then addHardwareInternal
        // We'll need to split this to get proper timing for library changes
        emit logMessage(QString("Removing hardware for replacement: %1").arg(hwKey), LogHandler::Normal);
        removeHardwareInternal(hwKey);
    }
    
    // Phase 2: Apply vendor library configuration changes at SAFE timing
    // All hardware objects that could use the libraries are now destroyed
    if (!applyVendorLibraryChanges()) {
        emit logMessage("Warning: Some vendor library changes failed to apply, continuing with hardware synchronization", LogHandler::Warning);
        // Continue with hardware sync even if library changes fail - user should be notified but not blocked
    }
    
    // Phase 3: Create new hardware objects (using updated libraries if any changes were applied)
    for(const auto& [hwKey, impl] : toReplace) {
        emit logMessage(QString("Adding replacement hardware: %1 with implementation: %2").arg(hwKey, impl), LogHandler::Normal);
        addHardwareInternal(hwKey, impl);
    }
    
    for(const auto& [hwKey, impl] : toAdd) {
        addHardwareInternal(hwKey, impl);
    }
    
    emit logMessage("Hardware synchronization changes applied successfully", LogHandler::Normal);

    // Report threading status for all hardware objects (Debug level)
    {
        QReadLocker locker(&d_hardwareMapLock);
        QThread* managerThread = QThread::currentThread();
        QString managerThreadName = managerThread->objectName().isEmpty()
                                    ? QString("0x%1").arg(reinterpret_cast<quintptr>(managerThread), 0, 16)
                                    : managerThread->objectName();
        emit logMessage(QString("Thread report - HardwareManager: %1").arg(managerThreadName),
                        LogHandler::Debug);
        for (auto& [hwKey, hwObj] : d_hardwareMap) {
            QThread* hwThread = hwObj->thread();
            QString hwThreadName = hwThread->objectName().isEmpty()
                                   ? QString("0x%1").arg(reinterpret_cast<quintptr>(hwThread), 0, 16)
                                   : hwThread->objectName();
            bool ownThread = (hwThread != managerThread);
            emit logMessage(QString("Thread report - %1: %2 (%3)")
                            .arg(hwKey, hwThreadName, ownThread ? "own thread" : "manager thread"),
                            LogHandler::Debug);
        }
    }

    // Resolve GPIB controllers for instruments before connection testing
    resolveGpibControllersForInstruments();  // Now uses its own read lock
    
    // Update ClockManager with current clocks
    updateClockManager();
    
    emit logMessage("Starting connection testing after hardware synchronization", LogHandler::Normal);
    testAll();
    
    emit logMessage("Hardware synchronization with runtime configuration complete", LogHandler::Normal);
}

void HardwareManager::updateClockManager()
{
    if (!pu_clockManager) {
        emit logMessage("ClockManager not initialized - cannot update clocks", LogHandler::Error);
        return;
    }
    
    // Find all Clock objects in the hardware map
    QVector<Clock*> clocks;
    {
        QReadLocker locker(&d_hardwareMapLock);
        for (const auto& [key, hwObj] : d_hardwareMap) {
            if (Clock* clock = qobject_cast<Clock*>(hwObj)) {
                clocks.append(clock);
            }
        }
    }
    
    emit logMessage(QString("Updating ClockManager with %1 clock(s)").arg(clocks.size()), LogHandler::Normal);
    pu_clockManager->setClocksFromHardwareManager(clocks);
}

std::vector<QString> HardwareManager::findHardwareToRemove(const std::map<QString, QString>& targetHardware)
{
    std::vector<QString> toRemove;
    
    // Hardware in current map but not in target should be removed
    for(const auto& [currentKey, hwObj] : d_hardwareMap) {
        Q_UNUSED(hwObj)
        if(targetHardware.find(currentKey) == targetHardware.end()) {
            toRemove.push_back(currentKey);
        }
    }
    
    return toRemove;
}

std::vector<std::pair<QString, QString>> HardwareManager::findHardwareToAdd(const std::map<QString, QString>& targetHardware)
{
    std::vector<std::pair<QString, QString>> toAdd;
    
    // Hardware in target but not in current map should be added
    for(const auto& [targetKey, targetImpl] : targetHardware) {
        if(d_hardwareMap.find(targetKey) == d_hardwareMap.end()) {
            toAdd.emplace_back(targetKey, targetImpl);
        }
    }
    
    return toAdd;
}

std::vector<std::pair<QString, QString>> HardwareManager::findHardwareToReplace(const std::map<QString, QString>& targetHardware)
{
    std::vector<std::pair<QString, QString>> toReplace;
    
    // Hardware in both maps but with different implementations or threading settings should be replaced
    for(const auto& [targetKey, targetImpl] : targetHardware) {
        auto currentIt = d_hardwareMap.find(targetKey);
        if(currentIt != d_hardwareMap.end()) {
            HardwareObject* currentObj = currentIt->second;

            // Implementation changed
            if(currentObj->d_model != targetImpl) {
                toReplace.emplace_back(targetKey, targetImpl);
                continue;
            }

            // Threading setting changed (only if a stored override differs from current state)
            auto threadedOverride = RuntimeHardwareConfig::constInstance().getThreaded(targetKey);
            if(threadedOverride.has_value() && *threadedOverride != currentObj->d_threaded) {
                toReplace.emplace_back(targetKey, targetImpl);
            }
        }
    }
    
    return toReplace;
}

void HardwareManager::resolveGpibControllersForInstruments()
{
    emit logMessage("Resolving GPIB controllers for GPIB instruments", LogHandler::Normal);
    
    // Use read lock for safe iteration over hardware map
    QReadLocker locker(&d_hardwareMapLock);
    
    // Iterate through all hardware objects to find GPIB instruments
    for(auto& [hwKey, hwObj] : d_hardwareMap) {
        // Check if this hardware uses GPIB communication
        if(hwObj->d_commType != CommunicationProtocol::Gpib) {
            continue;  // Skip non-GPIB hardware
        }
        
        // Read the controller key from the instrument's settings
        SettingsStorage s(hwKey, SettingsStorage::Hardware);
        QString controllerKey = s.getGroupValue(BC::Key::Comm::gpib, BC::Key::GPIB::gpibController, QString());
        
        if(controllerKey.isEmpty()) {
            emit logMessage(QString("GPIB instrument %1 has no controller configured - connection testing will fail").arg(hwKey), LogHandler::Warning);
            continue;
        }
        
        // Use the callback-based resolution to safely get the controller
        resolveGpibController(controllerKey, [this, hwKey, hwObj, controllerKey](GpibController* controller) {
            if(controller) {
                // Controller found - complete the deferred initialization
                emit logMessage(QString("Resolved GPIB controller %1 for instrument %2").arg(controllerKey, hwKey), LogHandler::Normal);
                hwObj->buildCommunication(controller);
            } else {
                // Controller not found - emit informative error message
                emit logMessage(QString("GPIB controller %1 not found for instrument %2 - connection testing will fail").arg(controllerKey, hwKey), LogHandler::Warning);
                // Allow testConnection to fail naturally - don't force failure here
            }
        });
    }
    
    emit logMessage("GPIB controller resolution complete", LogHandler::Normal);
}

// ============================================================================
// Task 3.3.8: Communication Protocol Management API
// ============================================================================

void HardwareManager::getHardwareCommunicationInfo(const QString& hwKey)
{
    QReadLocker locker(&d_hardwareMapLock);
    
    auto it = d_hardwareMap.find(hwKey);
    if (it == d_hardwareMap.end()) {
        emit logMessage(QString("Hardware %1 not found for communication info query").arg(hwKey), LogHandler::Warning);
        return;
    }
    
    HardwareObject* hwObj = it->second;
    if (!hwObj) {
        emit logMessage(QString("Hardware object %1 is null").arg(hwKey), LogHandler::Warning);
        return;
    }
    
    // Get current protocol and connection status
    CommunicationProtocol::CommType currentProtocol = hwObj->d_commType;
    bool connected = hwObj->isConnected();
    
    // Get supported protocols from the hardware object
    QVector<CommunicationProtocol::CommType> supportedProtocols = hwObj->supportedProtocols();
    
    // Emit the response signal with the collected information
    emit hardwareCommunicationInfoReady(hwKey, currentProtocol, supportedProtocols, connected);
}

void HardwareManager::setHardwareProtocol(const QString& hwKey, CommunicationProtocol::CommType protocol, const QString& gpibControllerKey)
{
    QReadLocker locker(&d_hardwareMapLock);
    
    auto it = d_hardwareMap.find(hwKey);
    if (it == d_hardwareMap.end()) {
        emit protocolSetResult(hwKey, false, QString("Hardware %1 not found").arg(hwKey));
        return;
    }
    
    HardwareObject* hwObj = it->second;
    if (!hwObj) {
        emit protocolSetResult(hwKey, false, QString("Hardware object %1 is null").arg(hwKey));
        return;
    }
    
    // Check if the protocol is supported by this hardware
    QVector<CommunicationProtocol::CommType> supportedProtocols = hwObj->supportedProtocols();
    if (!supportedProtocols.contains(protocol)) {
        emit protocolSetResult(hwKey, false, QString("Protocol not supported by %1").arg(hwKey));
        return;
    }
    
    // If protocol is already current, success
    if (hwObj->d_commType == protocol) {
        emit protocolSetResult(hwKey, true, QString("Protocol already set for %1").arg(hwKey));
        return;
    }
    
    // Protocol change needed
    if (protocol == CommunicationProtocol::CommType::Gpib) {
        // For GPIB protocol, use provided controller key or get from settings
        QString controllerKey = gpibControllerKey;
        if (controllerKey.isEmpty()) {
            SettingsStorage s(hwKey, SettingsStorage::Hardware);
            controllerKey = s.getGroupValue(BC::Key::Comm::gpib, BC::Key::GPIB::gpibController, QString());
        }
        
        if (controllerKey.isEmpty()) {
            emit protocolSetResult(hwKey, false, QString("No GPIB controller specified for %1").arg(hwKey));
            return;
        }
        
        locker.unlock(); // Release read lock before callback
        
        // Use existing resolution function with callback
        resolveGpibController(controllerKey, [this, hwKey, hwObj, protocol](GpibController* controller) {
            bool success = false;
            if (hwObj->thread() != this->thread()) {
                QMetaObject::invokeMethod(hwObj, [hwObj, protocol, controller]() -> bool {
                    return hwObj->setCommProtocol(protocol, controller);
                }, Qt::BlockingQueuedConnection, &success);
            } else {
                success = hwObj->setCommProtocol(protocol, controller);
            }
            
            if (success) {
                emit protocolSetResult(hwKey, true, QString("Protocol set successfully for %1").arg(hwKey));
            } else {
                emit protocolSetResult(hwKey, false, QString("Failed to set protocol for %1").arg(hwKey));
            }
        });
    } else {
        // Non-GPIB protocol - no controller needed
        locker.unlock(); // Release read lock before potentially calling across threads
        
        bool success = false;
        if (hwObj->thread() != this->thread()) {
            QMetaObject::invokeMethod(hwObj, [hwObj, protocol]() -> bool {
                return hwObj->setCommProtocol(protocol, nullptr);
            }, Qt::BlockingQueuedConnection, &success);
        } else {
            success = hwObj->setCommProtocol(protocol, nullptr);
        }
        
        if (success) {
            emit protocolSetResult(hwKey, true, QString("Protocol set successfully for %1").arg(hwKey));
        } else {
            emit protocolSetResult(hwKey, false, QString("Failed to set protocol for %1").arg(hwKey));
        }
    }
}

void HardwareManager::getActiveGpibControllers()
{
    QReadLocker locker(&d_hardwareMapLock);
    
    QStringList controllerKeys;
    
    // Iterate through all hardware objects to find GPIB controllers
    for (const auto& [hwKey, hwObj] : d_hardwareMap) {
        if (!hwObj) continue;
        
        // Check if this is a GPIB controller
        // We can identify GPIB controllers by checking their type
        GpibController* controller = dynamic_cast<GpibController*>(hwObj);
        if (controller) {
            controllerKeys.append(hwKey);
        }
    }
    
    // Sort the keys for consistent ordering
    controllerKeys.sort();
    
    // Emit the list of available GPIB controllers
    emit gpibControllersAvailable(controllerKeys);
}

bool HardwareManager::allCriticalHardwareConnected() const
{
    QReadLocker locker(&d_hardwareMapLock);
    
    for (const auto& [key, hwObj] : d_hardwareMap) {
        if (hwObj->d_critical && !hwObj->isConnected()) {
            return false;
        }
    }
    
    return true;
}

void HardwareManager::addLibraryDependentHardwareToRecreation(const std::map<QString, QString>& /* targetHardware */,
                                                            std::vector<QString>& toRemove,
                                                            std::vector<std::pair<QString, QString>>& /* toAdd */,
                                                            std::vector<std::pair<QString, QString>>& toReplace)
{
    const HardwareRegistry& registry = HardwareRegistry::instance();
    
    // Get all libraries that have staged changes (completely generic!)
    QStringList changedLibraries = registry.getLibrariesWithChanges();
    
    if (changedLibraries.isEmpty()) {
        return; // No library changes, nothing to do
    }
    
    for (const QString& libName : changedLibraries) {
        emit logMessage(QString("Library %1 has staged changes - dependent hardware will be recreated")
                       .arg(libName), LogHandler::Normal);
    }
    
    // Find current hardware that depends on changed libraries
    const auto& currentHardware = RuntimeHardwareConfig::constInstance().getCurrentHardware();
    
    for (const auto& [hwKey, impl] : currentHardware) {
        QStringList deps = registry.getLibraryDependencies(impl);
        
        bool needsRecreation = false;
        for (const QString& changedLib : changedLibraries) {
            if (deps.contains(changedLib)) {
                needsRecreation = true;
                emit logMessage(QString("Hardware %1 depends on changed library %2 - adding to recreation list")
                               .arg(hwKey, changedLib), LogHandler::Normal);
                break;
            }
        }
        
        if (needsRecreation) {
            // Check if this hardware is already in one of the recreation lists
            bool alreadyInRemoval = std::find(toRemove.begin(), toRemove.end(), hwKey) != toRemove.end();
            bool alreadyInReplacement = std::find_if(toReplace.begin(), toReplace.end(), 
                                                   [&hwKey](const auto& pair) { return pair.first == hwKey; }) != toReplace.end();
            
            if (!alreadyInRemoval && !alreadyInReplacement) {
                // Add to replacement list (same implementation, but needs recreation for library changes)
                toReplace.emplace_back(hwKey, impl);
                emit logMessage(QString("Added %1 to replacement list due to library dependency").arg(hwKey), LogHandler::Normal);
            }
        }
    }
}
