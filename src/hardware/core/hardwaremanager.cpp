#include <hardware/core/hardwaremanager.h>
#include <hardware/core/runtimehardwareconfig.h>
#include <hardware/core/hardwareregistry.h>
#include <data/settings/hardwarekeys.h>

#include <hardware/core/hardwareobject.h>
#include <hardware/core/ftmwdigitizer/ftmwscope.h>
#include <hardware/core/clock/clockmanager.h>
#include <hardware/optional/chirpsource/awg.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <hardware/optional/ioboard/ioboard.h>
#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>
#include <hardware/optional/tempcontroller/temperaturecontroller.h>

#include <hardware/hw_h.h>
#include <hardware/opthw_h.h>
#include <hardware/core/clock/clock_h.h>

#include <boost/preprocessor/iteration/local.hpp>

#include <QThread>

#include <hardware/core/lifdigitizer/lifscope.h>
#include <hardware/core/liflaser/liflaser.h>
#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>

// Phase 2.4.2: Virtual hardware includes for capability discovery
#include <hardware/core/ftmwdigitizer/virtualftmwscope.h>
#include <hardware/optional/chirpsource/virtualawg.h>
#include <hardware/optional/pulsegenerator/virtualpulsegenerator.h>
#include <hardware/optional/flowcontroller/virtualflowcontroller.h>
#include <hardware/optional/ioboard/virtualioboard.h>
#include <hardware/optional/gpibcontroller/virtualgpibcontroller.h>
#include <hardware/optional/pressurecontroller/virtualpressurecontroller.h>
#include <hardware/optional/tempcontroller/virtualtempcontroller.h>
#include <hardware/core/lifdigitizer/virtuallifscope.h>
#include <hardware/core/liflaser/virtualliflaser.h>

// Static instance for const access
HardwareManager* HardwareManager::s_instance = nullptr;

HardwareManager::HardwareManager(QObject *parent) : QObject(parent), SettingsStorage(BC::Key::hw),
    d_optHwTypes{QString(FlowController::staticMetaObject.className()),QString(IOBoard::staticMetaObject.className()),QString(PressureController::staticMetaObject.className()),QString(PulseGenerator::staticMetaObject.className()),QString(TemperatureController::staticMetaObject.className())}
{
    // Set static instance for const access
    s_instance = this;
    
    // Lock mutex for entire initialization - no concurrency issues during startup
    QMutexLocker locker(&d_accessMutex);
    
    // Phase 2.4.2: Refactored constructor using extracted methods
    // Create virtual hardware for capability discovery (replaces compile-time flags)
    createVirtualHardwareForCapabilityDiscovery();
    
    // Finalize initialization with signal connections and threading setup
    finalizeInitialization();
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
        return hw->d_name;

    return QString();
}

void HardwareManager::initialize()
{
    //start all threads and initialize hw
    for(auto it = d_hardwareMap.cbegin(); it != d_hardwareMap.cend(); ++it)
    {
        auto hw = it->second;
        if(hw->d_commType == CommunicationProtocol::Virtual)
            emit logMessage(QString("%1 is a virtual instrument. Be cautious about taking real measurements!")
                            .arg(hw->d_name),LogHandler::Warning);
        if(hw->d_threaded)
        {
            if(!hw->thread()->isRunning())
                hw->thread()->start();
        }
        else
        {
            hw->setParent(this);
            QMetaObject::invokeMethod(hw,&HardwareObject::bcInitInstrument);
        }
    }

    emit hwInitializationComplete();
}

void HardwareManager::connectionResult(HardwareObject *obj, bool success, QString msg)
{
    if(d_responseCount < d_hardwareMap.size())
        d_responseCount++;

    if(success)
    {
        connect(obj,&HardwareObject::hardwareFailure,this,&HardwareManager::hardwareFailure,Qt::UniqueConnection);
        emit logMessage(obj->d_name + QString(": Connected successfully."));
    }
    else
    {
        disconnect(obj,&HardwareObject::hardwareFailure,this,&HardwareManager::hardwareFailure);
        LogHandler::MessageCode code = LogHandler::Error;
        if(!obj->d_critical)
            code = LogHandler::Warning;
        emit logMessage(obj->d_name + QString(": Connection failed!"),code);
        if(!msg.isEmpty())
            emit logMessage(msg,code);
    }

    emit testComplete(obj->d_name,success,msg);
    checkStatus();
}

void HardwareManager::hardwareFailure()
{
    HardwareObject *obj = dynamic_cast<HardwareObject*>(sender());
    if(obj == nullptr)
        return;

    disconnect(obj,&HardwareObject::hardwareFailure,this,&HardwareManager::hardwareFailure);

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
                emit logMessage(QString("Error initializing %1").arg(obj->d_name),LogHandler::Error);
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
            else
                connect(ll,&LifLaser::laserPosUpdate,this,&HardwareManager::lifLaserSetComplete,Qt::UniqueConnection);
        }
    }
    //any additional synchronous initialization can be performed here, before experimentInitialized() is emitted


    emit experimentInitialized(exp);

}

void HardwareManager::experimentComplete()
{
    auto activeKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<LifLaser>();
    if (!activeKeys.isEmpty()) {
        auto ll = findHardware<LifLaser>(activeKeys.first());
        if(ll)
            disconnect(ll,&LifLaser::laserPosUpdate,this,&HardwareManager::lifLaserSetComplete);
    }
}

void HardwareManager::testAll()
{
    for(auto it = d_hardwareMap.cbegin(); it != d_hardwareMap.cend(); ++it)
    {
        auto obj = it->second;
        QMetaObject::invokeMethod(obj,&HardwareObject::bcTestConnection);
    }

    checkStatus();
}

void HardwareManager::testObjectConnection(const QString hwKey)
{
    HardwareObject* obj = nullptr;
    {
        QMutexLocker locker(&d_accessMutex);
        auto it = d_hardwareMap.find(hwKey);
        if(it == d_hardwareMap.end()) {
            emit testComplete(hwKey,false,QString("Device not found!"));
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

QStringList HardwareManager::getForbiddenKeys(const QString key) const
{
    auto hw = findHardware<HardwareObject>(key);
    if(hw)
        return hw->forbiddenKeys();

    return {};
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

    for(auto it = clocks.begin(); it != clocks.end(); ++it)
        it.value().desiredFreqMHz = pu_clockManager->setClockFrequency(it.key(),it.value().desiredFreqMHz);

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
    PulseGenConfig out("PulseGenerator", "virtual", "temp"); // Dummy constructor, will be overwritten
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
    FlowConfig out("FlowController", "virtual", "temp"); // Dummy constructor, will be overwritten
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
    QMutexLocker locker(&d_accessMutex);
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
    PressureControllerConfig out("PressureController", "virtual", "temp"); // Dummy constructor, will be overwritten
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
    TemperatureControllerConfig out("TemperatureController", "virtual", "temp"); // Dummy constructor, will be overwritten
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
    IOBoardConfig out("IOBoard", "virtual", "temp"); // Dummy constructor, will be overwritten
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
        auto t = BC::Key::parseIndexKey(hwKey);
        auto type = t.first;
        auto index = t.second;

        if((d_optHwTypes.find(type) == d_optHwTypes.end()) || index < 0 )
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
    //gotta wait until all instruments have responded
    if(d_responseCount < d_hardwareMap.size())
        return;

    bool success = true;
    for(auto &[key,obj] : d_hardwareMap)
    {
        Q_UNUSED(key)
        if(!obj->isConnected() && obj->d_critical)
            success = false;
    }

    emit allHardwareConnected(success);
}

void HardwareManager::setLifParameters(double delay, double pos)
{
    bool success = true;
    success &= setLifLaserPos(pos);
    if(success)
        success &= setPGenLifDelay(delay);

    emit lifSettingsComplete(success);
}

bool HardwareManager::setPGenLifDelay(double d)
{
#ifndef BC_PGEN
    emit logMessage(QString("Could not set LIF delay because no pulse generator is avaialble."),LogHandler::Error);
    return false;
#else
    bool out = true;
    auto activeKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<PulseGenerator>();
    for(const auto& key : activeKeys)
    {
        auto pGen = findHardware<PulseGenerator>(key);

        if(pGen->thread() == QThread::currentThread())
            out &= pGen->setLifDelay(d);
        else
            QMetaObject::invokeMethod(pGen,[pGen,d](){ return pGen->setLifDelay(d); },Qt::BlockingQueuedConnection,&out);
    }

    return out;
#endif
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

void HardwareManager::lifLaserSetComplete(double pos)
{
    emit lifSettingsComplete(pos > 0.0);
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
    GpibController* controller = nullptr;
    {
        QMutexLocker locker(&d_accessMutex);
        auto it = d_hardwareMap.find(controllerKey);
        if (it != d_hardwareMap.end()) {
            controller = static_cast<GpibController*>(it->second);
        }
    }
    
    // Call the callback with the resolved controller (or nullptr if not found)
    callback(controller);
}

// Phase 2.4.2: Constructor refactoring methods

void HardwareManager::createVirtualHardwareForCapabilityDiscovery()
{
    // Create virtual instances of all hardware types to discover capabilities
    // This replaces the compile-time flag-based hardware creation
    
    // Required hardware: FtmwScope (always virtual for discovery)
    auto ftmwScope = new VirtualFtmwScope("temp");
    d_hardwareMap.emplace(ftmwScope->d_key, ftmwScope);

    // Clock Manager (creates virtual Clock instances for discovery)
    pu_clockManager = std::make_unique<ClockManager>();
    auto cl = pu_clockManager->getClockList();
    for(int i = 0; i < cl.size(); i++)
        d_hardwareMap.emplace(cl.at(i)->d_key, cl.at(i));

    // Optional hardware - create virtual instances of each type
    // This ensures all hardware types are available for discovery
    
    // Chirp Source (AWG)
    auto awg = new VirtualAwg("temp");
    d_hardwareMap.emplace(awg->d_key, awg);
    
    // GPIB Controller
    auto gpib = new VirtualGpibController("temp");
    d_hardwareMap.emplace(gpib->d_key, gpib);
    
    // Pulse Generator
    auto pGen = new VirtualPulseGenerator("temp");
    d_hardwareMap.emplace(pGen->d_key, pGen);
    
    // Flow Controller
    auto flowController = new VirtualFlowController("temp");
    d_hardwareMap.emplace(flowController->d_key, flowController);
    
    // Pressure Controller
    auto pressureController = new VirtualPressureController("temp");
    d_hardwareMap.emplace(pressureController->d_key, pressureController);
    
    // Temperature Controller
    auto tempController = new VirtualTemperatureController("temp");
    d_hardwareMap.emplace(tempController->d_key, tempController);
    
    // IO Board
    auto ioBoard = new VirtualIOBoard("temp");
    d_hardwareMap.emplace(ioBoard->d_key, ioBoard);
    
    // LIF Hardware
    auto lifScope = new VirtualLifScope("temp");
    d_hardwareMap.emplace(lifScope->d_key, lifScope);
    
    auto lifLaser = new VirtualLifLaser("temp");
    d_hardwareMap.emplace(lifLaser->d_key, lifLaser);
}

void HardwareManager::setupHardwareObject(HardwareObject* obj)
{
    // Common signal connections for all hardware objects
    connect(obj, &HardwareObject::logMessage, [this, obj](QString msg, LogHandler::MessageCode mc){
        emit logMessage(QString("%1: %2").arg(obj->d_name).arg(msg), mc);
    });
    connect(obj, &HardwareObject::connected, [obj, this](bool success, QString msg){
        connectionResult(obj, success, msg);
    });
    connect(obj, &HardwareObject::auxDataRead, [obj, this](AuxDataStorage::AuxDataMap m){
        AuxDataStorage::AuxDataMap out;
        for(auto it = m.cbegin(); it != m.cend(); ++it)
            out.insert({AuxDataStorage::makeKey(obj->d_key, obj->d_subKey, it->first), it->second});
        emit auxData(out);
    });
    connect(obj, &HardwareObject::auxDataRead, [obj, this](AuxDataStorage::AuxDataMap m){
        AuxDataStorage::AuxDataMap out;
        for(auto it = m.cbegin(); it != m.cend(); ++it)
            out.insert({AuxDataStorage::makeKey(obj->d_key, obj->d_subKey, it->first), it->second});
        emit validationData(out);
    });
    connect(obj, &HardwareObject::rollingDataRead, [obj, this](AuxDataStorage::AuxDataMap m){
        AuxDataStorage::AuxDataMap out;
        for(auto it = m.cbegin(); it != m.cend(); ++it)
            out.insert({AuxDataStorage::makeKey(obj->d_key, obj->d_subKey, it->first), it->second});
        emit rollingData(out, QDateTime::currentDateTime());
    });
    connect(this, &HardwareManager::beginAcquisition, obj, &HardwareObject::beginAcquisition);
    connect(this, &HardwareManager::endAcquisition, obj, &HardwareObject::endAcquisition);
}

void HardwareManager::finalizeInitialization()
{
    // Resolve GPIB controller for communication setup
    GpibController* gpib = nullptr;
    auto gpibIt = std::find_if(d_hardwareMap.begin(), d_hardwareMap.end(), 
        [](const auto& pair) { return qobject_cast<GpibController*>(pair.second) != nullptr; });
    if (gpibIt != d_hardwareMap.end()) {
        gpib = static_cast<GpibController*>(gpibIt->second);
    }

    // Setup hardware-specific signal connections and communication
    for(auto& [key, obj] : d_hardwareMap) {
        // Setup common hardware object
        setupHardwareObject(obj);
        
        // Setup hardware-specific signal connections
        if (auto ftmwScope = qobject_cast<FtmwScope*>(obj)) {
            connect(ftmwScope, &FtmwScope::shotAcquired, this, &HardwareManager::ftmwScopeShotAcquired);
        }
        else if (auto pGen = qobject_cast<PulseGenerator*>(obj)) {
            QString k = pGen->d_key;
            connect(pGen, &PulseGenerator::settingUpdate, [this, k](const int ch, const PulseGenConfig::Setting set, const QVariant val){
                emit pGenSettingUpdate(k, ch, set, val);
            });
            connect(pGen, &PulseGenerator::configUpdate, [this, k](const PulseGenConfig cfg){
                emit pGenConfigUpdate(k, cfg);
            });
        }
        else if (auto flowController = qobject_cast<FlowController*>(obj)) {
            QString k = flowController->d_key;
            connect(flowController, &FlowController::flowUpdate, [this, k](int i, double d){
                emit flowUpdate(k, i, d);
            });
            connect(flowController, &FlowController::flowSetpointUpdate, [this, k](int i, double d){
                emit flowSetpointUpdate(k, i, d);
            });
            connect(flowController, &FlowController::pressureUpdate, [this, k](double d){
                emit gasPressureUpdate(k, d);
            });
            connect(flowController, &FlowController::pressureSetpointUpdate, [this, k](double d){
               emit gasPressureSetpointUpdate(k, d);
            });
            connect(flowController, &FlowController::pressureControlMode, [this, k](bool b){
                emit gasPressureControlMode(k, b);
            });
        }
        else if (auto pressureController = qobject_cast<PressureController*>(obj)) {
            QString k = pressureController->d_key;
            connect(pressureController, &PressureController::pressureUpdate, this, [this, k](double d){
                emit pressureUpdate(k, d);
            });
            connect(pressureController, &PressureController::pressureSetpointUpdate, this, [this, k](double d){
                emit pressureSetpointUpdate(k, d);
            });
            connect(pressureController, &PressureController::pressureControlMode, this, [this, k](bool b){
                emit pressureControlMode(k, b);
            });
        }
        else if (auto tempController = qobject_cast<TemperatureController*>(obj)) {
            QString k = tempController->d_key;
            connect(tempController, &TemperatureController::channelEnableUpdate, this, [this, k](uint i, bool en){
                emit temperatureEnableUpdate(k, i, en);
            });
            connect(tempController, &TemperatureController::temperatureUpdate, this, [this, k](uint i, double t) {
                emit temperatureUpdate(k, i, t);
            });
        }
        else if (auto lifScope = qobject_cast<LifScope*>(obj)) {
            connect(lifScope, &LifScope::waveformRead, this, &HardwareManager::lifScopeShotAcquired);
            connect(lifScope, &LifScope::configAcqComplete, this, &HardwareManager::lifConfigAcqStarted);
        }
        else if (auto lifLaser = qobject_cast<LifLaser*>(obj)) {
            connect(lifLaser, &LifLaser::laserPosUpdate, this, &HardwareManager::lifLaserPosUpdate);
            connect(lifLaser, &LifLaser::laserFlashlampUpdate, this, &HardwareManager::lifLaserFlashlampUpdate);
        }

        // Build communication protocol for hardware object
        obj->buildCommunication(gpib);

        // Handle threaded hardware
        if(obj->d_threaded) {
            auto t = new QThread(this);
            t->setObjectName(obj->d_key + "Thread");
            obj->moveToThread(t);
            connect(t, &QThread::started, obj, &HardwareObject::bcInitInstrument);
        } else {
            obj->setParent(this);
        }
    }
    
    // Setup ClockManager signals
    if (pu_clockManager) {
        connect(pu_clockManager.get(), &ClockManager::logMessage, this, &HardwareManager::logMessage);
        connect(pu_clockManager.get(), &ClockManager::clockFrequencyUpdate, this, &HardwareManager::clockFrequencyUpdate);
    }

    // TEMPORARY: Populate RuntimeHardwareConfig with virtual hardware selections
    auto& runtimeConfig = RuntimeHardwareConfig::instance();
    int mapIndex = 0;
    for(auto& [key, obj] : d_hardwareMap) {
        auto [hardwareType, index] = BC::Key::parseIndexKey(obj->d_key);
        QString implementation = obj->d_subKey.isEmpty() ? "virtual" : obj->d_subKey;
        runtimeConfig.registerHardwareForTesting(hardwareType, implementation, mapIndex++);
    }
    
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
    
    emit logMessage(QString("Successfully created hardware: %1 (%2.%3)")
                   .arg(hwObj->d_name, type, label), LogHandler::Normal);
    
    return hwObj;
}
