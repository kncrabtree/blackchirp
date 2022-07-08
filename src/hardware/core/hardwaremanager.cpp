#include <hardware/core/hardwaremanager.h>

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

#include <QThread>

#ifdef BC_LIF
#include <modules/lif/hardware/lifdigitizer/lifscope.h>
#include <modules/lif/hardware/liflaser/liflaser.h>
#endif

HardwareManager::HardwareManager(QObject *parent) : QObject(parent), SettingsStorage(BC::Key::hw)
{
    //Required hardware: FtmwScope and Clocks
    auto ftmwScope = new FtmwScopeHardware;
    connect(ftmwScope,&FtmwScope::shotAcquired,this,&HardwareManager::ftmwScopeShotAcquired);
    d_hardwareMap.emplace(ftmwScope->d_key,ftmwScope);

    pu_clockManager = std::make_unique<ClockManager>();
    connect(pu_clockManager.get(),&ClockManager::logMessage,this,&HardwareManager::logMessage);
    connect(pu_clockManager.get(),&ClockManager::clockFrequencyUpdate,this,&HardwareManager::clockFrequencyUpdate);
    auto cl = pu_clockManager->d_clockList;
    for(int i=0; i<cl.size(); i++)
        d_hardwareMap.emplace(cl.at(i)->d_key,cl.at(i));

#ifdef BC_AWG
    auto awg =new AwgHardware;
    d_hardwareMap.emplace(awg->d_key,awg);
#endif

    QThread* gpibThread = nullptr;
#ifdef BC_GPIBCONTROLLER
    auto gpib = new GpibControllerHardware;
    gpibThread = new QThread(this);
    gpibThread->setObjectName(gpib->d_key+"Thread");
    connect(gpibThread,&QThread::started,gpib,&HardwareObject::bcInitInstrument);
    d_hardwareMap.emplace(gpib->d_key,gpib);
#else
    auto gpib = nullptr;
#endif

#ifdef BC_PGEN
    auto pGen = new PulseGeneratorHardware;
    connect(pGen,&PulseGenerator::settingUpdate,this,&HardwareManager::pGenSettingUpdate);
    connect(pGen,&PulseGenerator::configUpdate,this,&HardwareManager::pGenConfigUpdate);
    connect(pGen,&PulseGenerator::repRateUpdate,this,&HardwareManager::pGenRepRateUpdate);
    d_hardwareMap.emplace(pGen->d_key,pGen);
#endif

#ifdef BC_FLOWCONTROLLER
    auto flow = new FlowControllerHardware;
    connect(flow,&FlowController::flowUpdate,this,&HardwareManager::flowUpdate);
    connect(flow,&FlowController::flowSetpointUpdate,this,&HardwareManager::flowSetpointUpdate);
    connect(flow,&FlowController::pressureUpdate,this,&HardwareManager::gasPressureUpdate);
    connect(flow,&FlowController::pressureSetpointUpdate,this,&HardwareManager::gasPressureSetpointUpdate);
    connect(flow,&FlowController::pressureControlMode,this,&HardwareManager::gasPressureControlMode);
    d_hardwareMap.emplace(flow->d_key,flow);
#endif

#ifdef BC_PCONTROLLER
    auto pc = new PressureControllerHardware;
    connect(pc,&PressureController::pressureUpdate,this,&HardwareManager::pressureUpdate);
    connect(pc,&PressureController::pressureSetpointUpdate,this,&HardwareManager::pressureSetpointUpdate);
    connect(pc,&PressureController::pressureControlMode,this,&HardwareManager::pressureControlMode);
    d_hardwareMap.emplace(pc->d_key,pc);
#endif

#ifdef BC_TEMPCONTROLLER
    auto tc = new TemperatureControllerHardware;
    connect(tc,&TemperatureController::channelEnableUpdate,this,&HardwareManager::temperatureEnableUpdate);
    connect(tc,&TemperatureController::temperatureUpdate,this,&HardwareManager::temperatureUpdate);
    d_hardwareMap.emplace(tc->d_key,tc);
#endif

#ifdef BC_IOBOARD
    auto iob = new IOBoardHardware;
    d_hardwareMap.emplace(iob->d_key,iob);
#endif

#ifdef BC_LIF
    auto lsc = new LifScopeHardware();
    connect(lsc,&LifScope::waveformRead,this,&HardwareManager::lifScopeShotAcquired);
    connect(lsc,&LifScope::configAcqComplete,this,&HardwareManager::lifConfigAcqStarted);
    d_hardwareMap.emplace(lsc->d_key,lsc);

    auto ll = new LifLaserHardware();
    connect(ll,&LifLaser::laserPosUpdate,this,&HardwareManager::lifLaserPosUpdate);
    d_hardwareMap.emplace(ll->d_key,ll);
#endif

    //write arrays of the connected devices for use in the Hardware Settings menu
    //first array is for all objects accessible to the hardware manager
    setArray(BC::Key::allHw,{},false);
    setArray(BC::Key::Comm::tcp,{},false);
    setArray(BC::Key::Comm::rs232,{},false);
    setArray(BC::Key::Comm::gpib,{},false);
    setArray(BC::Key::Comm::custom,{},false);
    for(auto hwit = d_hardwareMap.cbegin(); hwit != d_hardwareMap.cend(); ++hwit)
    {        
        auto obj = hwit->second;
        connect(obj,&HardwareObject::logMessage,[this,obj](QString msg, LogHandler::MessageCode mc){
            emit logMessage(QString("%1: %2").arg(obj->d_name).arg(msg),mc);
        });
        connect(obj,&HardwareObject::connected,[obj,this](bool success, QString msg){
            connectionResult(obj,success,msg);
        });
        connect(obj,&HardwareObject::auxDataRead,[obj,this](AuxDataStorage::AuxDataMap m){
            AuxDataStorage::AuxDataMap out;
            for(auto it = m.cbegin(); it != m.cend(); ++it)
                out.insert({AuxDataStorage::makeKey(obj->d_key,obj->d_subKey,it->first),it->second});
            emit auxData(out);
        });
        connect(obj,&HardwareObject::auxDataRead,[obj,this](AuxDataStorage::AuxDataMap m){
            AuxDataStorage::AuxDataMap out;
            for(auto it = m.cbegin(); it != m.cend(); ++it)
                out.insert({AuxDataStorage::makeKey(obj->d_key,obj->d_subKey,it->first),it->second});
            emit validationData(out);
        });
        connect(obj,&HardwareObject::rollingDataRead,[obj,this](AuxDataStorage::AuxDataMap m){
            AuxDataStorage::AuxDataMap out;
            for(auto it = m.cbegin(); it != m.cend(); ++it)
                out.insert({AuxDataStorage::makeKey(obj->d_key,obj->d_subKey,it->first),it->second});
            emit rollingData(out,QDateTime::currentDateTime());
        });
        connect(this,&HardwareManager::beginAcquisition,obj,&HardwareObject::beginAcquisition);
        connect(this,&HardwareManager::endAcquisition,obj,&HardwareObject::endAcquisition);



        appendArrayMap(BC::Key::allHw,{
                           {BC::Key::HW::key,obj->d_key},
                           {BC::Key::HW::subKey,obj->d_subKey},
                           {BC::Key::HW::name,obj->d_name},
                           {BC::Key::HW::critical,obj->d_critical}
                       });
        switch(obj->d_commType)
        {
        case CommunicationProtocol::Tcp:
            appendArrayMap(BC::Key::Comm::tcp,{
                               {BC::Key::HW::key,obj->d_key},
                               {BC::Key::HW::subKey,obj->d_subKey},
                               {BC::Key::HW::name,obj->d_name}
                           });
            break;
        case CommunicationProtocol::Rs232:
            appendArrayMap(BC::Key::Comm::rs232,{
                               {BC::Key::HW::key,obj->d_key},
                               {BC::Key::HW::subKey,obj->d_subKey},
                               {BC::Key::HW::name,obj->d_name}
                           });
            break;
        case CommunicationProtocol::Gpib:
            appendArrayMap(BC::Key::Comm::gpib,{
                               {BC::Key::HW::key,obj->d_key},
                               {BC::Key::HW::subKey,obj->d_subKey},
                               {BC::Key::HW::name,obj->d_name}
                           });
            break;
        case CommunicationProtocol::Custom:
            appendArrayMap(BC::Key::Comm::custom,{
                               {BC::Key::HW::key,obj->d_key},
                               {BC::Key::HW::subKey,obj->d_subKey},
                               {BC::Key::HW::name,obj->d_name}
                           });
            break;
        default:
            break;
        }

        obj->buildCommunication(gpib);

        if(hwit->first == BC::Key::gpibController && gpibThread)
            obj->moveToThread(gpibThread);
        else if(obj->d_commType == CommunicationProtocol::Gpib && gpibThread)
            obj->moveToThread(gpibThread);
        else if(obj->d_threaded)
        {
            auto t = new QThread(this);
            t->setObjectName(obj->d_key+"Thread");
            obj->moveToThread(t);
            connect(t,&QThread::started,obj,&HardwareObject::bcInitInstrument);
        }
        else
            obj->setParent(this);
    }

    save();
}

HardwareManager::~HardwareManager()
{
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
            if(hw->d_key != BC::Key::gpibController && hw->d_commType != CommunicationProtocol::Gpib)
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
                    return obj->prepareForExperiment(*exp);
                },Qt::BlockingQueuedConnection,&success);
            else
                success = obj->prepareForExperiment(*exp);

            if(!success)
            {
                emit logMessage(QString("Error initializing %1").arg(obj->d_name),LogHandler::Error);
                break;
            }
        }
    }

    exp->d_hardwareSuccess = success;
    exp->d_hardware = currentHardware();
    //any additional synchronous initialization can be performed here, before experimentInitialized() is emitted


    emit experimentInitialized(exp);

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

void HardwareManager::testObjectConnection(const QString type, const QString key)
{
    Q_UNUSED(type)
    auto it = d_hardwareMap.find(key);
    if(it == d_hardwareMap.end())
        emit testComplete(key,false,QString("Device not found!"));
    else
    {
        auto obj = it->second;
        QMetaObject::invokeMethod(obj,&HardwareObject::bcTestConnection);
    }
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

void HardwareManager::setPGenSetting(int index, PulseGenConfig::Setting s, QVariant val)
{
    auto pGen = findHardware<PulseGenerator>(BC::Key::PGen::key);
    if(pGen)
        QMetaObject::invokeMethod(pGen,[pGen,index,s,val](){ pGen->setPGenSetting(index,s,val); });

}

void HardwareManager::setPGenConfig(const PulseGenConfig &c)
{
    auto pGen = findHardware<PulseGenerator>(BC::Key::PGen::key);
    if(pGen)
        QMetaObject::invokeMethod(pGen,[pGen,c](){ pGen->setAll(c); });
}

void HardwareManager::setPGenRepRate(double r)
{
    auto pGen = findHardware<PulseGenerator>(BC::Key::PGen::key);
    if(pGen)
        QMetaObject::invokeMethod(pGen,[pGen,r](){ pGen->setRepRate(r); });
}

PulseGenConfig HardwareManager::getPGenConfig()
{
    PulseGenConfig out;
    auto pg = findHardware<PulseGenerator>(BC::Key::PGen::key);
    if(pg)
    {
        if(pg->thread() != QThread::currentThread())
            QMetaObject::invokeMethod(pg,&PulseGenerator::config,Qt::BlockingQueuedConnection,&out);
        else
            out = pg->config();
    }

    return out;
}

void HardwareManager::setFlowSetpoint(int index, double val)
{
    auto flow = findHardware<FlowController>(BC::Key::Flow::flowController);
    if(flow)
        QMetaObject::invokeMethod(flow,[flow,index,val](){flow->setFlowSetpoint(index,val);});
}

void HardwareManager::setFlowChannelName(int index, QString name)
{
    auto flow = findHardware<FlowController>(BC::Key::Flow::flowController);
    if(flow)
        QMetaObject::invokeMethod(flow,[flow,index,name](){flow->setChannelName(index,name);});
}

void HardwareManager::setGasPressureSetpoint(double val)
{
    auto flow = findHardware<FlowController>(BC::Key::Flow::flowController);
    if(flow)
        QMetaObject::invokeMethod(flow,[flow,val](){flow->setPressureSetpoint(val);});
}

void HardwareManager::setGasPressureControlMode(bool en)
{
    auto flow = findHardware<FlowController>(BC::Key::Flow::flowController);
    if(flow)
        QMetaObject::invokeMethod(flow,[flow,en](){flow->setPressureControlMode(en);});
}

FlowConfig HardwareManager::getFlowConfig()
{
    FlowConfig out;
    auto fc = findHardware<FlowController>(BC::Key::Flow::flowController);
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
    std::map<QString, QStringList> out;
    for(auto &[key,obj] : d_hardwareMap)
        out.insert_or_assign(key,obj->validationKeys());

    return out;
}

std::map<QString, QString> HardwareManager::currentHardware() const
{
    std::map<QString,QString> out;
    for(auto &[key,obj] : d_hardwareMap)
        out.insert_or_assign(key,obj->d_subKey);

    return out;
}


void HardwareManager::setPressureSetpoint(double val)
{
    auto pc = findHardware<PressureController>(BC::Key::PController::key);
    if(pc)
        QMetaObject::invokeMethod(pc,[pc,val](){pc->setPressureSetpoint(val);});
}

void HardwareManager::setPressureControlMode(bool en)
{
    auto pc = findHardware<PressureController>(BC::Key::PController::key);
    if(pc)
        QMetaObject::invokeMethod(pc,[pc,en](){pc->setPressureControlMode(en);});
}

void HardwareManager::openGateValve()
{
    auto pc = findHardware<PressureController>(BC::Key::PController::key);
    if(pc)
        QMetaObject::invokeMethod(pc,&PressureController::openGateValve);
}

void HardwareManager::closeGateValve()
{
    auto pc = findHardware<PressureController>(BC::Key::PController::key);
    if(pc)
        QMetaObject::invokeMethod(pc,&PressureController::closeGateValve);
}

PressureControllerConfig HardwareManager::getPressureControllerConfig()
{
    PressureControllerConfig out;
    auto pc = findHardware<PressureController>(BC::Key::PController::key);
    if(pc)
    {
        if(pc->thread() != QThread::currentThread())
            QMetaObject::invokeMethod(pc,&PressureController::getConfig,Qt::BlockingQueuedConnection,&out);
        else
            out = pc->getConfig();
    }

    return out;
}


void HardwareManager::setTemperatureChannelEnabled(int ch, bool en)
{
    auto tc = findHardware<TemperatureController>(BC::Key::TC::key);
    if(tc)
        QMetaObject::invokeMethod(tc,[tc,ch,en](){ tc->setChannelEnabled(ch,en);});
}

void HardwareManager::setTemperatureChannelName(int ch, const QString name)
{
    auto tc = findHardware<TemperatureController>(BC::Key::TC::key);
    if(tc)
        QMetaObject::invokeMethod(tc,[tc,ch,name](){ tc->setChannelName(ch,name);});
}

TemperatureControllerConfig HardwareManager::getTemperatureControllerConfig()
{
    TemperatureControllerConfig out;
    auto tc = findHardware<TemperatureController>(BC::Key::TC::key);
    if(tc)
    {
        if(tc->thread() == QThread::currentThread())
            QMetaObject::invokeMethod(tc,&TemperatureController::getConfig,Qt::BlockingQueuedConnection,&out);
        else
            out = tc->getConfig();
    }

    return out;
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

#ifdef BC_LIF
void HardwareManager::setLifParameters(double delay, double pos)
{
    bool success = true;

    if(!setPGenLifDelay(delay))
        success = false;

    if(!setLifLaserPos(pos))
        success = false;

    emit lifSettingsComplete(success);
}

bool HardwareManager::setPGenLifDelay(double d)
{
    auto pGen = findHardware<PulseGenerator>(BC::Key::PGen::key);
    if(!pGen)
    {
        emit logMessage(QString("Could not set LIF delay because no pulse generator is avaialble."),LogHandler::Error);
        return false;
    }

    if(pGen->thread() == QThread::currentThread())
        return pGen->setLifDelay(d);


    bool out;
    QMetaObject::invokeMethod(pGen,[pGen,d](){ return pGen->setLifDelay(d); },Qt::BlockingQueuedConnection,&out);
    return out;

}

bool HardwareManager::setLifLaserPos(double pos)
{
    auto ll = findHardware<LifLaser>(BC::Key::LifLaser::key);
    if(!ll)
    {
        emit logMessage(QString("Could not set LIF Laser position because no laser is avaialble."),LogHandler::Error);
        return false;
    }

    if(ll->thread() == QThread::currentThread())
    {
        auto p = ll->setPosition(pos);
        return p > 0.0;
    }

    double out;
    QMetaObject::invokeMethod(ll,[ll,pos](){ return ll->setPosition(pos); },Qt::BlockingQueuedConnection,&out);
    return out > 0.0;

}

void HardwareManager::startLifConfigAcq(const LifDigitizerConfig &c)
{
    auto ld = findHardware<LifScope>(BC::Key::LifDigi::lifScope);
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
    auto ld = findHardware<LifScope>(BC::Key::LifDigi::lifScope);
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
#endif
