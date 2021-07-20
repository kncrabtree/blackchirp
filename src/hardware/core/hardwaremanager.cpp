#include <hardware/core/hardwaremanager.h>

#include <hardware/core/hardwareobject.h>
#include <hardware/core/ftmwdigitizer/ftmwscope.h>
#include <hardware/core/clock/clockmanager.h>
#include <hardware/core/chirpsource/awg.h>
#include <hardware/core/pulsegenerator/pulsegenerator.h>
#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <hardware/core/ioboard/ioboard.h>
#include <hardware/optional/gpibcontroller/gpibcontroller.h>

#include <QThread>

#ifdef BC_PCONTROLLER
#include <hardware/optional/pressurecontroller/pressurecontroller.h>
#endif

#ifdef BC_TEMPCONTROLLER
#include <hardware/optional/tempcontroller/temperaturecontroller.h>
#endif



#ifdef BC_LIF
#include <modules/lif/hardware/lifdigitizer/lifscope.h>
#include <modules/lif/hardware/liflaser/liflaser.h>
#endif

#ifdef BC_MOTOR
#include <modules/motor/hardware/motorcontroller/motorcontroller.h>
#include <modules/motor/hardware/motordigitizer/motoroscilloscope.h>
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

#ifdef BC_GPIBCONTROLLER
    auto gpib = new GpibControllerHardware;
    QThread *gpibThread = new QThread(this);
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
    d_hardwareMap.emplace(tc->d_key,tc);
#endif

#ifdef BC_IOBOARD
    auto iob = new IOBoardHardware;
    d_hardwareMap.emplace(iob->d_key,iob);
#endif

#ifdef BC_LIF
    p_lifScope = new LifScopeHardware();
    connect(p_lifScope,&LifScope::waveformRead,this,&HardwareManager::lifScopeShotAcquired);
    connect(p_lifScope,&LifScope::configUpdated,this,&HardwareManager::lifScopeConfigUpdated);
    d_hardwareList.append(p_lifScope);

    p_lifLaser = new LifLaserHardware();
    connect(p_lifLaser,&LifLaser::laserPosUpdate,this,&HardwareManager::lifLaserPosUpdate);
    d_hardwareList.append(p_lifLaser);
#endif

#ifdef BC_MOTOR
    p_mc = new MotorControllerHardware();
    connect(p_mc,&MotorController::motionComplete,this,&HardwareManager::motorMoveComplete);
    connect(this,&HardwareManager::moveMotorToPosition,p_mc,&MotorController::moveToPosition);
    connect(this,&HardwareManager::motorRest,p_mc,&MotorController::moveToRestingPos);
    connect(p_mc,&MotorController::posUpdate,this,&HardwareManager::motorPosUpdate);
    connect(p_mc,&MotorController::limitStatus,this,&HardwareManager::motorLimitStatus);
    d_hardwareList.append(p_mc);

    p_motorScope = new MotorScopeHardware();
    connect(p_motorScope,&MotorOscilloscope::traceAcquired,this,&HardwareManager::motorTraceAcquired);
    d_hardwareList.append(p_motorScope);
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
        connect(obj,&HardwareObject::logMessage,[this,obj](QString msg, BlackChirp::LogMessageCode mc){
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
                           {BC::Key::HW::critical,obj->d_critical},
                           {BC::Key::HW::threaded,obj->d_threaded}
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

        if(hwit->first == BC::Key::gpibController)
            obj->moveToThread(gpibThread);
        else if(obj->d_commType == CommunicationProtocol::Gpib)
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
    //start all threads and initizlize hw
    for(auto it = d_hardwareMap.cbegin(); it != d_hardwareMap.cend(); ++it)
    {
        auto hw = it->second;
        if(hw->d_commType == CommunicationProtocol::Virtual)
            emit logMessage(QString("%1 is a virtual instrument. Be cautious about taking real measurements!")
                            .arg(hw->d_name),BlackChirp::LogWarning);
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
        BlackChirp::LogMessageCode code = BlackChirp::LogError;
        if(!obj->d_critical)
            code = BlackChirp::LogWarning;
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
            if(obj->thread() != thread())
                QMetaObject::invokeMethod(obj,[obj,exp](){
                    return obj->prepareForExperiment(*exp);
                },Qt::BlockingQueuedConnection,&success);
            else
                success = obj->prepareForExperiment(*exp);

            if(!success)
                break;
        }
    }

    exp->d_hardwareSuccess = success;
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

void HardwareManager::getAuxData()
{
    for(auto it = d_hardwareMap.cbegin(); it != d_hardwareMap.cend(); ++it)
    {
        auto obj = it->second;
        QMetaObject::invokeMethod(obj,&HardwareObject::bcReadAuxData);
    }
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

void HardwareManager::setPGenConfig(const PulseGenConfig c)
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

void HardwareManager::setFlowSetpoint(int index, double val)
{
    auto flow = findHardware<FlowController>(BC::Key::Flow::flowController);
    if(flow)
        QMetaObject::invokeMethod(flow,[flow,index,val](){flow->setFlowSetpoint(index,val);});
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

std::map<QString, QStringList> HardwareManager::validationKeys() const
{
    std::map<QString, QStringList> out;
    for(auto &[key,obj] : d_hardwareMap)
        out.insert_or_assign(key,obj->validationKeys());

    return out;
}

#ifdef BC_PCONTROLLER
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
#endif

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
    if(p_pGen->thread() == thread())
        return p_pGen->setLifDelay(d);


    bool out;
    QMetaObject::invokeMethod(p_pGen,"setLifDelay",Qt::BlockingQueuedConnection,
                              Q_RETURN_ARG(bool,out),Q_ARG(double,d));
    return out;

}

void HardwareManager::setLifScopeConfig(const BlackChirp::LifScopeConfig c)
{
    if(p_lifScope->thread() == thread())
        p_lifScope->setAll(c);
    else
        QMetaObject::invokeMethod(p_lifScope,"setAll",Q_ARG(BlackChirp::LifScopeConfig,c));
}

bool HardwareManager::setLifLaserPos(double pos)
{
    if(p_lifLaser->thread() == thread())
    {
        auto p = p_lifLaser->setPosition(pos);
        return p > 0.0;
    }

    double out;
    QMetaObject::invokeMethod(p_lifLaser,"setPosition",Qt::BlockingQueuedConnection,
                              Q_RETURN_ARG(double,out),Q_ARG(double,pos));
    return out > 0.0;

}
#endif
