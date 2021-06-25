#include <hardware/core/hardwaremanager.h>

HardwareManager::HardwareManager(QObject *parent) : QObject(parent), SettingsStorage(BC::Key::hw), d_responseCount(0)
{

#ifdef BC_GPIBCONTROLLER
    GpibController *gpib = new GpibControllerHardware();
    QThread *gpibThread = new QThread(this);
    d_hardwareList.append(gpib);
#endif

    p_ftmwScope = new FtmwScopeHardware();
    connect(p_ftmwScope,&FtmwScope::shotAcquired,this,&HardwareManager::ftmwScopeShotAcquired);
    d_hardwareList.append(p_ftmwScope);

    p_awg = new AwgHardware();
    d_hardwareList.append(p_awg);

    p_clockManager = new ClockManager(this);
    connect(p_clockManager,&ClockManager::logMessage,this,&HardwareManager::logMessage);
    connect(p_clockManager,&ClockManager::clockFrequencyUpdate,this,&HardwareManager::clockFrequencyUpdate);
    auto cl = p_clockManager->clockList();
    for(int i=0; i<cl.size(); i++)
        d_hardwareList.append(cl.at(i));

    p_pGen = new PulseGeneratorHardware();
    connect(p_pGen,&PulseGenerator::settingUpdate,this,&HardwareManager::pGenSettingUpdate);
    connect(p_pGen,&PulseGenerator::configUpdate,this,&HardwareManager::pGenConfigUpdate);
    connect(p_pGen,&PulseGenerator::repRateUpdate,this,&HardwareManager::pGenRepRateUpdate);
    d_hardwareList.append(p_pGen);

    p_flow = new FlowControllerHardware();
    connect(p_flow,&FlowController::flowUpdate,this,&HardwareManager::flowUpdate);
    connect(p_flow,&FlowController::flowSetpointUpdate,this,&HardwareManager::flowSetpointUpdate);
    connect(p_flow,&FlowController::pressureUpdate,this,&HardwareManager::gasPressureUpdate);
    connect(p_flow,&FlowController::pressureSetpointUpdate,this,&HardwareManager::gasPressureSetpointUpdate);
    connect(p_flow,&FlowController::pressureControlMode,this,&HardwareManager::gasPressureControlMode);
    d_hardwareList.append(p_flow);

#ifdef BC_PCONTROLLER
    p_pc = new PressureControllerHardware();
    connect(p_pc,&PressureController::pressureUpdate,this,&HardwareManager::pressureUpdate);
    connect(p_pc,&PressureController::pressureSetpointUpdate,this,&HardwareManager::pressureSetpointUpdate);
    connect(p_pc,&PressureController::pressureControlMode,this,&HardwareManager::pressureControlMode);
    d_hardwareList.append(p_pc);
#endif

#ifdef BC_TEMPCONTROLLER
    p_tc = new TemperatureControllerHardware();
    d_hardwareList.append(p_tc);
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

    p_iob = new IOBoardHardware();
    d_hardwareList.append(p_iob);

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
    setArray(BC::Key::tcp,{},false);
    setArray(BC::Key::rs232,{},false);
    setArray(BC::Key::gpib,{},false);
    setArray(BC::Key::custom,{},false);
    for(int i=0;i<d_hardwareList.size();i++)
    {
        HardwareObject *obj = d_hardwareList.at(i);

        connect(obj,&HardwareObject::logMessage,[=](QString msg, BlackChirp::LogMessageCode mc){
            emit logMessage(QString("%1: %2").arg(obj->d_name).arg(msg),mc);
        });
        connect(obj,&HardwareObject::connected,[=](bool success, QString msg){ connectionResult(obj,success,msg); });
        connect(obj,&HardwareObject::timeDataRead,[=](const QList<QPair<QString,QVariant>> l,bool plot){ emit timeData(l,plot); });
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
            appendArrayMap(BC::Key::tcp,{
                               {BC::Key::HW::key,obj->d_key},
                               {BC::Key::HW::subKey,obj->d_subKey},
                               {BC::Key::HW::name,obj->d_name}
                           });
            break;
        case CommunicationProtocol::Rs232:
            appendArrayMap(BC::Key::rs232,{
                               {BC::Key::HW::key,obj->d_key},
                               {BC::Key::HW::subKey,obj->d_subKey},
                               {BC::Key::HW::name,obj->d_name}
                           });
            break;
        case CommunicationProtocol::Gpib:
            appendArrayMap(BC::Key::gpib,{
                               {BC::Key::HW::key,obj->d_key},
                               {BC::Key::HW::subKey,obj->d_subKey},
                               {BC::Key::HW::name,obj->d_name}
                           });
            break;
        case CommunicationProtocol::Custom:
            appendArrayMap(BC::Key::custom,{
                               {BC::Key::HW::key,obj->d_key},
                               {BC::Key::HW::subKey,obj->d_subKey},
                               {BC::Key::HW::name,obj->d_name}
                           });
            break;
        default:
            break;
        }



#ifdef BC_GPIBCONTROLLER
        obj->buildCommunication(gpib);
#else
        obj->buildCommunication();
#endif

#ifdef BC_GPIBCONTROLLER
        if(obj == gpib)
            obj->moveToThread(gpibThread);
        else if(obj->d_commType == CommunicationProtocol::Gpib)
            obj->moveToThread(gpibThread);
        else
#endif
        if(obj->d_threaded)
            obj->moveToThread(new QThread(this));
        else
            obj->setParent(this);

        QThread *t = d_hardwareList.at(i)->thread();
        if(t != thread())
        {
            connect(t,&QThread::started,obj,&HardwareObject::bcInitInstrument);
            connect(t,&QThread::finished,obj,&HardwareObject::deleteLater);
        }
    }


    save();
}

HardwareManager::~HardwareManager()
{
    //note, the hardwareObjects are deleted when the threads exit
    while(!d_hardwareList.isEmpty())
    {
        auto hw = d_hardwareList.takeFirst();
        if(hw->thread() != thread())
        {
            hw->thread()->quit();
            hw->thread()->wait();
        }
        else
            hw->deleteLater();
    }
}

void HardwareManager::initialize()
{
    //start all threads and initizlize hw
    for(int i=0;i<d_hardwareList.size();i++)
    {
        auto hw = d_hardwareList.at(i);
        if(hw->d_commType == CommunicationProtocol::Virtual)
            emit logMessage(QString("%1 is a virtual instrument. Be cautious about taking real measurements!")
                            .arg(hw->d_name),BlackChirp::LogWarning);
        if(hw->thread() != thread())
        {
            if(!hw->thread()->isRunning())
                hw->thread()->start();
        }
        else
            hw->bcInitInstrument();
    }

    emit hwInitializationComplete();
}

void HardwareManager::connectionResult(HardwareObject *obj, bool success, QString msg)
{
    if(d_responseCount < d_hardwareList.size())
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

   //TODO: implement re-test like in QtFTM?
    emit abortAcquisition();

    checkStatus();
}

void HardwareManager::sleep(bool b)
{
    for(int i=0; i<d_hardwareList.size(); i++)
    {
        HardwareObject *obj = d_hardwareList.at(i);
        if(obj->isConnected())
        {
            if(obj->thread() == thread())
                obj->sleep(b);
            else
                QMetaObject::invokeMethod(obj,"sleep",Q_ARG(bool,b));
        }
    }
}

void HardwareManager::initializeExperiment(Experiment exp)
{
    //do initialization
    bool success = p_clockManager->prepareForExperiment(exp);

    if(success) {
        for(int i=0;i<d_hardwareList.size();i++)
        {
            HardwareObject *obj = d_hardwareList.at(i);
            if(obj->thread() != thread())
                QMetaObject::invokeMethod(obj,"prepareForExperiment",Qt::BlockingQueuedConnection,Q_RETURN_ARG(bool,success),Q_ARG(Experiment &,exp));
            else
                success = obj->prepareForExperiment(exp);

            if(!success)
                break;
        }
    }

    //any additional synchronous initialization can be performed here
    emit experimentInitialized(exp);

}

void HardwareManager::testAll()
{
    for(int i=0; i<d_hardwareList.size(); i++)
    {
        HardwareObject *obj = d_hardwareList.at(i);
        if(obj->thread() == thread())
            obj->bcTestConnection();
        else
            QMetaObject::invokeMethod(obj,"bcTestConnection");
    }

    checkStatus();
}

void HardwareManager::testObjectConnection(const QString type, const QString key)
{
    Q_UNUSED(type)
    HardwareObject *obj = nullptr;
    for(int i=0; i<d_hardwareList.size();i++)
    {
        if(d_hardwareList.at(i)->d_key == key)
            obj = d_hardwareList.at(i);
    }
    if(obj == nullptr)
        emit testComplete(key,false,QString("Device not found!"));
    else
    {
        if(obj->thread() == thread())
            obj->bcTestConnection();
        else
            QMetaObject::invokeMethod(obj,"bcTestConnection");
    }
}

void HardwareManager::getTimeData()
{
    for(int i=0; i<d_hardwareList.size(); i++)
    {
        HardwareObject *obj = d_hardwareList.at(i);
        if(obj->thread() == thread())
            obj->bcReadTimeData();
        else
            QMetaObject::invokeMethod(obj,"bcReadTimeData");
    }
}

void HardwareManager::setClocks(const RfConfig rfc)
{
    auto l = rfc.getClocks();
    for(auto it = l.constBegin(); it != l.constEnd(); it++)
        p_clockManager->setClockFrequency(it.key(),it.value().desiredFreqMHz);

    emit allClocksReady();
}

void HardwareManager::setPGenSetting(int index, PulseGenConfig::Setting s, QVariant val)
{
    QMetaObject::invokeMethod(p_pGen,[this,index,s,val](){ p_pGen->setPGenSetting(index,s,val); });
}

void HardwareManager::setPGenConfig(const PulseGenConfig c)
{
    QMetaObject::invokeMethod(p_pGen,[this,c](){ p_pGen->setAll(c); });
}

void HardwareManager::setPGenRepRate(double r)
{
    QMetaObject::invokeMethod(p_pGen,[this,r](){ p_pGen->setRepRate(r); });
}

void HardwareManager::setFlowSetpoint(int index, double val)
{
    if(p_flow->thread() == thread())
        p_flow->setFlowSetpoint(index,val);
    else
        QMetaObject::invokeMethod(p_flow,"setFlowSetpoint",Q_ARG(int,index),Q_ARG(double,val));
}

void HardwareManager::setGasPressureSetpoint(double val)
{
    if(p_flow->thread() == thread())
        p_flow->setPressureSetpoint(val);
    else
        QMetaObject::invokeMethod(p_flow,"setPressureSetpoint",Q_ARG(double,val));
}

void HardwareManager::setGasPressureControlMode(bool en)
{
    if(p_flow->thread() == thread())
        p_flow->setPressureControlMode(en);
    else
        QMetaObject::invokeMethod(p_flow,"setPressureControlMode",Q_ARG(bool,en));
}

#ifdef BC_PCONTROLLER
void HardwareManager::setPressureSetpoint(double val)
{
    if(p_pc->thread() == thread())
        p_pc->setPressureSetpoint(val);
    else
        QMetaObject::invokeMethod(p_pc,"setPressureSetpoint",Q_ARG(double,val));
}

void HardwareManager::setPressureControlMode(bool en)
{
    if(p_pc->thread() == thread())
        p_pc->setPressureControlMode(en);
    else
        QMetaObject::invokeMethod(p_pc,"setPressureControlMode",Q_ARG(bool,en));
}

void HardwareManager::openGateValve()
{
    if(p_pc->thread() == thread())
        p_pc->openGateValve();
    else
        QMetaObject::invokeMethod(p_pc,"openGateValve");
}

void HardwareManager::closeGateValve()
{
    if(p_pc->thread() == thread())
        p_pc->closeGateValve();
    else
        QMetaObject::invokeMethod(p_pc,"closeGateValve");
}
#endif

void HardwareManager::checkStatus()
{
    //gotta wait until all instruments have responded
    if(d_responseCount < d_hardwareList.size())
        return;

    bool success = true;
    for(int i=0; i<d_hardwareList.size(); i++)
    {
        HardwareObject *obj = d_hardwareList.at(i);
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
