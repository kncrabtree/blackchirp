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

#include <hardware/hw_h.h>
#include <hardware/core/clock/clock_h.h>

#include <boost/preprocessor/iteration/local.hpp>

#include <QThread>

#ifdef BC_LIF
#include <modules/lif/hardware/lifhw_h.h>
#include <modules/lif/hardware/lifdigitizer/lifscope.h>
#include <modules/lif/hardware/liflaser/liflaser.h>
#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#endif

HardwareManager::HardwareManager(QObject *parent) : QObject(parent), SettingsStorage(BC::Key::hw),
    d_optHwTypes{BC::Key::Flow::flowController,BC::Key::IOB::ioboard,BC::Key::PController::key,BC::Key::PGen::key,BC::Key::TC::key}
{
    //Required hardware: FtmwScope and Clocks
    auto ftmwScope = new BC_FTMWSCOPE;
    connect(ftmwScope,&FtmwScope::shotAcquired,this,&HardwareManager::ftmwScopeShotAcquired);
    d_hardwareMap.emplace(ftmwScope->d_key,ftmwScope);

    pu_clockManager = std::make_unique<ClockManager>();
    connect(pu_clockManager.get(),&ClockManager::logMessage,this,&HardwareManager::logMessage);
    connect(pu_clockManager.get(),&ClockManager::clockFrequencyUpdate,this,&HardwareManager::clockFrequencyUpdate);
    auto cl = pu_clockManager->d_clockList;
    for(int i=0; i<cl.size(); i++)
        d_hardwareMap.emplace(cl.at(i)->d_key,cl.at(i));

#ifdef BC_AWG
    auto awg = new BC_AWG;
    d_hardwareMap.emplace(awg->d_key,awg);
#endif

    QThread* gpibThread = nullptr;
#ifdef BC_GPIBCONTROLLER
    auto gpib = new BC_GPIBCONTROLLER;
    gpibThread = new QThread(this);
    gpibThread->setObjectName(gpib->d_key+"Thread");
    connect(gpibThread,&QThread::started,gpib,&HardwareObject::bcInitInstrument);
    d_hardwareMap.emplace(gpib->d_key,gpib);
#else
    auto gpib = nullptr;
#endif

#ifdef BC_PGEN
    QList<PulseGenerator*> pGenList;

#define BOOST_PP_LOCAL_MACRO(n) pGenList << new BC_PGEN_##n;
#define BOOST_PP_LOCAL_LIMITS (0,BC_NUM_PGEN-1)
#include BOOST_PP_LOCAL_ITERATE()
#undef BOOST_PP_LOCAL_MACRO
#undef BOOST_PP_LOCAL_LIMITS

    for( auto &pGen : pGenList)
    {
        auto k = pGen->d_key;
        connect(pGen,&PulseGenerator::settingUpdate,[this,k](const int ch, const PulseGenConfig::Setting set, const QVariant val){
            emit pGenSettingUpdate(k,ch,set,val);
        });
        connect(pGen,&PulseGenerator::configUpdate,[this,k](const PulseGenConfig cfg){
            emit pGenConfigUpdate(k,cfg);
        });
        d_hardwareMap.emplace(k,pGen);
    }
#endif

#ifdef BC_FLOWCONTROLLER
    QList<FlowController*> fcList;

#define BOOST_PP_LOCAL_MACRO(n) fcList << new BC_FLOWCONTROLLER_##n;
#define BOOST_PP_LOCAL_LIMITS (0,BC_NUM_FLOWCONTROLLER-1)
#include BOOST_PP_LOCAL_ITERATE()
#undef BOOST_PP_LOCAL_MACRO
#undef BOOST_PP_LOCAL_LIMITS

    for(auto flow : fcList)
    {
        auto k = flow->d_key;
        connect(flow,&FlowController::flowUpdate,[this,k](int i, double d){
            emit flowUpdate(k,i,d);
        });
        connect(flow,&FlowController::flowSetpointUpdate,[this,k](int i, double d){
            emit flowSetpointUpdate(k,i,d);
        });
        connect(flow,&FlowController::pressureUpdate,[this,k](double d){
            emit gasPressureUpdate(k,d);
        });
        connect(flow,&FlowController::pressureSetpointUpdate,[this,k](double d){
           emit gasPressureSetpointUpdate(k,d);
        });
        connect(flow,&FlowController::pressureControlMode,[this,k](bool b){
            emit gasPressureControlMode(k,b);
        });
        d_hardwareMap.emplace(k,flow);
    }
#endif

#ifdef BC_PCONTROLLER
    QList<PressureController*> pcList;

#define BOOST_PP_LOCAL_MACRO(n) pcList << new BC_PCONTROLLER_##n;
#define BOOST_PP_LOCAL_LIMITS (0,BC_NUM_PCONTROLLER-1)
#include BOOST_PP_LOCAL_ITERATE()
#undef BOOST_PP_LOCAL_MACRO
#undef BOOST_PP_LOCAL_LIMITS

    for(auto pc : pcList)
    {
        auto k = pc->d_key;
        connect(pc,&PressureController::pressureUpdate,this,[this,k](double d){
            emit pressureUpdate(k,d);
        });
        connect(pc,&PressureController::pressureSetpointUpdate,this,[this,k](double d){
            emit pressureSetpointUpdate(k,d);
        });
        connect(pc,&PressureController::pressureControlMode,this,[this,k](bool b){
            emit pressureControlMode(k,b);
        });
        d_hardwareMap.emplace(pc->d_key,pc);
    }
#endif

#ifdef BC_TEMPCONTROLLER
    QList<TemperatureController*> tcList;

#define BOOST_PP_LOCAL_MACRO(n) tcList << new BC_TEMPCONTROLLER_##n;
#define BOOST_PP_LOCAL_LIMITS (0,BC_NUM_TEMPCONTROLLER-1)
#include BOOST_PP_LOCAL_ITERATE()
#undef BOOST_PP_LOCAL_MACRO
#undef BOOST_PP_LOCAL_LIMITS

    for(auto tc : tcList)
    {
        auto k = tc->d_key;
        connect(tc,&TemperatureController::channelEnableUpdate,this,[this,k](uint i,bool en){
            emit temperatureEnableUpdate(k,i,en);
        });
        connect(tc,&TemperatureController::temperatureUpdate,this,[this,k](uint i, double t) {
            emit temperatureUpdate(k,i,t);
        });
        d_hardwareMap.emplace(k,tc);
    }
#endif

#ifdef BC_IOBOARD
    QList<IOBoard*> iobList;

#define BOOST_PP_LOCAL_MACRO(n) iobList << new BC_IOBOARD_##n;
#define BOOST_PP_LOCAL_LIMITS (0,BC_NUM_IOBOARD-1)
#include BOOST_PP_LOCAL_ITERATE()
#undef BOOST_PP_LOCAL_MACRO
#undef BOOST_PP_LOCAL_LIMITS

    for(auto iob : iobList)
        d_hardwareMap.emplace(iob->d_key,iob);
#endif

#ifdef BC_LIF
    auto lsc = new BC_LIFSCOPE;
    connect(lsc,&LifScope::waveformRead,this,&HardwareManager::lifScopeShotAcquired);
    connect(lsc,&LifScope::configAcqComplete,this,&HardwareManager::lifConfigAcqStarted);
    d_hardwareMap.emplace(lsc->d_key,lsc);

    auto ll = new BC_LIFLASER;
    connect(ll,&LifLaser::laserPosUpdate,this,&HardwareManager::lifLaserPosUpdate);
    connect(ll,&LifLaser::laserFlashlampUpdate,this,&HardwareManager::lifLaserFlashlampUpdate);
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
    exp->d_hardware = currentHardware();

#ifdef BC_LIF
    if(exp->lifEnabled())
    {
        auto ll = findHardware<LifLaser>(BC::Key::hwKey(BC::Key::LifLaser::key,0));
        if(!ll)
        {
            emit logMessage(QString("Could not perform LIF experiment because no laser is avaialble."),LogHandler::Error);
            emit lifSettingsComplete(false);
            exp->d_hardwareSuccess = false;
        }
        else
            connect(ll,&LifLaser::laserPosUpdate,this,&HardwareManager::lifLaserSetComplete,Qt::UniqueConnection);
    }
#endif
    //any additional synchronous initialization can be performed here, before experimentInitialized() is emitted


    emit experimentInitialized(exp);

}

void HardwareManager::experimentComplete()
{
#ifdef BC_LIF
    auto ll = findHardware<LifLaser>(BC::Key::hwKey(BC::Key::LifLaser::key,0));
    if(ll)
        disconnect(ll,&LifLaser::laserPosUpdate,this,&HardwareManager::lifLaserSetComplete);
#endif
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
    PulseGenConfig out;
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
    FlowConfig out;
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
    PressureControllerConfig out;
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
    TemperatureControllerConfig out;
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
    IOBoardConfig out;
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
    for(auto const &[hwKey,_] : d_hardwareMap)
    {
        auto t = BC::Key::parseKey(hwKey);
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
            if(type == BC::Key::PGen::key)
                exp->addOptHwConfig(getPGenConfig(hwKey));
            else if(type == BC::Key::Flow::flowController)
                exp->addOptHwConfig(getFlowConfig(hwKey));
            else if(type == BC::Key::TC::key)
                exp->addOptHwConfig(getTemperatureControllerConfig(hwKey));
            else if(type == BC::Key::PController::key)
                exp->addOptHwConfig(getPressureControllerConfig(hwKey));
            else if(type == BC::Key::IOB::ioboard)
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

#ifdef BC_LIF
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
#endif

    bool out = true;
    for(uint i=0; i<BC_NUM_PGEN; i++)
    {
        auto pGen = findHardware<PulseGenerator>(BC::Key::hwKey(BC::Key::PGen::key,i));


        if(pGen->thread() == QThread::currentThread())
            out &= pGen->setLifDelay(d);
        else
            QMetaObject::invokeMethod(pGen,[pGen,d](){ return pGen->setLifDelay(d); },Qt::BlockingQueuedConnection,&out);
    }

    return out;
}

bool HardwareManager::setLifLaserPos(double pos)
{
    auto ll = findHardware<LifLaser>(BC::Key::hwKey(BC::Key::LifLaser::key,0));
    if(!ll)
    {
        emit logMessage(QString("Could not set LIF Laser position because no laser is avaialble."),LogHandler::Error);
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
    auto ld = findHardware<LifScope>(BC::Key::hwKey(BC::Key::LifDigi::lifScope,0));
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
    auto ld = findHardware<LifScope>(BC::Key::hwKey(BC::Key::LifDigi::lifScope,0));
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
    auto ll = findHardware<LifLaser>(BC::Key::hwKey(BC::Key::LifLaser::key,0));
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
    auto ll = findHardware<LifLaser>(BC::Key::hwKey(BC::Key::LifLaser::key,0));
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
    auto ll = findHardware<LifLaser>(BC::Key::hwKey(BC::Key::LifLaser::key,0));
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
#endif
