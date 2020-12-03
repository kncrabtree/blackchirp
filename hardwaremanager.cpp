#include "hardwaremanager.h"

#include <QSettings>

#include "hardwareobject.h"
#include "ftmwscope.h"
#include "clockmanager.h"
#include "awg.h"
#include "pulsegenerator.h"
#include "flowcontroller.h"
#include "ioboard.h"

#ifdef BC_PCONTROLLER
#include "pressurecontroller.h"
#endif

#ifdef BC_TEMPCONTROLLER
#include "temperaturecontroller.h"
#endif

#ifdef BC_GPIBCONTROLLER
#include "gpibcontroller.h"
#endif

#ifdef BC_LIF
#include "lifscope.h"
#endif

#ifdef BC_MOTOR
#include "motorcontroller.h"
#include "motoroscilloscope.h"
#endif

HardwareManager::HardwareManager(QObject *parent) : QObject(parent), d_responseCount(0)
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
    connect(p_flow,&FlowController::channelNameUpdate,this,&HardwareManager::flowNameUpdate);
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
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("hardware"));
    s.remove("");
    s.beginWriteArray("instruments");
    for(int i=0;i<d_hardwareList.size();i++)
    {
        HardwareObject *obj = d_hardwareList.at(i);
        s.setArrayIndex(i);
        s.setValue(QString("key"),obj->key());
        s.setValue(QString("subKey"),obj->subKey());
        s.setValue(QString("prettyName"),obj->name());
        s.setValue(QString("critical"),obj->isCritical());
    }
    s.endArray();
    s.endGroup();

    //now an array for all TCP instruments
    s.beginGroup(QString("tcp"));
    s.remove("");
    s.beginWriteArray("instruments");
    int index=0;
    for(int i=0;i<d_hardwareList.size();i++)
    {
        if(d_hardwareList.at(i)->type() == CommunicationProtocol::Tcp)
        {
            s.setArrayIndex(index);
            s.setValue(QString("key"),d_hardwareList.at(i)->key());
            s.setValue(QString("subKey"),d_hardwareList.at(i)->subKey());
            index++;
        }
    }
    s.endArray();
    s.endGroup();

    //now an array for all RS232 instruments
    s.beginGroup(QString("rs232"));
    s.remove("");
    s.beginWriteArray("instruments");
    index=0;
    for(int i=0;i<d_hardwareList.size();i++)
    {
        if(d_hardwareList.at(i)->type() == CommunicationProtocol::Rs232)
        {
            s.setArrayIndex(index);
            s.setValue(QString("key"),d_hardwareList.at(i)->key());
            s.setValue(QString("subKey"),d_hardwareList.at(i)->subKey());
            index++;
        }
    }
    s.endArray();
    s.endGroup();

    //now an array for all GPIB instruments
    s.beginGroup(QString("gpib"));
    s.remove("");
    s.beginWriteArray("instruments");
    index=0;
    for(int i=0;i<d_hardwareList.size();i++)
    {
       if(d_hardwareList.at(i)->type() == CommunicationProtocol::Gpib)
        {
            s.setArrayIndex(index);
            s.setValue(QString("key"),d_hardwareList.at(i)->key());
            s.setValue(QString("subKey"),d_hardwareList.at(i)->subKey());
            index++;
        }
    }
    s.endArray();
    s.endGroup();

    //now an array for all custom instruments
    s.beginGroup(QString("custom"));
    s.remove("");
    s.beginWriteArray("instruments");
    index = 0;
    for(int i=0;i<d_hardwareList.size();i++)
    {
       if(d_hardwareList.at(i)->type() == CommunicationProtocol::Custom)
        {
            s.setArrayIndex(index);
            s.setValue(QString("key"),d_hardwareList.at(i)->key());
            s.setValue(QString("subKey"),d_hardwareList.at(i)->subKey());
            index++;
        }
    }
    s.endArray();
    s.endGroup();

    //write settings relevant for configuring UI
    s.beginGroup(QString("hwUI"));
    s.setValue(QString("flowChannels"),p_flow->numChannels());
    s.setValue(QString("pGenChannels"),p_pGen->numChannels());
#ifdef BC_PCONTROLLER
    s.setValue(QString("pControllerReadOnly"),p_pc->isReadOnly());
#endif
    s.endGroup();

    for(int i=0;i<d_hardwareList.size();i++)
    {
        HardwareObject *obj = d_hardwareList.at(i);


        s.setValue(QString("%1/prettyName").arg(obj->key()),obj->name());
        s.setValue(QString("%1/subKey").arg(obj->key()),obj->subKey());
        s.setValue(QString("%1/connected").arg(obj->key()),false);
        s.setValue(QString("%1/critical").arg(obj->key()),obj->isCritical());

        connect(obj,&HardwareObject::logMessage,[=](QString msg, BlackChirp::LogMessageCode mc){
            emit logMessage(QString("%1: %2").arg(obj->name()).arg(msg),mc);
        });
        connect(obj,&HardwareObject::connected,[=](bool success, QString msg){ connectionResult(obj,success,msg); });
        connect(obj,&HardwareObject::timeDataRead,[=](const QList<QPair<QString,QVariant>> l,bool plot){ emit timeData(l,plot); });
        connect(this,&HardwareManager::beginAcquisition,obj,&HardwareObject::beginAcquisition);
        connect(this,&HardwareManager::endAcquisition,obj,&HardwareObject::endAcquisition);

#ifdef BC_GPIBCONTROLLER
        obj->buildCommunication(gpib);
#else
        obj->buildCommunication();
#endif

#ifdef BC_GPIBCONTROLLER
        if(obj == gpib)
            obj->moveToThread(gpibThread);
        else if(obj->type() == CommunicationProtocol::Gpib)
            obj->moveToThread(gpibThread);
        else
#endif
        if(obj->isThreaded())
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

    s.sync();
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
        if(hw->type() == CommunicationProtocol::Virtual)
            emit logMessage(QString("%1 is a virtual instrument. Be cautious about taking real measurements!")
                            .arg(hw->name()),BlackChirp::LogWarning);
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
        emit logMessage(obj->name().append(QString(": Connected successfully.")));
    }
    else
    {
        disconnect(obj,&HardwareObject::hardwareFailure,this,&HardwareManager::hardwareFailure);
        BlackChirp::LogMessageCode code = BlackChirp::LogError;
        if(!obj->isCritical())
            code = BlackChirp::LogWarning;
        emit logMessage(obj->name().append(QString(": Connection failed!")),code);
        if(!msg.isEmpty())
            emit logMessage(msg,code);
    }

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.setValue(QString("%1/connected").arg(obj->key()),success);
    s.sync();

    emit testComplete(obj->name(),success,msg);
    checkStatus();
}

void HardwareManager::hardwareFailure()
{
    HardwareObject *obj = dynamic_cast<HardwareObject*>(sender());
    if(obj == nullptr)
        return;

    disconnect(obj,&HardwareObject::hardwareFailure,this,&HardwareManager::hardwareFailure);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.setValue(QString("%1/connected").arg(obj->key()),false);
    s.sync();

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
    p_clockManager->prepareForExperiment(exp);

    for(int i=0;i<d_hardwareList.size();i++)
    {
        HardwareObject *obj = d_hardwareList.at(i);
        if(obj->thread() != thread())
            QMetaObject::invokeMethod(obj,"prepareForExperiment",Qt::BlockingQueuedConnection,Q_RETURN_ARG(Experiment,exp),Q_ARG(Experiment,exp));
        else
            exp = obj->prepareForExperiment(exp);

        if(!exp.hardwareSuccess())
            break;
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
        if(d_hardwareList.at(i)->key() == key)
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

void HardwareManager::setPGenSetting(int index, BlackChirp::PulseSetting s, QVariant val)
{
    if(p_pGen->thread() == thread())
    {
        p_pGen->set(index,s,val);
        return;
    }

    QMetaObject::invokeMethod(p_pGen,"set",Q_ARG(int,index),Q_ARG(BlackChirp::PulseSetting,s),Q_ARG(QVariant,val));
}

void HardwareManager::setPGenConfig(const PulseGenConfig c)
{
    if(p_pGen->thread() == thread())
        p_pGen->setAll(c);
    else
        QMetaObject::invokeMethod(p_pGen,"setAll",Q_ARG(PulseGenConfig,c));
}

void HardwareManager::setPGenRepRate(double r)
{
    if(p_pGen->thread() == thread())
        p_pGen->setRepRate(r);
    else
        QMetaObject::invokeMethod(p_pGen,"setRepRate",Q_ARG(double,r));
}

void HardwareManager::setFlowChannelName(int index, QString name)
{
    if(p_flow->thread() == thread())
        p_flow->setChannelName(index,name);
    else
        QMetaObject::invokeMethod(p_flow,"setChannelName",Q_ARG(int,index),Q_ARG(QString,name));
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
        if(!obj->isConnected() && obj->isCritical())
            success = false;
    }

    emit allHardwareConnected(success);
}

#ifdef BC_LIF
void HardwareManager::setLifParameters(double delay, double frequency)
{
    bool success = true;

    if(!setPGenLifDelay(delay))
        success = false;
    Q_UNUSED(frequency)

    emit lifSettingsComplete(success);
}

bool HardwareManager::setPGenLifDelay(double d)
{
    if(p_pGen->thread() == thread())
        return p_pGen->setLifDelay(d);
    else
    {
        bool out;
        QMetaObject::invokeMethod(p_pGen,"setLifDelay",Qt::BlockingQueuedConnection,
                                  Q_RETURN_ARG(bool,out),Q_ARG(double,d));
        return out;
    }
}

void HardwareManager::setLifScopeConfig(const BlackChirp::LifScopeConfig c)
{
    if(p_lifScope->thread() == thread())
        p_lifScope->setAll(c);
    else
        QMetaObject::invokeMethod(p_lifScope,"setAll",Q_ARG(BlackChirp::LifScopeConfig,c));
}
#endif
