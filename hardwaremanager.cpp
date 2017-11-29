#include "hardwaremanager.h"

#include <QSettings>

#include "hardwareobject.h"
#include "ftmwscope.h"
#include "synthesizer.h"
#include "awg.h"
#include "pulsegenerator.h"
#include "flowcontroller.h"
#include "ioboard.h"

#ifdef BC_PCONTROLLER
#include "pressurecontroller.h"
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

}

HardwareManager::~HardwareManager()
{
    //note, the hardwareObjects are deleted when the threads exit
    while(!d_hardwareList.isEmpty())
    {
        QPair<HardwareObject*,QThread*> p = d_hardwareList.takeFirst();
        if(p.second != nullptr)
        {
            p.second->quit();
            p.second->wait();
        }
        else
            p.first->deleteLater();
    }
}

void HardwareManager::initialize()
{

#ifdef BC_GPIBCONTROLLER
    GpibController *gpib = new GpibControllerHardware();
    QThread *gpibThread = new QThread(this);
    d_hardwareList.append(qMakePair(gpib,gpibThread));
#endif

    p_ftmwScope = new FtmwScopeHardware();
    connect(p_ftmwScope,&FtmwScope::shotAcquired,this,&HardwareManager::ftmwScopeShotAcquired);
    d_hardwareList.append(qMakePair(p_ftmwScope,nullptr));

    p_awg = new AwgHardware();
    d_hardwareList.append(qMakePair(p_awg,nullptr));

    p_synth = new SynthesizerHardware();
    connect(p_synth,&Synthesizer::txFreqRead,this,&HardwareManager::valonTxFreqRead);
    connect(p_synth,&Synthesizer::rxFreqRead,this,&HardwareManager::valonRxFreqRead);
    d_hardwareList.append(qMakePair(p_synth,nullptr));

    p_pGen = new PulseGeneratorHardware();
    connect(p_pGen,&PulseGenerator::settingUpdate,this,&HardwareManager::pGenSettingUpdate);
    connect(p_pGen,&PulseGenerator::configUpdate,this,&HardwareManager::pGenConfigUpdate);
    connect(p_pGen,&PulseGenerator::repRateUpdate,this,&HardwareManager::pGenRepRateUpdate);
    d_hardwareList.append(qMakePair(p_pGen,nullptr));

    p_flow = new FlowControllerHardware();
    connect(p_flow,&FlowController::flowUpdate,this,&HardwareManager::flowUpdate);
    connect(p_flow,&FlowController::channelNameUpdate,this,&HardwareManager::flowNameUpdate);
    connect(p_flow,&FlowController::flowSetpointUpdate,this,&HardwareManager::flowSetpointUpdate);
    connect(p_flow,&FlowController::pressureUpdate,this,&HardwareManager::gasPressureUpdate);
    connect(p_flow,&FlowController::pressureSetpointUpdate,this,&HardwareManager::gasPressureSetpointUpdate);
    connect(p_flow,&FlowController::pressureControlMode,this,&HardwareManager::gasPressureControlMode);
    d_hardwareList.append(qMakePair(p_flow,nullptr));

#ifdef BC_PCONTROLLER
    p_pc = new PressureControllerHardware();
    connect(p_pc,&PressureController::pressureUpdate,this,&HardwareManager::pressureUpdate);
    connect(p_pc,&PressureController::pressureSetpointUpdate,this,&HardwareManager::pressureSetpointUpdate);
    connect(p_pc,&PressureController::pressureControlMode,this,&HardwareManager::pressureControlMode);
    d_hardwareList.append(qMakePair(p_pc,nullptr));
    emit pressureControlReadOnly(p_pc->isReadOnly());
#endif

#ifdef BC_LIF
    p_lifScope = new LifScopeHardware();
    connect(p_lifScope,&LifScope::waveformRead,this,&HardwareManager::lifScopeShotAcquired);
    connect(p_lifScope,&LifScope::configUpdated,this,&HardwareManager::lifScopeConfigUpdated);
    d_hardwareList.append(qMakePair(p_lifScope,nullptr));
#endif

    p_iob = new IOBoardHardware();
    d_hardwareList.append(qMakePair(p_iob,nullptr));

#ifdef BC_MOTOR
    p_mc = new MotorControllerHardware();
    connect(p_mc,&MotorController::motionComplete,this,&HardwareManager::motorMoveComplete);
    connect(this,&HardwareManager::moveMotorToPosition,p_mc,&MotorController::moveToPosition);
    connect(this,&HardwareManager::motorRest,p_mc,&MotorController::moveToRestingPos);
    connect(p_mc,&MotorController::posUpdate,this,&HardwareManager::motorPosUpdate);
    connect(p_mc,&MotorController::limitStatus,this,&HardwareManager::motorLimitStatus);
    d_hardwareList.append(qMakePair(p_mc,nullptr));

    p_motorScope = new MotorScopeHardware();
    connect(p_motorScope,&MotorOscilloscope::traceAcquired,this,&HardwareManager::motorTraceAcquired);
    d_hardwareList.append(qMakePair(p_motorScope,nullptr));
#endif


	//write arrays of the connected devices for use in the Hardware Settings menu
	//first array is for all objects accessible to the hardware manager
	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	s.beginGroup(QString("hardware"));
	s.remove("");
	s.beginWriteArray("instruments");
	for(int i=0;i<d_hardwareList.size();i++)
	{
        HardwareObject *obj = d_hardwareList.at(i).first;
		s.setArrayIndex(i);
        s.setValue(QString("key"),obj->key());
        s.setValue(QString("subKey"),obj->subKey());
        s.setValue(QString("prettyName"),obj->name());
        s.setValue(QString("critical"),obj->isCritical());
        if(obj->type() == CommunicationProtocol::Virtual)
            emit logMessage(QString("%1 is a virtual instrument. Be cautious about taking real measurements!")
                            .arg(obj->name()),BlackChirp::LogWarning);
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
        if(d_hardwareList.at(i).first->type() == CommunicationProtocol::Tcp)
		{
			s.setArrayIndex(index);
			s.setValue(QString("key"),d_hardwareList.at(i).first->key());
            s.setValue(QString("subKey"),d_hardwareList.at(i).first->subKey());
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
        if(d_hardwareList.at(i).first->type() == CommunicationProtocol::Rs232)
		{
			s.setArrayIndex(index);
			s.setValue(QString("key"),d_hardwareList.at(i).first->key());
            s.setValue(QString("subKey"),d_hardwareList.at(i).first->subKey());
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
       if(d_hardwareList.at(i).first->type() == CommunicationProtocol::Gpib)
        {
            s.setArrayIndex(index);
            s.setValue(QString("key"),d_hardwareList.at(i).first->key());
            s.setValue(QString("subKey"),d_hardwareList.at(i).first->subKey());
            index++;
        }
    }
    s.endArray();
    s.endGroup();

	s.sync();

    for(int i=0;i<d_hardwareList.size();i++)
    {
        QThread *thread = d_hardwareList.at(i).second;
        HardwareObject *obj = d_hardwareList.at(i).first;
#ifdef BC_GPIBCONTROLLER
        if(obj->type() == CommunicationProtocol::Gpib)
        {
            thread = gpibThread;
            d_hardwareList[i].second = thread;
        }
        else if(obj->isThreaded() && obj != gpib)
#else
        if(obj->isThreaded())
#endif
        {
            thread = new QThread(this);
            d_hardwareList[i].second = thread;
        }

#ifdef BB_GPIBCONTROLLER
        obj->buildCommunication(gpib);
#else
        obj->buildCommunication();
#endif

        s.setValue(QString("%1/prettyName").arg(obj->key()),obj->name());
        s.setValue(QString("%1/subKey").arg(obj->key()),obj->subKey());
        s.setValue(QString("%1/connected").arg(obj->key()),false);
        s.setValue(QString("%1/critical").arg(obj->key()),obj->isCritical());

        connect(obj,&HardwareObject::logMessage,[=](QString msg, BlackChirp::LogMessageCode mc){
            emit logMessage(QString("%1: %2").arg(obj->name()).arg(msg),mc);
        });
        connect(obj,&HardwareObject::connected,[=](bool success, QString msg){ connectionResult(obj,success,msg); });
        connect(obj,&HardwareObject::timeDataRead,[=](const QList<QPair<QString,QVariant>> l){ emit timeData(l,true); });
        connect(obj,&HardwareObject::timeDataReadNoPlot,[=](const QList<QPair<QString,QVariant>> l){ emit timeData(l,false); });
        connect(this,&HardwareManager::beginAcquisition,obj,&HardwareObject::beginAcquisition);
        connect(this,&HardwareManager::endAcquisition,obj,&HardwareObject::endAcquisition);
        connect(this,&HardwareManager::readTimeData,obj,&HardwareObject::readTimeData);

        if(thread != nullptr)
        {
            connect(thread,&QThread::started,obj,&HardwareObject::initialize);
            connect(thread,&QThread::finished,obj,&HardwareObject::deleteLater);
            obj->moveToThread(thread);
        }
        else
            obj->initialize();

    }

    //now, start all threads
    for(int i=0;i<d_hardwareList.size();i++)
    {
        QThread *thread = d_hardwareList.at(i).second;
        if(thread != nullptr)
        {
            if(!thread->isRunning())
                thread->start();
        }
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

    obj->setConnected(success);

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
    obj->setConnected(false);

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
        HardwareObject *obj = d_hardwareList.at(i).first;
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
    for(int i=0;i<d_hardwareList.size();i++)
    {
        QThread *t = d_hardwareList.at(i).second;
        HardwareObject *obj = d_hardwareList.at(i).first;
        if(t != nullptr)
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
        HardwareObject *obj = d_hardwareList.at(i).first;
        obj->setConnected(false);
        if(obj->thread() == thread())
            obj->testConnection();
        else
            QMetaObject::invokeMethod(obj,"testConnection");
    }

    checkStatus();
}

void HardwareManager::testObjectConnection(const QString type, const QString key)
{
    Q_UNUSED(type)
    HardwareObject *obj = nullptr;
    for(int i=0; i<d_hardwareList.size();i++)
    {
        if(d_hardwareList.at(i).first->key() == key)
            obj = d_hardwareList.at(i).first;
    }
    if(obj == nullptr)
        emit testComplete(key,false,QString("Device not found!"));
    else
    {
        if(obj->thread() == thread())
            obj->testConnection();
        else
            QMetaObject::invokeMethod(obj,"testConnection");
    }
}

void HardwareManager::getTimeData()
{
    for(int i=0; i<d_hardwareList.size(); i++)
    {
        HardwareObject *obj = d_hardwareList.at(i).first;
        if(obj->thread() == thread())
            obj->readTimeData();
        else
            QMetaObject::invokeMethod(obj,"readTimeData");
    }
}


double HardwareManager::setValonTxFreq(const double d)
{
    if(p_synth->thread() == thread())
        return p_synth->setTxFreq(d);

    double out;
    QMetaObject::invokeMethod(p_synth,"setTxFreq",Qt::BlockingQueuedConnection,Q_RETURN_ARG(double,out),Q_ARG(double,d));
    return out;
}

double HardwareManager::setValonRxFreq(const double d)
{
    if(p_synth->thread() == thread())
        return p_synth->setRxFreq(d);

    double out;
    QMetaObject::invokeMethod(p_synth,"setRxFreq",Qt::BlockingQueuedConnection,Q_RETURN_ARG(double,out),Q_ARG(double,d));
    return out;
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

void HardwareManager::checkStatus()
{
    //gotta wait until all instruments have responded
    if(d_responseCount < d_hardwareList.size())
        return;

    bool success = true;
    for(int i=0; i<d_hardwareList.size(); i++)
    {
        HardwareObject *obj = d_hardwareList.at(i).first;
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
