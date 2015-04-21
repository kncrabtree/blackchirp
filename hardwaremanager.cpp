#include "hardwaremanager.h"
#include <QSettings>
#include "dsa71604c.h"
#include "virtualawg.h"
#include "virtualftmwscope.h"
#include "virtualpulsegenerator.h"
#include "virtualvalonsynth.h"

HardwareManager::HardwareManager(QObject *parent) : QObject(parent)
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
    p_scope = new FtmwScopeHardware();
    connect(p_scope,&FtmwScope::shotAcquired,this,&HardwareManager::scopeShotAcquired);

    QThread *scopeThread = new QThread(this);
    d_hardwareList.append(qMakePair(p_scope,scopeThread));

    //awg does not need to be in its own thread
    p_awg = new AwgHardware();
    d_hardwareList.append(qMakePair(p_awg,nullptr));

    //valon synth does not need to be in its own thread
    p_synth = new SynthesizerHardware();
    connect(p_synth,&Synthesizer::txFreqRead,this,&HardwareManager::valonTxFreqRead);
    connect(p_synth,&Synthesizer::rxFreqRead,this,&HardwareManager::valonRxFreqRead);
    d_hardwareList.append(qMakePair(p_synth,nullptr));

    //pulse generator does not need to be in its own thread
    p_pGen = new PulseGeneratorHardware();
    connect(p_pGen,&PulseGenerator::settingUpdate,this,&HardwareManager::pGenSettingUpdate);
    connect(p_pGen,&PulseGenerator::configUpdate,this,&HardwareManager::pGenConfigUpdate);
    connect(p_pGen,&PulseGenerator::repRateUpdate,this,&HardwareManager::pGenRepRateUpdate);
    d_hardwareList.append(qMakePair(p_pGen,nullptr));


	//write arrays of the connected devices for use in the Hardware Settings menu
	//first array is for all objects accessible to the hardware manager
	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	s.beginGroup(QString("hardware"));
	s.remove("");
	s.beginWriteArray("instruments");
	for(int i=0;i<d_hardwareList.size();i++)
	{
		s.setArrayIndex(i);
		s.setValue(QString("key"),d_hardwareList.at(i).first->key());
        s.setValue(QString("type"),d_hardwareList.at(i).first->subKey());
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

        connect(obj,&HardwareObject::logMessage,[=](QString msg, LogHandler::MessageCode mc){
            emit logMessage(QString("%1: %2").arg(obj->name()).arg(msg),mc);
        });
        connect(obj,&HardwareObject::connected,[=](bool success, QString msg){ connectionResult(obj,success,msg); });
        connect(obj,&HardwareObject::hardwareFailure,[=](bool abort){ hardwareFailure(obj,abort); });
        connect(obj,&HardwareObject::timeDataRead,this,&HardwareManager::timeData);
        connect(this,&HardwareManager::beginAcquisition,obj,&HardwareObject::beginAcquisition);
        connect(this,&HardwareManager::endAcquisition,obj,&HardwareObject::endAcquisition);
        connect(this,&HardwareManager::readTimeData,obj,&HardwareObject::readTimeData);

        if(thread != nullptr)
        {
            connect(thread,&QThread::started,obj,&HardwareObject::initialize);
            connect(thread,&QThread::finished,obj,&HardwareObject::deleteLater);
            obj->moveToThread(thread);
            thread->start();
        }
        else
            obj->initialize();

    }
}

void HardwareManager::connectionResult(HardwareObject *obj, bool success, QString msg)
{
    if(success)
        emit logMessage(obj->name().append(QString(" connected successfully.")));
    else
    {
        emit logMessage(obj->name().append(QString(" connection failed!")),LogHandler::Error);
        emit logMessage(msg,LogHandler::Error);
    }

    bool ok = success;
    if(!obj->isCritical())
        ok = true;

    if(d_status.contains(obj->key()))
        d_status[obj->key()] = ok;
    else
        d_status.insert(obj->key(),ok);



    emit testComplete(obj->name(),success,msg);
    checkStatus();
}

void HardwareManager::hardwareFailure(HardwareObject *obj, bool abort)
{
    if(abort)
        emit abortAcquisition();

    if(!obj->isCritical())
        return;

    d_status[obj->key()] = false;
    checkStatus();
}

void HardwareManager::initializeExperiment(Experiment exp)
{
    //do initialization
    //if successful, call Experiment::setInitialized()
    bool success = true;
    for(int i=0;i<d_hardwareList.size();i++)
    {
        QThread *t = d_hardwareList.at(i).second;
        HardwareObject *obj = d_hardwareList.at(i).first;
        if(t != nullptr)
            QMetaObject::invokeMethod(obj,"prepareForExperiment",Qt::BlockingQueuedConnection,Q_RETURN_ARG(Experiment,exp),Q_ARG(Experiment,exp));
        else
            exp = obj->prepareForExperiment(exp);

        if(!exp.hardwareSuccess())
        {
            success = false;
            break;
        }
    }

    //any additional synchronous initialization can be performed here

    if(success)
        exp.setInitialized();

    emit experimentInitialized(exp);

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
        QMetaObject::invokeMethod(obj,"testConnection");
}

void HardwareManager::getTimeData()
{
    for(int i=0; i<d_hardwareList.size(); i++)
        QMetaObject::invokeMethod(d_hardwareList.at(i).first,"readTimeData");
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

void HardwareManager::setPGenSetting(int index, PulseGenConfig::Setting s, QVariant val)
{
    if(p_pGen->thread() == thread())
    {
        p_pGen->set(index,s,val);
        return;
    }

    QMetaObject::invokeMethod(p_pGen,"set",Q_ARG(int,index),Q_ARG(PulseGenConfig::Setting,s),Q_ARG(QVariant,val));
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

void HardwareManager::checkStatus()
{
    //gotta wait until all instruments have responded
    if(d_status.size() < d_hardwareList.size())
        return;

    bool success = true;
    foreach (bool b, d_status)
    {
        if(!b)
            success = false;
    }

    emit allHardwareConnected(success);
}
