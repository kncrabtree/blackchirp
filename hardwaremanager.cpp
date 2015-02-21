#include "hardwaremanager.h"
#include <QSettings>

HardwareManager::HardwareManager(QObject *parent) : QObject(parent)
{

}

HardwareManager::~HardwareManager()
{
    //note, the hardwareObjects are deleted when the threads exit
    while(!d_hardwareList.isEmpty())
    {
        QPair<HardwareObject*,QThread*> p = d_hardwareList.takeFirst();
        p.second->quit();
        p.second->wait();
    }
}

void HardwareManager::initialize()
{    
    p_scope = new Oscilloscope();
    connect(p_scope,&Oscilloscope::shotAcquired,this,&HardwareManager::scopeShotAcquired);

    QThread *scopeThread = new QThread(this);
    d_hardwareList.append(qMakePair(p_scope,scopeThread));

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
        s.setValue(QString("virtual"),d_hardwareList.at(i).first->isVirtual());
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
		if(d_hardwareList.at(i).first->inherits("TcpInstrument"))
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
		if(d_hardwareList.at(i).first->inherits("Rs232Instrument"))
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
        connect(d_hardwareList.at(i).second,&QThread::started,d_hardwareList.at(i).first,&HardwareObject::initialize);
        connect(d_hardwareList.at(i).second,&QThread::finished,d_hardwareList.at(i).first,&HardwareObject::deleteLater);
        connect(d_hardwareList.at(i).first,&HardwareObject::logMessage,this,&HardwareManager::logMessage);
        connect(d_hardwareList.at(i).first,&HardwareObject::connectionResult,this,&HardwareManager::connectionResult);
        connect(d_hardwareList.at(i).first,&HardwareObject::hardwareFailure,this,&HardwareManager::hardwareFailure);
        connect(this,&HardwareManager::beginAcquisition,d_hardwareList.at(i).first,&HardwareObject::beginAcquisition);

        d_hardwareList.at(i).first->moveToThread(d_hardwareList.at(i).second);
        d_hardwareList.at(i).second->start();
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
        if(t != NULL)
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
