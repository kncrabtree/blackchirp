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
			s.setValue(QString("virtual"),d_hardwareList.at(i).first->isVirtual());
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

        d_hardwareList.at(i).first->moveToThread(d_hardwareList.at(i).second);
        d_hardwareList.at(i).second->start();
    }
}

