#include "hardwaremanager.h"

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
    for(int i=0;i<d_hardwareList.size();i++)
    {
        connect(d_hardwareList.at(i).second,&QThread::started,d_hardwareList.at(i).first,&HardwareObject::initialize);
        connect(d_hardwareList.at(i).second,&QThread::finished,d_hardwareList.at(i).first,&HardwareObject::deleteLater);
        connect(d_hardwareList.at(i).first,&HardwareObject::logMessage,this,&HardwareManager::logMessage);

        d_hardwareList.at(i).first->moveToThread(d_hardwareList.at(i).second);
        d_hardwareList.at(i).second->start();
    }
}

