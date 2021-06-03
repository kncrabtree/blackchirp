#include "virtualliflaser.h"

VirtualLifLaser::VirtualLifLaser(QObject *parent) : LifLaser (parent), d_pos(0.0)
{
    d_subKey = QString("virtualLifLaser");
    d_prettyName = QString("Virtual LIF Laser");
    d_commType = CommunicationProtocol::Virtual;

}


void VirtualLifLaser::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    d_minPos = s.value(QString("minPos"),200.0).toDouble();
    d_maxPos = s.value(QString("maxPos"),2000.0).toDouble();
    d_decimals = s.value(QString("decimals"),2).toInt();
    d_units = s.value(QString("units"),QString("nm")).toString();
    s.setValue(QString("minPos"),d_minPos);
    s.setValue(QString("maxPos"),d_maxPos);
    s.setValue(QString("decimals"),d_decimals);
    s.setValue(QString("units"),d_units);
    s.endGroup();
    s.endGroup();
    s.sync();
}

void VirtualLifLaser::sleep(bool b)
{
    Q_UNUSED(b)
}

void VirtualLifLaser::initialize()
{
    d_pos = d_minPos;
}

bool VirtualLifLaser::testConnection()
{
    return true;
}

double VirtualLifLaser::readPos()
{
    return  d_pos;
}

void VirtualLifLaser::setPos(const double pos)
{
    d_pos = pos;
}
