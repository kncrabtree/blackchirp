#include "virtualliflaser.h"

VirtualLifLaser::VirtualLifLaser(QObject *parent) :
    LifLaser (BC::Key::Comm::hwVirtual,BC::Key::vLifLaser,CommunicationProtocol::Virtual,parent), d_pos(0.0), d_fl(false)
{
    using namespace BC::Key::LifLaser;
    setDefault(minPos,250.);
    setDefault(maxPos,2000.);
    setDefault(units,QString("nm"));
    setDefault(decimals,2);
}

void VirtualLifLaser::initialize()
{
}

bool VirtualLifLaser::testConnection()
{
    d_pos = 500.0;
    d_fl = false;

    readPosition();
    readFlashLamp();

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

bool VirtualLifLaser::readFl()
{
    return d_fl;
}

bool VirtualLifLaser::setFl(bool en)
{
    d_fl = en;
    return true;
}
