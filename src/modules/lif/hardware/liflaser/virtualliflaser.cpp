#include "virtualliflaser.h"

VirtualLifLaser::VirtualLifLaser(QObject *parent) :
    LifLaser (BC::Key::Comm::hwVirtual,BC::Key::vLifLaser,CommunicationProtocol::Virtual,parent), d_pos(0.0)
{
    using namespace BC::Key::LifLaser;
    setDefault(minPos,250.);
    setDefault(maxPos,2000.);
    setDefault(units,QString("nm"));
    setDefault(decimals,2);
}

void VirtualLifLaser::sleep(bool b)
{
    Q_UNUSED(b)
}

void VirtualLifLaser::initialize()
{
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
