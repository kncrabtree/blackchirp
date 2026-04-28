#include "virtualliflaser.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualLifLaser, "Virtual LIF Laser for Testing")

VirtualLifLaser::VirtualLifLaser(const QString& label, QObject *parent) :
    LifLaser(QString(VirtualLifLaser::staticMetaObject.className()), label, parent), d_pos(0.0), d_fl(false)
{
    save();
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
