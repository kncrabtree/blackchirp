#include "virtualliflaser.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualLifLaser, "Virtual LIF Laser for Testing")

REGISTER_HARDWARE_SETTINGS(VirtualLifLaser,
    {BC::Key::LifLaser::minPos, "Min Position", "Minimum laser wavelength/position", 250.0, QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::LifLaser::maxPos, "Max Position", "Maximum laser wavelength/position", 2000.0, QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::LifLaser::units, "Position Units", "Units for position display (e.g. nm, cm-1)", QString("nm"), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::LifLaser::decimals, "Display Decimals", "Number of decimal places for position display", 2, 0, 8, HwSettingPriority::Optional},
    {BC::Key::LifLaser::hasFl, "Has Flashlamp", "Laser has a software-controlled flashlamp", true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

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
