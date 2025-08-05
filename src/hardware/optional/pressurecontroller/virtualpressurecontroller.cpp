#include "virtualpressurecontroller.h"
#include <hardware/core/hardwareregistration.h>
#include <data/settings/hardwarekeys.h>

#include <QTimer>

using namespace BC::Key::PController;

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualPressureController, "Virtual Pressure Controller for testing and development")

VirtualPressureController::VirtualPressureController(const QString& label, QObject *parent) :
    PressureController(QString(VirtualPressureController::staticMetaObject.className()), label, false, parent)
{
    setDefault(min,-1.0);
    setDefault(max,20.0);
    setDefault(decimals,4);
    setDefault(units,QString("Torr"));
    setDefault(readInterval,200);
    setDefault(hasValve,true);

    save();
}

VirtualPressureController::~VirtualPressureController()
{
}


bool VirtualPressureController::pcTestConnection()
{
    return true;
}

void VirtualPressureController::pcInitialize()
{
}

double VirtualPressureController::hwReadPressure()
{
    return d_config.d_pressure;
}

double VirtualPressureController::hwSetPressureSetpoint(const double val)
{
    d_config.d_setPoint = val;
    return d_config.d_setPoint;
}

double VirtualPressureController::hwReadPressureSetpoint()
{
    return d_config.d_setPoint;
}

void VirtualPressureController::hwSetPressureControlMode(bool enabled)
{
    d_config.d_pressureControlMode = enabled;
}

int VirtualPressureController::hwReadPressureControlMode()
{
    return d_config.d_pressureControlMode;
}

void VirtualPressureController::hwOpenGateValve()
{
}

void VirtualPressureController::hwCloseGateValve()
{
}


