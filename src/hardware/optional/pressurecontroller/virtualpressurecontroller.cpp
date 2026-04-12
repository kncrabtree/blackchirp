#include "virtualpressurecontroller.h"
#include <hardware/core/hardwareregistration.h>
#include <data/settings/hardwarekeys.h>

#include <QTimer>

using namespace BC::Key::PController;

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualPressureController, "Virtual Pressure Controller for testing and development")
REGISTER_HARDWARE_SETTINGS(VirtualPressureController,
    {BC::Key::PController::min, "Min Pressure",
     "Minimum pressure reading (display range lower bound).",
     -1.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PController::max, "Max Pressure",
     "Maximum pressure reading (display range upper bound).",
     20.0, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PController::decimals, "Display Decimals",
     "Number of decimal places in pressure display.",
     4, 0, 8, HwSettingPriority::Optional},
    {BC::Key::PController::units, "Pressure Units",
     "Pressure units for display.",
     QString("Torr"), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::PController::readInterval, "Read Interval (ms)",
     "Polling interval for pressure readings in milliseconds.",
     200, 1, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PController::hasValve, "Has Valve",
     "Device includes a controlled valve output.",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

VirtualPressureController::VirtualPressureController(const QString& label, QObject *parent) :
    PressureController(QString(VirtualPressureController::staticMetaObject.className()), label, false, parent)
{
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


