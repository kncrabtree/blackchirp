#include "virtualpressurecontroller.h"

#include <QTimer>

using namespace BC::Key::PController;

VirtualPressureController::VirtualPressureController(QObject *parent) :
    PressureController(BC::Key::Comm::hwVirtual,BC::Key::vpcName,CommunicationProtocol::Virtual,false,parent)
{
    setDefault(min,-1.0);
    setDefault(max,20.0);
    setDefault(decimals,4);
    setDefault(units,QString("Torr"));
    setDefault(readInterval,200);
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


