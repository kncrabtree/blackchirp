#include "virtualmotorcontroller.h"

#include <math.h>

#include <hardware/core/communication/virtualinstrument.h>


using namespace BC::Key::MC;

VirtualMotorController::VirtualMotorController(QObject *parent) :
    MotorController(BC::Key::hwVirtual,vmcName,CommunicationProtocol::Virtual,parent)
{
    d_pos[MotorScan::MotorX] = getArrayValue(channels,get(xIndex,0),min,0.0);
    d_pos[MotorScan::MotorY] = getArrayValue(channels,get(yIndex,1),min,0.0);
    d_pos[MotorScan::MotorZ] = getArrayValue(channels,get(zIndex,2),min,0.0);
}

bool VirtualMotorController::mcTestConnection()
{
    return true;
}

void VirtualMotorController::mcInitialize()
{
}

bool VirtualMotorController::prepareForMotorScan(Experiment &exp)
{
    Q_UNUSED(exp)
    return true;
}


bool VirtualMotorController::hwMoveToPosition(double x, double y, double z)
{
    d_pos[MotorScan::MotorX] = x;
    d_pos[MotorScan::MotorY] = y;
    d_pos[MotorScan::MotorZ] = z;

    return true;
}

Limits VirtualMotorController::hwCheckLimits(MotorScan::MotorAxis axis)
{
    QString ax;
    auto p = d_pos.value(axis);
    switch(axis)
    {
    case MotorScan::MotorX:
        return { fabs(p-getArrayValue(channels,get(xIndex,0),min,0.0)) < 0.001,
                    fabs(p-getArrayValue(channels,get(xIndex,0),max,100.0)) < 0.001};
    case MotorScan::MotorY:
        return { fabs(p-getArrayValue(channels,get(yIndex,1),min,0.0)) < 0.001,
                    fabs(p-getArrayValue(channels,get(yIndex,1),max,100.0)) < 0.001};
    case MotorScan::MotorZ:
        return { fabs(p-getArrayValue(channels,get(zIndex,2),min,0.0)) < 0.001,
                    fabs(p-getArrayValue(channels,get(zIndex,2),max,100.0)) < 0.001};
    default:
        break;
    }

    return {false,false};
}

double VirtualMotorController::hwReadPosition(MotorScan::MotorAxis axis)
{
    return d_pos.value(axis);
}

bool VirtualMotorController::hwCheckAxisMotion(MotorScan::MotorAxis axis)
{
    Q_UNUSED(axis)
    return false;
}

bool VirtualMotorController::hwStopMotion(MotorScan::MotorAxis axis)
{
    Q_UNUSED(axis)
    return true;
}
