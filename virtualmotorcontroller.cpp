#include "virtualmotorcontroller.h"

#include "virtualinstrument.h"

VirtualMotorController::VirtualMotorController(QObject *parent) :
    MotorController(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Motor Controller");

    p_comm = new VirtualInstrument(d_key,this);
    connect(p_comm,&CommunicationProtocol::logMessage,this,&HardwareObject::logMessage);
    connect(p_comm,&CommunicationProtocol::hardwareFailure,this,&HardwareObject::hardwareFailure);

}



bool VirtualMotorController::testConnection()
{
    emit connected();
    return true;
}

void VirtualMotorController::initialize()
{
    testConnection();
}

Experiment VirtualMotorController::prepareForExperiment(Experiment exp)
{
    return exp;
}

void VirtualMotorController::beginAcquisition()
{
}

void VirtualMotorController::endAcquisition()
{
}

void VirtualMotorController::readTimeData()
{
}

void VirtualMotorController::moveToNextPosition()
{
}
