#include "virtualgpibcontroller.h"

VirtualGpibController::VirtualGpibController(QObject *parent) : GpibController(parent)
{
	d_subKey = QString("virtual");
	d_prettyName = QString("Virtual GPIB Controller");
    d_isCritical = false;
    d_commType = CommunicationProtocol::Virtual;
}

VirtualGpibController::~VirtualGpibController()
{

}



bool VirtualGpibController::testConnection()
{
	return true;
}

void VirtualGpibController::initialize()
{
}

Experiment VirtualGpibController::prepareForExperiment(Experiment exp)
{
   return exp;
}

void VirtualGpibController::beginAcquisition()
{
}

void VirtualGpibController::endAcquisition()
{
}

bool VirtualGpibController::readAddress()
{
    return true;
}

bool VirtualGpibController::setAddress(int a)
{
	d_currentAddress = a;
    return true;
}

