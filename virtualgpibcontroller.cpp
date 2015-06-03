#include "virtualgpibcontroller.h"

#include "virtualinstrument.h"

VirtualGpibController::VirtualGpibController(QObject *parent) : GpibController(parent)
{
	d_subKey = QString("virtual");
	d_prettyName = QString("Virtual GPIB Controller");
    d_isCritical = false;

	p_comm = new VirtualInstrument(d_key,this);
	connect(p_comm,&CommunicationProtocol::logMessage,this,&VirtualGpibController::logMessage);
	connect(p_comm,&CommunicationProtocol::hardwareFailure,[=](){ emit hardwareFailure(); });
}

VirtualGpibController::~VirtualGpibController()
{

}



bool VirtualGpibController::testConnection()
{
	emit connected();
	return true;
}

void VirtualGpibController::initialize()
{
    testConnection();
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

void VirtualGpibController::readPointData()
{
}

void VirtualGpibController::readAddress()
{
}

void VirtualGpibController::setAddress(int a)
{
	d_currentAddress = a;
}
