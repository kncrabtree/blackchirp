#include "virtualgpibcontroller.h"

VirtualGpibController::VirtualGpibController(QObject *parent) :
    GpibController(BC::Key::Comm::hwVirtual,BC::Key::vgpibName,CommunicationProtocol::Virtual,parent)
{
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

bool VirtualGpibController::readAddress()
{
    return true;
}

bool VirtualGpibController::setAddress(int a)
{
	d_currentAddress = a;
    return true;
}

