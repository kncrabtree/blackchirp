#include "gpibcontroller.h"

GpibController::GpibController(QObject *parent) :
	HardwareObject(parent)
{
	d_key = QString("gpibController");
}

GpibController::~GpibController()
{

}

bool GpibController::writeCmd(int address, QString cmd)
{
	if(address != d_currentAddress)
    {
        if(!setAddress(address))
            return false;
    }

    return p_comm->writeCmd(cmd);
}

bool GpibController::writeBinary(int address, QByteArray dat)
{
    if(address != d_currentAddress)
    {
        if(!setAddress(address))
            return false;
    }

    return p_comm->writeBinary(dat);
}

QByteArray GpibController::queryCmd(int address, QString cmd)
{
	if(address != d_currentAddress)
    {
        if(!setAddress(address))
            return QByteArray();
    }

	return p_comm->queryCmd(cmd);
}

