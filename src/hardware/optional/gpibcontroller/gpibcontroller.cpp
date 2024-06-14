#include <hardware/optional/gpibcontroller/gpibcontroller.h>

GpibController::GpibController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent) :
    HardwareObject(BC::Key::gpibController,subKey,name,commType,parent,true,true,d_count)
{
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

QByteArray GpibController::queryCmd(int address, QString cmd, bool suppressError)
{
	if(address != d_currentAddress)
    {
        if(!setAddress(address))
            return QByteArray();
    }

    return p_comm->queryCmd(cmd.append(queryTerminator()), suppressError);
}

