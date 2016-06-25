#include "hardwareobject.h"

#include "virtualinstrument.h"
#include "tcpinstrument.h"
#include "rs232instrument.h"
#include "gpibinstrument.h"
#include "custominstrument.h"

HardwareObject::HardwareObject(QObject *parent) :
    QObject(parent), d_isCritical(true), d_threaded(true)
{
}

HardwareObject::~HardwareObject()
{

}


void HardwareObject::buildCommunication(QObject *gc)
{
    GpibController *c = dynamic_cast<GpibController*>(gc);
    switch(d_commType)
    {
    case CommunicationProtocol::Rs232:
        p_comm = new Rs232Instrument(d_key,d_subKey,this);
        break;
    case CommunicationProtocol::Tcp:
        p_comm = new TcpInstrument(d_key,d_subKey,this);
        break;
    case CommunicationProtocol::Gpib:
        p_comm = new GpibInstrument(d_key,d_subKey,c,this);
        setParent(c);
        break;
    case CommunicationProtocol::Custom:
        p_comm = new CustomInstrument(d_key,d_subKey,this);
    case CommunicationProtocol::Virtual:
    default:
        p_comm = new VirtualInstrument(d_key,this);
        break;
    }

    connect(p_comm,&CommunicationProtocol::logMessage,this,&HardwareObject::logMessage);
    connect(p_comm,&CommunicationProtocol::hardwareFailure,this,&HardwareObject::hardwareFailure);
}

void HardwareObject::sleep(bool b)
{
	if(b)
		emit logMessage(name().append(QString(" is asleep.")));
	else
		emit logMessage(name().append(QString(" is awake.")));
}
