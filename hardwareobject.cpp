#include "hardwareobject.h"

#include "virtualinstrument.h"
#include "tcpinstrument.h"
#include "rs232instrument.h"
#include "custominstrument.h"

#ifdef BC_GPIBCONTROLLER
#include "gpibinstrument.h"
#endif

HardwareObject::HardwareObject(QObject *parent) :
    QObject(parent), d_isCritical(true), d_threaded(true), d_enabledForExperiment(true)
{
}

HardwareObject::~HardwareObject()
{

}


void HardwareObject::buildCommunication(QObject *gc)
{
#ifdef BC_GPIBCONTROLLER
    GpibController *c = dynamic_cast<GpibController*>(gc);
#else
    Q_UNUSED(gc)
#endif
    switch(d_commType)
    {
    case CommunicationProtocol::Rs232:
        p_comm = new Rs232Instrument(d_key,d_subKey,this);
        break;
    case CommunicationProtocol::Tcp:
        p_comm = new TcpInstrument(d_key,d_subKey,this);
        break;
#ifdef BC_GPIBCONTROLLER
    case CommunicationProtocol::Gpib:
        p_comm = new GpibInstrument(d_key,d_subKey,c,this);
        setParent(c);
        break;
#endif
    case CommunicationProtocol::Custom:
        p_comm = new CustomInstrument(d_key,d_subKey,this);
        break;
    case CommunicationProtocol::Virtual:
        p_comm = new VirtualInstrument(d_key,this);
        break;
    case CommunicationProtocol::None:
    default:
        p_comm = nullptr;
        break;
    }

    if(p_comm)
    {
        connect(p_comm,&CommunicationProtocol::logMessage,this,&HardwareObject::logMessage);
        connect(p_comm,&CommunicationProtocol::hardwareFailure,this,&HardwareObject::hardwareFailure);
    }
}

void HardwareObject::sleep(bool b)
{
	if(b)
		emit logMessage(name().append(QString(" is asleep.")));
	else
		emit logMessage(name().append(QString(" is awake.")));
}
