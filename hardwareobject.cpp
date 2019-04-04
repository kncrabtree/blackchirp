#include "hardwareobject.h"

#include "virtualinstrument.h"
#include "tcpinstrument.h"
#include "rs232instrument.h"
#include "custominstrument.h"

#ifdef BC_GPIBCONTROLLER
#include "gpibinstrument.h"
#endif

HardwareObject::HardwareObject(QObject *parent) :
    QObject(parent), d_isCritical(true), d_threaded(true), d_enabledForExperiment(true), d_isConnected(false)
{
}

HardwareObject::~HardwareObject()
{

}

QString HardwareObject::errorString()
{
    QString out = d_errorString;
    d_errorString.clear();
    return out;
}

void HardwareObject::bcInitInstrument()
{
    if(p_comm)
        p_comm->initialize();

    readSettings();
    initialize();
    bcTestConnection();

    connect(this,&HardwareObject::hardwareFailure,[=](){ d_isConnected = false; });
}

void HardwareObject::bcTestConnection()
{
    d_isConnected = false;
    if(p_comm)
    {
        if(!p_comm->bcTestConnection())
        {
            emit connected(false,p_comm->errorString(),QPrivateSignal());
            return;
        }
    }
    bool success = testConnection();
    d_isConnected = success;
    emit connected(success,errorString(),QPrivateSignal());
}

void HardwareObject::readSettings()
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
