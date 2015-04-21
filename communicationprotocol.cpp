#include "communicationprotocol.h"

CommunicationProtocol::CommunicationProtocol(CommType type, QString key, QString subKey, QObject *parent) :
    QObject(parent), d_type(type), d_key(QString("%1/%2").arg(key,subKey)),
    d_useTermChar(false), d_timeOut(1000)
{

}

CommunicationProtocol::~CommunicationProtocol()
{

}

