#include "virtualinstrument.h"

VirtualInstrument::VirtualInstrument(QString key, QObject *parent) :
    CommunicationProtocol(CommunicationProtocol::Virtual,key,QString("virtual"),parent)
{

}

VirtualInstrument::~VirtualInstrument()
{

}



bool VirtualInstrument::writeCmd(QString cmd)
{
    Q_UNUSED(cmd)
    return true;
}

bool VirtualInstrument::writeBinary(QByteArray dat)
{
    Q_UNUSED(dat)
    return true;
}

QByteArray VirtualInstrument::queryCmd(QString cmd)
{
    Q_UNUSED(cmd)
    return QByteArray();
}

void VirtualInstrument::initialize()
{
}

bool VirtualInstrument::testConnection()
{
    return true;
}
