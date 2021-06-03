#include "virtualinstrument.h"

VirtualInstrument::VirtualInstrument(QString key, QObject *parent) :
    CommunicationProtocol(CommunicationProtocol::Virtual,key,QString("virtual"),parent)
{

}

VirtualInstrument::~VirtualInstrument()
{

}

void VirtualInstrument::initialize()
{
}

bool VirtualInstrument::testConnection()
{
    return true;
}
