#include <src/hardware/core/communication/virtualinstrument.h>

VirtualInstrument::VirtualInstrument(QString key, QObject *parent) :
    CommunicationProtocol(key,parent)
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
