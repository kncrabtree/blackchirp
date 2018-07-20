#include "custominstrument.h"

CustomInstrument::CustomInstrument(QString key, QString subKey, QObject *parent) :
    CommunicationProtocol(CommunicationProtocol::Custom,key,subKey,parent)
{
}


void CustomInstrument::initialize()
{
}

bool CustomInstrument::testConnection()
{
    return true;
}
