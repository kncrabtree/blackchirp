#include "custominstrument.h"

CustomInstrument::CustomInstrument(QString key, QObject *parent) :
    CommunicationProtocol(key,parent)
{
}


void CustomInstrument::initialize()
{
}

bool CustomInstrument::testConnection()
{
    return true;
}
