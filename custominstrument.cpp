#include "custominstrument.h"

CustomInstrument::CustomInstrument(QString key, QString subKey, QObject *parent) :
    CommunicationProtocol(CommunicationProtocol::Custom,key,subKey,parent)
{
}



bool CustomInstrument::writeCmd(QString cmd)
{
    Q_UNUSED(cmd)
    return true;
}

bool CustomInstrument::writeBinary(QByteArray dat)
{
    Q_UNUSED(dat)
    return true;
}

QByteArray CustomInstrument::queryCmd(QString cmd)
{
    Q_UNUSED(cmd)
    return QByteArray();
}

QIODevice *CustomInstrument::device()
{
    return nullptr;
}

void CustomInstrument::initialize()
{
}

bool CustomInstrument::testConnection()
{
    return true;
}
