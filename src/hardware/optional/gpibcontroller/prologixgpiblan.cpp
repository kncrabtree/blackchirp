#include "prologixgpiblan.h"

PrologixGpibLan::PrologixGpibLan(QObject *parent) :
    GpibController(BC::Key::prologix,BC::Key::prologixName,CommunicationProtocol::Tcp,parent,true,true)
{
}



bool PrologixGpibLan::testConnection()
{

    QByteArray resp = p_comm->queryCmd(QString("++ver\n"));
    if(resp.isEmpty())
    {
        d_errorString = QString("%1 gave a null response to ID query").arg(d_name);
        return false;
    }
    if(!resp.startsWith("Prologix GPIB-ETHERNET Controller"))
    {
        d_errorString = QString("%1 response invalid. Received: %2").arg(d_name).arg(QString(resp));
        return false;
    }

    emit logMessage(QString("%1 ID response: %2").arg(d_name).arg(QString(resp)));

    p_comm->writeCmd(QString("++auto 0\n"));
    p_comm->writeCmd(QString("++savecfg 0\n"));
    p_comm->writeCmd(QString("++read_tmo_ms 50\n"));

    readAddress();

    return true;
}

void PrologixGpibLan::initialize()
{
    p_comm->setReadOptions(1000,true,QByteArray("\n"));
}

bool PrologixGpibLan::readAddress()
{
    bool success = false;
    QByteArray resp = p_comm->queryCmd(QString("++addr\n"));
    d_currentAddress = resp.trimmed().toInt(&success);
    if(!success)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read address. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())));
    }
    return success;
}

bool PrologixGpibLan::setAddress(int a)
{
    if(!p_comm->writeCmd(QString("++addr %1\n").arg(a)))
        return false;

    if(!readAddress())
        return false;

    if(d_currentAddress != a)
    {
        emit hardwareFailure();
        emit logMessage(QString("Address was not set to %1. Current address: %2").arg(a).arg(d_currentAddress),BlackChirp::LogError);
        return false;
    }

    return true;
}

QString PrologixGpibLan::queryTerminator() const
{
    return QString("++read eoi\n");
}
