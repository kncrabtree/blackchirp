#include "awg70002a.h"

#include "tcpinstrument.h"

AWG70002a::AWG70002a(QObject *parent) :
    AWG(parent)
{
    d_subKey = QString("awg70002a");
    d_prettyName = QString("Arbitrary Waveform Generator AWG70002A");

    p_comm = new TcpInstrument(d_key,d_subKey,this);
    connect(p_comm,&CommunicationProtocol::logMessage,this,&AWG70002a::logMessage);
    connect(p_comm,&CommunicationProtocol::hardwareFailure,[=](){ emit hardwareFailure(); });
}



bool AWG70002a::testConnection()
{
    if(!p_comm->testConnection())
    {
        emit connected(false,QString("TCP error."));
        return false;
    }

    QByteArray resp = p_comm->queryCmd(QString("*IDN?\n"));

    if(resp.isEmpty())
    {
        emit connected(false,QString("Did not respond to ID query."));
        return false;
    }

    if(!resp.startsWith(QByteArray("TEKTRONIX,AWG70002A")))
    {
        emit connected(false,QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp)));
    emit connected();
    return true;
}

void AWG70002a::initialize()
{
    p_comm->initialize();
    testConnection();
}

Experiment AWG70002a::prepareForExperiment(Experiment exp)
{
    return exp;
}

void AWG70002a::beginAcquisition()
{
}

void AWG70002a::endAcquisition()
{
}

void AWG70002a::readTimeData()
{
}
