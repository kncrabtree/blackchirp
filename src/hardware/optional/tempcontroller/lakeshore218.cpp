#include "lakeshore218.h"

using namespace BC::Key::TC;

Lakeshore218::Lakeshore218(QObject *parent) :
    TemperatureController(lakeshore218,lakeshore218Name,
                          CommunicationProtocol::Rs232,8,parent)
{
}

bool Lakeshore218::tcTestConnection()
{
    QByteArray resp = p_comm->queryCmd(QString("QIDN?\r\n"));

    if (resp.isEmpty())
    {
        d_errorString = QString("Did not respond to ID query.");
        return false;

    }
    if (!resp.startsWith(QByteArray("LSCI,MODEL218S")))
    {
        d_errorString= QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex()));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp)));
    return true;

}

void Lakeshore218::tcInitialize()
{
    p_comm->setReadOptions(500, true,QByteArray("\r\n"));
}

QList<double> Lakeshore218::readHWTemperatures()
{
    QByteArray temp = p_comm->queryCmd(QString("KRDG?0\r\n"));
    auto l = temp.split(',');
    QList<double> out;
    if (l.size() != numChannels())
    {
        emit logMessage(QString("Could not parse temperature response. The response was %1").arg(QString(temp)),LogHandler::Error);
        emit hardwareFailure();
        return out;
    }

    for (int i=0;i<l.size();i++)
        out.append(l.at(i).toDouble());
    return out;
}

double Lakeshore218::readHwTemperature(const int ch)
{
    QByteArray temp = p_comm->queryCmd(QString("KRDG?%1\r\n").arg(ch+1));
    bool ok = false;
    double t = temp.toDouble(&ok);
    if(!ok)
    {
        emit logMessage(QString("Could not parse temperature response (%1)").arg(QString(temp)),LogHandler::Error);
        emit hardwareFailure();
        return nan("");
    }

    return t;
}

QStringList Lakeshore218::channelNames()
{
    return QStringList{ QString("Channel 1"), QString("Channel 2"), QString("Channel 3"), QString("Channel 4"), QString("Channel 5"), QString("Channel 6"), QString("Channel 7"), QString("Channel 8")};
}

