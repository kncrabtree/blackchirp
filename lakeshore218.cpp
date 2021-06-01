#include "lakeshore218.h"

Lakeshore218::Lakeshore218(QObject *parent) : TemperatureController(parent)
{
    d_subKey = QString("lakeshore218");
    d_prettyName = QString("Lakeshore 218");
    d_commType = CommunicationProtocol::Rs232;
    d_numChannels = 8;
}
void Lakeshore218::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(d_key);
    s.beginGroup(d_subKey);


    QString units = s.value(QString("units"),QString("K")).toString();
    s.setValue(QString("units"), units);
    s.endGroup();
    s.endGroup();
}

bool Lakeshore218::testConnection()
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

QList<double> Lakeshore218::readHWTemperature()
{
    QByteArray temp = p_comm->queryCmd(QString("KRDG?0\r\n"));
    auto l = temp.split(',');
    QList<double> out;
    if (l.size() != d_numChannels)
    {
        emit logMessage(QString("Could not parse temperature response. The response was %1").arg(QString(temp)));
        emit hardwareFailure();
        return out;
    }

    for (int i=0;i<l.size();i++)
        out.append(l.at(i).toDouble());
    return out;
}

QStringList Lakeshore218::channelNames()
{
    return QStringList{ QString("Channel 1"), QString("Channel 2"), QString("Channel 3"), QString("Channel 4"), QString("Channel 5"), QString("Channel 6"), QString("Channel 7"), QString("Channel 8")};
}

