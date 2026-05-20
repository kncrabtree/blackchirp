#include "lakeshore218.h"
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key::TC;

// Register hardware implementation
REGISTER_HARDWARE_META(Lakeshore218, "Lakeshore 218 Temperature Controller")
REGISTER_HARDWARE_PROTOCOLS(Lakeshore218, CommunicationProtocol::Rs232)

Lakeshore218::Lakeshore218(const QString& label, QObject *parent) :
    TemperatureController(QString(Lakeshore218::staticMetaObject.className()), label, 8, parent)
{
    save();
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

    hwDebug(u"ID response: %1"_s.arg(QString(resp)));
    return true;

}

void Lakeshore218::tcInitialize()
{
}

double Lakeshore218::readHwTemperature(const uint ch)
{
    QByteArray temp = p_comm->queryCmd(QString("KRDG?%1\r\n").arg(ch+1));
    bool ok = false;
    double t = temp.toDouble(&ok);
    if(!ok)
    {
        hwError("Could not parse temperature response."_L1);
        hwDebug(u"Could not parse temperature response. Response = %1 (Hex: %2)"_s.arg(QString(temp), QString(temp.toHex())));
        emit hardwareFailure();
        return nan("");
    }

    return t;
}

QStringList Lakeshore218::channelNames()
{
    return QStringList{ QString("Channel 1"), QString("Channel 2"), QString("Channel 3"), QString("Channel 4"), QString("Channel 5"), QString("Channel 6"), QString("Channel 7"), QString("Channel 8")};
}

