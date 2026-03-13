#include "opolette.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Opolette, "Opolette LIF Laser")
REGISTER_HARDWARE_PROTOCOLS(Opolette, CommunicationProtocol::Tcp)

Opolette::Opolette(const QString& label, QObject *parent)
    : LifLaser(QString(Opolette::staticMetaObject.className()), label, parent)
{
    using namespace BC::Key::LifLaser;
    setDefault(decimals,2);
    setDefault(units,QString("nm"));
    setDefault(minPos,250);
    setDefault(maxPos,2000);
    setDefault(hasFl,true);

    // Communication defaults
    setDefault(BC::Key::Comm::timeout, 20000);
    setDefault(BC::Key::Comm::termChar, QString("\n"));

    save();
}

void Opolette::initialize()
{
}

bool Opolette::testConnection()
{
    auto resp = p_comm->queryCmd(QString("*IDN?\n"));
    if(!resp.startsWith("Crabtree Opolette Server"))
        return false;

    readPosition();
    readFlashLamp();

    return true;
}

double Opolette::readPos()
{
    auto resp = p_comm->queryCmd("LP?\n");
    if(resp.startsWith("ERROR:"))
    {
        emit logMessage(QString("Could not read laser position. %1").arg(QString(resp.mid(7))),LogHandler::Error);
        return -1.0;
    }

    bool ok = false;
    auto out = QString(resp.trimmed()).toDouble(&ok);
    if(!ok)
    {
        emit logMessage(QString("Could not parse wavelength response (%1)").arg(QString(resp)),LogHandler::Error);
                return -1.0;
    }
    return out;
}

void Opolette::setPos(double pos)
{
    auto resp = p_comm->queryCmd(QString("LP %1\n").arg(pos,0,'f',2));
    if(resp.startsWith("ERROR:"))
        emit logMessage(QString("Could not set laser position. %1").arg(QString(resp.mid(7))),LogHandler::Error);
}

bool Opolette::readFl()
{
    auto resp = p_comm->queryCmd(QString("FL?\n"));
    if(resp.startsWith("ERROR:"))
        emit logMessage(QString("Could not read flashlamp status. %1").arg(QString(resp.mid(7))),LogHandler::Error);
    else if(resp.startsWith("1"))
        return true;

    return false;
}

bool Opolette::setFl(bool en)
{
    QByteArray resp;
    if(en)
        resp = p_comm->queryCmd(QString("FL 1\n"));
    else
        resp = p_comm->queryCmd(QString("FL 0\n"));

    if(resp.startsWith("ERROR:"))
    {
        emit logMessage(QString("Could not set flashlamp status. %1").arg(QString(resp.mid(7))),LogHandler::Error);
        return false;
    }

    return true;
}
