#include "opolette.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Opolette, "Opolette LIF Laser")
REGISTER_HARDWARE_PROTOCOLS(Opolette, CommunicationProtocol::Tcp)

REGISTER_HARDWARE_SETTINGS(Opolette,
    {BC::Key::LifLaser::minPos, "Min Position", "Minimum laser wavelength/position", 250.0, QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::LifLaser::maxPos, "Max Position", "Maximum laser wavelength/position", 2000.0, QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::LifLaser::units, "Position Units", "Units for position display (e.g. nm, cm-1)", QString("nm"), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::LifLaser::decimals, "Display Decimals", "Number of decimal places for position display", 2, 0, 8, HwSettingPriority::Optional},
    {BC::Key::LifLaser::hasFl, "Has Flashlamp", "Laser has a software-controlled flashlamp", true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

Opolette::Opolette(const QString& label, QObject *parent)
    : LifLaser(QString(Opolette::staticMetaObject.className()), label, parent)
{
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
        hwError(u"Could not read laser position. %1"_s.arg(QString(resp.mid(7))));
        return -1.0;
    }

    bool ok = false;
    auto out = QString(resp.trimmed()).toDouble(&ok);
    if(!ok)
    {
        hwError("Could not parse wavelength response."_L1);
        hwDebug(u"Could not parse wavelength response. Response = %1 (Hex: %2)"_s.arg(QString(resp), QString(resp.toHex())));
                return -1.0;
    }
    return out;
}

void Opolette::setPos(double pos)
{
    auto resp = p_comm->queryCmd(QString("LP %1\n").arg(pos,0,'f',2));
    if(resp.startsWith("ERROR:"))
        hwError(u"Could not set laser position. %1"_s.arg(QString(resp.mid(7))));
}

bool Opolette::readFl()
{
    auto resp = p_comm->queryCmd(QString("FL?\n"));
    if(resp.startsWith("ERROR:"))
        hwError(u"Could not read flashlamp status. %1"_s.arg(QString(resp.mid(7))));
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
        hwError(u"Could not set flashlamp status. %1"_s.arg(QString(resp.mid(7))));
        return false;
    }

    return true;
}
