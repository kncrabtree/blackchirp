#include "prologixgpiblan.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(PrologixGpibLan, "Prologix GPIB-LAN Controller")
REGISTER_HARDWARE_PROTOCOLS(PrologixGpibLan, CommunicationProtocol::Tcp)
REGISTER_COMM_DEFAULTS(PrologixGpibLan, CommunicationProtocol::Tcp,
    {BC::Key::Comm::timeout, 1000},
    {BC::Key::Comm::termChar, QString("\n")})
REGISTER_HARDWARE_SETTINGS(PrologixGpibLan)

PrologixGpibLan::PrologixGpibLan(const QString& label, QObject *parent) :
    PrologixGpibController(QString(PrologixGpibLan::staticMetaObject.className()), label, 
                          CommunicationProtocol::Tcp, parent)
{
}

QString PrologixGpibLan::expectedIdResponse() const
{
    return QString("Prologix GPIB-ETHERNET Controller");
}

bool PrologixGpibLan::shouldSendSaveCfg() const
{
    return true;
}
