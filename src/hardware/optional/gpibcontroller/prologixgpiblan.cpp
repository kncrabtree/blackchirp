#include "prologixgpiblan.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(PrologixGpibLan, "Prologix GPIB-LAN Controller")

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
