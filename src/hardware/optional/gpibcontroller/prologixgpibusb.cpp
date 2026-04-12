#include "prologixgpibusb.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(PrologixGpibUsb, "Prologix GPIB-USB Controller")
REGISTER_HARDWARE_PROTOCOLS(PrologixGpibUsb, CommunicationProtocol::Rs232)
REGISTER_HARDWARE_SETTINGS(PrologixGpibUsb)

PrologixGpibUsb::PrologixGpibUsb(const QString& label, QObject *parent) :
    PrologixGpibController(QString(PrologixGpibUsb::staticMetaObject.className()), label, 
                          CommunicationProtocol::Rs232, parent)
{
}

QString PrologixGpibUsb::expectedIdResponse() const
{
    return QString("Prologix GPIB-USB Controller");
}

bool PrologixGpibUsb::shouldSendSaveCfg() const
{
    return false;  // USB version doesn't send savecfg command
}
