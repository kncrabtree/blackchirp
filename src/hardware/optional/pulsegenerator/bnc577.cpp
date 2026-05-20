#include "bnc577.h"
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key::PGen;

// Register hardware implementation
REGISTER_HARDWARE_META(Bnc577, "BNC 577 Pulse Generator")
REGISTER_HARDWARE_PROTOCOLS(Bnc577,
    CommunicationProtocol::Rs232,
    CommunicationProtocol::Tcp,
    CommunicationProtocol::Gpib)

Bnc577::Bnc577(const QString& label, QObject *parent)
    : QCPulseGenerator{QString(Bnc577::staticMetaObject.className()), label, 8, parent}
{
}

void Bnc577::initializePGen()
{
}

bool Bnc577::pGenWriteCmd(const QString &cmd)
{
    auto resp = pGenQueryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    hwError(u"Error writing command %1"_s.arg(cmd));
    return false;
}

QByteArray Bnc577::pGenQueryCmd(const QString &cmd)
{
    return p_comm->queryCmd(cmd + "\r\n"_L1);
}
