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
    setDefault(minWidth,0.01);
    setDefault(maxWidth,1e5);
    setDefault(minDelay,0.0);
    setDefault(maxDelay,1e5);
    setDefault(minRepRate,0.01);
    setDefault(maxRepRate,1e5);
    setDefault(lockExternal,false);
    setDefault(canDutyCycle,true);
    setDefault(canTrigger,true);
    setDefault(dutyMax,100000);
    setDefault(canSyncToChannel,true);
    setDefault(canDisableChannels,true);

    // Communication defaults
    setDefault(BC::Key::Comm::timeout, 200);
    setDefault(BC::Key::Comm::termChar, QString("\r\n"));

    save();
}

void Bnc577::initializePGen()
{
}

bool Bnc577::pGenWriteCmd(QString cmd)
{
    auto resp = pGenQueryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    emit logMessage(QString("Error writing command %1").arg(cmd),LogHandler::Error);
    return false;
}

QByteArray Bnc577::pGenQueryCmd(QString cmd)
{
    return p_comm->queryCmd(cmd.append(QString("\r\n")));
}
