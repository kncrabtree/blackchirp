#include "bnc577.h"
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key::PGen;

// Register hardware implementations
REGISTER_HARDWARE_META(Bnc577_4, "BNC 577 4-channel pulse generator")
REGISTER_HARDWARE_PROTOCOLS(Bnc577_4, CommunicationProtocol::Rs232)
REGISTER_HARDWARE_META(Bnc577_8, "BNC 577 8-channel pulse generator")
REGISTER_HARDWARE_PROTOCOLS(Bnc577_8, CommunicationProtocol::Rs232)

Bnc577_4::Bnc577_4(const QString& label, QObject *parent)
    : QCPulseGenerator{QString(Bnc577_4::staticMetaObject.className()), label, 4, parent}
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

void Bnc577_4::initializePGen()
{
}

bool Bnc577_4::pGenWriteCmd(QString cmd)
{
    auto resp = pGenQueryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    emit logMessage(QString("Error writing command %1").arg(cmd),LogHandler::Error);
    return false;
}

QByteArray Bnc577_4::pGenQueryCmd(QString cmd)
{
    return p_comm->queryCmd(cmd.append(QString("\r\n")));
}


Bnc577_8::Bnc577_8(const QString& label, QObject *parent)
    : QCPulseGenerator{QString(Bnc577_8::staticMetaObject.className()), label, 8, parent}
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

void Bnc577_8::initializePGen()
{
}

bool Bnc577_8::pGenWriteCmd(QString cmd)
{
    auto resp = pGenQueryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    emit logMessage(QString("Error writing command %1").arg(cmd),LogHandler::Error);
    return false;
}

QByteArray Bnc577_8::pGenQueryCmd(QString cmd)
{
    return p_comm->queryCmd(cmd.append(QString("\r\n")));
}
