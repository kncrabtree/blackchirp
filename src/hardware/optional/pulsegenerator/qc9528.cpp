#include "qcpulsegenerator.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Qc9528, "QuantumComposers 9528 Pulse Generator")

Qc9528::Qc9528(const QString& label, QObject *parent) :
    QCPulseGenerator(QString(Qc9528::staticMetaObject.className()), label, 8, parent)
{
    using namespace BC::Key::PGen;
    setDefault(minWidth,0.004);
    setDefault(maxWidth,1e5);
    setDefault(minDelay,0.0);
    setDefault(maxDelay,1e5);
    setDefault(minRepRate,0.01);
    setDefault(maxRepRate,1e5);
    setDefault(lockExternal,true);
    setDefault(canDutyCycle,true);
    setDefault(canTrigger,true);
    setDefault(dutyMax,100000);
    setDefault(canSyncToChannel,true);
    setDefault(canDisableChannels,true);

    save();
}

Qc9528::~Qc9528()
{

}

void Qc9528::initializePGen()
{
    setDefault(BC::Key::Comm::timeout, 200);
    setDefault(BC::Key::Comm::termChar, QString("\r\n"));
}

void Qc9528::beginAcquisition()
{
    lockKeys(true);
}

void Qc9528::endAcquisition()
{
    lockKeys(false);
}

bool Qc9528::pGenWriteCmd(QString cmd)
{
    auto resp = pGenQueryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    emit logMessage(QString("Error writing command %1").arg(cmd),LogHandler::Error);
    return false;
}

QByteArray Qc9528::pGenQueryCmd(QString cmd)
{
    return p_comm->queryCmd(cmd.append(QString("\r\n")));
}


