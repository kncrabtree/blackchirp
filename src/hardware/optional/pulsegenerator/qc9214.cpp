#include "qcpulsegenerator.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Qc9214, "QuantumComposers 9214 Pulse Generator")
REGISTER_HARDWARE_PROTOCOLS(Qc9214, CommunicationProtocol::Rs232)

Qc9214::Qc9214(const QString& label, QObject *parent) :
    QCPulseGenerator(QString(Qc9214::staticMetaObject.className()), label, 4, parent)
{
    using namespace BC::Key::PGen;
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

    save();
}

Qc9214::~Qc9214()
{

}

void Qc9214::initializePGen()
{
    setDefault(BC::Key::Comm::timeout, 200);
    setDefault(BC::Key::Comm::termChar, QString("\r\n"));
}

void Qc9214::beginAcquisition()
{
}

void Qc9214::endAcquisition()
{
}

bool Qc9214::pGenWriteCmd(QString cmd)
{
    auto resp = pGenQueryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    emit logMessage(QString("Error writing command %1").arg(cmd),LogHandler::Error);
    return false;
}

QByteArray Qc9214::pGenQueryCmd(QString cmd)
{
    return p_comm->queryCmd(cmd.append(QString("\r\n")));
}
