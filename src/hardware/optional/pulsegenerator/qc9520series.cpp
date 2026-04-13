#include "qcpulsegenerator.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Qc9520Series, "QuantumComposers 9520 Series Pulse Generator")
REGISTER_HARDWARE_PROTOCOLS(Qc9520Series,
    CommunicationProtocol::Rs232,
    CommunicationProtocol::Tcp,
    CommunicationProtocol::Gpib)

Qc9520Series::Qc9520Series(const QString& label, QObject *parent) :
    QCPulseGenerator(QString(Qc9520Series::staticMetaObject.className()), label, 8, parent)
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

Qc9520Series::~Qc9520Series()
{

}

void Qc9520Series::initializePGen()
{
    setDefault(BC::Key::Comm::timeout, 200);
    setDefault(BC::Key::Comm::termChar, QString("\r\n"));
}

void Qc9520Series::beginAcquisition()
{
    lockKeys(true);
}

void Qc9520Series::endAcquisition()
{
    lockKeys(false);
}

bool Qc9520Series::pGenWriteCmd(QString cmd)
{
    auto resp = pGenQueryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    emit logMessage(QString("Error writing command %1").arg(cmd),LogHandler::Error);
    return false;
}

QByteArray Qc9520Series::pGenQueryCmd(QString cmd)
{
    return p_comm->queryCmd(cmd.append(QString("\r\n")));
}
