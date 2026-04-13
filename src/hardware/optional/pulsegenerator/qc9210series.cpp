#include "qcpulsegenerator.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Qc9210Series, "QuantumComposers 9210 Series Pulse Generator")
REGISTER_HARDWARE_PROTOCOLS(Qc9210Series,
    CommunicationProtocol::Rs232,
    CommunicationProtocol::Tcp,
    CommunicationProtocol::Gpib)

Qc9210Series::Qc9210Series(const QString& label, QObject *parent) :
    QCPulseGenerator(QString(Qc9210Series::staticMetaObject.className()), label, 4, parent)
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

Qc9210Series::~Qc9210Series()
{

}

void Qc9210Series::initializePGen()
{
    setDefault(BC::Key::Comm::timeout, 200);
    setDefault(BC::Key::Comm::termChar, QString("\r\n"));
}

void Qc9210Series::beginAcquisition()
{
}

void Qc9210Series::endAcquisition()
{
}

bool Qc9210Series::pGenWriteCmd(QString cmd)
{
    auto resp = pGenQueryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    emit logMessage(QString("Error writing command %1").arg(cmd),LogHandler::Error);
    return false;
}

QByteArray Qc9210Series::pGenQueryCmd(QString cmd)
{
    return p_comm->queryCmd(cmd.append(QString("\r\n")));
}
