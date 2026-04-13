#include "qcpulsegenerator.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Qc9510Series, "QuantumComposers 9510 Series Pulse Generator")
REGISTER_HARDWARE_PROTOCOLS(Qc9510Series,
    CommunicationProtocol::Rs232,
    CommunicationProtocol::Tcp,
    CommunicationProtocol::Gpib)

Qc9510Series::Qc9510Series(const QString& label, QObject *parent) :
    QCPulseGenerator(QString(Qc9510Series::staticMetaObject.className()), label, 8, parent)
{
    using namespace BC::Key::PGen;
    setDefault(minWidth,0.004);
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

Qc9510Series::~Qc9510Series()
{

}

void Qc9510Series::initializePGen()
{
    setDefault(BC::Key::Comm::timeout, 100);
    setDefault(BC::Key::Comm::termChar, QString("\r\n"));
}

bool Qc9510Series::pGenWriteCmd(QString cmd)
{
    int maxAttempts = 10;
    for(int i=0; i<maxAttempts; i++)
    {
        QByteArray resp = pGenQueryCmd(cmd);
        if(resp.isEmpty())
            break;

        if(resp.startsWith("ok"))
            return true;
    }

    emit hardwareFailure();
    emit logMessage(QString("Error writing command %1").arg(cmd),LogHandler::Error);
    return false;
}

QByteArray Qc9510Series::pGenQueryCmd(QString cmd)
{
    return p_comm->queryCmd(cmd.append(QString("\n")));
}

void Qc9510Series::beginAcquisition()
{
    lockKeys(true);
}

void Qc9510Series::endAcquisition()
{
    lockKeys(false);
}
