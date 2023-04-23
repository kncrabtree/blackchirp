#include "qcpulsegenerator.h"

Qc9214::Qc9214(QObject *parent) :
    QCPulseGenerator(BC::Key::PGen::qc9214,BC::Key::PGen::qc9214Name,CommunicationProtocol::Rs232,4,parent)
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
}

Qc9214::~Qc9214()
{

}

void Qc9214::initializePGen()
{
    p_comm->setReadOptions(200,true,QByteArray("\r\n"));
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
