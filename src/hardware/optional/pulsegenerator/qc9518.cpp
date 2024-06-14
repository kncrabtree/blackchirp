#include "qcpulsegenerator.h"

Qc9518::Qc9518(QObject *parent) :
    QCPulseGenerator(BC::Key::PGen::qc9518,BC::Key::PGen::qc9518Name,CommunicationProtocol::Rs232,8,parent)
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
}

Qc9518::~Qc9518()
{

}

void Qc9518::initializePGen()
{
    p_comm->setReadOptions(100,true,QByteArray("\r\n"));
}

bool Qc9518::pGenWriteCmd(QString cmd)
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

QByteArray Qc9518::pGenQueryCmd(QString cmd)
{
    return p_comm->queryCmd(cmd.append(QString("\n")));
}

void Qc9518::beginAcquisition()
{
    lockKeys(true);
}

void Qc9518::endAcquisition()
{
    lockKeys(false);
}

