#include "qc9518.h"

Qc9518::Qc9518(QObject *parent) :
    PulseGenerator(BC::Key::qc9518,BC::Key::qc9518Name,CommunicationProtocol::Rs232,8,parent)
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
}

Qc9518::~Qc9518()
{

}

bool Qc9518::testConnection()
{
    QByteArray resp = p_comm->queryCmd(QString("*IDN?\n"));

    if(resp.isEmpty())
    {
        d_errorString = QString("No response to ID query.");
        return false;
    }

    if(!resp.startsWith(QByteArray("9518+")))
    {
        d_errorString = QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp.trimmed())).arg(QString(resp.toHex()));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));

    pGenWriteCmd(QString(":SPULSE:STATE 1\n"));
    pGenWriteCmd(QString(":SYSTEM:KLOCK 0\n"));
    readAll();

    return true;

}

void Qc9518::initializePGen()
{
    //set up config
    PulseGenerator::initialize();

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

