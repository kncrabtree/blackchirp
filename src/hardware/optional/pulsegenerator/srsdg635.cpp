#include "srsdg635.h"

using namespace BC::Key::PGen;

SRSDG635::SRSDG635(QObject *parent)
    : PulseGenerator{dg635,dg635Name,CommunicationProtocol::Rs232,4,parent,false,true}
{
    setDefault(minWidth,0.0001);
    setDefault(maxWidth,1e5);
    setDefault(minDelay,0.0);
    setDefault(maxDelay,1e5);
    setDefault(minRepRate,0.01);
    setDefault(maxRepRate,1e5);
    setDefault(lockExternal,false);
    setDefault(canDutyCycle,false);
    setDefault(canTrigger,true);
    setDefault(dutyMax,100000);
    setDefault(canSyncToChannel,true);
}


bool SRSDG635::testConnection()
{
    auto resp = p_comm->queryCmd("*IDN?\n");

    if(resp.isEmpty())
    {
        d_errorString = "No response to ID query.";
        return false;
    }

    if(!resp.startsWith("Stanford Research Systems,DG645"))
    {
        d_errorString = QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp.trimmed())).arg(QString(resp.toHex()));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));

    if(get(lockExternal,true))
    {
        resp = p_comm->queryCmd("TIMB?\n");
        if(!resp.contains('3'))
            emit logMessage("Timebase is not set to external.",LogHandler::Warning);
    }

    readAll();

    return true;
}

void SRSDG635::initializePGen()
{
    p_comm->setReadOptions(500,true,"\r\n");
}

bool SRSDG635::setChWidth(const int index, const double width)
{
    if(index >= d_numChannels)
        return false;

    //widths correspond to delay for channels B (3), D (5), F (7), and H (9)
    //indices are 0=AB, 1=CD, 2=EF, 3=GH.
    //Delay command takes the target channel, then the reference, then the time
    //So target channel is 2*index+3, ref channel is 2*index+2
    return p_comm->writeCmd(QString("DLAY %1,%2,%3\n").arg(2*index+3).arg(2*index+2).arg(width/1e6,0,'f',12));
}

bool SRSDG635::setChDelay(const int index, const double delay)
{
    if(index >= d_numChannels)
        return false;

    const auto &cfg = getConfig();
    auto targetCh = 2*index+2;
    auto syncCh = cfg.d_channels.at(index).syncCh > 0 ? cfg.d_channels.at(index).syncCh*2+2 : 0;
    return p_comm->writeCmd(QString("DLAY %1,%2,%3\n").arg(targetCh).arg(syncCh).arg(delay/1e6,0,'f',12));
}

bool SRSDG635::setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level)
{
    if(index >= d_numChannels)
        return false;

    int pol = 1;
    if(level == PulseGenConfig::ActiveLow)
        pol = 0;

    //when referring to outputs on the DG635, 1=AB, 2=CD, 3=EF, 4=GH
    return p_comm->writeCmd(QString("LPOL %1,%2\n").arg(index+1).arg(pol));
}

bool SRSDG635::setChEnabled(const int index, const bool en)
{
}

bool SRSDG635::setChSyncCh(const int index, const int syncCh)
{
}

bool SRSDG635::setChMode(const int index, const PulseGenConfig::ChannelMode mode)
{
}

bool SRSDG635::setChDutyOn(const int index, const int pulses)
{
}

bool SRSDG635::setChDutyOff(const int index, const int pulses)
{
}

bool SRSDG635::setHwPulseMode(PulseGenConfig::PGenMode mode)
{
}

bool SRSDG635::setHwRepRate(double rr)
{
}

bool SRSDG635::setHwPulseEnabled(bool en)
{
}

double SRSDG635::readChWidth(const int index)
{
}

double SRSDG635::readChDelay(const int index)
{
}

PulseGenConfig::ActiveLevel SRSDG635::readChActiveLevel(const int index)
{
}

bool SRSDG635::readChEnabled(const int index)
{
}

int SRSDG635::readChSynchCh(const int index)
{
}

PulseGenConfig::ChannelMode SRSDG635::readChMode(const int index)
{
}

int SRSDG635::readChDutyOn(const int index)
{
}

int SRSDG635::readChDutyOff(const int index)
{
}

PulseGenConfig::PGenMode SRSDG635::readHwPulseMode()
{
}

double SRSDG635::readHwRepRate()
{
}

bool SRSDG635::readHwPulseEnabled()
{
}
