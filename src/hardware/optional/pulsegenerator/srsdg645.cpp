#include "srsdg645.h"

using namespace BC::Key::PGen;

SRSDG645::SRSDG645(QObject *parent)
    : PulseGenerator{dg645,dg645Name,CommunicationProtocol::Rs232,4,parent,false,true}
{
    setDefault(minWidth,0.0000);
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
    setDefault(canDisableChannels,false);
}


bool SRSDG645::testConnection()
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

    p_comm->writeCmd("*CLS\n");

    readAll();

    return true;
}

void SRSDG645::initializePGen()
{
    p_comm->setReadOptions(500,true,"\r\n");
}

bool SRSDG645::setChWidth(const int index, const double width)
{
    if(index >= d_numChannels)
        return false;

    //widths correspond to delay for channels B (3), D (5), F (7), and H (9)
    //indices are 0=AB, 1=CD, 2=EF, 3=GH.
    //Delay command takes the target channel, then the reference, then the time
    //So target channel is 2*index+3, ref channel is 2*index+2
    return p_comm->writeCmd(QString("DLAY %1,%2,%3\n").arg(2*index+3).arg(2*index+2).arg(width/1e6,0,'f',12));
}

bool SRSDG645::setChDelay(const int index, const double delay)
{
    if(index >= d_numChannels)
        return false;

    const auto &cfg = getConfig();
    auto targetCh = 2*index+2;
    auto syncCh = cfg.d_channels.at(index).syncCh > 0 ? cfg.d_channels.at(index).syncCh*2+2 : 0;
    return p_comm->writeCmd(QString("DLAY %1,%2,%3\n").arg(targetCh).arg(syncCh).arg(delay/1e6,0,'f',12));
}

bool SRSDG645::setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level)
{
    if(index >= d_numChannels)
        return false;

    int pol = 1;
    if(level == PulseGenConfig::ActiveLow)
        pol = 0;

    //when referring to outputs on the DG635, 1=AB, 2=CD, 3=EF, 4=GH
    return p_comm->writeCmd(QString("LPOL %1,%2\n").arg(index+1).arg(pol));
}

bool SRSDG645::setChEnabled(const int index, const bool en)
{
    Q_UNUSED(index)
    Q_UNUSED(en)

    //DG 645 output channels are always enabled
    return true;
}

bool SRSDG645::setChSyncCh(const int index, const int syncCh)
{
    if(index >= d_numChannels)
        return false;

    //note that syncCh is indexed from 1 so that 0 = T0
    const auto &cfg = getConfig();
    auto d = cfg.setting(index,PulseGenConfig::DelaySetting).toDouble();
    auto targetCh = 2*index+2;
    auto sc = 2*syncCh; // don't add 2 because 0 is T0, not Ch A

    return p_comm->writeCmd(QString("DLAY %1,%2,%3\n").arg(targetCh).arg(sc).arg(d/1e6,0,'f',12));
}

bool SRSDG645::setChMode(const int index, const PulseGenConfig::ChannelMode mode)
{
    Q_UNUSED(index)
    Q_UNUSED(mode)
    //not supported
    return true;
}

bool SRSDG645::setChDutyOn(const int index, const int pulses)
{
    Q_UNUSED(index)
    Q_UNUSED(pulses)
    //not supported
    return true;
}

bool SRSDG645::setChDutyOff(const int index, const int pulses)
{
    Q_UNUSED(index)
    Q_UNUSED(pulses)
    //not supported
    return true;
}

bool SRSDG645::setHwPulseMode(PulseGenConfig::PGenMode mode)
{
    QString cmd;
    switch(mode)
    {
    case PulseGenConfig::Continuous:
        cmd = "TSRC 0\n";
        break;
    case PulseGenConfig::Triggered_Rising:
        cmd = "TSRC 3\n";
        break;
    case PulseGenConfig::Triggered_Falling:
        cmd = "TSRC 4\n";
        break;
    }

    return p_comm->writeCmd(cmd);
}

bool SRSDG645::setHwRepRate(double rr)
{
    return p_comm->writeCmd(QString("TRAT %1\n").arg(rr,0,'f',2));
}

bool SRSDG645::setHwPulseEnabled(bool en)
{
    //the DG 645 doesn't have a proper enable/disable
    //as a workaround, we can set it to single shot mode to disable it, and return to regular
    //triggered mode to enable it.

    if(en)
    {
        const auto &cfg = getConfig();
        return setHwPulseMode(cfg.d_mode);
    }

    return p_comm->writeCmd("TSRC 5\n");
}

double SRSDG645::readChWidth(const int index)
{
    if(index >= d_numChannels)
        return -1.0;

    auto targetCh = 2*index+3;
    auto refCh = 2*index+2;
    double out = -1.0;
    bool retrying = false;
    while(true)
    {
        QByteArray resp = p_comm->queryCmd(QString("DLAY?%1\n").arg(targetCh));
        auto l = resp.split(',');
        if(l.size() < 2)
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not read Ch %1 width. Response: %2").arg(index).arg(QString(resp)),LogHandler::Error);
            return -1.0;
        }
        auto i = l.constFirst().toInt();
        if(i != refCh)
        {
            if(!retrying)
            {
                //link channels and try again
                p_comm->writeCmd(QString("LINK %1,%2\n").arg(targetCh).arg(refCh));
                retrying = true;
                continue;
            }
            else
            {
                emit hardwareFailure();
                emit logMessage(QString("Could not read Ch %1 width. Response: %2").arg(index).arg(QString(resp)),LogHandler::Error);
                return -1.0;
            }
        }
        else
        {
            bool ok = false;
            out = l.at(1).trimmed().toDouble(&ok)*1e6;
            if(!ok)
            {
                emit hardwareFailure();
                emit logMessage(QString("Could not read Ch %1 width. Response: %2").arg(index).arg(QString(resp)),LogHandler::Error);
                return -1.0;
            }
            else
                break;
        }
    }

    return out;


}

double SRSDG645::readChDelay(const int index)
{
    if(index >= d_numChannels)
        return -1.0;

    auto targetCh = 2*index+2;
    double out = -1.0;
    QByteArray resp = p_comm->queryCmd(QString("DLAY?%1\n").arg(targetCh));
    auto l = resp.split(',');
    if(l.size() < 2)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read Ch %1 delay. Response: %2").arg(index).arg(QString(resp)),LogHandler::Error);
        return -1.0;
    }

    bool ok = false;
    out = l.at(1).trimmed().toDouble(&ok)*1e6;
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read Ch %1 delay. Response: %2").arg(index).arg(QString(resp)),LogHandler::Error);
        return -1.0;
    }

    return out;

}

PulseGenConfig::ActiveLevel SRSDG645::readChActiveLevel(const int index)
{
    if(index >= d_numChannels)
        return PulseGenConfig::ActiveHigh;

    auto resp = p_comm->queryCmd(QString("LPOL?%1\n").arg(index+1)).trimmed();
    if(resp.contains('0'))
        return PulseGenConfig::ActiveLow;
    else if(resp.contains('1'))
        return PulseGenConfig::ActiveHigh;

    emit hardwareFailure();
    emit logMessage(QString("Could not read Ch %1 active level. Response: %2").arg(index).arg(QString(resp)),LogHandler::Error);
    return PulseGenConfig::ActiveHigh;

}

bool SRSDG645::readChEnabled(const int index)
{
    //not supported
    Q_UNUSED(index)
    return true;
}

int SRSDG645::readChSynchCh(const int index)
{
    auto targetCh = 2*index+2;
    auto resp = p_comm->queryCmd(QString("DLAY?%1\n").arg(targetCh)).trimmed();
    auto l = resp.split(',');
    if(l.size() < 2)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read Ch %1 sync channel. Response: %2").arg(index).arg(QString(resp)),LogHandler::Error);
        return -1;
    }

    bool ok = false;
    auto ch = l.constFirst().toInt(&ok);
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read Ch %1 sync channel. Response: %2").arg(index).arg(QString(resp)),LogHandler::Error);
        return -1;
    }

    return ch/2;

}

PulseGenConfig::ChannelMode SRSDG645::readChMode(const int index)
{
    //not supported
    Q_UNUSED(index)
    return PulseGenConfig::Normal;
}

int SRSDG645::readChDutyOn(const int index)
{
    //not supported
    Q_UNUSED(index)
    return 1;
}

int SRSDG645::readChDutyOff(const int index)
{
    //not supported
    Q_UNUSED(index)
    return 1;
}

PulseGenConfig::PGenMode SRSDG645::readHwPulseMode()
{
    auto resp = p_comm->queryCmd("TSRC?\n").trimmed();
    bool ok = false;
    auto src = resp.toInt(&ok);
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage("Could not read pulse mode.",LogHandler::Error);
        return PulseGenConfig::Continuous;
    }

    switch(src)
    {
    case 0:
        return PulseGenConfig::Continuous;
        break;
    case 3:
        return PulseGenConfig::Triggered_Rising;
        break;
    case 4:
        return PulseGenConfig::Triggered_Falling;
        break;
    case 5:
    {
        //pulsing is disabled, so read value from config
        const auto &cfg = getConfig();
        return cfg.d_mode;
        break;
    }
    default:
        //triggering mode is set to something else; change it and throw a warning
        emit logMessage(QString("Device is set to an unsupported trigger source. Disabling pulses."),LogHandler::Warning);
        setPulseEnabled(false);
        return PulseGenConfig::Continuous;
    }

    //not reached
    return PulseGenConfig::Continuous;
}

double SRSDG645::readHwRepRate()
{
    auto resp = p_comm->queryCmd("TRAT?\n").trimmed();
    bool ok = false;
    auto rr = resp.toDouble(&ok);
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read rep rate. Response: %1").arg(QString(resp)),LogHandler::Error);
        return -1.0;
    }

    return rr;
}

bool SRSDG645::readHwPulseEnabled()
{
    auto resp = p_comm->queryCmd("TSRC?\n").trimmed();
    bool ok = false;
    auto src = resp.toInt(&ok);
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage("Could not read pulse enabled status.",LogHandler::Error);
        return false;
    }

    return src != 5;
}
