#include "qcpulsegenerator.h"

#include <QThread>

QCPulseGenerator::QCPulseGenerator(const QString subKey, const QString name, CommunicationProtocol::CommType commType, int numChannels, QObject *parent, bool threaded, bool critical) : PulseGenerator(subKey,name,commType,numChannels,parent,threaded,critical)
{
}

QCPulseGenerator::~QCPulseGenerator()
{
}

bool QCPulseGenerator::testConnection()
{
    QByteArray resp = pGenQueryCmd(QString("*IDN?"));

    if(resp.isEmpty())
    {
        d_errorString = QString("No response to ID query.");
        return false;
    }

    if(!resp.startsWith(idResponse().toLatin1()))
    {
        d_errorString = QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp.trimmed())).arg(QString(resp.toHex()));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));

    if(get(BC::Key::PGen::lockExternal,true))
    {
        if(!pGenWriteCmd(QString(":%1:ICL %2").arg(sysStr()).arg(clock10MHzStr())))
        {
            d_errorString = QString("Could not set clock source to external 10 MHz.");
            return false;
        }
    }

    readAll();

    return true;
}

void QCPulseGenerator::sleep(bool b)
{
    if(b)
        setPulseEnabled(false);
}

bool QCPulseGenerator::setChWidth(const int index, const double width)
{
    return pGenWriteCmd(QString(":PULSE%1:WIDTH %2").arg(index+1).arg(width/1e6,0,'f',9));
}

bool QCPulseGenerator::setChDelay(const int index, const double delay)
{
    return pGenWriteCmd(QString(":PULSE%1:DELAY %2").arg(index+1).arg(delay/1e6,0,'f',9));
}

bool QCPulseGenerator::setChActiveLevel(const int index, const ActiveLevel level)
{
    if(level == PulseGenConfig::ActiveHigh)
        return pGenWriteCmd(QString(":PULSE%1:POLARITY NORM").arg(index+1));

    return pGenWriteCmd(QString(":PULSE%1:POLARITY INV").arg(index+1));
}

bool QCPulseGenerator::setChEnabled(const int index, const bool en)
{
    if(en)
        return pGenWriteCmd(QString(":PULSE%1:STATE 1").arg(index+1));

    return pGenWriteCmd(QString(":PULSE%1:STATE 0").arg(index+1));
}

bool QCPulseGenerator::setChSyncCh(const int index, const int syncCh)
{
    return pGenWriteCmd(QString(":PULSE%1:SYNC %2").arg(index+1).arg(d_channels.at(syncCh)));
}

bool QCPulseGenerator::setChMode(const int index, const ChannelMode mode)
{
    QString mstr("NORM");
    if(mode == PulseGenConfig::DutyCycle)
        mstr = QString("DCYC");
    return pGenWriteCmd(QString(":PULSE%1:CMODE %2").arg(index+1).arg(mstr));
}

bool QCPulseGenerator::setChDutyOn(const int index, const int pulses)
{
    return pGenWriteCmd(QString(":PULSE%1:PCO %2").arg(index+1).arg(pulses));
}

bool QCPulseGenerator::setChDutyOff(const int index, const int pulses)
{
    return pGenWriteCmd(QString(":PULSE%1:OCO %2").arg(index+1).arg(pulses));
}

bool QCPulseGenerator::setHwPulseMode(PGenMode mode)
{
    QString mstr("DIS");
    QString smstr("NORM");
    if(mode == Triggered)
    {
        mstr = QString("TRIG");
        smstr = QString("SINGLE");
    }

    bool success = pGenWriteCmd(QString("%1 %2").arg(trigBase()).arg(mstr));
    success &= pGenWriteCmd(QString(":%1:MOD %2").arg(sysStr()).arg(smstr));
    QThread::msleep(5);
    return success;
}

bool QCPulseGenerator::setHwRepRate(double rr)
{
    return pGenWriteCmd(QString(":%1:PERIOD %2").arg(sysStr()).arg(1.0/rr,0,'f',9));
}

bool QCPulseGenerator::setHwPulseEnabled(bool en)
{
    if(en)
        return pGenWriteCmd(QString(":%1:STATE 1").arg(sysStr()));

    return pGenWriteCmd(QString(":%1:STATE 0").arg(sysStr()));
}

double QCPulseGenerator::readChWidth(const int index)
{
    auto resp = pGenQueryCmd(QString(":PULSE%1:WIDTH?").arg(index+1));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double val = resp.trimmed().toDouble(&ok)*1e6;
        if(ok)
            return val;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 width. Response: %2").arg(index+1).arg(QString(resp)));
    return nan("");
}

double QCPulseGenerator::readChDelay(const int index)
{
    auto resp = pGenQueryCmd(QString(":PULSE%1:DELAY?").arg(index+1));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double val = resp.trimmed().toDouble(&ok)*1e6;
        if(ok)
            return val;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 delay. Response: %2").arg(index+1).arg(QString(resp)));
    return nan("");
}

PulseGenConfig::ActiveLevel QCPulseGenerator::readChActiveLevel(const int index)
{
    auto resp = pGenQueryCmd(QString(":PULSE%1:POLARITY?").arg(index+1));
    if(!resp.isEmpty())
    {
        if(QString(resp).startsWith(QString("NORM"),Qt::CaseInsensitive))
            return PulseGenConfig::ActiveHigh;
        else if(QString(resp).startsWith(QString("INV"),Qt::CaseInsensitive))
            return PulseGenConfig::ActiveLow;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 active level. Response: %2").arg(index+1).arg(QString(resp)));
    return PulseGenConfig::ActiveHigh;
}

bool QCPulseGenerator::readChEnabled(const int index)
{
    auto resp = pGenQueryCmd(QString(":PULSE%1:STATE?").arg(index+1));
    if(!resp.isEmpty())
    {
        bool ok = false;
        int val = resp.trimmed().toInt(&ok);
        if(ok)
            return static_cast<bool>(val);
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 enabled state. Response: %2").arg(index+1).arg(QString(resp)));
    return false;
}

int QCPulseGenerator::readChSynchCh(const int index)
{
    auto resp = pGenQueryCmd(QString(":PULSE%1:SYNC?").arg(index+1));
    if(!resp.isEmpty())
    {
        QString val = QString(resp.trimmed());
        int idx = d_channels.indexOf(val);
        if(idx >= 0)
            return idx;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 sync channel. Response: %2").arg(index+1).arg(QString(resp)));
    return -1;
}

PulseGenConfig::ChannelMode QCPulseGenerator::readChMode(const int index)
{
    auto resp = pGenQueryCmd(QString(":PULSE%1:CMOD?").arg(index+1));
    if(!resp.isEmpty())
    {
        QString val = QString(resp.trimmed());
        if(resp.contains("DCYC"))
            return DutyCycle;
        else if(resp.contains("NORM"))
            return Normal;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 mode. Response: %2").arg(index+1).arg(QString(resp)));
    return Normal;
}

int QCPulseGenerator::readChDutyOn(const int index)
{
    auto resp = pGenQueryCmd(QString(":PULSE%1:PCO?").arg(index+1));
    if(!resp.isEmpty())
    {
        bool ok = false;
        auto val = resp.trimmed().toInt(&ok);

        if(ok)
            return val;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 duty cycle on pulses. Response: %2").arg(index).arg(QString(resp)));
    return -1;
}

int QCPulseGenerator::readChDutyOff(const int index)
{
    auto resp = pGenQueryCmd(QString(":PULSE%1:OCO?").arg(index+1));
    if(!resp.isEmpty())
    {
        bool ok = false;
        auto val = resp.trimmed().toInt(&ok);

        if(ok)
            return val;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 duty cycle off pulses. Response: %2").arg(index).arg(QString(resp)));
    return -1;
}

PulseGenConfig::PGenMode QCPulseGenerator::readHwPulseMode()
{
    auto resp = pGenQueryCmd(QString("%1?").arg(trigBase()));
    if(!resp.isEmpty())
    {
        if(resp.contains("DIS"))
            return Continuous;
        if(resp.contains("TRIG"))
            return Triggered;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read system pulse mode. Response: %1").arg(QString(resp)));
    return Continuous;
}

double QCPulseGenerator::readHwRepRate()
{
    QByteArray resp = pGenQueryCmd(QString(":%1:PERIOD?").arg(sysStr()));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double period = resp.trimmed().toDouble(&ok);
        if(ok && period > 0.0)
            return 1.0/period;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read rep rate. Response: %1").arg(QString(resp)));
    return nan("");
}

bool QCPulseGenerator::readHwPulseEnabled()
{
    QByteArray resp = pGenQueryCmd(QString(":%1:STATE?").arg(sysStr()));
    if(!resp.isEmpty())
    {
        bool ok = false;
        int en = resp.trimmed().toInt(&ok);
        if(ok)
            return static_cast<bool>(en);
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read system pulse enabled status. Response: %1").arg(QString(resp)));
    return false;
}

void QCPulseGenerator::lockKeys(bool lock)
{
    if(lock)
        pGenWriteCmd(QString(":SYSTEM:KLOCK 1").arg(sysStr()));
    else
        pGenWriteCmd(QString(":SYSTEM:KLOCK 0").arg(sysStr()));
}
