#include "qc9528.h"

Qc9528::Qc9528(QObject *parent) :
    PulseGenerator(BC::Key::qc9528,BC::Key::qc9528Name,CommunicationProtocol::Rs232,8,parent)
{
    setDefault(BC::Key::PGen::minWidth,0.004);
    setDefault(BC::Key::PGen::maxWidth,1e5);
    setDefault(BC::Key::PGen::minDelay,0.0);
    setDefault(BC::Key::PGen::maxDelay,1e5);
    setDefault(BC::Key::PGen::minRepRate,0.01);
    setDefault(BC::Key::PGen::maxRepRate,1e5);
    setDefault(BC::Key::PGen::lockExternal,true);
}

bool Qc9528::testConnection()
{
    QByteArray resp = p_comm->queryCmd(QString("*IDN?\r\n"));

    if(resp.isEmpty())
    {
        d_errorString = QString("No response to ID query.");
        return false;
    }

    if(!resp.startsWith(QByteArray("QC,9528")))
    {
        d_errorString = QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp.trimmed())).arg(QString(resp.toHex()));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));

    if(get(BC::Key::PGen::lockExternal,true))
    {
        resp = p_comm->queryCmd(QString(":PULSE0:ICLOCK?\r\n"));
        if(resp.isEmpty())
        {
            d_errorString = QString("No response to external clock source query.");
            return false;
        }
        if(!resp.startsWith("EXT10"))
        {
            if(!pGenWriteCmd(QString(":PULSE0:ICL EXT10\r\n")))
            {
                d_errorString = QString("Could not set clock source to external 10 MHz.");
                return false;
            }
        }
    }

    if(!pGenWriteCmd(QString(":PULSE0:GATE:MODE DIS\r\n")))
    {
        d_errorString = QString("Could not disable gate mode.");
        return false;
    }

    if(!pGenWriteCmd(QString(":PULSE0:TRIG:MODE DIS\r\n")))
    {
        d_errorString = QString("Could not disable external trigger mode.");
        return false;
    }

    if(!pGenWriteCmd(QString(":PULSE0:STATE 1\n")))
    {
        d_errorString = QString("Could not start pulsing.");
        return false;
    }

    readAll();

    return true;

}

void Qc9528::initializePGen()
{
    p_comm->setReadOptions(100,true,QByteArray("\r\n"));
}

void Qc9528::beginAcquisition()
{
    pGenWriteCmd(QString(":SYSTEM:KLOCK 1\r\n"));
}

void Qc9528::endAcquisition()
{
    pGenWriteCmd(QString(":SYSTEM:KLOCK 0\r\n"));
}

void Qc9528::sleep(bool b)
{
    if(b)
        pGenWriteCmd(QString(":PULSE0:STATE 0\r\n"));
    else
        pGenWriteCmd(QString(":PULSE0:STATE 1\r\n"));
}

bool Qc9528::pGenWriteCmd(QString cmd)
{
    QByteArray resp = p_comm->queryCmd(cmd);

    if(resp.startsWith("ok"))
        return true;

    emit hardwareFailure();
    emit logMessage(QString("Error writing command %1").arg(cmd),BlackChirp::LogError);
    return false;
}


bool Qc9528::setChWidth(const int index, const double width)
{
    return pGenWriteCmd(QString(":PULSE%1:WIDTH %2\r\n").arg(index+1).arg(width/1e6,0,'f',9));
}

bool Qc9528::setChDelay(const int index, const double delay)
{
     return pGenWriteCmd(QString(":PULSE%1:DELAY %2\r\n").arg(index+1).arg(delay,0,'f',9));
}

bool Qc9528::setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level)
{
    if(level == PulseGenConfig::ActiveHigh)
        return pGenWriteCmd(QString(":PULSE%1:POLARITY NORM\r\n").arg(index+1));

    return pGenWriteCmd(QString(":PULSE%1:POLARITY INV\r\n").arg(index+1));
}

bool Qc9528::setChEnabled(const int index, const bool en)
{
    if(en)
        return pGenWriteCmd(QString(":PULSE%1:STATE 1\r\n").arg(index+1));

    return pGenWriteCmd(QString(":PULSE%1:STATE 0\r\n").arg(index+1));
}

bool Qc9528::setHwRepRate(double rr)
{
    return pGenWriteCmd(QString(":PULSE0:PERIOD %1\r\n").arg(1.0/rr,0,'f',9));
}

double Qc9528::readChWidth(const int index)
{
    auto resp = p_comm->queryCmd(QString(":PULSE%1:DELAY?\r\n").arg(index+1));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double val = resp.trimmed().toDouble(&ok)*1e6;
        if(ok)
            return val;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 width. Response: %2").arg(index).arg(QString(resp)));
    return nan("");
}

double Qc9528::readChDelay(const int index)
{
    auto resp = p_comm->queryCmd(QString(":PULSE%1:WIDTH?\r\n").arg(index+1));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double val = resp.trimmed().toDouble(&ok)*1e6;
        if(ok)
            return val;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 delay. Response: %2").arg(index).arg(QString(resp)));
    return nan("");
}

PulseGenConfig::ActiveLevel Qc9528::readChActiveLevel(const int index)
{
    auto resp = p_comm->queryCmd(QString(":PULSE%1:POLARITY?\r\n").arg(index+1));
    if(!resp.isEmpty())
    {
        if(QString(resp).startsWith(QString("NORM"),Qt::CaseInsensitive))
            return PulseGenConfig::ActiveHigh;
        else if(QString(resp).startsWith(QString("INV"),Qt::CaseInsensitive))
            return PulseGenConfig::ActiveLow;
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 active level. Response: %2").arg(index).arg(QString(resp)));
    return PulseGenConfig::ActiveInvalid;
}

bool Qc9528::readChEnabled(const int index)
{
    auto resp = p_comm->queryCmd(QString(":PULSE%1:STATE?\r\n").arg(index+1));
    if(!resp.isEmpty())
    {
        bool ok = false;
        int val = resp.trimmed().toInt(&ok);
        if(ok)
            return static_cast<bool>(val);
    }

    emit hardwareFailure();
    emit logMessage(QString("Could not read channel %1 enabled state. Response: %2").arg(index).arg(QString(resp)));
    return false;
}

double Qc9528::readHwRepRate()
{
    QByteArray resp = p_comm->queryCmd(QString(":PULSE0:PERIOD?\r\n"));
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
