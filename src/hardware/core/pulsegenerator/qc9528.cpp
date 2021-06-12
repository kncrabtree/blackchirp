#include "qc9528.h"

Qc9528::Qc9528(QObject *parent) :
    PulseGenerator(BC::Key::qc9528,BC::Key::qc9528Name,CommunicationProtocol::Rs232,parent)
{
    d_numChannels = 8;

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

    blockSignals(true);
    readAll();
    blockSignals(false);

    if(d_forceExtClock)
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

    emit configUpdate(d_config);
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

QVariant Qc9528::read(const int index, const BlackChirp::PulseSetting s)
{
    QVariant out;
    QByteArray resp;
    if(index < 0 || index >= d_config.size())
        return out;

    switch (s) {
    case BlackChirp::PulseDelaySetting:
        resp = p_comm->queryCmd(QString(":PULSE%1:DELAY?\r\n").arg(index+1));
        if(!resp.isEmpty())
        {
            bool ok = false;
            double val = resp.trimmed().toDouble(&ok)*1e6;
            if(ok)
            {
                out = val;
                d_config.set(index,s,val);
            }
        }
        break;
    case BlackChirp::PulseWidthSetting:
        resp = p_comm->queryCmd(QString(":PULSE%1:WIDTH?\r\n").arg(index+1));
        if(!resp.isEmpty())
        {
            bool ok = false;
            double val = resp.trimmed().toDouble(&ok)*1e6;
            if(ok)
            {
                out = val;
                d_config.set(index,s,val);
            }
        }
        break;
    case BlackChirp::PulseEnabledSetting:
        resp = p_comm->queryCmd(QString(":PULSE%1:STATE?\r\n").arg(index+1));
        if(!resp.isEmpty())
        {
            bool ok = false;
            int val = resp.trimmed().toInt(&ok);
            if(ok)
            {
                out = static_cast<bool>(val);
                d_config.set(index,s,val);
            }
        }
        break;
    case BlackChirp::PulseLevelSetting:
        resp = p_comm->queryCmd(QString(":PULSE%1:POLARITY?\r\n").arg(index+1));
        if(!resp.isEmpty())
        {
            if(QString(resp).startsWith(QString("NORM"),Qt::CaseInsensitive))
            {
                out = static_cast<int>(BlackChirp::PulseLevelActiveHigh);
                d_config.set(index,s,out);
            }
            else if(QString(resp).startsWith(QString("INV"),Qt::CaseInsensitive))
            {
                out = static_cast<int>(BlackChirp::PulseLevelActiveLow);
                d_config.set(index,s,out);
            }
        }
        break;
    case BlackChirp::PulseNameSetting:
        out = d_config.at(index).channelName;
        break;
    case BlackChirp::PulseRoleSetting:
        out = d_config.at(index).role;
        break;
    default:
        break;
    }

    if(out.isValid())
        emit settingUpdate(index,s,out);

    return out;
}

double Qc9528::readRepRate()
{
    QByteArray resp = p_comm->queryCmd(QString(":PULSE0:PERIOD?\r\n"));
    if(resp.isEmpty())
        return -1.0;

    bool ok = false;
    double period = resp.trimmed().toDouble(&ok);
    if(!ok || period < 0.000001)
        return -1.0;

    double rr = 1.0/period;
    d_config.setRepRate(rr);
    emit repRateUpdate(rr);
    return rr;
}

bool Qc9528::set(const int index, const BlackChirp::PulseSetting s, const QVariant val)
{
    if(index < 0 || index >= d_config.size())
        return false;

    bool out = true;
    QString setting;
    QString target;

    switch (s) {
    case BlackChirp::PulseDelaySetting:
        setting = QString("delay");
        target = QString::number(val.toDouble());
        if(val.toDouble() < d_minDelay || val.toDouble() > d_maxDelay)
        {
            emit logMessage(QString("Requested delay (%1) is outside valid range (%2 - %3)").arg(target).arg(d_minDelay).arg(d_maxDelay));
            out = false;
        }
        else if(qAbs(val.toDouble() - d_config.at(index).delay) > 0.001)
        {
            bool success = pGenWriteCmd(QString(":PULSE%1:DELAY %2\r\n").arg(index+1).arg(val.toDouble()/1e6,0,'f',9));
            if(!success)
                out = false;
            else
            {
                double newVal = read(index,s).toDouble();
                if(qAbs(newVal-val.toDouble()) > 0.001)
                    out = false;
            }
        }
        break;
    case BlackChirp::PulseWidthSetting:
        setting = QString("width");
        target = QString::number(val.toDouble());
        if(val.toDouble() < d_minWidth || val.toDouble() > d_maxWidth)
        {
            emit logMessage(QString("Requested width (%1) is outside valid range (%2 - %3)").arg(target).arg(d_minWidth).arg(d_maxWidth));
            out = false;
        }
        else if(qAbs(val.toDouble() - d_config.at(index).width) > 0.001)
        {
            bool success = pGenWriteCmd(QString(":PULSE%1:WIDTH %2\r\n").arg(index+1).arg(val.toDouble()/1e6,0,'f',9));
            if(!success)
                out = false;
            else
            {
                double newVal = read(index,s).toDouble();
                if(qAbs(newVal-val.toDouble()) > 0.001)
                    out = false;
            }
        }
        break;
    case BlackChirp::PulseLevelSetting:
        setting = QString("active level");
        target = val.toInt() == static_cast<int>(d_config.at(index).level) ? QString("active high") : QString("active low");
        if(val.toInt() != static_cast<int>(d_config.at(index).level))
        {
            bool success = false;
            if(val.toInt() == static_cast<int>(BlackChirp::PulseLevelActiveHigh))
                success = pGenWriteCmd(QString(":PULSE%1:POLARITY NORM\r\n").arg(index+1));
            else
                success = pGenWriteCmd(QString(":PULSE%1:POLARITY INV\r\n").arg(index+1));

            if(!success)
                out = false;
            else
            {
                int lvl = read(index,s).toInt();
                if(lvl != val.toInt())
                    out = false;
            }
        }
        break;
    case BlackChirp::PulseEnabledSetting:
        setting = QString("enabled");
        target = val.toBool() ? QString("true") : QString("false");
        if(val.toBool() != d_config.at(index).enabled)
        {
            bool success = false;
            if(val.toBool())
                success = pGenWriteCmd(QString(":PULSE%1:STATE 1\r\n").arg(index+1));
            else
                success = pGenWriteCmd(QString(":PULSE%1:STATE 0\r\n").arg(index+1));

            if(!success)
                out = false;
            else
            {
                bool en = read(index,s).toBool();
                if(en != val.toBool())
                    out = false;
            }

        }
        break;
    case BlackChirp::PulseNameSetting:
        d_config.set(index,s,val);
        read(index,s);
        break;
    case BlackChirp::PulseRoleSetting:
        d_config.set(index,s,val);
        if(static_cast<BlackChirp::PulseRole>(val.toInt()) != BlackChirp::NoPulseRole)
        {
            d_config.set(index,BlackChirp::PulseNameSetting,BlackChirp::getPulseName(static_cast<BlackChirp::PulseRole>(val.toInt())));
            read(index,BlackChirp::PulseNameSetting);
        }
        read(index,s);
    default:
        break;
    }

    if(!out)
        emit logMessage(QString("Could not set %1 to %2. Current value is %3.")
                        .arg(setting).arg(target).arg(read(index,s).toString()));

    return out;
}

bool Qc9528::setRepRate(double d)
{
    if(d < 0.01 || d > 100000.0)
        return false;

    if(!pGenWriteCmd(QString(":PULSE0:PERIOD %1\r\n").arg(1.0/d,0,'f',9)))
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not set reprate to %1 Hz (%2 s)").arg(d,0,'f',1).arg(1.0/d,0,'f',9));
        return false;
    }

    double rr = readRepRate();
    if(rr < 0.0)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not set reprate to %1 Hz (%2 s), Value is %3 Hz.").arg(d,0,'f',1).arg(1.0/d,0,'f',9).arg(rr,0,'f',1));
        return false;
    }

    return true;
}

void Qc9528::readSettings()
{
    PulseGenerator::readSettings();

    QSettings s(QSettings::SystemScope, QApplication::organizationName(), QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    d_forceExtClock = s.value(QString("forceExtClock"),true).toBool();
    s.setValue(QString("forceExtClock"),d_forceExtClock);
    s.endGroup();
    s.endGroup();

    s.sync();
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

    return false;
}
