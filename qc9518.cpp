#include "qc9518.h"

Qc9518::Qc9518(QObject *parent) :
    PulseGenerator(parent)
{
    d_subKey = QString("QC9518");
    d_prettyName = QString("Pulse Generator QC 9518");
    d_commType = CommunicationProtocol::Rs232;
    d_threaded = false;

    QSettings s(QSettings::SystemScope, QApplication::organizationName(), QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    d_minWidth = s.value(QString("minWidth"),0.004).toDouble();
    d_maxWidth = s.value(QString("maxWidth"),100000.0).toDouble();
    d_minDelay = s.value(QString("minDelay"),0.0).toDouble();
    d_maxDelay = s.value(QString("maxDelay"),100000.0).toDouble();

    s.setValue(QString("minWidth"),d_minWidth);
    s.setValue(QString("maxWidth"),d_maxWidth);
    s.setValue(QString("minDelay"),d_minDelay);
    s.setValue(QString("maxDelay"),d_maxDelay);

    s.endGroup();
    s.endGroup();
    s.sync();

}



bool Qc9518::testConnection()
{
    if(!p_comm->testConnection())
    {
	   emit connected(false);
        return false;
    }

    QByteArray resp = p_comm->queryCmd(QString("*IDN?\n"));

    if(resp.isEmpty())
    {
        emit connected(false,QString("No response to ID query."));
        return false;
    }

    if(!resp.startsWith(QByteArray("9518+")))
    {
        emit connected(false,QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp.trimmed())).arg(QString(resp.toHex())));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));

    blockSignals(true);
    readAll();
    blockSignals(false);

    pGenWriteCmd(QString(":SPULSE:STATE 1\n"));
    pGenWriteCmd(QString(":SYSTEM:KLOCK 0\n"));

    emit configUpdate(d_config);
    emit connected();
    return true;

}

void Qc9518::initialize()
{
    //set up config
    PulseGenerator::initialize();

    p_comm->initialize();
    p_comm->setReadOptions(100,true,QByteArray("\r\n"));
    testConnection();
}

QVariant Qc9518::read(const int index, const BlackChirp::PulseSetting s)
{
    QVariant out;
    QByteArray resp;
    if(index < 0 || index >= d_config.size())
        return out;

    switch (s) {
    case BlackChirp::PulseDelaySetting:
        resp = p_comm->queryCmd(QString(":PULSE%1:DELAY?\n").arg(index+1));
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
        resp = p_comm->queryCmd(QString(":PULSE%1:WIDTH?\n").arg(index+1));
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
        resp = p_comm->queryCmd(QString(":PULSE%1:STATE?\n").arg(index+1));
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
        resp = p_comm->queryCmd(QString(":PULSE%1:POLARITY?\n").arg(index+1));
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

double Qc9518::readRepRate()
{
    QByteArray resp = p_comm->queryCmd(QString(":SPULSE:PERIOD?\n"));
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

bool Qc9518::set(const int index, const BlackChirp::PulseSetting s, const QVariant val)
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
            bool success = pGenWriteCmd(QString(":PULSE%1:DELAY %2\n").arg(index+1).arg(val.toDouble()/1e6,0,'f',9));
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
            bool success = pGenWriteCmd(QString(":PULSE%1:WIDTH %2\n").arg(index+1).arg(val.toDouble()/1e6,0,'f',9));
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
                success = pGenWriteCmd(QString(":PULSE%1:POLARITY NORM\n").arg(index+1));
            else
                success = pGenWriteCmd(QString(":PULSE%1:POLARITY INV\n").arg(index+1));

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
                success = pGenWriteCmd(QString(":PULSE%1:STATE 1\n").arg(index+1));
            else
                success = pGenWriteCmd(QString(":PULSE%1:STATE 0\n").arg(index+1));

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

bool Qc9518::setRepRate(double d)
{
    if(d < 0.01 || d > 20.0)
        return false;

    if(!pGenWriteCmd(QString(":SPULSE:PERIOD %1\n").arg(1.0/d,0,'f',9)))
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

void Qc9518::sleep(bool b)
{
    if(b)
        pGenWriteCmd(QString(":SPULSE:STATE 0\n"));
    else
        pGenWriteCmd(QString(":SPULSE:STATE 1\n"));

    HardwareObject::sleep(b);
}

bool Qc9518::pGenWriteCmd(QString cmd)
{
    int maxAttempts = 10;
    for(int i=0; i<maxAttempts; i++)
    {
        QByteArray resp = p_comm->queryCmd(cmd);
        if(resp.isEmpty())
            return false;

        if(resp.startsWith("ok"))
            return true;
    }
    return false;
}

Experiment Qc9518::prepareForExperiment(Experiment exp)
{
    bool success = setAll(exp.pGenConfig());
    if(!success)
        exp.setHardwareFailed();

    return exp;
}

void Qc9518::beginAcquisition()
{
    pGenWriteCmd(QString(":SYSTEM:KLOCK 1\n"));
}

void Qc9518::endAcquisition()
{
    pGenWriteCmd(QString(":SYSTEM:KLOCK 0\n"));
}

void Qc9518::readTimeData()
{

}
