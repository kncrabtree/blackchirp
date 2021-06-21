#include "qc9518.h"

Qc9518::Qc9518(QObject *parent) :
    PulseGenerator(BC::Key::qc9518,BC::Key::qc9518Name,CommunicationProtocol::Rs232,8,parent)
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

    blockSignals(true);
    readAll();
    blockSignals(false);

    pGenWriteCmd(QString(":SPULSE:STATE 1\n"));
    pGenWriteCmd(QString(":SYSTEM:KLOCK 0\n"));

    emit configUpdate(d_config);
    return true;

}

void Qc9518::initializePGen()
{
    //set up config
    PulseGenerator::initialize();

    p_comm->setReadOptions(100,true,QByteArray("\r\n"));
}

QVariant Qc9518::read(const int index, const PulseGenConfig::Setting s)
{
    QVariant out;
    QByteArray resp;
    if(index < 0 || index >= d_config.size())
        return out;

    switch (s) {
    case PulseGenConfig::DelaySetting:
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
    case PulseGenConfig::WidthSetting:
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
    case PulseGenConfig::EnabledSetting:
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
    case PulseGenConfig::LevelSetting:
        resp = p_comm->queryCmd(QString(":PULSE%1:POLARITY?\n").arg(index+1));
        if(!resp.isEmpty())
        {
            if(QString(resp).startsWith(QString("NORM"),Qt::CaseInsensitive))
                d_config.set(index,s,QVariant::fromValue(PulseGenConfig::ActiveHigh));
            else if(QString(resp).startsWith(QString("INV"),Qt::CaseInsensitive))
                d_config.set(index,s,QVariant::fromValue(PulseGenConfig::ActiveLow));
        }
        break;
    case PulseGenConfig::NameSetting:
        out = d_config.at(index).channelName;
        break;
    case PulseGenConfig::RoleSetting:
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

bool Qc9518::set(const int index, const PulseGenConfig::Setting s, const QVariant val)
{
    if(index < 0 || index >= d_config.size())
        return false;

    bool out = true;
    QString setting;
    QString target;

    switch (s) {
    case PulseGenConfig::DelaySetting:
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
    case PulseGenConfig::WidthSetting:
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
    case PulseGenConfig::LevelSetting:
        setting = QString("active level");
        target = val.value<PulseGenConfig::ActiveLevel>() == d_config.at(index).level ? QString("active high") : QString("active low");
        if(val.value<PulseGenConfig::ActiveLevel>() !=d_config.at(index).level)
        {
            bool success = false;
            if(val.value<PulseGenConfig::ActiveLevel>() == PulseGenConfig::ActiveHigh)
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
    case PulseGenConfig::EnabledSetting:
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
    case PulseGenConfig::NameSetting:
        d_config.set(index,s,val);
        read(index,s);
        break;
    case PulseGenConfig::RoleSetting:
        d_config.set(index,s,val);
        if(val.value<PulseGenConfig::Role>() != PulseGenConfig::NoRole)
        {
            d_config.set(index,PulseGenConfig::NameSetting,PulseGenConfig::roles.value(val.value<PulseGenConfig::Role>()));
            read(index,PulseGenConfig::NameSetting);
        }
        read(index,s);
        break;
    default:
        break;
    }

    if(!out)
        emit logMessage(QString("Could not set %1 to %2. Current value is %3.")
                        .arg(setting).arg(target).arg(read(index,s).toString()),BlackChirp::LogWarning);

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

void Qc9518::beginAcquisition()
{
    pGenWriteCmd(QString(":SYSTEM:KLOCK 1\n"));
}

void Qc9518::endAcquisition()
{
    pGenWriteCmd(QString(":SYSTEM:KLOCK 0\n"));
}
