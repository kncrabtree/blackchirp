#include "qc9528.h"

#include "rs232instrument.h"

Qc9528::Qc9528(QObject *parent) :
    PulseGenerator(parent)
{
    d_subKey = QString("qc9528");
    d_prettyName = QString("Pulse Generator QC 9528");

    p_comm = new Rs232Instrument(d_key,d_subKey,this);
    connect(p_comm,&CommunicationProtocol::logMessage,this,&HardwareObject::logMessage);
    connect(p_comm,&CommunicationProtocol::hardwareFailure,this,&HardwareObject::hardwareFailure);

    p_comm->setReadOptions(100,true,QByteArray("\r\n"));
}



bool Qc9528::testConnection()
{
    if(!p_comm->testConnection())
    {
        emit connected(false,QString("RS232 error."));
        return false;
    }

    QByteArray resp = p_comm->queryCmd(QString("*IDN?\n"));

    if(resp.isEmpty())
    {
        emit connected(false,QString("No response to ID query."));
        return false;
    }

    if(!resp.startsWith(QByteArray("QC,9528")))
    {
        emit connected(false,QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp.trimmed())).arg(QString(resp.toHex())));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));

    blockSignals(true);
    readAll();
    blockSignals(false);

    pGenWriteCmd(QString(":PULSE0:STATE 1\n"));

    emit configUpdate(d_config);
    emit connected();
    return true;

}

void Qc9528::initialize()
{
    //set up config
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    s.beginReadArray(QString("channels"));
    for(int i=0; i<BC_PGEN_NUMCHANNELS; i++)
    {
        s.setArrayIndex(i);
        QString name = s.value(QString("name"),QString("Ch%1").arg(i)).toString();
        double d = s.value(QString("defaultDelay"),0.0).toDouble();
        double w = s.value(QString("defaultWidth"),0.050).toDouble();
        QVariant lvl = s.value(QString("level"),BlackChirp::PulseLevelActiveHigh);
        bool en = s.value(QString("defaultEnabled"),false).toBool();

        if(lvl == QVariant(BlackChirp::PulseLevelActiveHigh))
            d_config.add(name,en,d,w,BlackChirp::PulseLevelActiveHigh);
        else
            d_config.add(name,en,d,w,BlackChirp::PulseLevelActiveLow);
    }
    s.endArray();

    d_config.setRepRate(s.value(QString("repRate"),10.0).toDouble());
    s.endGroup();
    s.endGroup();

    p_comm->initialize();
    testConnection();
}

Experiment Qc9528::prepareForExperiment(Experiment exp)
{
    bool success = setAll(exp.pGenConfig());
    if(!success)
        exp.setHardwareFailed();

    return exp;
}

void Qc9528::beginAcquisition()
{
    pGenWriteCmd(QString(":SYSTEM:KLOCK 1\n"));
}

void Qc9528::endAcquisition()
{
    pGenWriteCmd(QString(":SYSTEM:KLOCK 0\n"));
}

void Qc9528::readTimeData()
{
}

QVariant Qc9528::read(const int index, const BlackChirp::PulseSetting s)
{
    QVariant out;
    QByteArray resp;
    if(index < 0 || index >= d_config.size())
        return out;

    switch (s) {
    case BlackChirp::PulseDelay:
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
    case BlackChirp::PulseWidth:
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
    case BlackChirp::PulseEnabled:
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
    case BlackChirp::PulseLevel:
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
    case BlackChirp::PulseName:
        out = d_config.at(index).channelName;
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
    QByteArray resp = p_comm->queryCmd(QString(":PULSE0:PERIOD?\n"));
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
    case BlackChirp::PulseDelay:
        setting = QString("delay");
        target = QString::number(val.toDouble());
        if(qAbs(val.toDouble() - d_config.at(index).delay) > 0.001)
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
    case BlackChirp::PulseWidth:
        setting = QString("width");
        target = QString::number(val.toDouble());
        if(qAbs(val.toDouble() - d_config.at(index).width) > 0.001)
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
    case BlackChirp::PulseLevel:
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
    case BlackChirp::PulseEnabled:
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
    case BlackChirp::PulseName:
        d_config.set(index,s,val);
        read(index,s);
        break;
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
    if(d < 0.01 || d > 20.0)
        return false;

    if(!pGenWriteCmd(QString(":PULSE0:PERIOD %1\n").arg(1.0/d,0,'f',9)))
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

void Qc9528::sleep(bool b)
{
    if(b)
        pGenWriteCmd(QString(":PULSE0:STATE 0\n"));
    else
        pGenWriteCmd(QString(":PULSE0:STATE 1\n"));

    HardwareObject::sleep(b);
}

bool Qc9528::pGenWriteCmd(QString cmd)
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
