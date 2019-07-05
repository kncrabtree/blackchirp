#include "mks946.h"

Mks946::Mks946(QObject *parent) : FlowController(parent), d_nextRead(0)
{
    d_subKey = QString("mks946");
    d_prettyName = QString("MKS 946 Flow Controller");
    d_threaded = false;
    d_commType = CommunicationProtocol::Rs232;
    d_isCritical = false;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    d_numChannels = s.value(QString("numChannels"),4).toInt();
    d_address = s.value(QString("address"),253).toInt();
    d_pressureChannel = s.value(QString("pressureChannel"),5).toInt();
    s.setValue(QString("numChannels"),d_numChannels);
    s.setValue(QString("address"),d_address);
    s.setValue(QString("pressureChannel"),d_pressureChannel);
    s.endGroup();
    s.endGroup();

    s.sync();

}


bool Mks946::fcTestConnection()
{
    QByteArray resp = mksQuery(QString("MD?"));
    if(!resp.contains(QByteArray("946")))
    {
        d_errorString = QString("Received invalid response to model query. Response: %1").arg(QString(resp));
        return false;
    }

    emit logMessage(QString("Response: %1").arg(QString(resp)));
    return true;
}

void Mks946::hwSetFlowSetpoint(const int ch, const double val)
{
    if(!isConnected())
    {
        emit logMessage(QString("Cannot set flow setpoint due to a previous communication failure. Reconnect and try again."),BlackChirp::LogError);
        return;
    }

    //first ensure recipe 1 is active
    if(!mksWrite(QString("RCP1!")))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return;
    }

    //make sure ratio recipe 1 is active
    if(!mksWrite(QString("RRCP1!")))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return;
    }

    if(!mksWrite(QString("RRQ%1:%2").arg(ch).arg(val,0,'E',2,QLatin1Char(' '))))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return;
    }
}

void Mks946::hwSetPressureSetpoint(const double val)
{
    if(!isConnected())
    {
        emit logMessage(QString("Cannot set pressure setpoint due to a previous communication failure. Reconnect and try again."),BlackChirp::LogError);
        return;
    }

    //first ensure recipe 1 is active
    if(!mksWrite(QString("RCP1!")))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return;
    }

    if(!mksWrite(QString("RPSP!:%1").arg(val,1,'E',2,QLatin1Char(' '))))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return;
    }
}

double Mks946::hwReadFlowSetpoint(const int ch)
{
    if(!isConnected())
        return 0.0;

    //first ensure recipe 1 is active
    if(!mksWrite(QString("RCP1!")))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return -1.0;
    }

    //make sure ratio recipe 1 is active
    if(!mksWrite(QString("RRCP1!")))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return -1.0;
    }

    QByteArray resp = mksQuery(QString("RRQ%1?").arg(ch));
    bool ok = false;
    double out = resp.toDouble(&ok);
    if(!ok)
    {
        emit logMessage(QString("Received invalid response to channel %1 setpoint query. Response: %2").arg(ch).arg(QString(resp)),BlackChirp::LogError);
        emit hardwareFailure();
        return -1.0;
    }

    return out;
}

double Mks946::hwReadPressureSetpoint()
{
    if(!isConnected())
        return 0.0;

    //first ensure recipe 1 is active
    if(!mksWrite(QString("RCP1!")))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return -1.0;
    }

    QByteArray resp = mksQuery(QString("RPSP?"));
    bool ok = false;
    double out = resp.toDouble(&ok);
    if(!ok)
    {
        emit logMessage(QString("Received invalid response to pressure setpoint query. Response: %1").arg(QString(resp)),BlackChirp::LogError);
        emit hardwareFailure();
        return -1.0;
    }

    return out;
}

double Mks946::hwReadFlow(const int ch)
{
    if(!isConnected())
        return 0.0;

    QByteArray resp = mksQuery(QString("FR%1").arg(ch));
    bool ok = false;
    double out = resp.toDouble(&ok);
    if(!ok)
    {
        emit logMessage(QString("Received invalid response to flow query for channel %1. Response: %2").arg(ch).arg(QString(resp)),BlackChirp::LogError);
        emit hardwareFailure();
        return -1.0;
    }

    return out;
}

double Mks946::hwReadPressure()
{
    if(!isConnected())
        return 0.0;

    QByteArray resp = mksQuery(QString("PR%1?").arg(d_pressureChannel));
    if(resp.contains(QByteArray("LO")))
        return 0.0;

    if(resp.contains(QByteArray("MISCONN")))
    {
        emit logMessage(QString("No pressure gauge connected."),BlackChirp::LogWarning);
        setPressureControlMode(false);
        emit hardwareFailure();
        return -0.0;
    }

    if(resp.contains(QByteArray("ATM")))
        return 10.0;

    bool ok = false;
    double out = resp.toDouble(&ok);
    if(ok)
        return out/1000.0; //convert to kTorr

    emit logMessage(QString("Could not parse reply to pressure query. Response: %1").arg(QString(resp)),BlackChirp::LogError);
    emit hardwareFailure();
    return -1.0;
}

void Mks946::hwSetPressureControlMode(bool enabled)
{
    if(!isConnected())
    {
        emit logMessage(QString("Cannot set pressure control mode due to a previous communication failure. Reconnect and try again."),BlackChirp::LogError);
        return;
    }

    //first ensure recipe 1 is active
    if(!mksWrite(QString("RCP1!")))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return;
    }

    //ensure Ratio mode is selected
    if(!mksWrite(QString("RPCH!:Rat")))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return;
    }

    QString s = QString("PID!:ON");
    if(!enabled)
        s = QString("PID!:OFF");

    if(!mksWrite(s))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return;
    }
}

int Mks946::hwReadPressureControlMode()
{
    if(!isConnected())
    {
        return -1;
    }

    //first ensure recipe 1 is active
    if(!mksWrite(QString("RCP1!")))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return -1;
    }

    QByteArray resp = mksQuery(QString("PID?"));
    if(resp.contains(QByteArray("ON")))
        return 1;
    else if(resp.contains(QByteArray("OFF")))
        return 0;
    else
        emit logMessage(QString("Received invalid response to pressure control mode query. Response: %1").arg(QString(resp)),BlackChirp::LogError);

    return -1;

}

void Mks946::poll()
{
    if(d_nextRead < 0 || d_nextRead >= d_numChannels)
    {
        readPressure();
        d_nextRead = 0;
    }
    else
    {
        readFlow(d_nextRead);
        d_nextRead++;
    }
}

void Mks946::fcInitialize()
{
    p_comm->setReadOptions(100,true,QByteArray(";FF"));
}

bool Mks946::mksWrite(QString cmd)
{
    QByteArray resp = p_comm->queryCmd(QString("@%1%2;FF").arg(d_address,3,10,QChar('0')).arg(cmd));
    if(resp.contains(QByteArray("ACK")))
        return true;

    d_errorString = QString("Received invalid response to command %1. Response: %2").arg(cmd).arg(QString(resp));
    return false;
}

QByteArray Mks946::mksQuery(QString cmd)
{
    QByteArray resp = p_comm->queryCmd(QString("@%1%2;FF").arg(d_address,3,10,QChar('0')).arg(cmd));

    if(!resp.startsWith(QString("@%1ACK").arg(d_address,3,10,QChar('0')).toLatin1()))
        return resp;

    //chop off prefix
    return resp.mid(7);
}

void Mks946::readSettings()
{
}

void Mks946::sleep(bool b)
{
    Q_UNUSED(b)
}
