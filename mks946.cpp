#include "mks946.h"

Mks946::Mks946(QObject *parent) : FlowController(parent)
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


bool Mks946::testConnection()
{
    p_readTimer->stop();

    QByteArray resp = mksQuery(QString("MD?"));
    if(!resp.contains(QByteArray("946")))
    {
        d_errorString = QString("Received invalid response to model query. Response: %1").arg(QString(resp));
        return false;
    }

    emit logMessage(QString("Response: %1").arg(QString(resp)));

    p_readTimer->start();

    return true;
}

double Mks946::setFlowSetpoint(const int ch, const double val)
{
    return 0.0;
}

double Mks946::setPressureSetpoint(const double val)
{
    return 0.0;
}

double Mks946::readFlowSetpoint(const int ch)
{
    if(!isConnected())
        return 0.0;

    //first ensure recipe 1 is active
    if(!mksWrite(QString("RCP1!")))
    {
        emit logMessage(d_errorString,BlackChirp::LogError);
        emit hardwareFailure();
        return 0.0;
    }

    QByteArray resp = mksQuery(QString("RPSP?1"));
    bool ok = false;
    double out = resp.toDouble(&ok);
    if(!ok)
    {
        emit logMessage(QString("Received invalid response to pressure setpoint query. Response: %1").arg(QString(resp)));
        emit hardwareFailure();
        return 0.0;
    }

    emit pressureSetpointUpdate(out);
    return out;
}

double Mks946::readPressureSetpoint()
{
    return 0.0;
}

double Mks946::readFlow(const int ch)
{
    return 0.0;
}

double Mks946::readPressure()
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
        return 0.0;
    }

    if(resp.contains(QByteArray("ATM")))
        return 10.0;

    bool ok = false;
    double out = resp.toDouble(&ok);
    if(ok)
        return out/1000.0; //convert to kTorr

    emit logMessage(QString("Could not parse reply to pressure query. Response: %1").arg(QString(resp)));
    return 0.0;
}

void Mks946::setPressureControlMode(bool enabled)
{
    if(!isConnected())
    {
        emit logMessage(QString("Cannot set pressure control mode due to a previous communication failure. Reconnect and try again."),BlackChirp::LogError);
        return;
    }
}

bool Mks946::readPressureControlMode()
{
    return false;
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
}
