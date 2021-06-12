#include "intellisysiqplus.h"

IntellisysIQPlus::IntellisysIQPlus(QObject *parent) :
    PressureController(BC::Key::iqplus,BC::Key::iqplusName,CommunicationProtocol::Rs232,parent)
{
    d_readOnly = false;
}

void IntellisysIQPlus::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    double min = s.value(QString("min"),0.0).toDouble();
    double max = s.value(QString("max"),20.0).toDouble();
    int decimal = s.value(QString("decimal"),4).toInt();
    QString units = s.value(QString("units"),QString("Torr")).toString();

    s.setValue(QString("min"),min);
    s.setValue(QString("max"),max);
    s.setValue(QString("decimal"),decimal);
    s.setValue(QString("units"),units);

    d_fullScale = s.value(QString("fullScale"),10.0).toDouble();
    s.setValue(QString("fullScale"),10.0);

    s.endGroup();
    s.endGroup();
}


bool IntellisysIQPlus::testConnection()
{
    p_readTimer->stop();

    QByteArray resp = p_comm->queryCmd(QString("R38\n"));
    if(resp.isEmpty())
    {
        d_errorString = QString("Pressure controller did not respond");
        return false;
    }
    if(!resp.startsWith("XIQ")) // should be "IQ+3" based on manual, real resp: "XIQ Rev 4.33 27-Feb-2013"
    {
        d_errorString = QString("Wrong pressure controller connected. Model received: %1").arg(QString(resp.trimmed()));
        return false;
    }

    resp = p_comm->queryCmd(QString("R26\n"));
    if(!resp.startsWith("T11"))
    {
        p_comm->writeCmd(QString("T11\n"));
        resp = p_comm->queryCmd(QString("R26\n"));
        if(!resp.startsWith("T11"))
        {
            d_errorString = QString("Could not set the pressure controller to pressure control mode").arg(QString(resp.trimmed()));
            return false;
        }
    }

    resp = p_comm->queryCmd(QString("RN1\n"));
    int f = resp.indexOf('N');
    int l = resp.size();
    double num = resp.mid(f+2,l-f-2).trimmed().toDouble();
    if(num != d_fullScale)
    {
        d_errorString = QString("Full scale (%1 Torr) does not match value in settings (%2 Torr)").arg(num,0,'f',2).arg(d_fullScale,0,'f',2);
        return false;
    }

    if(readPressureSetpoint() < 0.0)
        return false;

    //hold the valve at its current position to ensure pressure control is disabled
    setPressureControlMode(false);

    p_readTimer->start();
    return true;
}

void IntellisysIQPlus::pcInitialize()
{
    p_comm->setReadOptions(1000,true,QByteArray("\r\n"));

    p_readTimer = new QTimer(this);
    p_readTimer->setInterval(200);
    connect(p_readTimer,&QTimer::timeout,this,&IntellisysIQPlus::readPressure);
    connect(this,&IntellisysIQPlus::hardwareFailure,p_readTimer,&QTimer::stop);
}

double IntellisysIQPlus::readPressure()
{
    QByteArray resp = p_comm->queryCmd(QString("R5\n"));
    if((resp.isEmpty()) || (!resp.startsWith("P+")))
    {
        emit logMessage(QString("Could not read chamber pressure"),BlackChirp::LogError);
        emit hardwareFailure();
        return -1.0;
    }
    int f = resp.indexOf('+');
    int l = resp.size();
    double num = resp.mid(f+1,l-f-1).trimmed().toDouble();
    d_pressure = num * d_fullScale/100.0;
    emit pressureUpdate(d_pressure);
    return d_pressure;
}

double IntellisysIQPlus::setPressureSetpoint(const double val)
{
    d_setPoint = val;
    double num = val * 100.0 / d_fullScale;
    p_readTimer->stop();

    p_comm->writeCmd(QString("S1%1\n").arg(num,0,'f',2,'0'));
    double num_check = readPressureSetpoint();
    if(qAbs(num_check - num)>=0.01)
    {
        emit logMessage(QString("Failed to set chamber pressure set point"),BlackChirp::LogError);
        emit hardwareFailure();
        return -1.0;
    }

    p_readTimer->start();
    return num_check;
}

double IntellisysIQPlus::readPressureSetpoint()
{
    QByteArray resp = p_comm->queryCmd(QString("R1\n"));
    if((resp.isEmpty()) || (!resp.startsWith("S1+")))
    {
        emit logMessage(QString("Could not read chamber pressure set point"),BlackChirp::LogError);
        emit hardwareFailure();
        return -1.0;
    }
    int f = resp.indexOf('+');
    int l = resp.size();
    bool ok = false;
    double out = resp.mid(f+1,l-f-1).trimmed().toDouble(&ok);
    if(!ok)
    {
        emit logMessage(QString("Could not parse pressure setpoint response. Received: %1").arg(QString(resp),BlackChirp::LogError));
        emit hardwareFailure();
        return -1.0;
    }
    emit pressureSetpointUpdate(out/100.0*d_fullScale);
    return out;
}

void IntellisysIQPlus::setPressureControlMode(bool enabled)
{
    d_pressureControlMode = enabled;
    if (enabled)
    {
        p_comm->writeCmd(QString("D1\n"));
    }
    else
    {
        p_comm->writeCmd(QString("H\n"));
    }
    readPressureControlMode();
}

bool IntellisysIQPlus::readPressureControlMode()
{
    emit pressureControlMode(d_pressureControlMode);
    return d_pressureControlMode;
}

void IntellisysIQPlus::openGateValve()
{
    this->setPressureControlMode(false);
    p_comm->writeCmd(QString("O\n"));
}

void IntellisysIQPlus::closeGateValve()
{
    this->setPressureControlMode(false);
    p_comm->writeCmd(QString("C\n"));
}
