#include "intellisysiqplus.h"

IntellisysIQPlus::IntellisysIQPlus(QObject *parent) : PressureController(parent)
{
    d_subKey = QString("IntellisysIQPlus");
    d_prettyName = QString("Intellisys IQ Plus Pressure Controller");
    d_commType = CommunicationProtocol::Rs232;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    double min = s.value(QString("min"),-1.0).toDouble();
    double max = s.value(QString("max"),20.0).toDouble();
    int decimal = s.value(QString("decimal"),4).toInt();
    QString units = s.value(QString("units"),QString("Torr")).toString();

    s.setValue(QString("min"),min);
    s.setValue(QString("max"),max);
    s.setValue(QString("decimal"),decimal);
    s.setValue(QString("units"),units);

    fullScale = s.value(QString("fullScale"),10.0).toDouble();
    s.setValue(QString("fullScale"),10.0);

    s.endGroup();
    s.endGroup();

    d_readOnly = true;
}


bool IntellisysIQPlus::testConnection()
{
    p_readTimer->stop();

    if(!p_comm->testConnection())
    {
        emit connected(false,QString("RS232 error"));
        return false;
    }

    emit connected();

    p_readTimer->start();
    return true;
}

void IntellisysIQPlus::initialize()
{
    p_comm->initialize();
    p_comm->setReadOptions(1000,true,QByteArray("\r\n"));

    p_readTimer = new QTimer(this);
    p_readTimer->setInterval(200);
    connect(p_readTimer,&QTimer::timeout,this,&IntellisysIQPlus::readPressure);
    connect(this,&IntellisysIQPlus::hardwareFailure,p_readTimer,&QTimer::stop);
    testConnection();
}

Experiment IntellisysIQPlus::prepareForExperiment(Experiment exp)
{
    return exp;
}

void IntellisysIQPlus::beginAcquisition()
{
}

void IntellisysIQPlus::endAcquisition()
{
}

double IntellisysIQPlus::readPressure()
{
    QByteArray resp = p_comm->queryCmd(QString("R5\n"));
    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read chamber pressure"),BlackChirp::LogError);
        return false;
    }
    int f = resp.indexOf('+');
    int l = resp.size();
    double num = resp.mid(f+1,l-f-1).trimmed().toDouble();
    d_pressure = num * fullScale/100.0;
    emit pressureUpdate(d_pressure);
    return d_pressure;
}

double IntellisysIQPlus::setPressureSetpoint(const double val)
{
    d_setPoint = val;
    return readPressureSetpoint();
}

double IntellisysIQPlus::readPressureSetpoint()
{
    emit pressureSetpointUpdate(d_setPoint);
    return d_setPoint;
}

void IntellisysIQPlus::setPressureControlMode(bool enabled)
{
    d_pressureControlMode = enabled;
    readPressureControlMode();
}

bool IntellisysIQPlus::readPressureControlMode()
{
    emit pressureControlMode(d_pressureControlMode);
    return d_pressureControlMode;
}
