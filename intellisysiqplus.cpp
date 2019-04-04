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

    d_readOnly = false;
}

void IntellisysIQPlus::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    d_minFreqMHz = s.value(QString("minFreqMHz"),0.0).toDouble();
    d_maxFreqMHz = s.value(QString("maxFreqMHz"),1e7).toDouble();
    s.setValue(QString("minFreqMHz"),d_minFreqMHz);
    s.setValue(QString("maxFreqMHz"),d_maxFreqMHz);
    s.endGroup();
    s.endGroup();
}


bool IntellisysIQPlus::testConnection()
{
    p_readTimer->stop();

    QByteArray resp = p_comm->queryCmd(QString("R38\n"));
    if(resp.isEmpty())
    {
        d_errorString = QString("Pressure controller did not response");
        return false;
    }
    if(!resp.startsWith("XIQ")) // should be "IQ+3" based on manual, real resp: "XIQ Rev 4.33 27-Feb-2013"
    {
        d_errorString = QString("Wrong pressure controller connected:%1").arg(QString(resp.trimmed()));
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
    if(num != fullScale)
    {
        d_errorString = QString("Full scale (%1 Torr) is not matched with setting value (%2 Torr)").arg(num,0,'f',2).arg(fullScale,0,'f',2);
        return false;
    }

    p_readTimer->start();
    return true;
}

void IntellisysIQPlus::initialize()
{
    p_comm->setReadOptions(1000,true,QByteArray("\r\n"));

    p_readTimer = new QTimer(this);
    p_readTimer->setInterval(200);
    connect(p_readTimer,&QTimer::timeout,this,&IntellisysIQPlus::readPressure);
    connect(this,&IntellisysIQPlus::hardwareFailure,p_readTimer,&QTimer::stop);
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
    if((resp.isEmpty()) || (!resp.startsWith("P+")))
    {
        d_errorString = QString("Could not read chamber pressure");
        emit hardwareFailure();
        return -1.0;
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
    double num = val * 100.0 / fullScale;
    p_readTimer->stop();

    p_comm->writeCmd(QString("S1%1\n").arg(num,0,'f',2,'0'));
    QByteArray resp = p_comm->queryCmd(QString("R1\n"));
    if((resp.isEmpty()) || (!resp.startsWith("S1+")))
    {
        d_errorString = QString("Could not read chamber pressure set point");
        emit hardwareFailure();
        return -1.0;
    }
    int f = resp.indexOf('+');
    int l = resp.size();
    double num_check = resp.mid(f+1,l-f-1).trimmed().toDouble();
    if(qAbs(num_check - num)>=0.01)
    {
        d_errorString = QString("Failed to set chamber pressure set point");
        emit hardwareFailure();
        return -1.0;
    }

    p_readTimer->start();
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
