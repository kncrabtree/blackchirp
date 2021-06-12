#include "virtualpressurecontroller.h"

#include <QTimer>

VirtualPressureController::VirtualPressureController(QObject *parent) : PressureController(BC::Key::hwVirtual,BC::Key::vpcName,CommunicationProtocol::Virtual,parent)
{
    d_readOnly = false;
}

VirtualPressureController::~VirtualPressureController()
{
}

void VirtualPressureController::readSettings()
{
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

    s.endGroup();
    s.endGroup();
}


bool VirtualPressureController::testConnection()
{
    p_readTimer->start();
    return true;
}

void VirtualPressureController::pcInitialize()
{
    p_readTimer = new QTimer(this);
    p_readTimer->setInterval(200);
    connect(p_readTimer,&QTimer::timeout,this,&VirtualPressureController::readPressure);
}
double VirtualPressureController::readPressure()
{
    d_pressure = static_cast<double>((qrand() % 65536) - 32768) / 65536.0 * 0.05 + randPressure;
   // emit logMessage(QString("%1").arg(d_pressure,0,'f',3));
    emit pressureUpdate(d_pressure);
    return d_pressure;
}

double VirtualPressureController::setPressureSetpoint(const double val)
{
    d_setPoint = val;
    return readPressureSetpoint();
}

double VirtualPressureController::readPressureSetpoint()
{
    emit pressureSetpointUpdate(d_setPoint);
    return d_setPoint;
}

void VirtualPressureController::setPressureControlMode(bool enabled)
{
    d_pressureControlMode = enabled;
    if(enabled)
    {
        randPressure = d_setPoint;
    }
    readPressureControlMode();
}

bool VirtualPressureController::readPressureControlMode()
{
    emit pressureControlMode(d_pressureControlMode);
    return d_pressureControlMode;
}

void VirtualPressureController::openGateValve()
{
    this->setPressureControlMode(false);
    randPressure = 0.05;
}

void VirtualPressureController::closeGateValve()
{
    this->setPressureControlMode(false);
}


