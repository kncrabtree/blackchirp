#include "virtualpressurecontroller.h"

#include <QTimer>

VirtualPressureController::VirtualPressureController(QObject *parent) : PressureController(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Pressure Controller");
    d_commType = CommunicationProtocol::Virtual;

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

    d_readOnly = false;

}

VirtualPressureController::~VirtualPressureController()
{
}


bool VirtualPressureController::testConnection()
{
    emit connected();
    return true;
}

void VirtualPressureController::initialize()
{

    testConnection();
}

Experiment VirtualPressureController::prepareForExperiment(Experiment exp)
{
    return exp;
}

void VirtualPressureController::beginAcquisition()
{
}

void VirtualPressureController::endAcquisition()
{
}


double VirtualPressureController::readPressure()
{
    d_pressure = static_cast<double>((qrand() % 65536)) / 65536.0 * 10.0;
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
    readPressureControlMode();
}

bool VirtualPressureController::readPressureControlMode()
{
    emit pressureControlMode(d_pressureControlMode);
    return d_pressureControlMode;
}
