#include "virtualtempcontroller.h"

#include <QTimer>

VirtualTemperatureController::VirtualTemperatureController(QObject *parent) : TemperatureController(BC::Key::hwVirtual,BC::Key::vtcName,CommunicationProtocol::Virtual,parent)
{
    d_numChannels= 4;
}

void VirtualTemperatureController::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    double min = s.value(QString("min"),1.0).toDouble();
    double max = s.value(QString("max"),10.0).toDouble();
    int decimal = s.value(QString("decimal"),4).toInt();
    QString units = s.value(QString("units"),QString("Kelvin")).toString();

    s.setValue(QString("min"),min);
    s.setValue(QString("max"),max);
    s.setValue(QString("decimal"),decimal);
    s.setValue(QString("units"),units);

    s.endGroup();
    s.endGroup();
}

VirtualTemperatureController::~VirtualTemperatureController()
{
}

bool VirtualTemperatureController::testConnection()
{
    p_readTimer->start();
    emit logMessage(QString("I'm the temperature controller!"));
    return true;
}
void VirtualTemperatureController::tcInitialize()
{
    p_readTimer = new QTimer(this);
    p_readTimer->setInterval(200);
    connect(p_readTimer,&QTimer::timeout,this,&VirtualTemperatureController::readTemperatures);
}


QList<double> VirtualTemperatureController::readHWTemperature()
{
    //not entirely sure what numbers to use here:
    QList<double> out;
    for (int i=0; i<d_numChannels;i++)
        out.append(static_cast<double>((qrand() % 65536) - 32768) / 32768.0 + 5.0);
    return out;
}
