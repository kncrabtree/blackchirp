#include "temperaturecontroller.h"

TemperatureController::TemperatureController(QObject *parent) : HardwareObject(parent)
{
    d_key= QString("tempController");

}

TemperatureController::~TemperatureController()
{
}

double TemperatureController::readTemperature()
{
   double T = readHWTemperature();
   emit temperatureUpdate(T, QPrivateSignal());
   return T;
}

QList<QPair<QString, QVariant> > TemperatureController::readAuxPlotData()
{
    QList<QPair<QString,QVariant>> out;
    out.append(qMakePair(QString("temperature"),d_temperature));
    return out;
}

void TemperatureController::initialize()
{
    tcInitialize();
}
