#include "temperaturecontroller.h"

TemperatureController::TemperatureController(QObject *parent) : HardwareObject(parent)
{
    d_key= QString("tempController");

}

TemperatureController::~TemperatureController()
{
}

QList<double> TemperatureController::readTemperatures()
{
   d_temperatureList = readHWTemperature();
   emit temperatureUpdate(d_temperatureList, QPrivateSignal());
   return d_temperatureList;
}

QList<QPair<QString, QVariant> > TemperatureController::readAuxPlotData()
{
    QList<QPair<QString,QVariant>> out;
    for (int i=0;i<d_temperatureList.size();i++)
        out.append(qMakePair(QString("temperature.%1").arg(i),d_temperatureList.at(i)));
    return out;
}

void TemperatureController::initialize()
{
    d_temperatureList.clear();
    for (int i=0; i<d_numChannels;i++)
        d_temperatureList.append(0.0);
    tcInitialize();
}
