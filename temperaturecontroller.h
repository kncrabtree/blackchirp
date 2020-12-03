#ifndef TEMPERATURECONTROLLER_H
#define TEMPERATURECONTROLLER_H
#include "hardwareobject.h"


class TemperatureController : public HardwareObject
{
    Q_OBJECT
public:
    explicit TemperatureController(QObject *parent =nullptr);
    virtual ~TemperatureController();

signals:
    void temperatureUpdate(double, QPrivateSignal);

signals:
    void temperatureUpdate(double);
    void temperatureSetpointUpdate(double);
    void temperatureControlMode(bool);

public slots:
    virtual double readTemperature();




protected:
    double d_temperature;

    virtual void tcInitialize() =0;


    // HardwareObject interface
    virtual QList<QPair<QString, QVariant> > readAuxPlotData() override;
    void initialize() override final;
protected:
    virtual double readHWTemperature() = 0;
};

#endif // TEMPERATURECONTROLLER_H


#if BC_TEMPCONTROLLER == 1
#include "lakeshore218.h"
class Lakeshore218;
typedef Lakeshore218 TemperatureControllerHardware;
#else
#include "virtualtempcontroller.h"
class VirtualTemperatureController;
typedef VirtualTemperatureController TemperatureControllerHardware;
#endif



