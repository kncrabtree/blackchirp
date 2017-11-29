#ifndef PRESSURECONTROLLER_H
#define PRESSURECONTROLLER_H

#include "hardwareobject.h"

class PressureController : public HardwareObject
{
    Q_OBJECT
public:
    PressureController(QObject *parent =nullptr);
    virtual ~PressureController();

    bool isReadOnly() const { return d_readOnly; }

signals:
    void pressureUpdate(double);
    void pressureSetpointUpdate(double);
    void pressureControlMode(bool);

public slots:
    virtual double readPressure() =0;
    virtual double setPressureSetpoint(const double val) =0;

    virtual double readPressureSetpoint() =0;

    virtual void setPressureControlMode(bool enabled) =0;
    virtual bool readPressureControlMode() =0;

protected:
    bool d_readOnly;
    double d_pressure;
    double d_setPoint;
    bool d_pressureControlMode;

    // HardwareObject interface
public slots:
    virtual void readTimeData();
};

#if BC_PCONTROLLER == 1
#include "intellisysiqplus.h"
class IntellisysIQPlus;
typedef IntellisysIQPlus PressureControllerHardware;
#else
#include "virtualpressurecontroller.h"
class VirtualPressureController;
typedef VirtualPressureController PressureControllerHardware;
#endif

#endif // PRESSURECONTROLLER_H
