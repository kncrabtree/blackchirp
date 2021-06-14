#ifndef PRESSURECONTROLLER_H
#define PRESSURECONTROLLER_H

#include <src/hardware/core/hardwareobject.h>

namespace BC::Key {
static const QString pController("pressureController");
static const QString pControllerMin("min");
static const QString pControllerMax("max");
static const QString pControllerDecimals("decimal");
static const QString pControllerUnits("units");
static const QString pControllerReadOnly("readOnly");
}

class PressureController : public HardwareObject
{
    Q_OBJECT
public:
    PressureController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, bool readOnly, QObject *parent =nullptr, bool threaded = false, bool critical=false);
    virtual ~PressureController();

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

    virtual void openGateValve() =0;
    virtual void closeGateValve() =0;

protected:
    const bool d_readOnly;
    double d_pressure;
    double d_setPoint;
    bool d_pressureControlMode;

    virtual void pcInitialize() =0;


    // HardwareObject interface
    virtual QList<QPair<QString, QVariant> > readAuxPlotData() override;
    void initialize() override final;
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
