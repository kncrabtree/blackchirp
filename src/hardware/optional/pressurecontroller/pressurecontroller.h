#ifndef PRESSURECONTROLLER_H
#define PRESSURECONTROLLER_H

#include <hardware/core/hardwareobject.h>

namespace BC::Key::PController {
static const QString key("PressureController");
static const QString min("min");
static const QString max("max");
static const QString decimals("decimal");
static const QString units("units");
static const QString readOnly("readOnly");
static const QString readInterval("intervalMs");
}

class PressureController : public HardwareObject
{
    Q_OBJECT
public:
    PressureController(const QString subKey, const QString name, CommunicationProtocol::CommType commType,
                       bool ro, QObject *parent =nullptr, bool threaded = false, bool critical=false);
    virtual ~PressureController();

signals:
    void pressureUpdate(double,QPrivateSignal);
    void pressureSetpointUpdate(double,QPrivateSignal);
    void pressureControlMode(bool,QPrivateSignal);

public slots:
    double readPressure();
    void setPressureSetpoint(const double val);
    void readPressureSetpoint();
    void setPressureControlMode(bool enabled);
    void readPressureControlMode();
    void openGateValve();
    void closeGateValve();

protected:
    const bool d_readOnly;
    virtual double hwReadPressure() =0;
    virtual double hwSetPressureSetpoint(const double val) =0;

    virtual double hwReadPressureSetpoint() =0;

    virtual void hwSetPressureControlMode(bool enabled) =0;
    virtual int hwReadPressureControlMode() =0;

    virtual void hwOpenGateValve() =0;
    virtual void hwCloseGateValve() =0;

    virtual void pcInitialize() =0;
    virtual bool pcTestConnection() =0;

    // HardwareObject interface
    virtual QList<QPair<QString, QVariant> > readAuxPlotData() override;
    void initialize() override final;
    bool testConnection() override final;

private:
    QTimer *p_readTimer;
    double d_pressure;
    double d_setPoint;
    bool d_pressureControlMode;

#if BC_PCONTROLLER == 0
    friend class VirtualPressureController;
#endif

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
