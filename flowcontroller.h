#ifndef FLOWCONTROLLER_H
#define FLOWCONTROLLER_H

#include "hardwareobject.h"

#include "flowconfig.h"

class FlowController : public HardwareObject
{
    Q_OBJECT
public:
    FlowController(QObject *parent = nullptr);
    ~FlowController();

    FlowConfig config() const { return d_config; }

signals:
    void flowUpdate(int,double);
    void pressureUpdate(double);
    void flowSetpointUpdate(int,double);
    void pressureSetpointUpdate(double);
    void pressureControlMode(bool);

public slots:
    virtual double setFlowSetpoint(const int ch, const double val) =0;
    virtual double setPressureSetpoint(const double val) =0;
    virtual void setChannelName(const int ch, const QString name);

    virtual double readFlowSetpoint(const int ch) =0;
    virtual double readPressureSetpoint() =0;
    virtual double readFlow(const int ch) =0;
    virtual double readPressure() =0;

    virtual void setPressureControlMode(bool enabled) =0;
    virtual bool readPressureControlMode() =0;


protected:
    FlowConfig d_config;
    bool d_pressureControlMode;

    void readAll();
};

#if BC_FLOWCONTROLLER == 1
#include "mks647c.h"
class Mks647c;
typedef Mks647c FlowControllerHardware

#define BC_FLOW_NUMCHANNELS 4
#else
#include "virtualflowcontroller.h"
class VirtualFlowController;
typedef VirtualFlowController FlowControllerHardware;

#define BC_FLOW_NUMCHANNELS 4
#endif

#endif // FLOWCONTROLLER_H
