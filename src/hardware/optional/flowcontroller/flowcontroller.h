#ifndef FLOWCONTROLLER_H
#define FLOWCONTROLLER_H

#include <hardware/core/hardwareobject.h>

#include <QTimer>

#include <hardware/optional/flowcontroller/flowconfig.h>

namespace BC::Key::Flow {
static const QString flowController("FlowController");
static const QString flowChannels("numChannels");
static const QString interval("intervalMs");
static const QString pUnits("pressureUnits");
static const QString pDec("pressureDecimals");
static const QString pMax("pressureMax");
static const QString channels("channels");
static const QString chUnits("units");
static const QString chDecimals("decimals");
static const QString chMax("max");
}

namespace BC::Aux::Flow {
static const QString pressure{"pressure"};
static const QString flow{"flow.%1"};
}

class FlowController : public HardwareObject
{
    Q_OBJECT
public:
    FlowController(const QString subKey, const QString name, CommunicationProtocol::CommType commType,
                   QObject *parent = nullptr, bool threaded = false, bool critical = false);
    virtual ~FlowController();

    FlowConfig config() const { return d_config; }

signals:
    void flowUpdate(int,double,QPrivateSignal);
    void pressureUpdate(double,QPrivateSignal);
    void flowSetpointUpdate(int,double,QPrivateSignal);
    void pressureSetpointUpdate(double,QPrivateSignal);
    void pressureControlMode(bool,QPrivateSignal);

public slots:
    void setAll(const FlowConfig &c);
    void setChannelName(const int ch, const QString name);
    void setPressureControlMode(bool enabled);
    void setFlowSetpoint(const int ch, const double val);
    void setPressureSetpoint(const double val);
    void readFlowSetpoint(const int ch);
    void readPressureSetpoint();
    void readFlow(const int ch);
    void readPressure();
    void readPressureControlMode();

    virtual void poll();

private:
    virtual void hwSetPressureControlMode(bool enabled) =0;
    virtual void hwSetFlowSetpoint(const int ch, const double val) =0;
    virtual void hwSetPressureSetpoint(const double val) =0;
    virtual double hwReadFlowSetpoint(const int ch) =0;
    virtual double hwReadPressureSetpoint() =0;
    virtual double hwReadFlow(const int ch) =0;
    virtual double hwReadPressure() =0;
    virtual int hwReadPressureControlMode() =0;

    FlowConfig d_config;
    QTimer *p_readTimer;
    const int d_numChannels;

protected:
    void initialize() override final;
    bool testConnection() override final;
    bool prepareForExperiment(Experiment &e) override final;
    virtual void fcInitialize() =0;
    virtual bool fcTestConnection() =0;

    void readAll();

    // HardwareObject interface
protected:
    virtual AuxDataStorage::AuxDataMap readAuxData() override;

#if BC_FLOWCONTROLLER == 0
    friend class VirtualFlowController;
#endif
};

#ifdef BC_FLOWCONTROLLER
#if BC_FLOWCONTROLLER == 1
#include "mks647c.h"
class Mks647c;
typedef Mks647c FlowControllerHardware;
#elif BC_FLOWCONTROLLER == 2
#include "mks946.h"
class Mks946;
typedef Mks946 FlowControllerHardware;
#else
#include "virtualflowcontroller.h"
class VirtualFlowController;
typedef VirtualFlowController FlowControllerHardware;
#endif
#endif

#endif // FLOWCONTROLLER_H
