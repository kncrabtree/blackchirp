#ifndef FLOWCONTROLLER_H
#define FLOWCONTROLLER_H

#include <hardware/core/hardwareobject.h>

#include <QTimer>

#include <hardware/optional/flowcontroller/flowconfig.h>

namespace BC::Key::Flow {
static const QString flowController{"FlowController"};
static const QString flowChannels{"numChannels"};
static const QString interval{"intervalMs"};
static const QString pUnits{"pressureUnits"};
static const QString pDec{"pressureDecimals"};
static const QString pMax{"pressureMax"};
static const QString channels{"channels"};
static const QString chUnits{"units"};
static const QString chDecimals{"decimals"};
static const QString chMax{"max"};
}

namespace BC::Aux::Flow {
static const QString pressure{"Pressure"};
static const QString flow{"Flow%1"};
}

class FlowController : public HardwareObject
{
    Q_OBJECT
public:
    FlowController(const QString subKey, const QString name, CommunicationProtocol::CommType commType,
                   QObject *parent = nullptr, bool threaded = false, bool critical = false);
    virtual ~FlowController();

    FlowConfig config() const { return d_config; }
    QStringList validationKeys() const override;

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
    QStringList forbiddenKeys() const override;

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

    friend class VirtualFlowController;


};

#ifdef BC_FLOWCONTROLLER
#include BC_STR(BC_FLOWCONTROLLER_H)
#endif

#endif // FLOWCONTROLLER_H
