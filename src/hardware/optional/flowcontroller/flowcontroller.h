#ifndef FLOWCONTROLLER_H
#define FLOWCONTROLLER_H

#include <hardware/core/hardwareobject.h>
#include <data/settings/hardwarekeys.h>
#include <data/experiment/auxdatakeys.h>

#include <QTimer>

#include <data/experiment/hardware/optional/flowcontroller/flowconfig.h>

class FlowController : public HardwareObject
{
    Q_OBJECT
public:
    FlowController(const QString& impl, const QString& label, QObject *parent = nullptr);
    virtual ~FlowController();

    QStringList validationKeys() const override;
    FlowConfig config() { readAll(); return d_config; }

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

    void poll();
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
    int d_numChannels;
    int d_nextRead{0};
    QTimer *p_readTimer;

protected:
    void initialize() override final;
    bool testConnection() override final;
    bool prepareForExperiment(Experiment &e) override final;
    void readSettings() override;
    virtual void fcInitialize() =0;
    virtual bool fcTestConnection() =0;

    void readAll();

    // HardwareObject interface
protected:
    virtual AuxDataStorage::AuxDataMap readAuxData() override;

private:
    friend class VirtualFlowController;


};

#endif // FLOWCONTROLLER_H
