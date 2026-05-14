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
    void channelEnableUpdate(int,bool,QPrivateSignal);

public slots:
    void storeOptHwConfig(Experiment *exp) override { exp->addOptHwConfig(config()); }
    void prepareForShutdown() override;
    void setAll(const FlowConfig &c);
    void setChannelName(const int ch, const QString name);
    void setChannelEnabled(const int ch, const bool en);
    void setPressureControlMode(bool enabled);
    void setFlowSetpoint(const int ch, const double val);
    void setPressureSetpoint(const double val);
    void readFlowSetpoint(const int ch);
    void readPressureSetpoint();
    void readFlow(const int ch);
    void readPressure();
    void readPressureControlMode();

    void poll();

private:
    virtual void hwSetPressureControlMode(bool enabled) =0;
    virtual void hwSetFlowSetpoint(const int ch, const double val) =0;
    virtual void hwSetPressureSetpoint(const double val) =0;
    virtual double hwReadFlowSetpoint(const int ch) =0;
    virtual double hwReadPressureSetpoint() =0;
    virtual double hwReadFlow(const int ch) =0;
    virtual double hwReadPressure() =0;
    virtual int hwReadPressureControlMode() =0;
    /*!
     * \brief Optional driver hook for hardware that exposes a separate
     * channel enable/disable command (independent of setpoint). Default
     * is a no-op; controllers that gate flow purely by setpoint do not
     * need to override this. The base FlowController re-issues the
     * channel's setpoint after toggling enable so such drivers can
     * close the valve via their hwSetFlowSetpoint override (which
     * may consult isChannelEnabled()).
     */
    virtual void hwSetChannelEnabled(const int ch, const bool en) { Q_UNUSED(ch); Q_UNUSED(en); }

    FlowConfig d_config;
    int d_numChannels;
    int d_nextRead{0};
    QTimer *p_readTimer;

protected:
    void initialize() override final;
    bool testConnection() override final;
    bool prepareForExperiment(Experiment &e) override final;
    void hwReadSettings() override final;
    virtual void fcInitialize() =0;
    virtual bool fcTestConnection() =0;

    /*!
     * \brief Returns the current enabled state of a channel from the
     * controller's cached configuration. Drivers without a hardware
     * enable command can use this in their hwSetFlowSetpoint() override
     * to force the setpoint to zero when the channel is disabled.
     */
    bool isChannelEnabled(const int ch) const;
    /*!
     * \brief Driver hook called after the base FlowController has refreshed
     * its poll interval and rebuilt its channel configuration. Default
     * is a no-op.
     */
    virtual void fcReadSettings() {}

    void readAll();

    // HardwareObject interface
protected:
    virtual AuxDataStorage::AuxDataMap readAuxData() override;

private:
    friend class VirtualFlowController;


};

#endif // FLOWCONTROLLER_H
