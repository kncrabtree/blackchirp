#ifndef PRESSURECONTROLLER_H
#define PRESSURECONTROLLER_H

#include <hardware/core/hardwareobject.h>
#include <data/experiment/hardware/optional/pressurecontroller/pressurecontrollerconfig.h>


namespace BC::Aux::PController {
inline constexpr QLatin1StringView pressure{"ChamberPressure"};
}

class PressureController : public HardwareObject
{
    Q_OBJECT
public:
    PressureController(const QString& impl, const QString& label, bool ro, QObject *parent = nullptr);
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
    PressureControllerConfig getConfig() const;
    void storeOptHwConfig(Experiment *exp) override { exp->addOptHwConfig(getConfig()); }
    void prepareForShutdown() override;

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
    bool prepareForExperiment(Experiment &e) override;
    virtual AuxDataStorage::AuxDataMap readAuxData() override;
    void initialize() override final;
    bool testConnection() override final;
    void hwReadSettings() override final;
    /*!
     * \brief Driver hook called after the base PressureController has
     * refreshed its poll interval. Default is a no-op.
     */
    virtual void pcReadSettings() {}

private:
    QTimer *p_readTimer;
    PressureControllerConfig d_config;
    friend class VirtualPressureController;


    // HardwareObject interface
public:
    QStringList validationKeys() const override;
};

#endif // PRESSURECONTROLLER_H
