#ifndef PRESSURECONTROLLER_H
#define PRESSURECONTROLLER_H

#include <hardware/core/hardwareobject.h>
#include <hardware/optional/pressurecontroller/pressurecontrollerconfig.h>

namespace BC::Key::PController {
static const QString key{"PressureController"};
static const QString min{"min"};
static const QString max{"max"};
static const QString decimals{"decimal"};
static const QString units{"units"};
static const QString readOnly{"readOnly"};
static const QString readInterval{"intervalMs"};
static const QString hasValve{"hasValve"};
}

namespace BC::Aux::PController {
static const QString pressure{"ChamberPressure"};
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
    PressureControllerConfig getConfig() const;

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

private:
    QTimer *p_readTimer;
    PressureControllerConfig d_config;
    friend class VirtualPressureController;


    // HardwareObject interface
public:
    QStringList validationKeys() const override;
};

#ifdef BC_PCONTROLLER
#include BC_STR(BC_PCONTROLLER_H)
#endif

#endif // PRESSURECONTROLLER_H
