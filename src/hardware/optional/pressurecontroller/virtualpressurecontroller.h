#ifndef VIRTUALPRESSURECONTROLLER_H
#define VIRTUALPRESSURECONTROLLER_H

#include <src/hardware/optional/pressurecontroller/pressurecontroller.h>

namespace BC::Key {
static const QString vpcName("Virtual Pressure Controller");
}

class VirtualPressureController : public PressureController
{
   Q_OBJECT
public:
    VirtualPressureController(QObject *parent =nullptr);
    ~VirtualPressureController();

    // PressureController interface
public slots:
    double hwReadPressure() override;
    double hwSetPressureSetpoint(const double val) override;
    double hwReadPressureSetpoint() override;
    void hwSetPressureControlMode(bool enabled) override;
    int hwReadPressureControlMode() override;
    void hwOpenGateValve() override;
    void hwCloseGateValve() override;

protected:
    bool pcTestConnection() override;
    void pcInitialize() override;
};

#endif // VIRTUALPRESSURECONTROLLER_H
