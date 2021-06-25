#ifndef VIRTUALMOTORCONTROLLER_H
#define VIRTUALMOTORCONTROLLER_H

#include <modules/motor/hardware/motorcontroller/motorcontroller.h>

namespace BC::Key {
static const QString vmcName("Virtual Motor Controller");
}
class VirtualMotorController : public MotorController
{
    Q_OBJECT
public:
    VirtualMotorController(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void readSettings() override;

    // MotorController interface
    bool moveToPosition(double x, double y, double z) override;
    void moveToRestingPos() override;
    void checkLimit() override;

protected:
    bool prepareForMotorScan(Experiment &exp) override;
    bool testConnection() override;
    void initialize() override;
};

#endif // VIRTUALMOTORCONTROLLER_H
