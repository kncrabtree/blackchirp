#ifndef VIRTUALMOTORCONTROLLER_H
#define VIRTUALMOTORCONTROLLER_H

#include <modules/motor/hardware/motorcontroller/motorcontroller.h>

namespace BC::Key::MC {
static const QString vmcName("Virtual Motor Controller");
}
class VirtualMotorController : public MotorController
{
    Q_OBJECT
public:
    VirtualMotorController(QObject *parent = nullptr);

protected:
    void mcInitialize() override;
    bool mcTestConnection() override;
    bool prepareForMotorScan(Experiment &exp) override;
    bool hwMoveToPosition(double x, double y, double z) override;
    Limits hwCheckLimits(MotorScan::MotorAxis axis) override;
    double hwReadPosition(MotorScan::MotorAxis axis) override;
    bool hwCheckAxisMotion(MotorScan::MotorAxis axis) override;
    bool hwStopMotion(MotorScan::MotorAxis axis) override;

private:
    QMap<MotorScan::MotorAxis,double> d_pos;
};

#endif // VIRTUALMOTORCONTROLLER_H
