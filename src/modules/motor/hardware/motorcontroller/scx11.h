#ifndef SCX11_H
#define SCX11_H

#include <QSerialPort>
#include <QTimer>

#include <modules/motor/hardware/motorcontroller/motorcontroller.h>

namespace BC::Key::MC {
static const QString scx11("scx11");
static const QString scx11Name("Motor Controller SCX11");
}

class Scx11 : public MotorController
{
    Q_OBJECT
public:
    Scx11(QObject *parent=nullptr);


    // MotorController interface
protected:
    bool prepareForMotorScan(Experiment &exp) override;
    bool mcTestConnection() override;
    void mcInitialize() override;
    bool hwMoveToPosition(double x, double y, double z) override;
    Limits hwCheckLimits(MotorScan::MotorAxis axis) override;
    double hwReadPosition(MotorScan::MotorAxis axis) override;
    bool hwCheckAxisMotion(MotorScan::MotorAxis axis) override;
    bool hwStopMotion(MotorScan::MotorAxis axis) override;
};

#endif // SCX11_H
