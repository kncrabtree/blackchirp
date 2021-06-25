#ifndef SCX11_H
#define SCX11_H

#include <QSerialPort>
#include <QTimer>

#include <modules/motor/hardware/motorcontroller/motorcontroller.h>

namespace BC::Key {
static const QString scx11("scx11");
static const QString scx11Name("Motor Controller SCX11");
}

class Scx11 : public MotorController
{
    Q_OBJECT
public:
    struct AxisInfo
    {
        int id;
        QString name;
        double min;
        double max;
        double rest;
        double offset;
        bool moving;
        double nextPos;
        MotorScan::MotorAxis axis;
    };

    Scx11(QObject *parent=nullptr);


    // HardwareObject interface
public slots:
    void readSettings() override;

    // MotorController interface
    bool moveToPosition(double x, double y, double z) override;
    void moveToRestingPos() override;
    void checkLimit() override;
    bool readCurrentPosition();
    void checkMotion();

protected:
    bool prepareForMotorScan(Experiment &exp) override;
    bool testConnection() override;
    void initialize() override;

private:
    bool d_idle;
    int d_nextRead;
    void checkLimitOneAxis(MotorScan::MotorAxis axis);
    QList<AxisInfo> d_channels;
    MotorScan::MotorAxis axisIndex(int id);
    AxisInfo axisInfo(MotorScan::MotorAxis axis);
    QString axisName(MotorScan::MotorAxis axis);

    QTimer *p_motionTimer;
    AxisInfo d_currentMotionChannel;
    bool moving();



};

#endif // SCX11_H
