#ifndef SCX11_H
#define SCX11_H

#include <QSerialPort>
#include <QTimer>

#include <src/modules/motor/hardware/motorcontroller/motorcontroller.h>

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
        BlackChirp::MotorAxis axis;
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
    bool prepareForMotorScan(const MotorScan ms) override;
    bool testConnection() override;
    void initialize() override;

private:
    bool d_idle;
    int d_nextRead;
    void checkLimitOneAxis(BlackChirp::MotorAxis axis);
    QList<AxisInfo> d_channels;
    BlackChirp::MotorAxis axisIndex(int id);
    AxisInfo axisInfo(BlackChirp::MotorAxis axis);
    QString axisName(BlackChirp::MotorAxis axis);

    QTimer *p_motionTimer;
    AxisInfo d_currentMotionChannel;
    bool moving();



};

#endif // SCX11_H
