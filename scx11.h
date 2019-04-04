#ifndef SCX11_H
#define SCX11_H

#include <QSerialPort>
#include <QTimer>

#include "motorcontroller.h"

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
    void readSettings();
    bool testConnection();
    void initialize();
    void beginAcquisition();
    void readTimeData();

    // MotorController interface
public slots:
    bool moveToPosition(double x, double y, double z);
    void moveToRestingPos();
    void checkLimit();
    bool readCurrentPosition();
    void checkMotion();

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
