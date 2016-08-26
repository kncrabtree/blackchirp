#ifndef SCX11_H
#define SCX11_H

#include <QSerialPort>
#include <QTimer>

#include "motorcontroller.h"

class Scx11 : public MotorController
{
    Q_OBJECT
public:
    Scx11(QObject *parent=nullptr);


    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    // MotorController interface
public slots:
    void moveToPosition(double x, double y, double z);
    bool prepareForMotorScan(const MotorScan ms);
    void moveToRestingPos();
    void checkLimit();

private:
    int d_xId;
    int d_yId;
    int d_zId;
    bool moveAxis(BlackChirp::MotorAxis axis, double pos);




};

#endif // SCX11_H
