#ifndef VIRTUALMOTORCONTROLLER_H
#define VIRTUALMOTORCONTROLLER_H

#include "motorcontroller.h"

class VirtualMotorController : public MotorController
{
    Q_OBJECT
public:
    VirtualMotorController(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void readSettings();
    bool testConnection();
    void initialize();
    void beginAcquisition();
    void endAcquisition();

    // MotorController interface
public slots:
    bool moveToPosition(double x, double y, double z);
    void moveToRestingPos();
    void checkLimit();
};

#endif // VIRTUALMOTORCONTROLLER_H
