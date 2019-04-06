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
    void beginAcquisition();
    void endAcquisition();

    // MotorController interface
    bool moveToPosition(double x, double y, double z);
    void moveToRestingPos();
    void checkLimit();

protected:
    bool testConnection();
    void initialize();
};

#endif // VIRTUALMOTORCONTROLLER_H
