#ifndef MOTORCONTROLLER_H
#define MOTORCONTROLLER_H

#include "hardwareobject.h"

class MotorController : public HardwareObject
{
public:
    MotorController(QObject *parent = nullptr);

signals:
    void motionComplete();

public slots:
    virtual void moveToNextPosition() =0;
};

#endif // MOTORCONTROLLER_H
