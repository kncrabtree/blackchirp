#include "motorcontroller.h"

MotorController::MotorController(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("motorController");
}

