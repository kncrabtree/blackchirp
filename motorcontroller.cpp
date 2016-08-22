#include "motorcontroller.h"

#include <QTimer>

MotorController::MotorController(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("motorController");

    p_limitTimer = new QTimer(this);
    connect(p_limitTimer,&QTimer::timeout,this,&MotorController::checkLimit);
}

Experiment MotorController::prepareForExperiment(Experiment exp)
{
    moveToRestingPos();

    return exp;
}
