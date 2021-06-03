#include "motorcontroller.h"

#include <QTimer>

MotorController::MotorController(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("motorController");

    p_limitTimer = new QTimer(this);
    p_limitTimer->setInterval(200);
    connect(p_limitTimer,&QTimer::timeout,this,&MotorController::checkLimit);
}

Experiment MotorController::prepareForExperiment(Experiment exp)
{

    d_enabledForExperiment = exp.motorScan().isEnabled();
    if(d_enabledForExperiment)
        prepareForMotorScan(exp.motorScan());
    return exp;
}
