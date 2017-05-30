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
    prepareForMotorScan(exp.motorScan());
    return exp;
}

bool MotorController::prepareForMotorScan(const MotorScan ms)
{
    Q_UNUSED(ms)
    return true;
}


void MotorController::endAcquisition()
{
}
