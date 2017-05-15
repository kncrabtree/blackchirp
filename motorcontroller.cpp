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
    if(exp.motorScan().isEnabled())
    {
        if(!prepareForMotorScan(exp.motorScan()))
        {
            exp.setHardwareFailed();
            exp.setErrorString(QString("Failed to prepare %1 for scan").arg(name()));
        }
    }
    else
        moveToRestingPos();

    return exp;
}
