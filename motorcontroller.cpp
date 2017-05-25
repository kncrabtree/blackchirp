#include "motorcontroller.h"

#include <QTimer>

MotorController::MotorController(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("motorController");

    p_limitTimer = new QTimer(this);
    p_limitTimer->setInterval(1000);
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

bool MotorController::prepareForMotorScan(const MotorScan ms)
{
    return moveToPosition(ms.xVal(0),ms.yVal(0),ms.zVal(0));
}


void MotorController::endAcquisition()
{
    moveToRestingPos();
}
