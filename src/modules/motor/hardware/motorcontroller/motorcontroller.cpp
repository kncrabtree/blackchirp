#include <src/modules/motor/hardware/motorcontroller/motorcontroller.h>

#include <QTimer>

MotorController::MotorController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical) : HardwareObject(BC::Key::mController,subKey,name,commType,parent,threaded,critical)
{
    p_limitTimer = new QTimer(this);
    p_limitTimer->setInterval(200);
    connect(p_limitTimer,&QTimer::timeout,this,&MotorController::checkLimit);
}

bool MotorController::prepareForExperiment(Experiment &exp)
{

    d_enabledForExperiment = exp.motorScan().isEnabled();
    if(d_enabledForExperiment)
        return prepareForMotorScan(exp);
    return true;
}
