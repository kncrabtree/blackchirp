#include "motoroscilloscope.h"

MotorOscilloscope::MotorOscilloscope(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("motorScope");
}



Experiment MotorOscilloscope::prepareForExperiment(Experiment exp)
{
    d_enabledForExperiment = exp.motorScan().isEnabled();
    if(d_enabledForExperiment)
    {
        MotorScan ms = prepareForMotorScan(exp.motorScan());
        if(ms.hardwareError())
        {
            exp.setHardwareFailed();
            exp.setErrorString(QString("Failed to prepare %1 for scan").arg(name()));
        }
        exp.setMotorScan(ms);
    }
    return exp;
}
