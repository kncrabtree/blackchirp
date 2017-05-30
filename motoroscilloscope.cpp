#include "motoroscilloscope.h"

MotorOscilloscope::MotorOscilloscope(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("motorScope");
}



Experiment MotorOscilloscope::prepareForExperiment(Experiment exp)
{
    if(exp.motorScan().isEnabled())
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

void MotorOscilloscope::beginAcquisition()
{
}

void MotorOscilloscope::endAcquisition()
{
}

void MotorOscilloscope::readTimeData()
{
}
