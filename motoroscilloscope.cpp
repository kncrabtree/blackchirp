#include "motoroscilloscope.h"

MotorOscilloscope::MotorOscilloscope(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("motorScope");
}



Experiment MotorOscilloscope::prepareForExperiment(Experiment exp)
{
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
