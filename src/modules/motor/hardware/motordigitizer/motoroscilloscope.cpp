#include <src/modules/motor/hardware/motordigitizer/motoroscilloscope.h>

MotorOscilloscope::MotorOscilloscope(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical) :
    HardwareObject(BC::Key::motorScope,subKey,name,commType,parent,threaded,critical)
{
}



bool MotorOscilloscope::prepareForExperiment(Experiment &exp)
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
    return exp.hardwareSuccess();
}
