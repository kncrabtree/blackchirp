#include "virtualflowcontroller.h"

VirtualFlowController::VirtualFlowController(QObject *parent) :
    FlowController(BC::Key::hwVirtual,BC::Key::virtFCName,CommunicationProtocol::Virtual,parent)
{
}

VirtualFlowController::~VirtualFlowController()
{
}



bool VirtualFlowController::fcTestConnection()
{
    return true;
}

void VirtualFlowController::fcInitialize()
{
}

void VirtualFlowController::hwSetFlowSetpoint(const int ch, const double val)
{
    if(ch<0 || ch >= d_config.size())
        return;

    d_config.set(ch,BlackChirp::FlowSettingSetpoint,val);
}

void VirtualFlowController::hwSetPressureSetpoint(const double val)
{
    d_config.setPressureSetpoint(val);
}

double VirtualFlowController::hwReadFlowSetpoint(const int ch)
{
    if(ch < 0 || ch >= d_config.size())
        return -1.0;

    return d_config.setting(ch,BlackChirp::FlowSettingSetpoint).toDouble();
}

double VirtualFlowController::hwReadPressureSetpoint()
{
    return d_config.pressureSetpoint();
}

double VirtualFlowController::hwReadFlow(const int ch)
{
    if(ch < 0 || ch >= d_config.size())
        return -1.0;

    double sp = d_config.setting(ch,BlackChirp::FlowSettingSetpoint).toDouble();
//    double noise = sp*((double)(qrand()%100)-50.0)/1000.0;
//    double flow = sp + noise;
    d_config.set(ch,BlackChirp::FlowSettingFlow,sp);

    return d_config.setting(ch,BlackChirp::FlowSettingFlow).toDouble();
}

double VirtualFlowController::hwReadPressure()
{
    return d_config.pressureSetpoint();
}

void VirtualFlowController::hwSetPressureControlMode(bool enabled)
{
    d_config.setPressureControlMode(enabled);
    readPressureControlMode();
}

int VirtualFlowController::hwReadPressureControlMode()
{
    return d_config.pressureControlMode() ? 1 : 0;
}

void VirtualFlowController::poll()
{
    readAll();
}
