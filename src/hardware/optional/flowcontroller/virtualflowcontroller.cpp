#include "virtualflowcontroller.h"

using namespace BC::Key::Flow;

VirtualFlowController::VirtualFlowController(QObject *parent) :
    FlowController(BC::Key::Comm::hwVirtual,virtFCName,CommunicationProtocol::Virtual,parent)
{

    if(!containsArray(channels))
    {
        std::vector<SettingsMap> l;
        int ch = get(flowChannels,4);
        l.reserve(ch);
        for(int i=0; i<ch; ++i)
            l.push_back({{chUnits,QString("sccm")},{chMax,10000.0},{chDecimals,3}});

        setArray(channels,l,true);
    }

    setDefault(pUnits,QString("kTorr"));
    setDefault(pMax,10.0);
    setDefault(pDec,3);
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

    d_config.set(ch,FlowConfig::Setpoint,val);
}

void VirtualFlowController::hwSetPressureSetpoint(const double val)
{
    d_config.setPressureSetpoint(val);
}

double VirtualFlowController::hwReadFlowSetpoint(const int ch)
{
    if(ch < 0 || ch >= d_config.size())
        return -1.0;

    return d_config.setting(ch,FlowConfig::Setpoint).toDouble();
}

double VirtualFlowController::hwReadPressureSetpoint()
{
    return d_config.pressureSetpoint();
}

double VirtualFlowController::hwReadFlow(const int ch)
{
    if(ch < 0 || ch >= d_config.size())
        return -1.0;

    double sp = d_config.setting(ch,FlowConfig::Setpoint).toDouble();
//    double noise = sp*((double)(qrand()%100)-50.0)/1000.0;
//    double flow = sp + noise;
    d_config.set(ch,FlowConfig::Flow,sp);

    return d_config.setting(ch,FlowConfig::Flow).toDouble();
}

double VirtualFlowController::hwReadPressure()
{
    return d_config.pressureSetpoint();
}

void VirtualFlowController::hwSetPressureControlMode(bool enabled)
{
    d_config.setPressureControlMode(enabled);
}

int VirtualFlowController::hwReadPressureControlMode()
{
    return d_config.pressureControlMode() ? 1 : 0;
}

void VirtualFlowController::poll()
{
    readAll();
}
