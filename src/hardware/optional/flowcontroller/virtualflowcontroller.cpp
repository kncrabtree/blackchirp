#include "virtualflowcontroller.h"
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key;

// Register hardware implementation using new metaobject system
REGISTER_HARDWARE_META(VirtualFlowController, "Virtual flow controller for testing and development")
REGISTER_HARDWARE_SETTINGS(VirtualFlowController,
    {BC::Key::Flow::flowChannels, "Flow Channels",
     "Number of mass flow controller channels connected.",
     4, 1, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Flow::pUnits, "Pressure Units",
     "Units for pressure reading display.",
     QString("kTorr"), QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Flow::pMax, "Max Pressure",
     "Full-scale pressure for display scaling.",
     10.0, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Flow::pDec, "Pressure Decimals",
     "Number of decimal places in pressure display.",
     3, 0, 10, HwSettingPriority::Optional}
)

VirtualFlowController::VirtualFlowController(const QString& label, QObject *parent) :
    FlowController(QString(VirtualFlowController::staticMetaObject.className()), label, parent)
{

    if(!containsArray(Flow::channels))
    {
        std::vector<SettingsMap> l;
        int ch = get(Flow::flowChannels,4);
        l.reserve(ch);
        for(int i=0; i<ch; ++i)
            l.push_back({{Flow::chUnits,QString("sccm")},{Flow::chMax,10000.0},{Flow::chDecimals,3}});

        setArray(Flow::channels,l,true);
    }

    save();
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

    d_config.setCh(ch,FlowConfig::Setpoint,val);
}

void VirtualFlowController::hwSetPressureSetpoint(const double val)
{
    d_config.d_pressureSetpoint = val;
}

double VirtualFlowController::hwReadFlowSetpoint(const int ch)
{
    if(ch < 0 || ch >= d_config.size())
        return -1.0;

    return d_config.setting(ch,FlowConfig::Setpoint).toDouble();
}

double VirtualFlowController::hwReadPressureSetpoint()
{
    return d_config.d_pressureSetpoint;
}

double VirtualFlowController::hwReadFlow(const int ch)
{
    if(ch < 0 || ch >= d_config.size())
        return -1.0;

    double sp = d_config.setting(ch,FlowConfig::Setpoint).toDouble();
//    double noise = sp*((double)(qrand()%100)-50.0)/1000.0;
//    double flow = sp + noise;
    d_config.setCh(ch,FlowConfig::Flow,sp);

    return d_config.setting(ch,FlowConfig::Flow).toDouble();
}

double VirtualFlowController::hwReadPressure()
{
    return d_config.d_pressureSetpoint;
}

void VirtualFlowController::hwSetPressureControlMode(bool enabled)
{
    d_config.d_pressureControlMode = enabled;
}

int VirtualFlowController::hwReadPressureControlMode()
{
    return d_config.d_pressureControlMode ? 1 : 0;
}
