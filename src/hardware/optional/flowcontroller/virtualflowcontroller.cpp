#include "virtualflowcontroller.h"
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key;

// Register hardware implementation using new metaobject system
REGISTER_HARDWARE_META(VirtualFlowController, "Virtual flow controller for testing and development")
REGISTER_HARDWARE_ARRAY(VirtualFlowController, BC::Key::Flow::channels,
    "Flow Channels", "Per-channel mass flow controller configuration.", HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualFlowController, BC::Key::Flow::channels,
    {{BC::Key::Flow::chUnits, QString("sccm")}, {BC::Key::Flow::chMax, 10000.0}, {BC::Key::Flow::chDecimals, 3}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualFlowController, BC::Key::Flow::channels,
    {{BC::Key::Flow::chUnits, QString("sccm")}, {BC::Key::Flow::chMax, 10000.0}, {BC::Key::Flow::chDecimals, 3}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualFlowController, BC::Key::Flow::channels,
    {{BC::Key::Flow::chUnits, QString("sccm")}, {BC::Key::Flow::chMax, 10000.0}, {BC::Key::Flow::chDecimals, 3}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualFlowController, BC::Key::Flow::channels,
    {{BC::Key::Flow::chUnits, QString("sccm")}, {BC::Key::Flow::chMax, 10000.0}, {BC::Key::Flow::chDecimals, 3}})

VirtualFlowController::VirtualFlowController(const QString& label, QObject *parent) :
    FlowController(QString(VirtualFlowController::staticMetaObject.className()), label, parent)
{

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
    bool enabled = d_config.setting(ch,FlowConfig::Enabled).toBool();
    double flow;
    if(enabled && sp > 0.0)
    {
        std::normal_distribution<double> dist(sp, sp * 0.01);
        flow = dist(d_rng);
    }
    else
    {
        std::normal_distribution<double> dist(0.0, 0.1);
        flow = std::abs(dist(d_rng));
    }
    d_config.setCh(ch,FlowConfig::Flow,flow);
    return flow;
}

double VirtualFlowController::hwReadPressure()
{
    double sp = d_config.d_pressureSetpoint;
    std::normal_distribution<double> dist(sp, 0.1);
    return qMax(0.0, dist(d_rng));
}

void VirtualFlowController::hwSetPressureControlMode(bool enabled)
{
    d_config.d_pressureControlMode = enabled;
}

int VirtualFlowController::hwReadPressureControlMode()
{
    return d_config.d_pressureControlMode ? 1 : 0;
}
