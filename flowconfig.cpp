#include "flowconfig.h"

FlowConfig::FlowConfig() : data(new FlowConfigData)
{
    for(int i=0; i<BC_FLOW_NUMCHANNELS; i++)
    {
        ChannelConfig cc;
        cc.enabled = false;
        cc.setpoint = 0.0;
        cc.flow = 0.0;
        cc.name = QString("");
        data->flowList.append(cc);
    }
}

FlowConfig::FlowConfig(const FlowConfig &rhs) : data(rhs.data)
{

}

FlowConfig &FlowConfig::operator=(const FlowConfig &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

FlowConfig::~FlowConfig()
{

}

QVariant FlowConfig::setting(int index, FlowConfig::Setting s) const
{
    QVariant out;
    if(index < 0 || index > data->flowList.size())
        return out;

    switch(s) {
    case Setpoint:
        out = data->flowList.at(index).setpoint;
        break;
    case Flow:
        out = data->flowList.at(index).flow;
        break;
    case Name:
        out = data->flowList.at(index).name;
        break;
    }

    return out;
}

double FlowConfig::pressure() const
{
    return data->pressure;
}

double FlowConfig::pressureSetPoint() const
{
    return data->pressureSetpoint;
}

void FlowConfig::set(int index, FlowConfig::Setting s, QVariant val)
{
    if(index < 0 || index > data->flowList.size())
        return;

    switch(s) {
    case Setpoint:
        data->flowList[index].setpoint = val.toDouble();
        break;
    case Flow:
        data->flowList[index].flow = val.toDouble();
        break;
    case Name:
        data->flowList[index].name = val.toString();
        break;
    }
}

void FlowConfig::setPressure(double p)
{
    data->pressure = p;
}

void FlowConfig::setPressureSetpoint(double s)
{
    data->pressureSetpoint = s;
}

