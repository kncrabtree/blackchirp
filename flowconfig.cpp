#include "flowconfig.h"

FlowConfig::FlowConfig() : data(new FlowConfigData)
{
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
    case Name:
        out = data->flowList.at(index).name;
        break;
    }

    return out;
}


double FlowConfig::pressureSetPoint() const
{
    return data->pressureSetpoint;
}

int FlowConfig::size() const
{
    return data->flowList.size();
}

void FlowConfig::add(double set, QString name)
{
    ChannelConfig cc;
    cc.enabled = qFuzzyCompare(1.0+set,1.0);
    cc.name = name;
    data->flowList.append(cc);
}

void FlowConfig::set(int index, FlowConfig::Setting s, QVariant val)
{
    if(index < 0 || index > data->flowList.size())
        return;

    switch(s) {
    case Setpoint:
        data->flowList[index].setpoint = val.toDouble();
        if(qFuzzyCompare(1.0+data->flowList.at(index).setpoint,1.0))
            data->flowList[index].enabled = false;
        break;
    case Name:
        data->flowList[index].name = val.toString();
        break;
    }
}


void FlowConfig::setPressureSetpoint(double s)
{
    data->pressureSetpoint = s;
}

