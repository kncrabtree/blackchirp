#include "temperaturecontrollerconfig.h"

#include <hardware/optional/tempcontroller/temperaturecontroller.h>

TemperatureControllerConfig::TemperatureControllerConfig(QString subKey, int index) : HeaderStorage(BC::Key::hwKey(BC::Key::TC::key,index),subKey)
{

}

void TemperatureControllerConfig::setNumChannels(int n)
{
    d_channels.resize(n);
}

void TemperatureControllerConfig::setTemperature(uint ch, double t)
{
    if(ch < d_channels.size())
        d_channels[ch].t = t;
}

void TemperatureControllerConfig::setName(uint ch, QString n)
{
    if(ch < d_channels.size())
        d_channels[ch].name = n;
}

void TemperatureControllerConfig::setEnabled(uint ch, bool en)
{
    if(ch < d_channels.size())
        d_channels[ch].enabled = en;
}

double TemperatureControllerConfig::temperature(uint ch) const
{
    return ch < d_channels.size() ? d_channels[ch].t : 0.0;
}

QString TemperatureControllerConfig::channelName(uint ch) const
{
    return ch < d_channels.size() ? d_channels[ch].name : "";
}

bool TemperatureControllerConfig::channelEnabled(uint ch) const
{
    return ch < d_channels.size() ? d_channels[ch].enabled : false;
}


void TemperatureControllerConfig::storeValues()
{
    using namespace BC::Store::TempControlConfig;

    for(uint i=0; i<d_channels.size(); ++i)
    {
        storeArrayValue(channel,i,name,channelName(i));
        storeArrayValue(channel,i,enabled,channelEnabled(i));
    }
}

void TemperatureControllerConfig::retrieveValues()
{
    using namespace BC::Store::TempControlConfig;
    auto n = arrayStoreSize(channel);
    d_channels.clear();
    d_channels.reserve(n);
    for(std::size_t i=0; i<n; ++i)
    {
        TcChannel c {
            0.0,
            retrieveArrayValue(channel,i,name,QString("")),
            retrieveArrayValue(channel,i,enabled,false)
        };
        d_channels.push_back(c);
    }
}
