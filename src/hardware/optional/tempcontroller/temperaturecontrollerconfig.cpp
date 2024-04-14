#include "temperaturecontrollerconfig.h"

#include <hardware/optional/tempcontroller/temperaturecontroller.h>

TemperatureControllerConfig::TemperatureControllerConfig(int index) : HeaderStorage(BC::Key::hwKey(BC::Key::TC::key,index))
{

}

void TemperatureControllerConfig::setNumChannels(int n)
{
    d_channels.resize(n);
}

void TemperatureControllerConfig::setTemperature(int ch, double t)
{
    if(ch >=0 && ch < d_channels.size())
        d_channels[ch].t = t;
}

void TemperatureControllerConfig::setName(int ch, QString n)
{
    if(ch >=0 && ch < d_channels.size())
        d_channels[ch].name = n;
}

void TemperatureControllerConfig::setEnabled(int ch, bool en)
{
    if(ch >=0 && ch < d_channels.size())
        d_channels[ch].enabled = en;
}

double TemperatureControllerConfig::temperature(int ch) const
{
    return d_channels.value(ch,{}).t;
}

QString TemperatureControllerConfig::channelName(int ch) const
{
    return d_channels.value(ch,{}).name;
}

bool TemperatureControllerConfig::channelEnabled(int ch) const
{
    return d_channels.value(ch,{}).enabled;
}


void TemperatureControllerConfig::storeValues()
{
    using namespace BC::Store::TempControlConfig;

    for(int i=0; i<d_channels.size(); ++i)
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
        d_channels << c;
    }
}
