#include "pulsegenerator.h"

PulseGenerator::PulseGenerator(QObject *parent) :
   HardwareObject(parent)
{
    d_key = QString("pGen");
}

PulseGenerator::~PulseGenerator()
{

}


PulseGenConfig::ChannelConfig PulseGenerator::read(const int index)
{
    PulseGenConfig::ChannelConfig out = d_config.settings(index);
    bool ok = false;
    out.channelName = read(index,PulseGenConfig::Name).toString();
    out.delay = read(index,PulseGenConfig::Delay).toDouble(&ok);
    if(!ok)
        return out;
    out.width = read(index,PulseGenConfig::Width).toDouble(&ok);
    if(!ok)
        return out;
    out.enabled = read(index,PulseGenConfig::Enabled).toBool();
    out.level = read(index,PulseGenConfig::Level).value<PulseGenConfig::ActiveLevel>();

    return out;
}


void PulseGenerator::setChannel(const int index, const PulseGenConfig::ChannelConfig cc)
{
    set(index,PulseGenConfig::Name,cc.channelName);
    set(index,PulseGenConfig::Enabled,cc.enabled);
    set(index,PulseGenConfig::Delay,cc.delay);
    set(index,PulseGenConfig::Width,cc.width);
    set(index,PulseGenConfig::Level,cc.level);
}

void PulseGenerator::setAll(const PulseGenConfig cc)
{
    for(int i=0; i<d_config.size(); i++)
        setChannel(i,cc.at(i));

    setRepRate(cc.repRate());

    return;
}

void PulseGenerator::readAll()
{
    for(int i=0;i<BC_PGEN_NUMCHANNELS; i++)
        read(i);
}
