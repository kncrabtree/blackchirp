#include "pulsegenerator.h"

PulseGenerator::PulseGenerator(QObject *parent) :
   HardwareObject(parent)
{
    d_key = QString("pGen");
}

PulseGenerator::~PulseGenerator()
{

}


BlackChirp::PulseChannelConfig PulseGenerator::read(const int index)
{
    BlackChirp::PulseChannelConfig out = d_config.settings(index);
    bool ok = false;
    out.channelName = read(index,BlackChirp::PulseName).toString();
    out.delay = read(index,BlackChirp::PulseDelay).toDouble(&ok);
    if(!ok)
        return out;
    out.width = read(index,BlackChirp::PulseWidth).toDouble(&ok);
    if(!ok)
        return out;
    out.enabled = read(index,BlackChirp::PulseEnabled).toBool();
    out.level = read(index,BlackChirp::PulseLevel).value<BlackChirp::PulseActiveLevel>();

    return out;
}


void PulseGenerator::setChannel(const int index, const BlackChirp::PulseChannelConfig cc)
{
    set(index,BlackChirp::PulseName,cc.channelName);
    set(index,BlackChirp::PulseEnabled,cc.enabled);
    set(index,BlackChirp::PulseDelay,cc.delay);
    set(index,BlackChirp::PulseWidth,cc.width);
    set(index,BlackChirp::PulseLevel,cc.level);
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
