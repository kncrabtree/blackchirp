#include "pulsegenerator.h"

PulseGenerator::PulseGenerator(QObject *parent) :
   HardwareObject(parent), d_minWidth(0.010), d_maxWidth(100000.0), d_minDelay(0.0), d_maxDelay(100000.0)
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


bool PulseGenerator::setChannel(const int index, const BlackChirp::PulseChannelConfig cc)
{
    bool success = true;

    success &= set(index,BlackChirp::PulseName,cc.channelName);
    success &= set(index,BlackChirp::PulseEnabled,cc.enabled);
    success &= set(index,BlackChirp::PulseDelay,cc.delay);
    success &= set(index,BlackChirp::PulseWidth,cc.width);
    success &= set(index,BlackChirp::PulseLevel,cc.level);

    return success;
}

bool PulseGenerator::setAll(const PulseGenConfig cc)
{
    bool success = true;
    for(int i=0; i<d_config.size(); i++)
        success &= setChannel(i,cc.at(i));

    success &= setRepRate(cc.repRate());

    return success;
}

bool PulseGenerator::setLifDelay(double d)
{
    return set(BC_PGEN_LIFCHANNEL,BlackChirp::PulseDelay,d);
}

void PulseGenerator::readAll()
{
    for(int i=0;i<BC_PGEN_NUMCHANNELS; i++)
        read(i);

    readRepRate();
}
