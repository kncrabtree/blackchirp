#include "virtualpulsegenerator.h"

VirtualPulseGenerator::VirtualPulseGenerator(QObject *parent) :
    PulseGenerator(BC::Key::Comm::hwVirtual,BC::Key::vpGen,CommunicationProtocol::Virtual,8,parent)
{
    setDefault(BC::Key::PGen::minWidth,0.010);
    setDefault(BC::Key::PGen::maxWidth,1e5);
    setDefault(BC::Key::PGen::minDelay,0.0);
    setDefault(BC::Key::PGen::maxDelay,1e5);
    setDefault(BC::Key::PGen::minRepRate,0.01);
    setDefault(BC::Key::PGen::maxRepRate,1e5);
    setDefault(BC::Key::PGen::lockExternal,false);
}

VirtualPulseGenerator::~VirtualPulseGenerator()
{

}

bool VirtualPulseGenerator::testConnection()
{
    for(int i=0; i<d_config.size(); i++)
        d_config.set(i,PulseGenConfig::ChannelConfig());

    readAll();
    return true;
}



bool VirtualPulseGenerator::setChWidth(const int index, const double width)
{
    d_config.set(index,PulseGenConfig::WidthSetting,width);
    return true;
}

bool VirtualPulseGenerator::setChDelay(const int index, const double delay)
{
    d_config.set(index,PulseGenConfig::DelaySetting,delay);
    return true;
}

bool VirtualPulseGenerator::setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level)
{
    d_config.set(index,PulseGenConfig::LevelSetting,level);
    return true;
}

bool VirtualPulseGenerator::setChEnabled(const int index, const bool en)
{
    d_config.set(index,PulseGenConfig::EnabledSetting,en);
    return true;
}

bool VirtualPulseGenerator::setHwRepRate(double rr)
{
    d_config.setRepRate(rr);
    return true;
}

double VirtualPulseGenerator::readChWidth(const int index)
{
    return d_config.at(index).width;
}

double VirtualPulseGenerator::readChDelay(const int index)
{
    return d_config.at(index).delay;
}

PulseGenConfig::ActiveLevel VirtualPulseGenerator::readChActiveLevel(const int index)
{
    return d_config.at(index).level;
}

bool VirtualPulseGenerator::readChEnabled(const int index)
{
    return d_config.at(index).enabled;
}

double VirtualPulseGenerator::readHwRepRate()
{
    return d_config.repRate();
}
