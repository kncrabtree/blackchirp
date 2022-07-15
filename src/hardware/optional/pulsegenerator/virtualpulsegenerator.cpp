#include "virtualpulsegenerator.h"

VirtualPulseGenerator::VirtualPulseGenerator(QObject *parent) :
    PulseGenerator(BC::Key::Comm::hwVirtual,BC::Key::vpGen,CommunicationProtocol::Virtual,8,parent)
{
    using namespace BC::Key::PGen;
    setDefault(minWidth,0.010);
    setDefault(maxWidth,1e5);
    setDefault(minDelay,0.0);
    setDefault(maxDelay,1e5);
    setDefault(minRepRate,0.01);
    setDefault(maxRepRate,1e5);
    setDefault(lockExternal,false);
    setDefault(canDutyCycle,true);
    setDefault(canTrigger,true);
    setDefault(dutyMax,100000);
    setDefault(canSyncToChannel,true);
}

VirtualPulseGenerator::~VirtualPulseGenerator()
{

}

bool VirtualPulseGenerator::testConnection()
{
    readAll();
    return true;
}



bool VirtualPulseGenerator::setChWidth(const int index, const double width)
{
    setCh(index,PulseGenConfig::WidthSetting,width);
    return true;
}

bool VirtualPulseGenerator::setChDelay(const int index, const double delay)
{
    setCh(index,PulseGenConfig::DelaySetting,delay);
    return true;
}

bool VirtualPulseGenerator::setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level)
{
    setCh(index,PulseGenConfig::LevelSetting,level);
    return true;
}

bool VirtualPulseGenerator::setChEnabled(const int index, const bool en)
{
    setCh(index,PulseGenConfig::EnabledSetting,en);
    return true;
}

bool VirtualPulseGenerator::setHwRepRate(double rr)
{
    setRepRate(rr);
    return true;
}

bool VirtualPulseGenerator::setChSyncCh(const int index, const int syncCh)
{
    setCh(index,PulseGenConfig::SyncSetting,syncCh);
    return true;
}

bool VirtualPulseGenerator::setChMode(const int index, const ChannelMode mode)
{
    setCh(index,PulseGenConfig::ModeSetting,mode);
    return true;
}

bool VirtualPulseGenerator::setChDutyOn(const int index, const int pulses)
{
    setCh(index,PulseGenConfig::DutyOnSetting,pulses);
    return true;
}

bool VirtualPulseGenerator::setChDutyOff(const int index, const int pulses)
{
    setCh(index,PulseGenConfig::DutyOffSetting,pulses);
    return true;
}

bool VirtualPulseGenerator::setHwPulseMode(PGenMode mode)
{
    d_mode = mode;
    return true;
}

bool VirtualPulseGenerator::setHwPulseEnabled(bool en)
{
    d_pulseEnabled = en;
    return true;
}

double VirtualPulseGenerator::readChWidth(const int index)
{
    return at(index).width;
}

double VirtualPulseGenerator::readChDelay(const int index)
{
    return at(index).delay;
}

PulseGenConfig::ActiveLevel VirtualPulseGenerator::readChActiveLevel(const int index)
{
    return at(index).level;
}

bool VirtualPulseGenerator::readChEnabled(const int index)
{
    return at(index).enabled;
}

double VirtualPulseGenerator::readHwRepRate()
{
    return d_repRate;
}

int VirtualPulseGenerator::readChSynchCh(const int index)
{
    return at(index).syncCh;
}

PulseGenConfig::ChannelMode VirtualPulseGenerator::readChMode(const int index)
{
    return at(index).mode;
}

int VirtualPulseGenerator::readChDutyOn(const int index)
{
    return at(index).dutyOn;
}

int VirtualPulseGenerator::readChDutyOff(const int index)
{
    return at(index).dutyOff;
}

PulseGenConfig::PGenMode VirtualPulseGenerator::readHwPulseMode()
{
    return d_mode;
}

bool VirtualPulseGenerator::readHwPulseEnabled()
{
    return d_pulseEnabled;
}
