#include "virtualpulsegenerator.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(VirtualPulseGenerator, "Virtual PulseGenerator for Testing")
REGISTER_HARDWARE_SETTINGS(VirtualPulseGenerator,
    {BC::Key::PGen::numChannels, "Number of Channels",
     "Number of pulse output channels",
     8, 1, 64, HwSettingPriority::Required},
    {BC::Key::PGen::minWidth, "Min Pulse Width (us)",
     "Minimum pulse width in microseconds",
     0.010, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::maxWidth, "Max Pulse Width (us)",
     "Maximum pulse width in microseconds",
     1e5, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::minDelay, "Min Delay (us)",
     "Minimum channel delay in microseconds",
     0.0, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::maxDelay, "Max Delay (us)",
     "Maximum channel delay in microseconds",
     1e5, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::minRepRate, "Min Rep Rate (Hz)",
     "Minimum repetition rate in Hz",
     0.01, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::maxRepRate, "Max Rep Rate (Hz)",
     "Maximum repetition rate in Hz",
     1e5, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::lockExternal, "External Clock Lock",
     "Default to external 10 MHz clock reference",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::canDutyCycle, "Duty Cycle Mode",
     "Supports duty-cycle triggered mode",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::canTrigger, "External Trigger",
     "Supports external trigger input",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::dutyMax, "Max Duty Cycles",
     "Maximum number of pulses per duty cycle burst",
     100000, 1, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::canSyncToChannel, "Sync to Channel",
     "Channels can be synchronized to another channel output",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PGen::canDisableChannels, "Can Disable Channels",
     "Individual channels can be independently enabled/disabled",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

VirtualPulseGenerator::VirtualPulseGenerator(const QString& label, QObject *parent) :
    PulseGenerator(QString(VirtualPulseGenerator::staticMetaObject.className()), label, 8, parent)
{
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
    d_config.setCh(index,PulseGenConfig::WidthSetting,width);
    return true;
}

bool VirtualPulseGenerator::setChDelay(const int index, const double delay)
{
    d_config.setCh(index,PulseGenConfig::DelaySetting,delay);
    return true;
}

bool VirtualPulseGenerator::setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level)
{
    d_config.setCh(index,PulseGenConfig::LevelSetting,level);
    return true;
}

bool VirtualPulseGenerator::setChEnabled(const int index, const bool en)
{
    d_config.setCh(index,PulseGenConfig::EnabledSetting,en);
    return true;
}

bool VirtualPulseGenerator::setHwRepRate(double rr)
{
    d_config.d_repRate = rr;
    return true;
}

bool VirtualPulseGenerator::setChSyncCh(const int index, const int syncCh)
{
    d_config.setCh(index,PulseGenConfig::SyncSetting,syncCh);
    return true;
}

bool VirtualPulseGenerator::setChMode(const int index, const PulseGenConfig::ChannelMode mode)
{
    d_config.setCh(index,PulseGenConfig::ModeSetting,mode);
    return true;
}

bool VirtualPulseGenerator::setChDutyOn(const int index, const int pulses)
{
    d_config.setCh(index,PulseGenConfig::DutyOnSetting,pulses);
    return true;
}

bool VirtualPulseGenerator::setChDutyOff(const int index, const int pulses)
{
    d_config.setCh(index,PulseGenConfig::DutyOffSetting,pulses);
    return true;
}

bool VirtualPulseGenerator::setHwPulseMode(PulseGenConfig::PGenMode mode)
{
    d_config.d_mode = mode;
    return true;
}

bool VirtualPulseGenerator::setHwPulseEnabled(bool en)
{
    d_config.d_pulseEnabled = en;
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
    return d_config.d_repRate;
}

int VirtualPulseGenerator::readChSynchCh(const int index)
{
    return d_config.at(index).syncCh;
}

PulseGenConfig::ChannelMode VirtualPulseGenerator::readChMode(const int index)
{
    return d_config.at(index).mode;
}

int VirtualPulseGenerator::readChDutyOn(const int index)
{
    return d_config.at(index).dutyOn;
}

int VirtualPulseGenerator::readChDutyOff(const int index)
{
    return d_config.at(index).dutyOff;
}

PulseGenConfig::PGenMode VirtualPulseGenerator::readHwPulseMode()
{
    return d_config.d_mode;
}

bool VirtualPulseGenerator::readHwPulseEnabled()
{
    return d_config.d_pulseEnabled;
}
