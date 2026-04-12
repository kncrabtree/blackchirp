#include "fixedclock.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(FixedClock, "Fixed Frequency Clock")
REGISTER_HARDWARE_PROTOCOLS(FixedClock, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_SETTINGS(FixedClock,
    {BC::Key::Clock::minFreq, "Min Frequency (MHz)", "Minimum output frequency in MHz", 0.0, 0.0, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Clock::maxFreq, "Max Frequency (MHz)", "Maximum output frequency in MHz", 1e7, 0.0, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Clock::lock, "Requires External Lock", "Clock references an external 10 MHz lock signal.", true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

FixedClock::FixedClock(const QString& label, QObject *parent) :
    Clock(6, true, QString(FixedClock::staticMetaObject.className()), label, parent)
{
    using namespace BC::Key::Clock;
    if(containsArray(ch))
    {
        for(int i=0; i<6; ++i)
            d_currentFrequencyList << getArrayValue(ch,i,freq,0.0);
    }
    else
    {
        for(int i=0; i<6; i++)
            d_currentFrequencyList << 0.0;
    }

    save();
}

FixedClock::~FixedClock()
{
    using namespace BC::Key::Clock;
    setArray(ch,{});
    for(int i=0; i<6; ++i)
        appendArrayMap(ch,{{freq,d_currentFrequencyList.value(i)}});
}


bool FixedClock::testClockConnection()
{
    return true;
}

void FixedClock::initializeClock()
{
}

bool FixedClock::setHwFrequency(double freqMHz, int outputIndex)
{
    d_currentFrequencyList[outputIndex] = freqMHz;

    return true;
}

double FixedClock::readHwFrequency(int outputIndex)
{
    return d_currentFrequencyList.at(outputIndex);
}


QStringList FixedClock::forbiddenKeys() const
{
    auto out = Clock::forbiddenKeys();
    out.append(BC::Key::Clock::ch);
    return out;
}

