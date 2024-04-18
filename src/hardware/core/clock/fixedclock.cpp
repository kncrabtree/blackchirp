#include "fixedclock.h"

FixedClock::FixedClock(QObject *parent) :
    Clock(6,false,BC::Key::Clock::fixed,BC::Key::Clock::fixedName,CommunicationProtocol::None,parent)
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

    setDefault(minFreq,0.0);
    setDefault(maxFreq,1e7);
    setDefault(lock,true);
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
