#include "fixedclock.h"

FixedClock::FixedClock(int clockNum, QObject *parent) : Clock(clockNum,5,false,BC::Key::fixed,BC::Key::fixedName.arg(clockNum),
                                                              CommunicationProtocol::None,parent)
{
    for(int i=0; i<5; i++)
        d_currentFrequencyList << 0.0;

    setDefault(BC::Key::Clock::minFreq,0.0);
    setDefault(BC::Key::Clock::maxFreq,1e7);
    setDefault(BC::Key::Clock::lock,true);
}


bool FixedClock::testConnection()
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
