#include "clock.h"

Clock::Clock(int clockNum, QObject *parent) : HardwareObject(parent)
{
    d_key = QString("clock%1").arg(clockNum);
}

QStringList Clock::channelNames()
{
    QStringList out;
    out << QString("");
    return out;
}

bool Clock::setRole(BlackChirp::ClockType t, int outputIndex)
{
    if(outputIndex >= d_numOutputs)
        return false;

    d_outputRoles.insert(outputIndex,t);
    return true;
}

void Clock::removeRole(BlackChirp::ClockType t)
{
    d_outputRoles.remove(d_outputRoles.key(t));
}

void Clock::clearRoles()
{
    d_outputRoles.clear();
}

bool Clock::hasRole(BlackChirp::ClockType t)
{
    return d_outputRoles.values().contains(t);
}

void Clock::readAll()
{
    for(int i=0; i<numOutputs(); i++)
    {
        double f = readHwFrequency(i);
        if(f < 0.0)
            break;
        if(d_outputRoles.contains(i))
            emit frequencyUpdate(d_outputRoles.value(i),f);
    }
}

double Clock::readFrequency(BlackChirp::ClockType t)
{
    if(!hasRole(t))
        return -1.0;

    double out = readHwFrequency(d_outputRoles.key(t));
    if(out > 0.0)
        emit frequencyUpdate(t,out);

    return out;
}

double Clock::setFrequency(BlackChirp::ClockType t, double freqMHz)
{
    if(!hasRole(t))
        return -1.0;

    if(freqMHz < d_minFreqMHz || freqMHz > d_maxFreqMHz)
    {
        emit logMessage(QString("Desired frequency (%1 MHz) is outside the clock's range (%2 - %3 MHz). Frequency has not been changed.")
                        .arg(freqMHz,0,'f',3).arg(d_minFreqMHz,0,'f',3).arg(d_maxFreqMHz,0,'f',3),
                        BlackChirp::LogWarning);
        return -1.0;
    }

    if(!setHwFrequency(freqMHz,d_outputRoles.key(t)))
    {
        emit logMessage(QString("Cannot set frequency to %1 because of a hardware error.")
                        .arg(freqMHz,0,'f',3),BlackChirp::LogError);
        return -1.0;
    }

    double out = readHwFrequency(d_outputRoles.key(t));
    if(out > 0.0)
        emit frequencyUpdate(t,out);

    return out;
}
