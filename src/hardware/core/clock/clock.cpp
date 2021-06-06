#include <src/hardware/core/clock/clock.h>

Clock::Clock(int clockNum, QObject *parent) : HardwareObject(parent)
{
    d_key = QString("clock%1").arg(clockNum);
}

void Clock::setMultFactor(double d, int output)
{
    if(d > 0.0 && output < d_multFactors.size())
        d_multFactors[output] = d;
}

QStringList Clock::channelNames()
{
    return QStringList();
}

void Clock::initialize()
{
    for(int i=0; i<d_numOutputs; i++)
        d_multFactors << 1.0;
    initializeClock();
}

bool Clock::addRole(BlackChirp::ClockType t, int outputIndex)
{
    if(outputIndex >= d_numOutputs)
        return false;

    d_outputRoles.insert(t,outputIndex);
    return true;
}

void Clock::removeRole(BlackChirp::ClockType t)
{
    d_outputRoles.remove(t);
}

void Clock::clearRoles()
{
    d_outputRoles.clear();
}

bool Clock::hasRole(BlackChirp::ClockType t)
{
    return d_outputRoles.contains(t);
}

void Clock::readAll()
{
    for(int i=0; i<numOutputs(); i++)
    {
        double f = readHwFrequency(i);
        f *= d_multFactors.at(i);
        if(f < 0.0)
            break;
        for(auto it = d_outputRoles.constBegin(); it != d_outputRoles.constEnd(); it++)
        {
            if(it.value() == 1)
                emit frequencyUpdate(it.key(),f);
        }
    }
}

double Clock::readFrequency(BlackChirp::ClockType t)
{
    if(!hasRole(t))
        return -1.0;

    int output = d_outputRoles.value(t);
    double out = readHwFrequency(output);
    out *= d_multFactors.at(output);
    emit frequencyUpdate(t,out);

    return out;
}

double Clock::setFrequency(BlackChirp::ClockType t, double freqMHz)
{
    if(!hasRole(t))
        return -1.0;

    int output = d_outputRoles.value(t);
    double hwFreqMHz = freqMHz/d_multFactors.at(output);

    if(hwFreqMHz < d_minFreqMHz || hwFreqMHz > d_maxFreqMHz)
    {
        emit logMessage(QString("Desired frequency (%1 MHz) is outside the clock's range (%2 - %3 MHz). Frequency has not been changed.")
                        .arg(hwFreqMHz,0,'f',3).arg(d_minFreqMHz,0,'f',3).arg(d_maxFreqMHz,0,'f',3),
                        BlackChirp::LogWarning);
        return -1.0;
    }

    if(!setHwFrequency(hwFreqMHz,d_outputRoles.value(t)))
    {
        emit logMessage(QString("Cannot set frequency to %1 because of a hardware error.")
                        .arg(hwFreqMHz,0,'f',3),BlackChirp::LogError);
        return -1.0;
    }

    double out = readHwFrequency(d_outputRoles.value(t));
    out *= d_multFactors.at(output);
    if(out > 0.0)
        emit frequencyUpdate(t,out);

    return out;
}


bool Clock::prepareForExperiment(Experiment &exp)
{
    if(exp.ftmwConfig().isEnabled())
    {
        auto rfc = exp.ftmwConfig().rfConfig();
        auto clocks = rfc.getClocks();
        for(auto it = clocks.constBegin(); it != clocks.constEnd(); it++)
        {
            if(hasRole(it.key()))
            {
                auto c = it.value();
                double val = setFrequency(it.key(),c.desiredFreqMHz);
                if(val < 0.0)
                {
                    exp.setErrorString(QString("Could not initialize %1 to %2 MHz").arg(BlackChirp::clockPrettyName(it.key())).arg(it.value().desiredFreqMHz,0,'f',6));
                    exp.setHardwareFailed();
                    return false;
                }
                rfc.setClockDesiredFreq(it.key(),val);
            }
        }
        exp.setRfConfig(rfc);
    }

    return prepareClock(exp);
}
