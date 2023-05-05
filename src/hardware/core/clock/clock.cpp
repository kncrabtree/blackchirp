#include <hardware/core/clock/clock.h>

#include <QMetaEnum>

Clock::Clock(int clockNum, int numOutputs, bool tunable, const QString subKey, const QString name, CommunicationProtocol::CommType commType,
             QObject *parent) :
    HardwareObject(BC::Key::Clock::clock,subKey,name,commType,parent,false,true,clockNum), d_numOutputs(numOutputs),
    d_isTunable(tunable)
{
    using namespace BC::Key::Clock;

    setDefault(BC::Key::Clock::tunable,d_isTunable);

    for(int i=0; i<d_numOutputs; i++)
        d_multFactors << 1.0;

    if(containsArray(outputs))
    {
        int n = getArraySize(outputs);
        for(int i=0; i<n && i<d_numOutputs; ++i)
        {
            QVariant type = getArrayValue(outputs,i,role);
            if(type.isValid())
                addRole(type.value<RfConfig::ClockType>(),i);
            auto factor = getArrayValue(outputs,i,mf,1.0);
            setMultFactor(factor,i);
        }
    }
}

Clock::~Clock()
{
    using namespace BC::Key::Clock;
    setArray(outputs,{});

    for(int i=0; i<d_numOutputs; ++i)
    {
        SettingsMap m;
        for(auto it = d_outputRoles.cbegin(); it != d_outputRoles.cend(); ++it)
        {
            if(it.value() == i)
                m.insert({role,it.key()});
        }
        m.insert({mf,d_multFactors.at(i)});
        appendArrayMap(outputs,m);
    }
}

void Clock::setMultFactor(double d, int output)
{
    if(d > 0.0 && output < d_multFactors.size())
        d_multFactors[output] = d;
}

double Clock::multFactor(int output)
{
    return d_multFactors.value(output);
}

int Clock::outputForRole(RfConfig::ClockType t)
{
    auto it = d_outputRoles.find(t);
    if(it == d_outputRoles.end())
        return -1;

    return it.value();
}

QStringList Clock::channelNames()
{
    return QStringList();
}

void Clock::initialize()
{
    initializeClock();
}

bool Clock::testConnection()
{
    auto out = testClockConnection();
    if(out)
        readAll();
    return out;
}

bool Clock::addRole(RfConfig::ClockType t, int outputIndex)
{
    if(outputIndex >= d_numOutputs)
        return false;

    d_outputRoles.insert(t,outputIndex);
    return true;
}

void Clock::removeRole(RfConfig::ClockType t)
{
    if(d_outputRoles.contains(t))
    {
        emit frequencyUpdate(t,-1.0);
        d_outputRoles.remove(t);
    }
}

void Clock::clearRoles()
{
    for(auto it = d_outputRoles.constBegin(); it != d_outputRoles.constEnd(); ++it)
        emit frequencyUpdate(it.key(),-1.0);

    d_outputRoles.clear();
}

bool Clock::hasRole(RfConfig::ClockType t)
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
            if(it.value() == i)
                emit frequencyUpdate(it.key(),f);
        }
    }
}

double Clock::readFrequency(RfConfig::ClockType t)
{
    if(!hasRole(t))
        return -1.0;

    int output = d_outputRoles.value(t);
    double out = readHwFrequency(output);
    out *= d_multFactors.at(output);
    emit frequencyUpdate(t,out);

    return out;
}

double Clock::setFrequency(RfConfig::ClockType t, double freqMHz)
{
    if(!hasRole(t))
        return -1.0;

    int output = d_outputRoles.value(t);
    double hwFreqMHz = freqMHz/d_multFactors.at(output);

    auto min = get<double>(BC::Key::Clock::minFreq);
    auto max = get<double>(BC::Key::Clock::maxFreq);

    if(hwFreqMHz < min || hwFreqMHz > max)
    {
        emit logMessage(QString("Desired frequency (%1 MHz) is outside the clock's range (%2 - %3 MHz). Frequency has not been changed.")
                        .arg(hwFreqMHz,0,'f',3).arg(min,0,'f',3).arg(max,0,'f',3),
                        LogHandler::Warning);
        return -1.0;
    }

    if(!setHwFrequency(hwFreqMHz,d_outputRoles.value(t)))
    {
        emit logMessage(QString("Cannot set frequency to %1 because of a hardware error.")
                        .arg(hwFreqMHz,0,'f',3),LogHandler::Error);
        return -1.0;
    }

    double out = readHwFrequency(d_outputRoles.value(t));
    out *= d_multFactors.value(output);
    if(!isnan(out))
        emit frequencyUpdate(t,out);

    return out;
}


bool Clock::prepareForExperiment(Experiment &exp)
{
    if(exp.ftmwEnabled())
    {
        auto clocks = exp.ftmwConfig()->d_rfConfig.getClocks();
        for(auto it = clocks.constBegin(); it != clocks.constEnd(); it++)
        {
            if(hasRole(it.key()))
            {
                auto c = it.value();
                double val = setFrequency(it.key(),c.desiredFreqMHz);
                if(val < 0.0)
                {
                    exp.d_errorString = QString("Could not initialize %1 to %2 MHz")
                                       .arg(QMetaEnum::fromType<RfConfig::ClockType>()
                                            .valueToKey(it.key()))
                                       .arg(it.value().desiredFreqMHz,0,'f',6);
                    return false;
                }
                exp.ftmwConfig()->d_rfConfig.setClockDesiredFreq(it.key(),val);
            }
        }
    }

    return prepareClock(exp);
}
