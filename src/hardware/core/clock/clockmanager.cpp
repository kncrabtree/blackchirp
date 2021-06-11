#include "clockmanager.h"

#include <QSettings>

#include <src/hardware/core/clock/clock.h>


ClockManager::ClockManager(QObject *parent) : QObject(parent),
    SettingsStorage(BC::Key::clockManager)
{
    d_clockTypes = BlackChirp::allClockTypes();

#ifdef BC_CLOCK_0
    d_clockList << new Clock0Hardware(0,this);
#endif

#ifdef BC_CLOCK_1
    d_clockList << new Clock1Hardware(1,this);
#endif

#ifdef BC_CLOCK_2
    d_clockList << new Clock2Hardware(2,this);
#endif

#ifdef BC_CLOCK_3
    d_clockList << new Clock3Hardware(3,this);
#endif

#ifdef BC_CLOCK_4
    d_clockList << new Clock4Hardware(4,this);
#endif

    setArray(BC::Key::hwClocks,{});

    for(auto c : d_clockList)
    {
        auto names = c->channelNames();
        for(int j=0; j<c->numOutputs(); j++)
        {
            QString pn = c->d_prettyName;
            pn.append(QString(" "));
            if(j < names.size())
                pn.append(names.at(j));
            else
                pn.append(QString("Output %1").arg(j));

            appendArrayMap(BC::Key::hwClocks,{
                               {BC::Key::clockKey,c->d_key},
                               {BC::Key::clockOutput,j},
                               {BC::Key::clockName,pn}
                           });
        }
    }
    save();

    for(int i=0; i<d_clockList.size(); i++)
    {
        auto c = d_clockList.at(i);
        connect(c,&Clock::frequencyUpdate,this,&ClockManager::clockFrequencyUpdate);
    }


}

double ClockManager::setClockFrequency(BlackChirp::ClockType t, double freqMHz)
{
    if(!d_clockRoles.contains(t))
    {
        emit logMessage(QString("No clock configured for use as %1").arg(BlackChirp::clockPrettyName(t)),BlackChirp::LogWarning);
        return -1.0;
    }

    return d_clockRoles.value(t)->setFrequency(t,freqMHz);
}

double ClockManager::readClockFrequency(BlackChirp::ClockType t)
{
    if(!d_clockRoles.contains(t))
    {
        emit logMessage(QString("No clock configured for use as %1").arg(BlackChirp::clockPrettyName(t)),BlackChirp::LogWarning);
        return -1.0;
    }

    return d_clockRoles.value(t)->readFrequency(t);
}

bool ClockManager::prepareForExperiment(Experiment &exp)
{
    if(!exp.ftmwConfig().isEnabled())
        return true;

    d_clockRoles.clear();
    for(int i=0; i<d_clockList.size(); i++)
        d_clockList[i]->clearRoles();

    auto rfc = exp.ftmwConfig().rfConfig();
    auto map = rfc.getClocks();
    for(auto i = map.constBegin(); i != map.constEnd(); i++)
    {
        auto type = i.key();
        auto d = i.value();

        if(d.hwKey.isEmpty())
            continue;

        //find correct clock
        Clock *c = nullptr;
        for(int j=0; j<d_clockList.size(); j++)
        {
            if(d.hwKey == d_clockList.at(j)->key())
            {
                c = d_clockList.at(j);
                break;
            }
        }

        if(c == nullptr)
        {
            exp.setErrorString(QString("Could not find hardware clock for %1 (%2 output %3)")
                               .arg(BlackChirp::clockPrettyName(type)).arg(d.hwKey).arg(d.output));
            exp.setHardwareFailed();
            return false;
        }

        if(!c->addRole(type,d.output))
        {
            exp.setErrorString(QString("The output number requested for %1 (%2) is out of range (only %2 outputs are available).")
                               .arg(c->name()).arg(d.output).arg(c->numOutputs()));
            exp.setHardwareFailed();
            return false;
        }

        d_clockRoles.insertMulti(type,c);

        double mf = d.factor;
        if(d.op == RfConfig::Divide)
            mf = 1.0/d.factor;

        c->setMultFactor(mf,d.output);

        double actualFreq = c->setFrequency(type,d.desiredFreqMHz);
        if(actualFreq < 0.0)
        {
            exp.setErrorString(QString("Could not set %1 to %2 MHz (raw frequency = %3 MHz).")
                               .arg(c->name())
                               .arg(d.desiredFreqMHz,0,'f',6)
                               .arg(rfc.rawClockFrequency(type),0,'f',6));
            exp.setHardwareFailed();
            return false;
        }

        if(qAbs(actualFreq-d.desiredFreqMHz) > 0.1)
        {
            emit logMessage(QString("Actual frequency of %1 (%2 MHz) is more than 100 kHz from desired frequency (%3 MHz)")
                            .arg(BlackChirp::clockPrettyName(type))
                            .arg(actualFreq,0,'f',6)
                            .arg(d.desiredFreqMHz,0,'f',6));
        }

        d.desiredFreqMHz = actualFreq;

        rfc.setClockFreqInfo(type,d);
    }

    exp.setRfConfig(rfc);
    return true;

}
