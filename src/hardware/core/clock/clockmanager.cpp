#include "clockmanager.h"

#include <QMetaEnum>

#include <hardware/core/clock/clock.h>
#include <hardware/core/runtimehardwareconfig.h>
#include <hardware/core/hardwareregistry.h>
#include <data/settings/hardwarekeys.h>

using namespace BC::Key::ClockManager;

ClockManager::ClockManager(QObject *parent) : QObject(parent),
    SettingsStorage(clockManager)
{
    // Clock instances will be provided by HardwareManager via setClocksFromHardwareManager()
    // This matches the uniform hardware lifecycle pattern where HardwareManager owns all hardware
}

void ClockManager::readActiveClocks()
{
    for(auto it = d_clockRoles.begin(); it != d_clockRoles.end(); ++it)
    {
        auto c = it.value();
        if(c->isConnected())
            c->readFrequency(it.key());
    }
}

QHash<RfConfig::ClockType, RfConfig::ClockFreq> ClockManager::getCurrentClocks()
{
    QHash<RfConfig::ClockType, RfConfig::ClockFreq> out;
    for(auto it = d_clockRoles.constBegin(); it != d_clockRoles.constEnd(); ++it)
    {
        auto type = it.key();
        auto clock = it.value();
        RfConfig::ClockFreq f;
        f.output = clock->outputForRole(type);
        f.hwKey = clock->d_key;
        f.factor = clock->multFactor(f.output);
        f.op = RfConfig::Multiply;
        if(f.factor > 0.0 && f.factor < 1.0)
            f.op = RfConfig::Divide;
        f.desiredFreqMHz = readClockFrequency(type);
        out.insert(type,f);
    }

    return out;
}

double ClockManager::setClockFrequency(RfConfig::ClockType t, double freqMHz)
{
    if(!d_clockRoles.contains(t))
    {
        bcWarn(u"No clock configured for use as %1"_s
               .arg(QMetaEnum::fromType<RfConfig::ClockType>().valueToKey(t)));
        return -1.0;
    }

    return d_clockRoles.value(t)->setFrequency(t,freqMHz);
}

double ClockManager::readClockFrequency(RfConfig::ClockType t)
{
    if(!d_clockRoles.contains(t))
    {
        bcWarn(u"No clock configured for use as %1"_s
               .arg(QMetaEnum::fromType<RfConfig::ClockType>().valueToKey(t)));
        return -1.0;
    }

    return d_clockRoles.value(t)->readFrequency(t);
}

bool ClockManager::configureClocks(QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks)
{
    d_clockRoles.clear();
    for(int i=0; i<d_clockList.size(); i++)
        d_clockList[i]->clearRoles();

    for(auto i = clocks.constBegin(); i != clocks.constEnd(); i++)
    {
        auto type = i.key();
        auto d = i.value();

        if(d.hwKey.isEmpty())
            continue;

        //find correct clock
        Clock *c = nullptr;
        for(int j=0; j<d_clockList.size(); j++)
        {
            if(d.hwKey == d_clockList.at(j)->d_key)
            {
                c = d_clockList.at(j);
                break;
            }
        }

        if(c == nullptr)
        {
            bcError(u"Could not find hardware clock for %1 (%2 output %3)"_s
                    .arg(QMetaEnum::fromType<RfConfig::ClockType>().valueToKey(type))
                    .arg(d.hwKey).arg(d.output));
            return false;
        }

        if(!c->addRole(type,d.output))
        {
            bcError(u"The output number requested for %1 (%2) is out of range (only %3 outputs are available)."_s
                    .arg(c->d_key).arg(d.output).arg(c->numOutputs()));
            return false;
        }

        d_clockRoles.insert(type,c);

        double mf = d.factor;
        if(d.op == RfConfig::Divide)
            mf = 1.0/d.factor;

        c->setMultFactor(mf,d.output);

        double actualFreq = c->setFrequency(type,d.desiredFreqMHz);
        if(actualFreq < 0.0)
        {
            bcError(u"Could not set %1 to %2 MHz (raw frequency = %3 MHz)."_s
                    .arg(c->d_key)
                    .arg(d.desiredFreqMHz,0,'f',6)
                    .arg(d.desiredFreqMHz/c->multFactor(d.output),0,'f',6));
            return false;
        }
        if(qAbs(actualFreq-d.desiredFreqMHz) > 0.1)
        {
            bcWarn(u"Actual frequency of %1 (%2 MHz) is more than 100 kHz from desired frequency (%3 MHz)"_s
                   .arg(QMetaEnum::fromType<RfConfig::ClockType>().valueToKey(type))
                   .arg(actualFreq,0,'f',6)
                   .arg(d.desiredFreqMHz,0,'f',6));
        }

        d.desiredFreqMHz = actualFreq;
    }

    return true;
}

bool ClockManager::prepareForExperiment(Experiment &exp)
{
    if(!exp.ftmwEnabled())
        return true;

    if(!configureClocks(exp.ftmwConfig()->d_rfConfig.getClocks()))
        return false;

    exp.ftmwConfig()->d_rfConfig.setCurrentClocks(getCurrentClocks());

    return true;

}

QVector<Clock*> ClockManager::getClockList() const
{
    return d_clockList;
}

void ClockManager::setClocksFromHardwareManager(const QVector<Clock*>& clocks)
{
    // Accept Clock pointers from HardwareManager
    // HardwareManager owns the clocks, ClockManager coordinates their activity
    d_clockList = clocks;
    
    // Set up clocks (role assignments and signal connections)
    setupClocks();
}

void ClockManager::reconfigureFromRuntimeConfig()
{
    // This method would be called by HardwareManager when clocks change
    // For now, just re-setup the existing clocks
    // In future, this could handle dynamic clock reconfiguration
    setupClocks();
}

void ClockManager::setupClocks()
{
    auto ct = QMetaEnum::fromType<RfConfig::ClockType>();

    d_clockRoles.clear();
    setArray(hwClocks,{});

    for(auto c : d_clockList)
    {
        auto names = c->channelNames();
        for(int j=0; j<c->numOutputs(); j++)
        {
            QString pn = c->d_key;
            pn.append(QString(" "));
            if(j < names.size())
                pn.append(names.at(j));
            else
                pn.append(QString("Output %1").arg(j+1));

            appendArrayMap(hwClocks,{
                               {clockKey,c->d_key},
                               {clockOutput,j},
                               {clockName,pn}
                           });
        }

        for(int i=0; i<ct.keyCount(); ++i)
        {
            auto type = static_cast<RfConfig::ClockType>(ct.value(i));
            if(c->hasRole(type))
                d_clockRoles.insert(type,c);
        }
        connect(c,&Clock::frequencyUpdate,this,&ClockManager::clockFrequencyUpdate);
    }
    save();
}
