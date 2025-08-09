#include "clockmanager.h"

#include <QMetaEnum>

#include "clock_h.h"
#include <hardware/core/clock/clock.h>
#include <hardware/core/clock/fixedclock.h>
#include <hardware/core/runtimehardwareconfig.h>
#include <hardware/core/hardwareregistry.h>
#include <data/settings/hardwarekeys.h>

using namespace BC::Key::ClockManager;

ClockManager::ClockManager(QObject *parent) : QObject(parent),
    SettingsStorage(clockManager)
{
    // Phase 2.4.5: Create virtual clocks for capability discovery (matches other hardware pattern)
    // This replaces the compile-time Boost.Preprocessor macro system
    
    // Create 2 FixedClock instances with "temp" labels for discovery
    // (matches the default BC_CLOCKS "fixed;fixed" configuration)
    auto clock1 = new FixedClock("temp");
    auto clock2 = new FixedClock("temp"); 
    d_clockList << clock1 << clock2;
    
    // Set up clocks (role assignments and signal connections)
    setupClocks();
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
        emit logMessage(QString("No clock configured for use as %1")
                        .arg(QMetaEnum::fromType<RfConfig::ClockType>().valueToKey(t)),
                        LogHandler::Warning);
        return -1.0;
    }

    return d_clockRoles.value(t)->setFrequency(t,freqMHz);
}

double ClockManager::readClockFrequency(RfConfig::ClockType t)
{
    if(!d_clockRoles.contains(t))
    {
        emit logMessage(QString("No clock configured for use as %1")
                        .arg(QMetaEnum::fromType<RfConfig::ClockType>().valueToKey(t)),
                        LogHandler::Warning);
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
           emit logMessage(QString("Could not find hardware clock for %1 (%2 output %3)")
                               .arg(QMetaEnum::fromType<RfConfig::ClockType>()
                                    .valueToKey(type))
                                    .arg(d.hwKey).arg(d.output),LogHandler::Error);
            return false;
        }

        if(!c->addRole(type,d.output))
        {
            emit logMessage(QString("The output number requested for %1 (%2) is out of range (only %2 outputs are available).")
                               .arg(c->d_name).arg(d.output).arg(c->numOutputs()),LogHandler::Error);
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
            emit logMessage(QString("Could not set %1 to %2 MHz (raw frequency = %3 MHz).")
                               .arg(c->d_name)
                               .arg(d.desiredFreqMHz,0,'f',6)
                               .arg(d.desiredFreqMHz/c->multFactor(d.output),0,'f',6),LogHandler::Error);
            return false;
        }
        if(qAbs(actualFreq-d.desiredFreqMHz) > 0.1)
        {
            emit logMessage(QString("Actual frequency of %1 (%2 MHz) is more than 100 kHz from desired frequency (%3 MHz)")
                            .arg(QMetaEnum::fromType<RfConfig::ClockType>().valueToKey(type))
                            .arg(actualFreq,0,'f',6)
                            .arg(d.desiredFreqMHz,0,'f',6),LogHandler::Warning);
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

//    d_clockRoles.clear();
//    for(int i=0; i<d_clockList.size(); i++)
//        d_clockList[i]->clearRoles();

//    auto map = exp.ftmwConfig()->d_rfConfig.getClocks();
//    for(auto i = map.constBegin(); i != map.constEnd(); i++)
//    {
//        auto type = i.key();
//        auto d = i.value();

//        if(d.hwKey.isEmpty())
//            continue;

//        //find correct clock
//        Clock *c = nullptr;
//        for(int j=0; j<d_clockList.size(); j++)
//        {
//            if(d.hwKey == d_clockList.at(j)->d_key)
//            {
//                c = d_clockList.at(j);
//                break;
//            }
//        }

//        if(c == nullptr)
//        {
//            exp.d_errorString = QString("Could not find hardware clock for %1 (%2 output %3)")
//                               .arg(QMetaEnum::fromType<RfConfig::ClockType>()
//                                    .valueToKey(type))
//                                    .arg(d.hwKey).arg(d.output);
//            return false;
//        }

//        if(!c->addRole(type,d.output))
//        {
//            exp.d_errorString = QString("The output number requested for %1 (%2) is out of range (only %2 outputs are available).")
//                               .arg(c->d_name).arg(d.output).arg(c->numOutputs());
//            return false;
//        }

//        d_clockRoles.insertMulti(type,c);

//        double mf = d.factor;
//        if(d.op == RfConfig::Divide)
//            mf = 1.0/d.factor;

//        c->setMultFactor(mf,d.output);

//        double actualFreq = c->setFrequency(type,d.desiredFreqMHz);
//        if(actualFreq < 0.0)
//        {
//            exp.d_errorString = QString("Could not set %1 to %2 MHz (raw frequency = %3 MHz).")
//                               .arg(c->d_name)
//                               .arg(d.desiredFreqMHz,0,'f',6)
//                               .arg(exp.ftmwConfig()->d_rfConfig.rawClockFrequency(type),0,'f',6);
//            return false;
//        }
//        if(qAbs(actualFreq-d.desiredFreqMHz) > 0.1)
//        {
//            emit logMessage(QString("Actual frequency of %1 (%2 MHz) is more than 100 kHz from desired frequency (%3 MHz)")
//                            .arg(QMetaEnum::fromType<RfConfig::ClockType>().valueToKey(type))
//                            .arg(actualFreq,0,'f',6)
//                            .arg(d.desiredFreqMHz,0,'f',6));
//        }

//        d.desiredFreqMHz = actualFreq;

//        exp.ftmwConfig()->d_rfConfig.setClockFreqInfo(type,d);
//    }

    exp.ftmwConfig()->d_rfConfig.setCurrentClocks(getCurrentClocks());

    return true;

}

QVector<Clock*> ClockManager::getClockList() const
{
    return d_clockList;
}

void ClockManager::createClocksFromRuntimeConfig()
{
    // Get active Clock configurations from RuntimeHardwareConfig
    const auto& config = RuntimeHardwareConfig::constInstance();
    auto activeClockKeys = config.getActiveKeys<Clock>();
    
    // Clear any existing clocks
    d_clockList.clear();
    
    // Create clocks based on RuntimeHardwareConfig only
    // RuntimeHardwareConfig is the authoritative source - no fallback behavior
    for (const QString& clockKey : activeClockKeys) {
        auto [type, label] = BC::Key::parseKey(clockKey);
        QString implementation = config.getHardwareImplementation<Clock>(label);
        
        // Create clock using HardwareRegistry
        HardwareObject* hwObj = HardwareRegistry::instance().createHardware(type, implementation, label);
        Clock* clock = qobject_cast<Clock*>(hwObj);
        
        if (clock) {
            clock->setParent(this);
            d_clockList << clock;
            emit logMessage(QString("Created clock: %1 (implementation: %2)").arg(label).arg(implementation));
        } else {
            emit logMessage(QString("Failed to create clock: type=%1, implementation=%2, label=%3")
                           .arg(type).arg(implementation).arg(label), LogHandler::Error);
            // Clean up failed creation
            if (hwObj) {
                hwObj->deleteLater();
            }
        }
    }
    
    // Set up clocks (role assignments and signal connections)
    setupClocks();
}

void ClockManager::setupClocks()
{
    auto ct = QMetaEnum::fromType<RfConfig::ClockType>();

    setArray(hwClocks,{});

    for(auto c : d_clockList)
    {
        auto names = c->channelNames();
        for(int j=0; j<c->numOutputs(); j++)
        {
            QString pn = c->d_name;
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
