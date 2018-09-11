#include "clockmanager.h"

#include <QSettings>

#include "clock.h"


ClockManager::ClockManager(QObject *parent) : QObject(parent)
{
    d_clockTypes = BlackChirp::allClockTypes();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("clockManager"));
    d_currentBand = s.value(QString("currentBand"),0).toInt();
    s.endGroup();

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

    readClockRoles();

    for(int i=0; i<d_clockList.size(); i++)
    {
        auto c = d_clockList.at(i);
        connect(c,&Clock::frequencyUpdate,this,&ClockManager::clockFrequencyUpdate);
    }


}

double ClockManager::setClockFrequency(BlackChirp::ClockType t, double freqMHz)
{
    if(!d_clockRoles.contains(d_clockTypes.indexOf(t)))
    {
        emit logMessage(QString("No clock configured for use as %1").arg(BlackChirp::clockPrettyName(t)),BlackChirp::LogWarning);
        return -1.0;
    }

    return d_clockRoles.value(d_clockTypes.indexOf(t))->setFrequency(t,freqMHz);
}

double ClockManager::readClockFrequency(BlackChirp::ClockType t)
{
    if(!d_clockRoles.contains(d_clockTypes.indexOf(t)))
    {
        emit logMessage(QString("No clock configured for use as %1").arg(BlackChirp::clockPrettyName(t)),BlackChirp::LogWarning);
        return -1.0;
    }

    return d_clockRoles.value(d_clockTypes.indexOf(t))->readFrequency(t);
}

void ClockManager::readClockRoles()
{
    d_clockRoles.clear();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("clockManager"));
    s.setValue(QString("currentBand"),d_currentBand);
    s.beginReadArray(QString("bands"));
    s.setArrayIndex(d_currentBand);
    s.beginReadArray(QString("clocks"));
    for(int index=0; index<d_clockList.size(); index++)
    {
        s.setArrayIndex(index);

        auto c = d_clockList.at(index);
        c->clearRoles();
        int outputs = c->numOutputs();
        s.beginReadArray(QString("outputs"));
        for(int i=0; i<outputs; i++)
        {
            s.setArrayIndex(i);
            double mult = s.value(QString("multFactor"),1.0).toDouble();
            c->setMultFactor(mult,i);
            auto role = static_cast<BlackChirp::ClockType>(s.value(QString("type"),QVariant(BlackChirp::UpConversionLO)).toInt());
            if(!d_clockRoles.contains(d_clockTypes.indexOf(role)))
            {
                d_clockRoles.insert(d_clockTypes.indexOf(role),c);
                c->setRole(role,i);
            }
        }
        s.endArray();
    }
    s.endArray();
    s.endArray();
    s.endGroup();

}

void ClockManager::setBand(int band)
{
    //Note, this function is intended to be called during experiment initialization.
    //The individual clock frequencies will be changed directly by
    //the HardwareManager
    ///TODO: figure out how to communicate with IO Board or something
    d_currentBand = band;
    readClockRoles();
}
