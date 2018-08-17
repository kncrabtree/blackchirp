#include "clockmanager.h"

#include <QSettings>

#include "clock.h"


ClockManager::ClockManager(QObject *parent) : QObject(parent)
{

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

}

double ClockManager::setClockFrequency(BlackChirp::ClockType t, double freqMHz)
{
    if(!d_usedClockTypes.contains(t))
    {
        emit logMessage(QString("No clock configured for use as %1").arg(BlackChirp::clockPrettyName(t)),BlackChirp::LogWarning);
        return -1.0;
    }
}

double ClockManager::readClockFrequency(BlackChirp::ClockType t)
{

}

void ClockManager::readClockRoles()
{
    d_usedClockTypes.clear();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("clocks"));
    for(int index=0; index<d_clockList.size(); index++)
    {
        s.setArrayIndex(index);

        int outputs = c->numOutputs();
        s.setValue(QString("subKey"),c->subKey());
        int no = s.beginReadArray(QString("outputs"));
        for(int i=0; i<outputs && i < no; i++)
        {
            s.setArrayIndex(i);
            auto role = static_cast<BlackChirp::ClockType>(s.value(QString("type"),QVariant(BlackChirp::UpConversionLO)).toInt());
            if(!d_usedClockTypes.contains(role))
                c->setRole(role,i);
        }
        s.endArray();
        s.endArray();
    }
    s.endGroup();

}
