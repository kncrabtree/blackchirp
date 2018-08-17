#include "clockmanager.h"

#include <QSettings>

#include "clock.h"

ClockManager::ClockManager(QObject *parent) : QObject(parent)
{


    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("clocks"));
    s.endGroup();


}

double ClockManager::setClockFrequency(BlackChirp::ClockType t, double freqMHz)
{

}

double ClockManager::readClockFrequency(BlackChirp::ClockType t)
{

}

void ClockManager::readClockRoles()
{

}
