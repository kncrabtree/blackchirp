#include "fixedclock.h"

FixedClock::FixedClock(int clockNum, QObject *parent) : Clock(clockNum, BC::Key::fixed,BC::Key::fixedName.arg(clockNum),
                                                              CommunicationProtocol::None,parent)
{
    d_numOutputs = 5;
    d_isTunable = false;

    for(int i=0; i<d_numOutputs; i++)
        d_currentFrequencyList << 0.0;
}

void FixedClock::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    d_minFreqMHz = s.value(QString("minFreqMHz"),0.0).toDouble();
    d_maxFreqMHz = s.value(QString("maxFreqMHz"),1e7).toDouble();
    s.setValue(QString("minFreqMHz"),d_minFreqMHz);
    s.setValue(QString("maxFreqMHz"),d_maxFreqMHz);
    s.endGroup();
    s.endGroup();
}


bool FixedClock::testConnection()
{
    return true;
}

void FixedClock::initializeClock()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.beginReadArray(QString("outputs"));
    for(int i=0; i<d_numOutputs; i++)
    {
        s.setArrayIndex(i);
        d_currentFrequencyList[i] = s.value(QString("frequencyMHz"),0.0).toDouble();
    }
    s.endArray();
    s.beginWriteArray(QString("outputs"));
    for(int i=0; i<d_numOutputs; i++)
    {
        s.setArrayIndex(i);
        s.setValue(QString("frequencyMHz"),d_currentFrequencyList.at(i));
    }
    s.endArray();
    s.endGroup();
    s.endGroup();
}

bool FixedClock::setHwFrequency(double freqMHz, int outputIndex)
{
    d_currentFrequencyList[outputIndex] = freqMHz;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.beginWriteArray(QString("outputs"));
    s.setArrayIndex(outputIndex);
    s.setValue(QString("frequencyMHz"),d_currentFrequencyList.at(outputIndex));
    s.endArray();
    s.endGroup();
    s.endGroup();
    s.sync();

    return true;
}

double FixedClock::readHwFrequency(int outputIndex)
{
    return d_currentFrequencyList.at(outputIndex);
}
