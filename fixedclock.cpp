#include "fixedclock.h"

FixedClock::FixedClock(int clockNum, QObject *parent) : Clock(clockNum, parent)
{
    d_subKey = QString("fixed");
    d_prettyName = QString("Fixed Clock (#%1)").arg(clockNum);
    d_threaded = false;
    d_commType = CommunicationProtocol::None;
    d_numOutputs = 5;
    d_isTunable = false;

    d_minFreqMHz = 0.0;
    d_maxFreqMHz = 1e20;

    for(int i=0; i<d_numOutputs; i++)
        d_currentFrequencyList << 0.0;

    Clock::prepareMultFactors();
}


bool FixedClock::testConnection()
{
    emit connected();
    return true;
}

void FixedClock::initialize()
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

Experiment FixedClock::prepareForExperiment(Experiment exp)
{
    return exp;
}

void FixedClock::beginAcquisition()
{
}

void FixedClock::endAcquisition()
{
}

void FixedClock::readTimeData()
{
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
