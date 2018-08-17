#include "fixedclock.h"

FixedClock::FixedClock(int clockNum, QObject *parent) : Clock(clockNum, parent)
{
    d_subKey = QString("fixed");
    d_prettyName = QString("Fixed Clock (#%1)").arg(clockNum);
    d_threaded = false;
    d_commType = CommunicationProtocol::None;
    d_numOutputs = 1;
    d_isTunable = false;

    d_minFreqMHz = 0.0;
    d_maxFreqMHz = 1e20;
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
    d_currentFrequency = s.value(QString("frequencyMHz"),0.0).toDouble();
    s.setValue(QString("frequencyMHz"),0.0);
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
    Q_UNUSED(outputIndex)
    d_currentFrequency = freqMHz;
    return true;
}

double FixedClock::readHwFrequency(int outputIndex)
{
    Q_UNUSED(outputIndex)
    return d_currentFrequency;
}
