#include "synthesizer.h"

Synthesizer::Synthesizer(QObject *parent)
 : HardwareObject(parent), d_minFreq(0.0), d_maxFreq(0.0)
{
    d_key = QString("synthesizer");
}

Synthesizer::~Synthesizer()
{

}

double Synthesizer::readTxFreq()
{
    double tx = readSynthTxFreq();
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.setValue(QString("txFreq"),tx);
    s.endGroup();
    s.endGroup();

    emit txFreqRead(tx);
    return tx;
}

double Synthesizer::readRxFreq()
{
    double rx = readSynthRxFreq();
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.setValue(QString("rxFreq"),rx);
    s.endGroup();
    s.endGroup();

    emit rxFreqRead(rx);
    return rx;
}

double Synthesizer::setTxFreq(const double f)
{
    if(f < d_minFreq || f > d_maxFreq)
    {
        emit hardwareFailure();
        emit logMessage(QString("The requested TX frequency (%1) is outside the valid range (%2 - %3).")
                        .arg(f).arg(d_minFreq).arg(d_maxFreq),BlackChirp::LogError);
        return -1.0;
    }

    return setSynthTxFreq(f);
}

double Synthesizer::setRxFreq(const double f)
{
    if(f < d_minFreq || f > d_maxFreq)
    {
        emit hardwareFailure();
        emit logMessage(QString("The requested RX frequency (%1) is outside the valid range (%2 - %3).")
                        .arg(f).arg(d_minFreq).arg(d_maxFreq),BlackChirp::LogError);
        return -1.0;
    }

    return setSynthRxFreq(f);
}
