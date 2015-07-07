#include "synthesizer.h"

Synthesizer::Synthesizer(QObject *parent)
 : HardwareObject(parent), d_minFreq(0.0), d_maxFreq(0.0)
{
    d_key = QString("synthesizer");
}

Synthesizer::~Synthesizer()
{

}

void Synthesizer::initialize()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    //allow hardware limits to be made in settings
    d_minFreq = s.value(QString("minFreq"),500.0).toDouble();
    d_maxFreq = s.value(QString("maxFreq"),6000.0).toDouble();
    //write the settings if they're not there
    s.setValue(QString("minFreq"),d_minFreq);
    s.setValue(QString("maxFreq"),d_maxFreq);
    s.endGroup();
    s.endGroup();
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
