#include "synthesizer.h"

Synthesizer::Synthesizer(QObject *parent)
 : HardwareObject(parent), d_minFreq(0.0), d_maxFreq(0.0)
{
    d_key = QString("synthesizer");
}

Synthesizer::~Synthesizer()
{

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
