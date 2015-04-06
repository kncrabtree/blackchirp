#include "chirpconfig.h"
#include <QSettings>
#include <QApplication>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_const.h>
#include <QList>

ChirpConfig::ChirpConfig() : data(new ChirpConfigData)
{

}

ChirpConfig::ChirpConfig(const ChirpConfig &rhs) : data(rhs.data)
{

}

ChirpConfig &ChirpConfig::operator=(const ChirpConfig &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

ChirpConfig::~ChirpConfig()
{

}

bool ChirpConfig::isValid() const
{
    return data->isValid;
}

double ChirpConfig::preChirpDelay() const
{
    return data->preChirpDelay;
}

double ChirpConfig::protectionDelay() const
{
    return data->protectionDelay;
}

int ChirpConfig::numChirps() const
{
    return data->numChirps;
}

double ChirpConfig::chirpInterval() const
{
    return data->chirpInterval;
}

double ChirpConfig::chirpDuration() const
{
    double out = 0.0;
    for(int i=0; i<data->segments.size(); i++)
        out += data->segments.at(i).durationUs;
    return out;
}

double ChirpConfig::totalDuration() const
{
    if(numChirps() > 1)
        return (static_cast<double>(data->numChirps)-1.0)*data->chirpInterval + data->preChirpDelay + chirpDuration() + data->protectionDelay;
    else
        return data->preChirpDelay + chirpDuration() + data->protectionDelay;
}

QList<ChirpConfig::ChirpSegment> ChirpConfig::segmentList() const
{
    return data->segments;
}

QVector<QPointF> ChirpConfig::getChirpMicroseconds() const
{
    return getChirpSegmentMicroSeconds(0.0,totalDuration());
}

QVector<QPointF> ChirpConfig::getChirpSegmentMicroSeconds(double t1, double t2) const
{
    //this function is to be called from the ChirpConfigPlot
    //t1 and t2 are the starting and ending time [t1,t2).
    //y values are ADC levels (10 bit)
    //x values are in microseconds
    if(!isValid())
        return QVector<QPointF>();

    int firstSample = getFirstSample(t1);
    int invalidSample = getFirstSample(t2); //invalid sample is the point AFTER the last point to be included

    //the actual number of samples may be different, if the requested t2 is after the total chirp duration
    int numSamples = qMin(invalidSample-firstSample - 1,
                          getLastSample(totalDuration())-firstSample);
    int currentSample = 0;

    QVector<QPointF> out(numSamples);

    //find where the first sample falls with respect to delays and chirps
    //find interval number

    bool done = false;

    while(!done) // loop that allows interval crossing
    {
        double currentTime = static_cast<double>(firstSample+currentSample)*data->sampleIntervalUS;
        int currentInterval = static_cast<int>(floor(currentTime/data->chirpInterval));
        double currentIntervalStartSample = getFirstSample(static_cast<double>(currentInterval)*data->chirpInterval);
        double nextIntervalStartSample = getFirstSample((static_cast<double>(currentInterval) + 1.0) * data->chirpInterval);
        if(nextIntervalStartSample > firstSample+numSamples)
        {
            //this is the last interval
            done = true;
        }
        int currentIntervalChirpStart = getFirstSample(static_cast<double>(currentIntervalStartSample)*data->sampleIntervalUS + data->preChirpDelay);
        int currentIntervalChirpEnd = getLastSample(static_cast<double>(currentIntervalChirpStart)*data->sampleIntervalUS + chirpDuration());
        int currentIntervalProtectionEnd = getLastSample(static_cast<double>(currentIntervalChirpEnd)*data->sampleIntervalUS + data->protectionDelay);

        //start times for each segment
        QList<int> segmentStarts;
        segmentStarts.append(currentIntervalChirpStart);
        for(int i=1; i<data->segments.size(); i++)
            segmentStarts.append(getFirstSample(static_cast<double>(segmentStarts.at(i-1))*data->sampleIntervalUS + data->segments.at(i-1).durationUs));

        //starting phase for each segment
        //want to transition between frequencies as smoothly as possible, so try to start each segment at the phase at which the previous
        //segment would have been if it were to continue
        QList<double> segmentPhasesRadians;
        segmentPhasesRadians.append(0.0);
        for(int i=1; i<data->segments.size(); i++)
            segmentPhasesRadians.append(GSL_REAL(gsl_complex_arcsin_real(calculateChirp(data->segments.at(i-1),data->segments.at(i-1).durationUs,segmentPhasesRadians.at(i-1)))));

        //determine current segment number
        //-1 means before chirp
        //0 - data->segments.size()-1 means during a segement
        //data->segments.size() means protection interval
        //data->segment.size() + 1 means waiting for next chirp

        int currentSegment = -1;
        int nextSectionSample = firstSample+currentSample + getFirstSample(data->preChirpDelay);
        if(currentSample >= currentIntervalChirpStart)
        {
            if(currentSample >= currentIntervalChirpEnd)
            {
                if(currentSample >= currentIntervalProtectionEnd)
                {
                    currentSegment = data->segments.size() + 1;
                    nextSectionSample = nextIntervalStartSample;
                }
                else
                {
                    currentSegment = data->segments.size();
                    nextSectionSample = currentIntervalProtectionEnd+1;
                }
            }
            currentSegment = 0;
            while(currentSegment + 1 < data->segments.size())
            {
                if(currentSample < segmentStarts.at(currentSegment+1))
                    break;

                currentSegment++;
                nextSectionSample += getFirstSample(data->segments.at(currentSegment).durationUs);
            }

        }

        //loop that increments time and calculates points
        while(currentSample < numSamples && currentSample < nextIntervalStartSample)
        {
            if(currentSegment < 0 || currentSegment >= data->segments.size())
                out[currentSample] = QPointF(currentTime,0.0);
            else if(currentSegment < data->segments.size())
                out[currentSample] = QPointF(currentTime,calculateChirp(data->segments.at(currentSegment),currentTime-segmentStarts.at(currentSegment)*data->sampleIntervalUS,segmentPhasesRadians.at(currentSegment)));

            currentSample++;
            currentTime = static_cast<double>(firstSample+currentSample)*data->sampleIntervalUS;

            if(currentSample >= numSamples)
                done = true;

            if(currentSample >= nextSectionSample)
            {
                currentSegment++;
                if(currentSample >= currentIntervalChirpEnd)
                {
                    if(currentSample >= currentIntervalProtectionEnd)
                        nextSectionSample = nextIntervalStartSample;
                    else
                        nextSectionSample = currentIntervalProtectionEnd+1;
                }
                else
                    nextSectionSample += getFirstSample(data->segments.at(currentSegment).durationUs);
            }
        }
    }


    return out;
}

bool ChirpConfig::validate()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("chirpConfig"));
    double minTwt = s.value(QString("minPreChirpDelay"),0.300).toDouble();
    double minProt = s.value(QString("minProtectionDelay"),0.300).toDouble();
    s.endGroup();

    double awgRate = s.value(QString("awg/sampleRate"),16e9).toDouble();
    double awgMaxSamples = s.value(QString("awg/maxSamples"),2e9).toDouble();

    data->sampleRateSperS = awgRate;
    data->sampleRateSperUS = awgRate/1e6;
    data->sampleIntervalS = 1.0/awgRate;
    data->sampleIntervalUS = 1.0/awgRate*1e6;

    //make sure all settings are possible
    if(data->preChirpDelay < minTwt)
        return false;

    if(data->protectionDelay < minProt)
        return false;

    if(data->numChirps < 1)
        return false;

    if(data->segments.isEmpty())
        return false;

    if(data->numChirps > 0 && data->chirpInterval < data->preChirpDelay + chirpDuration() + data->protectionDelay + 4.0)
        return false;

    if(totalDuration() >= awgMaxSamples/awgRate*1e6)
        return false;

    data->isValid = true;
    return true;
}

void ChirpConfig::setPreChirpDelay(const double d)
{
    data->preChirpDelay = d;
}

void ChirpConfig::setProtectionDelay(const double d)
{
    data->protectionDelay = d;
}

void ChirpConfig::setNumChirps(const int n)
{
    data->numChirps = n;
}

void ChirpConfig::setChirpInterval(const double i)
{
    data->chirpInterval = i;
}

void ChirpConfig::setSegmentList(const QList<ChirpConfig::ChirpSegment> l)
{
    data->segments = l;
}

int ChirpConfig::getFirstSample(double time) const
{
    //first sample is inclusive
    if(!data->isValid)
        return -1;

    double nearestSampleTime = round(data->sampleRateSperUS*time)*data->sampleIntervalUS;
    if(qFuzzyCompare(1.0 + time, 1.0 + nearestSampleTime))
        return static_cast<int>(round(data->sampleRateSperUS*time));
    else
        return static_cast<int>(ceil(data->sampleRateSperUS*time));
}

int ChirpConfig::getLastSample(double time) const
{
    //last sample is non-inclusive
    if(!data->isValid)
        return -1.0;

    double nearestSampleTime = round(data->sampleRateSperUS*time)*data->sampleIntervalUS;
    if(qFuzzyCompare(1.0 + time, 1.0 + nearestSampleTime))
        return static_cast<int>(round(data->sampleRateSperUS*time));
    else
        return static_cast<int>(floor(data->sampleRateSperUS*time));
}

double ChirpConfig::calculateChirp(const ChirpConfig::ChirpSegment segment, const double t, const double phase) const
{
    return gsl_sf_sin(gsl_sf_angle_restrict_pos(2.0*M_PI*(segment.startFreqMHz + 0.5*segment.alphaUs*t)*t + phase));
}

