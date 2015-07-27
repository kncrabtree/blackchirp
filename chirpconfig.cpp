#include "chirpconfig.h"

#include <QSettings>
#include <QApplication>
#include <QList>
#include <QCryptographicHash>
#include <QFile>

#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_const.h>

ChirpConfig::ChirpConfig() : data(new ChirpConfigData)
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("synthesizer"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    data->valonTxFreq = s.value(QString("txFreq"),5760.0).toDouble();
    s.endGroup();
    s.endGroup();

    s.beginGroup(QString("chirpConfig"));
    data->awgMult = s.value(QString("awgMult"),1.0).toDouble();
    data->valonTxMult = s.value(QString("txValonMult"),2.0).toDouble();
    data->totalMult = s.value(QString("txMult"),4.0).toDouble();
    data->mixerSideband = s.value(QString("txSidebandSign"),-1.0).toDouble();
    s.endGroup();

    //move this code to widget that configures TX chain...
//    s.beginGroup(QString("awg"));
//    double awgMin = s.value(QString("minFreq"),100.0).toDouble();
//    double awgMax = s.value(QString("maxFreq"),6250.0).toDouble();
//    s.endGroup();

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

ChirpConfig::ChirpConfig(int num, QString path) : data(new ChirpConfigData)
{
    QFile f(BlackChirp::getExptFile(num,BlackChirp::ChirpFile,path));
    if(!f.open(QIODevice::ReadOnly))
        return;

    while(!f.atEnd())
        parseFileLine(f.readLine().trimmed());

    validate();

    f.close();
}

ChirpConfig::~ChirpConfig()
{

}

bool ChirpConfig::compareTxParams(const ChirpConfig &other) const
{
    if(!qFuzzyCompare(awgMult(),other.awgMult()))
        return false;
    if(!qFuzzyCompare(synthTxMult(),other.synthTxMult()))
        return false;
    if(!qFuzzyCompare(mixerSideband(),other.mixerSideband()))
        return false;
    if(!qFuzzyCompare(totalMult(),other.totalMult()))
        return false;

    return true;
}

bool ChirpConfig::isValid() const
{
    return data->isValid;
}

double ChirpConfig::preChirpProtection() const
{
    return data->preChirpProtection;
}

double ChirpConfig::preChirpDelay() const
{
    return data->preChirpDelay;
}

double ChirpConfig::postChirpProtection() const
{
    return data->postChirpProtection;
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
        return (static_cast<double>(data->numChirps)-1.0)*data->chirpInterval + data->preChirpProtection + data->preChirpDelay + chirpDuration() + data->postChirpProtection;
    else
        return data->preChirpProtection + data->preChirpDelay + chirpDuration() + data->postChirpProtection;
}

QList<BlackChirp::ChirpSegment> ChirpConfig::segmentList() const
{
    return data->segments;
}

double ChirpConfig::segmentStartFreq(int i) const
{
    if(i < 0 || i >= data->segments.size())
        return -1.0;

    return awgToRealFreq(data->segments.at(i).startFreqMHz);
}

double ChirpConfig::segmentEndFreq(int i) const
{
    if(i < 0 || i >= data->segments.size())
        return -1.0;

    return awgToRealFreq(data->segments.at(i).endFreqMHz);
}

double ChirpConfig::segmentDuration(int i) const
{
    if(i < 0 || i >= data->segments.size())
        return -1.0;

    return data->segments.at(i).durationUs;
}

QByteArray ChirpConfig::waveformHash() const
{
    QCryptographicHash c(QCryptographicHash::Sha256);
    c.addData(QByteArray::number(data->preChirpProtection));
    c.addData(QByteArray::number(data->preChirpDelay));
    for(int i=0; i<data->segments.size(); i++)
    {
        c.addData(QByteArray::number(data->segments.at(i).startFreqMHz));
        c.addData(QByteArray::number(data->segments.at(i).endFreqMHz));
        c.addData(QByteArray::number(data->segments.at(i).durationUs));
    }
    c.addData(QByteArray::number(data->numChirps));
    c.addData(QByteArray::number(data->chirpInterval));
    c.addData(QByteArray::number(data->postChirpProtection));

    return c.result();
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
    if(qFuzzyCompare(t2,totalDuration()))
        invalidSample += 1;

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
        double currentTime = getSampleTime(firstSample+currentSample);
        int currentInterval = static_cast<int>(floor(currentTime/data->chirpInterval));
        int currentIntervalStartSample = getFirstSample(static_cast<double>(currentInterval)*data->chirpInterval);
        int nextIntervalStartSample = getFirstSample((static_cast<double>(currentInterval) + 1.0) * data->chirpInterval);
        if(nextIntervalStartSample > firstSample+numSamples)
        {
            //this is the last interval
            done = true;
        }
        int currentIntervalChirpStart = getFirstSample(getSampleTime(currentIntervalStartSample) + data->preChirpProtection + data->preChirpDelay);
        int currentIntervalChirpEnd = getLastSample(getSampleTime(currentIntervalChirpStart) + chirpDuration());

        //start times for each segment
        QList<int> segmentStarts;
        segmentStarts.append(currentIntervalChirpStart);
        for(int i=1; i<data->segments.size(); i++)
            segmentStarts.append(getFirstSample(getSampleTime(segmentStarts.at(i-1)) + data->segments.at(i-1).durationUs));

        //starting phase for each segment
        //want to transition between frequencies as smoothly as possible, so try to start each segment at the phase at which the previous
        //segment would have been if it were to continue
        QList<double> segmentPhasesRadians;
        segmentPhasesRadians.append(0.0);
        for(int i=1; i<data->segments.size(); i++)
        {
            double segmentEndTime = getSampleTime(segmentStarts.at(i)-segmentStarts.at(i-1));
            segmentPhasesRadians.append(calculateEndingPhaseRadians(data->segments.at(i-1),segmentEndTime,segmentPhasesRadians.at(i-1)));
        }

        //determine current segment number
        //-1 means before chirp
        //0 - data->segments.size()-1 means during a segement
        //data->segments.size() means protection interval
        //data->segment.size() + 1 means waiting for next chirp
        int currentSegment = 0;
        int nextSegmentSample = firstSample+currentSample + getFirstSample(data->preChirpProtection + data->preChirpDelay + data->segments.at(0).durationUs);
        if(data->segments.size() == 1)
            nextSegmentSample = currentIntervalChirpEnd;
        else
        {
            while(currentSegment + 1 < data->segments.size())
            {
                if(currentSample < segmentStarts.at(currentSegment+1))
                    break;

                currentSegment++;
                nextSegmentSample += getFirstSample(data->segments.at(currentSegment).durationUs);
            }
        }

        //loop that allows section/segment crossing
        while(currentSample < nextIntervalStartSample)
        {
            //loop that increments time and calculates points
            while(currentSample < numSamples && currentSample < nextSegmentSample)
            {
                if(currentSample < currentIntervalChirpStart || currentSample >= currentIntervalChirpEnd)
                    out[currentSample] = QPointF(currentTime,0.0);
                else
                    out[currentSample] = QPointF(currentTime,calculateChirp(data->segments.at(currentSegment),currentTime-getSampleTime(segmentStarts.at(currentSegment)),segmentPhasesRadians.at(currentSegment)));

                currentSample++;
                currentTime = getSampleTime(firstSample+currentSample);


            }

            if(currentSample >= numSamples)
            {
                done = true;
                break;
            }

            currentSegment++;
            if(currentSegment >= data->segments.size())
                nextSegmentSample = nextIntervalStartSample;
            else
                nextSegmentSample += getFirstSample(data->segments.at(currentSegment).durationUs);
        }
    }


    return out;
}

QVector<QPair<bool, bool> > ChirpConfig::getMarkerData() const
{
    if(!isValid())
        return QVector<QPair<bool,bool>>();

    int currentSample = 0;
    int firstSample = 0;
    int invalidSample = getFirstSample(totalDuration()) + 1; //invalid sample is the point AFTER the last point to be included
    int numSamples = invalidSample - 1;

    QVector<QPair<bool,bool>> out(numSamples);

    bool done = false;

    while(!done) // loop that allows interval crossing
    {
        //starting an interval
        double currentTime = getSampleTime(firstSample+currentSample);
        int currentInterval = static_cast<int>(floor(currentTime/data->chirpInterval));
        int currentIntervalStartSample = getFirstSample(static_cast<double>(currentInterval)*data->chirpInterval);
        int nextIntervalStartSample = getFirstSample((static_cast<double>(currentInterval) + 1.0) * data->chirpInterval);
        if(nextIntervalStartSample > firstSample+numSamples)
        {
            //this is the last interval
            done = true;
            nextIntervalStartSample = firstSample+numSamples;
        }
        int currentIntervalChirpStart = getFirstSample(getSampleTime(currentIntervalStartSample) + data->preChirpProtection + data->preChirpDelay);
        int currentIntervalChirpEnd = getLastSample(getSampleTime(currentIntervalChirpStart) + chirpDuration());
        int currentIntervalProtEnd = getLastSample(getSampleTime(currentIntervalChirpEnd) + data->postChirpProtection)-1;
        int currentIntervalTwtStart = getFirstSample(getSampleTime(currentIntervalStartSample) + data->preChirpProtection);
        //note: twt ends at the same time as the chirp

        //sections will correspond to marker states:
        //0 = prot high, twt low
        //1 = both high
        //2 = prot high, twt low
        //3 = both low

        bool prot = true;
        bool twt = false;

        while(currentSample < nextIntervalStartSample)
        {
            //assess section state
            if(currentSample == currentIntervalTwtStart)
                twt = true;
            if(currentSample == currentIntervalChirpEnd)
                twt = false;
            if(currentSample == currentIntervalProtEnd)
                prot = false;

            out[currentSample] = qMakePair(prot,twt);

            currentSample++;
        }


    }

    return out;

}

QMap<QString, QPair<QVariant, QString> > ChirpConfig::headerMap() const
{
    QMap<QString, QPair<QVariant,QString>> out;

    out.insert(QString("ChirpConfigPreChirpProtection"),qMakePair(QString::number(data->preChirpProtection,'f',3),QString::fromUtf16(u"μs")));
    out.insert(QString("ChirpConfigPreChirpDelay"),qMakePair(QString::number(data->preChirpDelay,'f',3),QString::fromUtf16(u"μs")));
    out.insert(QString("ChirpConfigPostChirpProtection"),qMakePair(QString::number(data->postChirpProtection,'f',3),QString::fromUtf16(u"μs")));
    out.insert(QString("ChirpConfigNumChirps"),qMakePair(data->numChirps,QString("")));
    out.insert(QString("ChirpConfigChirpInterval"),qMakePair(QString::number(data->chirpInterval,'f',3),QString::fromUtf16(u"μs")));
    out.insert(QString("ChirpConfigValonTxMult"),qMakePair(QString::number(data->valonTxMult,'f',1),QString("")));
    out.insert(QString("ChirpConfigAwgMult"),qMakePair(QString::number(data->awgMult,'f',1),QString("")));
    out.insert(QString("ChirpConfigTotalMult"),qMakePair(QString::number(data->totalMult,'f',1),QString("")));
    out.insert(QString("ChirpConfigMixerSideband"),qMakePair(QString::number(data->mixerSideband,'f',1),QString("")));
    out.insert(QString("ChirpConfigValonTxFreq"),qMakePair(QString::number(data->valonTxFreq,'f',3),QString("MHz")));

    return out;
}

QString ChirpConfig::toString() const
{
    QString out;
    QMap<QString, QPair<QVariant,QString>> header = headerMap();
    auto it = header.constBegin();
    while(it != header.constEnd())
    {
        out.append(QString("%1\t%2\t%3\n").arg(it.key()).arg(it.value().first.toString()).arg(it.value().second));
        it++;
    }

    for(int i=0; i<data->segments.size(); i++)
    {
        BlackChirp::ChirpSegment s = data->segments.at(i);
        out.append(QString("\nSegment\t%1\t%2\t%3").arg(s.startFreqMHz,0,'f',3).arg(s.endFreqMHz,0,'f',3).arg(s.durationUs,0,'f',4));
    }

    out.append(QString("\n"));
    for(int i=0; i<data->segments.size(); i++)
    {
        BlackChirp::ChirpSegment s = data->segments.at(i);
        out.append(QString("\n#Segment\t%1\t%2\t%3").arg(awgToRealFreq(s.startFreqMHz),0,'f',3).arg(awgToRealFreq(s.endFreqMHz),0,'f',3).arg(s.durationUs,0,'f',4));
    }

    return out;
}

double ChirpConfig::synthTxMult() const
{
    return data->valonTxMult;
}

double ChirpConfig::awgMult() const
{
    return data->awgMult;
}

double ChirpConfig::mixerSideband() const
{
    return data->mixerSideband;
}

double ChirpConfig::totalMult() const
{
    return data->totalMult;
}

double ChirpConfig::synthTxFreq() const
{
    return data->valonTxFreq;
}

bool ChirpConfig::validate()
{
    data->isValid = false;
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("chirpConfig"));
    double minPreProt = s.value(QString("minPreChirpProtection"),0.010).toDouble();
    double minTwt = s.value(QString("minPreChirpDelay"),0.100).toDouble();
    double minPostProt = s.value(QString("minPostChirpProtection"),0.100).toDouble();
    s.endGroup();

    s.beginGroup(QString("awg"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    double awgRate = s.value(QString("sampleRate"),16e9).toDouble();
    double awgMaxSamples = s.value(QString("maxSamples"),2e9).toDouble();
    double awgMinFreq = s.value(QString("minFreq"),100.0).toDouble();
    double awgMaxFreq = s.value(QString("maxFreq"),6250.0).toDouble();
    s.endGroup();
    s.endGroup();

    data->sampleRateSperS = awgRate;
    data->sampleRateSperUS = awgRate/1e6;
    data->sampleIntervalS = 1.0/awgRate;
    data->sampleIntervalUS = 1.0/awgRate*1e6;

    //make sure all settings are possible
    if(data->preChirpProtection < minPreProt)
        return false;

    if(data->preChirpDelay < minTwt)
        return false;

    if(data->postChirpProtection < minPostProt)
        return false;

    if(data->numChirps < 1)
        return false;

    if(data->segments.isEmpty())
        return false;

    for(int i=0; i<data->segments.size();i++)
    {
        if(data->segments.at(i).startFreqMHz > awgMaxFreq || data->segments.at(i).startFreqMHz < awgMinFreq)
            return false;

        if(data->segments.at(i).endFreqMHz > awgMaxFreq || data->segments.at(i).endFreqMHz < awgMinFreq)
            return false;
    }

    if(data->numChirps > 0 && data->chirpInterval < data->preChirpProtection + data->preChirpDelay + chirpDuration() + data->postChirpProtection + 4.0)
        return false;

    if(totalDuration() >= awgMaxSamples/awgRate*1e6)
        return false;

    data->isValid = true;
    return true;
}

void ChirpConfig::parseFileLine(QByteArray line)
{
    if(line.isEmpty() || line.startsWith('#'))
        return;

    if(line.startsWith(QByteArray("ChirpConfig")))
    {
        line.replace(QByteArray("ChirpConfig"),QByteArray(""));
        QByteArrayList l = line.split('\t');
        if(l.size() < 2)
            return;

        QByteArray key = l.first().trimmed();
        QByteArray val = l.at(1).trimmed();

        if(key.contains(QByteArray("PreChirpProtection")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->preChirpProtection = p;
        }
        else if(key.contains(QByteArray("PreChirpDelay")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->preChirpDelay = p;
        }
        else if(key.contains(QByteArray("PostChirpProtection")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->postChirpProtection = p;
        }
        else if(key.contains(QByteArray("NumChirps")))
        {
            bool ok;
            int p = val.toInt(&ok);
            if(ok)
                data->numChirps = p;
        }
        else if(key.contains(QByteArray("ChirpInterval")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->chirpInterval = p;
        }
        else if(key.contains(QByteArray("ValonTxMult")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->valonTxMult = p;
        }
        else if(key.contains(QByteArray("AwgMult")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->awgMult = p;
        }
        else if(key.contains(QByteArray("TotalMult")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->totalMult = p;
        }
        else if(key.contains(QByteArray("MixerSideband")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->mixerSideband = p;
        }
        else if(key.contains(QByteArray("ValonTxFreq")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->valonTxFreq = p;
        }
    }
    else if(line.startsWith(QByteArray("Segment")))
    {
        QByteArrayList l = line.split('\t');
        if(l.size() < 4)
            return;

        bool ok;
        double start = l.at(1).trimmed().toDouble(&ok);
        if(ok)
        {
            double stop = l.at(2).trimmed().toDouble(&ok);
            if(ok)
            {
                double dur = l.at(3).trimmed().toDouble(&ok);
                if(ok)
                {
                    BlackChirp::ChirpSegment cs;
                    cs.startFreqMHz = start;
                    cs.endFreqMHz = stop;
                    cs.durationUs = dur;
                    cs.alphaUs = (stop-start)/dur;
                    data->segments.append(cs);
                }
            }
        }
    }
}

void ChirpConfig::setPreChirpProtection(const double d)
{
    data->preChirpProtection = d;
    validate();
}

void ChirpConfig::setPreChirpDelay(const double d)
{
    data->preChirpDelay = d;
    validate();
}

void ChirpConfig::setPostChirpProtection(const double d)
{
    data->postChirpProtection = d;
    validate();
}

void ChirpConfig::setNumChirps(const int n)
{
    data->numChirps = n;
    validate();
}

void ChirpConfig::setChirpInterval(const double i)
{
    data->chirpInterval = i;
    validate();
}

void ChirpConfig::setSegmentList(const QList<BlackChirp::ChirpSegment> l)
{
    QList<BlackChirp::ChirpSegment> newSegList;
    for(int i=0; i<l.size(); i++)
    {
        BlackChirp::ChirpSegment seg;
        seg.startFreqMHz = realToAwgFreq(l.at(i).startFreqMHz);
        seg.endFreqMHz = realToAwgFreq(l.at(i).endFreqMHz);
        seg.durationUs = l.at(i).durationUs;
        seg.alphaUs = (seg.endFreqMHz - seg.startFreqMHz)/seg.durationUs;

        newSegList.append(seg);
    }
    data->segments = newSegList;
    validate();
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

double ChirpConfig::getSampleTime(const int sample) const
{
    return static_cast<double>(sample)*data->sampleIntervalUS;
}

double ChirpConfig::calculateChirp(const BlackChirp::ChirpSegment segment, const double t, const double phase) const
{
    return gsl_sf_sin(gsl_sf_angle_restrict_pos(2.0*M_PI*(segment.startFreqMHz + 0.5*segment.alphaUs*t)*t + phase));
}

double ChirpConfig::calculateEndingPhaseRadians(const BlackChirp::ChirpSegment segment, const double endingTime, const double startingPhase) const
{
    double sinVal = calculateChirp(segment,endingTime,startingPhase);
    if(qFuzzyCompare(sinVal,1.0))
        return 0.5*M_PI;
    if(qFuzzyCompare(sinVal,-1.0))
        return 1.5*M_PI;

    double cosVal = calculateChirp(segment,endingTime,startingPhase + 0.5*M_PI);

    if(cosVal > 0.0)
        return gsl_sf_angle_restrict_pos(GSL_REAL(gsl_complex_arcsin_real(sinVal)));
    else
        return gsl_sf_angle_restrict_pos(M_PI - GSL_REAL(gsl_complex_arcsin_real(sinVal)));

    //NOT REACHED
    return 0.0;
}

double ChirpConfig::realToAwgFreq(const double realFreq) const
{
    return data->mixerSideband*(realFreq/data->totalMult - data->valonTxMult*data->valonTxFreq)/data->awgMult;
}

double ChirpConfig::awgToRealFreq(const double awgFreq) const
{
    return data->totalMult*(data->mixerSideband*data->awgMult*awgFreq + data->valonTxFreq*data->valonTxMult);
}

