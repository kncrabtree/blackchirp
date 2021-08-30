#include <data/experiment/chirpconfig.h>


#include <QApplication>
#include <QList>
#include <QCryptographicHash>
#include <QFile>
#include <QSaveFile>
#include <math.h>

#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_const.h>

#include <data/datastructs.h>
#include <data/storage/blackchirpcsv.h>



ChirpConfig::ChirpConfig() : HeaderStorage(BC::Store::CC::key)
{
}

ChirpConfig::~ChirpConfig()
{

}

void ChirpConfig::readChirpFile(BlackchirpCSV *csv, int num, QString path)
{
    auto d = BlackchirpCSV::exptDir(num,path);
    QFile f(d.absoluteFilePath(BC::CSV::chirpFile));
    if(f.open(QIODevice::ReadOnly))
    {
        while(!f.atEnd())
        {
            auto l = csv->readLine(f);

            if(l.isEmpty())
                continue;

            if(l.constFirst().toString().startsWith("Chirp"))
                continue;


            if(l.size() == 7)
            {
                bool ok = false;
                int chirp = l.at(0).toInt(&ok);
                if(!ok)
                    continue;
                int seg = l.at(1).toInt(&ok);
                if(!ok)
                    continue;
                double start = l.at(2).toDouble(&ok);
                if(!ok)
                    continue;
                double end = l.at(3).toDouble(&ok);
                if(!ok)
                    continue;
                double dur = l.at(4).toDouble(&ok);
                if(!ok)
                    continue;
                double alpha = l.at(5).toDouble(&ok);
                if(!ok)
                    continue;
                bool empty = QVariant(l.at(6)).toBool();

                ChirpSegment s{start,end,dur,alpha,empty};
                while(d_chirpList.size() < chirp + 1) {
                    d_chirpList.append(QVector<ChirpSegment>());
                }
                while(d_chirpList.at(chirp).size() < seg + 1) {
                    d_chirpList[chirp].append(ChirpSegment());
                }

                d_chirpList[chirp][seg] = s;
            }
        }
    }
}

bool ChirpConfig::writeChirpFile(int num) const
{
    QDir d(BlackchirpCSV::exptDir(num));
    QSaveFile f(d.absoluteFilePath(BC::CSV::chirpFile));
    if(f.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        QTextStream t(&f);
        BlackchirpCSV::writeLine(t,{"Chirp","Segment","StartMHz","EndMHz","DurationUs","Alpha","Empty"});
        for(int i=0; i<d_chirpList.size(); ++i)
        {
            for(int j=0; j<d_chirpList.at(i).size(); ++j)
            {
                auto &seg = d_chirpList.at(i).at(j);
                BlackchirpCSV::writeLine(t,{i,j,seg.startFreqMHz,seg.endFreqMHz,
                                 seg.durationUs,seg.alphaUs,seg.empty});
            }
        }
        return f.commit();
    }

    return false;
}

double ChirpConfig::preChirpProtectionDelay() const
{
    return d_markers.preProt;
}

double ChirpConfig::preChirpGateDelay() const
{
    return d_markers.preGate;
}

double ChirpConfig::postChirpGateDelay() const
{
    return d_markers.postGate;
}

double ChirpConfig::postChirpProtectionDelay() const
{
    return d_markers.postProt;
}

double ChirpConfig::totalProtectionWidth() const
{
    return d_markers.preProt + d_markers.preGate + chirpDurationUs(0) + d_markers.postProt;
}

double ChirpConfig::totalGateWidth() const
{
    return d_markers.preGate + chirpDurationUs(0) + d_markers.postGate;
}

int ChirpConfig::numChirps() const
{
    return d_chirpList.size();
}

double ChirpConfig::chirpInterval() const
{
    return d_chirpInterval;
}

bool ChirpConfig::allChirpsIdentical() const
{
    if(d_chirpList.size() <= 1)
        return true;

    auto firstList = d_chirpList.constFirst();

    for(int i=1; i<d_chirpList.size(); i++)
    {
        auto thisList = d_chirpList.at(i);

        if(thisList.size() != firstList.size())
            return false;

        for(int j=0; j<thisList.size(); j++)
        {
            if(thisList.at(j).empty != firstList.at(j).empty)
                return false;

            if(!qFuzzyCompare(thisList.at(j).durationUs,firstList.at(j).durationUs))
                return false;

            if(!thisList.at(j).empty)
            {
                if(!qFuzzyCompare(thisList.at(j).startFreqMHz,firstList.at(j).startFreqMHz))
                    return false;

                if(!qFuzzyCompare(thisList.at(j).endFreqMHz,firstList.at(j).endFreqMHz))
                    return false;
            }
        }
    }

    return true;
}

double ChirpConfig::chirpDurationUs(int chirpNum) const
{
    double out = 0.0;
    if(chirpNum >= d_chirpList.size())
        return out;
    auto segments = d_chirpList.at(chirpNum);
    for(int i=0; i<segments.size(); i++)
        out += segments.at(i).durationUs;
    return out;
}

double ChirpConfig::totalDuration() const
{
    ///TODO: This should be an implementation detail of the AWG, not part of the chirpConfig
    double baseLength = 10.0;
    double length = preChirpProtectionDelay() + preChirpGateDelay() + postChirpProtectionDelay();
    length += (static_cast<double>(numChirps())-1.0)*d_chirpInterval + chirpDurationUs(numChirps()-1);

    return floor(length/baseLength + 1.0)*baseLength;
}

QVector<QVector<ChirpConfig::ChirpSegment> > ChirpConfig::chirpList() const
{
    return d_chirpList;
}

double ChirpConfig::segmentStartFreq(int chirp, int segment) const
{
    if(chirp < 0 || chirp >= d_chirpList.size())
        return -1.0;

    if(segment < 0 || segment >= d_chirpList.at(chirp).size())
        return -1.0;

    return d_chirpList.at(chirp).at(segment).startFreqMHz;
}

double ChirpConfig::segmentEndFreq(int chirp, int segment) const
{
    if(chirp < 0 || chirp >= d_chirpList.size())
        return -1.0;

    if(segment < 0 || segment >= d_chirpList.at(chirp).size())
        return -1.0;

    return d_chirpList.at(chirp).at(segment).endFreqMHz;
}

double ChirpConfig::segmentDuration(int chirp, int segment) const
{
    if(chirp < 0 || chirp >= d_chirpList.size())
        return -1.0;

    if(segment < 0 || segment >= d_chirpList.at(chirp).size())
        return -1.0;

    return d_chirpList.at(chirp).at(segment).durationUs;
}

bool ChirpConfig::segmentEmpty(int chirp, int segment) const
{
    if(chirp < 0 || chirp >= d_chirpList.size())
        return true;

    if(segment < 0 || segment >= d_chirpList.at(chirp).size())
        return true;

    return d_chirpList.at(chirp).at(segment).empty;
}

QByteArray ChirpConfig::waveformHash() const
{
    QCryptographicHash c(QCryptographicHash::Sha256);
    c.addData(QByteArray::number(preChirpProtectionDelay()));
    c.addData(QByteArray::number(preChirpGateDelay()));
    c.addData(QByteArray::number(postChirpGateDelay()));
    for(int j=0; j<d_chirpList.size(); j++)
    {
        for(int i=0; i<d_chirpList.at(j).size(); i++)
        {
            c.addData(QByteArray::number(d_chirpList.at(j).at(i).startFreqMHz));
            c.addData(QByteArray::number(d_chirpList.at(j).at(i).endFreqMHz));
            c.addData(QByteArray::number(d_chirpList.at(j).at(i).durationUs));
            c.addData(QByteArray::number(static_cast<int>(d_chirpList.at(j).at(i).empty)));
        }
    }
    c.addData(QByteArray::number(numChirps()));
    c.addData(QByteArray::number(d_chirpInterval));
    c.addData(QByteArray::number(postChirpProtectionDelay()));

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
    if(d_chirpList.isEmpty())
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
    if(numSamples == 0)
        return out;

    //find where the first sample falls with respect to delays and chirps
    //find interval number

    bool done = false;

    while(!done) // loop that allows interval crossing
    {
        double currentTime = getSampleTime(firstSample+currentSample);
        int currentInterval = static_cast<int>(floor(currentTime/d_chirpInterval));
        int currentIntervalStartSample = getFirstSample(static_cast<double>(currentInterval)*d_chirpInterval);
        int nextIntervalStartSample = getFirstSample((static_cast<double>(currentInterval) + 1.0) * d_chirpInterval);
        if(nextIntervalStartSample > firstSample+numSamples)
        {
            //this is the last interval
            done = true;
        }
        int currentIntervalChirpStart = getFirstSample(getSampleTime(currentIntervalStartSample) + preChirpProtectionDelay() + preChirpGateDelay());
        int currentIntervalChirpEnd = getLastSample(getSampleTime(currentIntervalChirpStart) + chirpDurationUs(currentInterval));

        //start times for each segment
        QList<int> segmentStarts;
        segmentStarts.append(currentIntervalChirpStart);
        for(int i=1; i<d_chirpList.at(currentInterval).size(); i++)
            segmentStarts.append(getFirstSample(getSampleTime(segmentStarts.at(i-1)) + d_chirpList.at(currentInterval).at(i-1).durationUs));

        //starting phase for each segment
        //want to transition between frequencies as smoothly as possible, so try to start each segment at the phase at which the previous
        //segment would have been if it were to continue
        QList<double> segmentPhasesRadians;
        segmentPhasesRadians.append(0.0);
        for(int i=1; i<d_chirpList.at(currentInterval).size(); i++)
        {
            double segmentEndTime = getSampleTime(segmentStarts.at(i)-segmentStarts.at(i-1));
            segmentPhasesRadians.append(calculateEndingPhaseRadians(d_chirpList.at(currentInterval).at(i-1),segmentEndTime,segmentPhasesRadians.at(i-1)));
        }

        //determine current segment number
        //-1 means before chirp
        //0 - segments.size()-1 means during a segement
        //segments.size() means protection interval
        //segment.size() + 1 means waiting for next chirp
        int currentSegment = 0;
        int nextSegmentSample;
        if(d_chirpList.at(currentInterval).isEmpty())
        {
            nextSegmentSample = firstSample+currentSample + getFirstSample(preChirpProtectionDelay() + preChirpGateDelay());
        }
        else
            nextSegmentSample = firstSample+currentSample + getFirstSample(preChirpProtectionDelay() + preChirpGateDelay() + d_chirpList.at(currentInterval).at(0).durationUs);
        if(d_chirpList.at(currentInterval).size() == 1)
            nextSegmentSample = currentIntervalChirpEnd;
        else
        {
            while(currentSegment + 1 < d_chirpList.at(currentInterval).size())
            {
                if(currentSample < segmentStarts.at(currentSegment+1))
                    break;

                currentSegment++;
                nextSegmentSample += getFirstSample(d_chirpList.at(currentInterval).at(currentSegment).durationUs);
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
                    out[currentSample] = QPointF(currentTime,calculateChirp(d_chirpList.at(currentInterval).at(currentSegment),currentTime-getSampleTime(segmentStarts.at(currentSegment)),segmentPhasesRadians.at(currentSegment)));

                currentSample++;
                currentTime = getSampleTime(firstSample+currentSample);


            }

            if(currentSample >= numSamples)
            {
                done = true;
                break;
            }

            currentSegment++;
            if(currentSegment >= d_chirpList.at(currentInterval).size())
                nextSegmentSample = nextIntervalStartSample;
            else
                nextSegmentSample += getFirstSample(d_chirpList.at(currentInterval).at(currentSegment).durationUs);
        }

    }
    //fill with zeroes until total length
    while(currentSample < invalidSample-1 && currentSample < out.size())
    {
        double currentTime = getSampleTime(firstSample+currentSample);
        out[currentSample] = QPointF(currentTime,0.0);
    }


    return out;
}

QVector<QPair<bool, bool> > ChirpConfig::getMarkerData() const
{
    if(d_chirpList.isEmpty())
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
        int currentInterval = static_cast<int>(floor(currentTime/d_chirpInterval));
        int currentIntervalStartSample = getFirstSample(static_cast<double>(currentInterval)*d_chirpInterval);
        int nextIntervalStartSample = getFirstSample((static_cast<double>(currentInterval) + 1.0) * d_chirpInterval);
        if(nextIntervalStartSample > firstSample+numSamples)
        {
            //this is the last interval
            done = true;
            nextIntervalStartSample = firstSample+numSamples;
        }
        int currentIntervalChirpStart = getFirstSample(getSampleTime(currentIntervalStartSample) + preChirpProtectionDelay() + preChirpGateDelay());
        int currentIntervalGateStart = getFirstSample(getSampleTime(currentIntervalStartSample) + preChirpProtectionDelay());
        int currentIntervalChirpEnd = getLastSample(getSampleTime(currentIntervalChirpStart) + chirpDurationUs(currentInterval));
        int currentIntervalGateEnd = getLastSample(getSampleTime(currentIntervalChirpEnd) + postChirpGateDelay())-1;
        int currentIntervalProtEnd = getLastSample(getSampleTime(currentIntervalChirpEnd) + postChirpProtectionDelay())-1;

        //sections will correspond to marker states:
        //0 = prot high, gate low
        //1 = both high
        //2 = prot high, gate low
        //3 = both low

        bool prot = true;
        bool gate = false;

        while(currentSample < nextIntervalStartSample)
        {
            //assess section state
            if(currentSample == currentIntervalGateStart)
                gate = true;
            if(currentSample == currentIntervalGateEnd)
                gate = false;
            if(currentSample == currentIntervalProtEnd)
                prot = false;

            out[currentSample] = qMakePair(prot,gate);

            currentSample++;
        }
    }
    //fill with zeroes until total length
    while(currentSample < invalidSample-1)
        out[currentSample] = qMakePair(false,false);

    return out;

}

void ChirpConfig::setAwgSampleRate(const double samplesPerSecond)
{
    d_sampleRateSperUS = samplesPerSecond/1e6;
    d_sampleIntervalUS = 1.0/d_sampleRateSperUS;
}

void ChirpConfig::setPreChirpProtectionDelay(const double d)
{
    d_markers.preProt = d;
}

void ChirpConfig::setPreChirpGateDelay(const double d)
{
    d_markers.preGate = d;
}

void ChirpConfig::setPostChirpGateDelay(const double d)
{
    d_markers.postGate = d;
}

void ChirpConfig::setPostChirpProtectionDelay(const double d)
{
    d_markers.postProt = d;
}

void ChirpConfig::setNumChirps(const int n)
{
    if(n > d_chirpList.size())
    {
        if(!d_chirpList.isEmpty())
        {
            for(int i=numChirps(); i<n; i++)
                d_chirpList.append(d_chirpList.constFirst());
        }
        else
        {
            for(int i=numChirps(); i<n; i++)
                d_chirpList.append(QVector<ChirpSegment>());
        }
    }
    else if(n < d_chirpList.size())
    {
        while(d_chirpList.size() > n)
            d_chirpList.removeLast();
    }    
}

void ChirpConfig::setChirpInterval(const double i)
{
    d_chirpInterval = i;
}

void ChirpConfig::addSegment(const double startMHz, const double endMHz, const double durationUs, const int chirpNum)
{
    if(startMHz < 0.0 || endMHz < 0.0 || durationUs < 0.0)
        return;

    ChirpSegment seg{startMHz,endMHz,durationUs,(endMHz-startMHz)/durationUs,false};

    if(chirpNum < 0 || chirpNum > d_chirpList.size())
    {
        for(int i=0; i<d_chirpList.size(); i++)
            d_chirpList[i].append(seg);
    }
    else
        d_chirpList[chirpNum].append(seg);

}

void ChirpConfig::addEmptySegment(const double durationUs, const int chirpNum)
{
    if(durationUs > 0.0)
    {
        ChirpSegment seg{0.0,0.0,durationUs,0.0,true};

        if(chirpNum < 0 || chirpNum > d_chirpList.size())
        {
            for(int i=0; i<d_chirpList.size(); i++)
                d_chirpList[i].append(seg);
        }
        else
            d_chirpList[chirpNum].append(seg);
    }
}

void ChirpConfig::setChirpList(const QVector<QVector<ChirpSegment>> l)
{
    d_chirpList = l;
}

int ChirpConfig::getFirstSample(double time) const
{
    //first sample is inclusive
    if(d_chirpList.isEmpty())
        return -1;

    double nearestSampleTime = round(d_sampleRateSperUS*time)*d_sampleIntervalUS;
    if(qFuzzyCompare(1.0 + time, 1.0 + nearestSampleTime))
        return static_cast<int>(round(d_sampleRateSperUS*time));
    else
        return static_cast<int>(ceil(d_sampleRateSperUS*time));
}

int ChirpConfig::getLastSample(double time) const
{
    //last sample is non-inclusive
    if(d_chirpList.isEmpty())
        return -1.0;

    double nearestSampleTime = round(d_sampleRateSperUS*time)*d_sampleIntervalUS;
    if(qFuzzyCompare(1.0 + time, 1.0 + nearestSampleTime))
        return static_cast<int>(round(d_sampleRateSperUS*time));
    else
        return static_cast<int>(floor(d_sampleRateSperUS*time));
}

double ChirpConfig::getSampleTime(const int sample) const
{
    return static_cast<double>(sample)*d_sampleIntervalUS;
}

double ChirpConfig::calculateChirp(const ChirpSegment segment, const double t, const double phase) const
{
    if(segment.empty)
        return 0.0;

    return gsl_sf_sin(gsl_sf_angle_restrict_pos(2.0*M_PI*(segment.startFreqMHz + 0.5*segment.alphaUs*t)*t + phase));
}

double ChirpConfig::calculateEndingPhaseRadians(const ChirpSegment segment, const double endingTime, const double startingPhase) const
{
    if(segment.empty)
        return 0.0;

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



void ChirpConfig::storeValues()
{
    using namespace BC::Store::CC;
    store(preProt,d_markers.preProt,QString::fromUtf8("μs"));
    store(postProt,d_markers.postProt,QString::fromUtf8("μs"));
    store(preGate,d_markers.preGate,QString::fromUtf8("μs"));
    store(postGate,d_markers.postGate,QString::fromUtf8("μs"));
    store(interval,d_chirpInterval,QString::fromUtf8("μs"));
    store(sampleRate,d_sampleRateSperUS,QString("MHz"));
    store(sampleInterval,d_sampleIntervalUS,QString::fromUtf8("μs"));
}

void ChirpConfig::retrieveValues()
{
    using namespace BC::Store::CC;
    d_markers.preProt = retrieve(preProt,0.5);
    d_markers.postProt = retrieve(postProt,0.5);
    d_markers.preGate = retrieve(preGate,0.5);
    d_markers.postGate = retrieve(postGate,0.5);
    d_chirpInterval = retrieve(interval,-1.0);
    d_sampleRateSperUS = retrieve(sampleRate,1.0);
    d_sampleIntervalUS = retrieve(sampleInterval,1.0);
}
