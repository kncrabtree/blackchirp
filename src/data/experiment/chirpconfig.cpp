#include <data/experiment/chirpconfig.h>

#include <QList>
#include <QCryptographicHash>
#include <QFile>
#include <QSaveFile>
#include <math.h>

#ifndef M_PI
#define M_PI 3.1415926535897323846
#endif

#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_const.h>

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

void ChirpConfig::readMarkersFile(BlackchirpCSV *csv, int num, QString path)
{
    auto d = BlackchirpCSV::exptDir(num,path);
    QFile f(d.absoluteFilePath(BC::CSV::markersFile));
    if(f.open(QIODevice::ReadOnly))
    {
        d_markerChannels.clear();
        while(!f.atEnd())
        {
            auto l = csv->readLine(f);
            if(l.isEmpty())
                continue;
            if(l.constFirst().toString().startsWith("Channel"))
                continue;
            if(l.size() == 7)
            {
                bool ok = false;
                int ch = l.at(0).toInt(&ok);
                if(!ok)
                    continue;
                QString name = l.at(1).toString();
                QString roleStr = l.at(2).toString();
                QString modeStr = l.at(3).toString();
                double start = l.at(4).toDouble(&ok);
                if(!ok)
                    continue;
                double end = l.at(5).toDouble(&ok);
                if(!ok)
                    continue;
                bool enabled = QVariant(l.at(6)).toBool();

                MarkerRole role = MarkerRole::Custom;
                if(roleStr == "Protection")
                    role = MarkerRole::Protection;
                else if(roleStr == "Gate")
                    role = MarkerRole::Gate;
                else if(roleStr == "Trigger")
                    role = MarkerRole::Trigger;

                MarkerChannel::TimingMode mode = MarkerChannel::ChirpRelative;
                if(modeStr == "Absolute")
                    mode = MarkerChannel::Absolute;

                MarkerChannel mc{name, mode, start, end, enabled, role};
                while(d_markerChannels.size() < ch + 1)
                    d_markerChannels.append(MarkerChannel{});
                d_markerChannels[ch] = mc;
            }
        }
    }
}

bool ChirpConfig::writeMarkersFile(int num) const
{
    QDir d(BlackchirpCSV::exptDir(num));
    QSaveFile f(d.absoluteFilePath(BC::CSV::markersFile));
    if(f.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        QTextStream t(&f);
        BlackchirpCSV::writeLine(t,{"Channel","Name","Role","TimingMode","StartUs","EndUs","Enabled"});
        for(int i=0; i<d_markerChannels.size(); ++i)
        {
            const auto &m = d_markerChannels.at(i);
            QString roleStr;
            switch(m.role)
            {
            case MarkerRole::Protection: roleStr = "Protection"; break;
            case MarkerRole::Gate:       roleStr = "Gate";       break;
            case MarkerRole::Trigger:    roleStr = "Trigger";    break;
            default:                     roleStr = "Custom";     break;
            }
            QString timingModeStr = (m.timingMode == MarkerChannel::Absolute) ? "Absolute" : "ChirpRelative";
            BlackchirpCSV::writeLine(t,{i,m.name,roleStr,timingModeStr,m.startTime,m.endTime,m.enabled});
        }
        return f.commit();
    }

    return false;
}

double ChirpConfig::leadTimeUs() const
{
    double lead = 0.0;
    for(const auto &m : d_markerChannels)
    {
        if(m.enabled)
            lead = qMax(lead, -m.startTime);
    }
    return qMax(0.0, lead);
}

double ChirpConfig::tailTimeUs() const
{
    double tail = 0.0;
    for(const auto &m : d_markerChannels)
    {
        if(m.enabled)
            tail = qMax(tail, m.endTime);
    }
    return qMax(0.0, tail);
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
    double length = leadTimeUs() + tailTimeUs();
    length += (static_cast<double>(numChirps())-1.0)*d_chirpInterval + chirpDurationUs(numChirps()-1);
    return length;
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
    for(const auto &m : d_markerChannels)
    {
        c.addData(m.name.toUtf8());
        c.addData(QByteArray::number(static_cast<int>(m.role)));
        c.addData(QByteArray::number(m.startTime));
        c.addData(QByteArray::number(m.endTime));
        c.addData(QByteArray::number(static_cast<int>(m.enabled)));
    }
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

    double lead = leadTimeUs();

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
        int currentIntervalChirpStart = getFirstSample(getSampleTime(currentIntervalStartSample) + lead);
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
            nextSegmentSample = firstSample+currentSample + getFirstSample(lead);
        }
        else
            nextSegmentSample = firstSample+currentSample + getFirstSample(lead + d_chirpList.at(currentInterval).at(0).durationUs);
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

QVector<QVector<bool>> ChirpConfig::getMarkerData() const
{
    if(d_chirpList.isEmpty() || d_markerChannels.isEmpty())
        return {};

    int numChannels = d_markerChannels.size();
    int invalidSample = getFirstSample(totalDuration()) + 1;
    int numSamples = invalidSample - 1;

    QVector<QVector<bool>> out(numChannels, QVector<bool>(numSamples, false));

    double lead = leadTimeUs();
    int currentSample = 0;
    bool done = false;

    while(!done)
    {
        double currentTime = getSampleTime(currentSample);
        int currentInterval = static_cast<int>(floor(currentTime/d_chirpInterval));
        double intervalStartTime = static_cast<double>(currentInterval)*d_chirpInterval;
        int nextIntervalStartSample = getFirstSample((static_cast<double>(currentInterval) + 1.0) * d_chirpInterval);
        if(nextIntervalStartSample > numSamples)
        {
            done = true;
            nextIntervalStartSample = numSamples;
        }

        double chirpStartTime = intervalStartTime + lead;
        double chirpEndTime = chirpStartTime + chirpDurationUs(currentInterval);

        QVector<int> markerStartSample(numChannels, 0);
        QVector<int> markerEndSample(numChannels, -1);
        for(int ch = 0; ch < numChannels; ++ch)
        {
            const auto &m = d_markerChannels.at(ch);
            if(!m.enabled)
                continue;
            markerStartSample[ch] = getFirstSample(chirpStartTime + m.startTime);
            markerEndSample[ch] = getLastSample(chirpEndTime + m.endTime) - 1;
        }

        while(currentSample < nextIntervalStartSample && currentSample < numSamples)
        {
            for(int ch = 0; ch < numChannels; ++ch)
            {
                const auto &m = d_markerChannels.at(ch);
                if(!m.enabled)
                    continue;
                out[ch][currentSample] = (currentSample >= markerStartSample[ch] &&
                                          currentSample <= markerEndSample[ch]);
            }
            currentSample++;
        }
    }

    return out;
}

QVector<quint32> ChirpConfig::getPackedMarkerData() const
{
    auto data = getMarkerData();
    if(data.isEmpty())
        return {};

    int numSamples = data.at(0).size();
    int numChannels = data.size();
    QVector<quint32> out(numSamples, 0u);

    for(int ch = 0; ch < numChannels; ++ch)
    {
        quint32 mask = (1u << ch);
        for(int s = 0; s < numSamples; ++s)
        {
            if(data.at(ch).at(s))
                out[s] |= mask;
        }
    }

    return out;
}

const QVector<MarkerChannel>& ChirpConfig::markerChannels() const
{
    return d_markerChannels;
}

const MarkerChannel* ChirpConfig::findEnabledMarkerByRole(MarkerRole role) const
{
    for(const auto &m : d_markerChannels)
    {
        if(m.enabled && m.role == role)
            return &m;
    }
    return nullptr;
}

void ChirpConfig::setMarkerChannels(const QVector<MarkerChannel>& channels)
{
    d_markerChannels = channels;
}

void ChirpConfig::setAwgSampleRate(const double samplesPerSecond)
{
    d_sampleRateSperUS = samplesPerSecond/1e6;
    d_sampleIntervalUS = 1.0/d_sampleRateSperUS;
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
}



void ChirpConfig::storeValues()
{
    using namespace BC::Store::CC;
    store(interval,d_chirpInterval,BC::Unit::us);
    store(sampleRate,d_sampleRateSperUS,BC::Unit::MHz);
    store(sampleInterval,d_sampleIntervalUS,BC::Unit::us);
}

void ChirpConfig::retrieveValues()
{
    using namespace BC::Store::CC;
    d_chirpInterval = retrieve(interval,-1.0);
    d_sampleRateSperUS = retrieve(sampleRate,1.0);
    d_sampleIntervalUS = retrieve(sampleInterval,1.0);
}
