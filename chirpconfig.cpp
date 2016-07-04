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
    data->synthTxFreq = s.value(QString("txFreq"),5760.0).toDouble();
    s.endGroup();
    s.endGroup();

    s.beginGroup(QString("chirpConfig"));
    data->awgMult = s.value(QString("awgMult"),1.0).toDouble();
    data->synthTxMult = s.value(QString("txValonMult"),2.0).toDouble();
    data->totalMult = s.value(QString("txMult"),4.0).toDouble();
    data->mixerTxSideband = s.value(QString("txSidebandSign"),-1.0).toDouble();
    s.endGroup();

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
    if(!qFuzzyCompare(synthTxFreq(),other.synthTxFreq()))
        return false;
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

bool ChirpConfig::allChirpsIdentical() const
{
    if(data->chirpList.size() == 1)
        return true;

    QList<BlackChirp::ChirpSegment> firstList = data->chirpList.first();

    for(int i=1; i<data->chirpList.size(); i++)
    {
        QList<BlackChirp::ChirpSegment> thisList = data->chirpList.at(i);

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

double ChirpConfig::chirpDuration(int chirpNum) const
{
    double out = 0.0;
    QList<BlackChirp::ChirpSegment> segments = data->chirpList.at(chirpNum);
    for(int i=0; i<segments.size(); i++)
        out += segments.at(i).durationUs;
    return out;
}

double ChirpConfig::totalDuration() const
{
    double baseLength = 10.0;
    double length = data->preChirpProtection + data->preChirpDelay + data->postChirpProtection;
    length += (static_cast<double>(data->numChirps)-1.0)*data->chirpInterval + chirpDuration(data->numChirps-1);

    return floor(length/baseLength + 1.0)*baseLength;
}

QList<QList<BlackChirp::ChirpSegment>> ChirpConfig::chirpList() const
{
    return data->chirpList;
}

double ChirpConfig::segmentStartFreq(int chirp, int segment) const
{
    if(chirp < 0 || chirp >= data->chirpList.size())
        return -1.0;

    if(segment < 0 || segment >= data->chirpList.at(chirp).size())
        return -1.0;

    return awgToRealFreq(data->chirpList.at(chirp).at(segment).startFreqMHz);
}

double ChirpConfig::segmentEndFreq(int chirp, int segment) const
{
    if(chirp < 0 || chirp >= data->chirpList.size())
        return -1.0;

    if(segment < 0 || segment >= data->chirpList.at(chirp).size())
        return -1.0;

    return awgToRealFreq(data->chirpList.at(chirp).at(segment).endFreqMHz);
}

double ChirpConfig::segmentDuration(int chirp, int segment) const
{
    if(chirp < 0 || chirp >= data->chirpList.size())
        return -1.0;

    if(segment < 0 || segment >= data->chirpList.at(chirp).size())
        return -1.0;

    return data->chirpList.at(chirp).at(segment).durationUs;
}

bool ChirpConfig::segmentEmpty(int chirp, int segment) const
{
    if(chirp < 0 || chirp >= data->chirpList.size())
        return true;

    if(segment < 0 || segment >= data->chirpList.at(chirp).size())
        return true;

    return data->chirpList.at(chirp).at(segment).empty;
}

QByteArray ChirpConfig::waveformHash() const
{
    QCryptographicHash c(QCryptographicHash::Sha256);
    c.addData(QByteArray::number(data->preChirpProtection));
    c.addData(QByteArray::number(data->preChirpDelay));
    for(int j=0; j<data->chirpList.size(); j++)
    {
        for(int i=0; i<data->chirpList.at(j).size(); i++)
        {
            c.addData(QByteArray::number(data->chirpList.at(j).at(i).startFreqMHz));
            c.addData(QByteArray::number(data->chirpList.at(j).at(i).endFreqMHz));
            c.addData(QByteArray::number(data->chirpList.at(j).at(i).durationUs));
            c.addData(QByteArray::number(static_cast<int>(data->chirpList.at(j).at(i).empty)));
        }
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
        int currentIntervalChirpEnd = getLastSample(getSampleTime(currentIntervalChirpStart) + chirpDuration(currentInterval));

        //start times for each segment
        QList<int> segmentStarts;
        segmentStarts.append(currentIntervalChirpStart);
        for(int i=1; i<data->chirpList.at(currentInterval).size(); i++)
            segmentStarts.append(getFirstSample(getSampleTime(segmentStarts.at(i-1)) + data->chirpList.at(currentInterval).at(i-1).durationUs));

        //starting phase for each segment
        //want to transition between frequencies as smoothly as possible, so try to start each segment at the phase at which the previous
        //segment would have been if it were to continue
        QList<double> segmentPhasesRadians;
        segmentPhasesRadians.append(0.0);
        for(int i=1; i<data->chirpList.at(currentInterval).size(); i++)
        {
            double segmentEndTime = getSampleTime(segmentStarts.at(i)-segmentStarts.at(i-1));
            segmentPhasesRadians.append(calculateEndingPhaseRadians(data->chirpList.at(currentInterval).at(i-1),segmentEndTime,segmentPhasesRadians.at(i-1)));
        }

        //determine current segment number
        //-1 means before chirp
        //0 - data->segments.size()-1 means during a segement
        //data->segments.size() means protection interval
        //data->segment.size() + 1 means waiting for next chirp
        int currentSegment = 0;
        int nextSegmentSample;
        if(data->chirpList.at(currentInterval).isEmpty())
        {
            nextSegmentSample = firstSample+currentSample + getFirstSample(data->preChirpProtection + data->preChirpDelay);
        }
        else
            nextSegmentSample = firstSample+currentSample + getFirstSample(data->preChirpProtection + data->preChirpDelay + data->chirpList.at(currentInterval).at(0).durationUs);
        if(data->chirpList.at(currentInterval).size() == 1)
            nextSegmentSample = currentIntervalChirpEnd;
        else
        {
            while(currentSegment + 1 < data->chirpList.at(currentInterval).size())
            {
                if(currentSample < segmentStarts.at(currentSegment+1))
                    break;

                currentSegment++;
                nextSegmentSample += getFirstSample(data->chirpList.at(currentInterval).at(currentSegment).durationUs);
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
                    out[currentSample] = QPointF(currentTime,calculateChirp(data->chirpList.at(currentInterval).at(currentSegment),currentTime-getSampleTime(segmentStarts.at(currentSegment)),segmentPhasesRadians.at(currentSegment)));

                currentSample++;
                currentTime = getSampleTime(firstSample+currentSample);


            }

            if(currentSample >= numSamples)
            {
                done = true;
                break;
            }

            currentSegment++;
            if(currentSegment >= data->chirpList.at(currentInterval).size())
                nextSegmentSample = nextIntervalStartSample;
            else
                nextSegmentSample += getFirstSample(data->chirpList.at(currentInterval).at(currentSegment).durationUs);
        }

    }
    //fill with zeroes until total length
    while(currentSample < invalidSample-1)
    {
        double currentTime = getSampleTime(firstSample+currentSample);
        out[currentSample] = QPointF(currentTime,0.0);
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
        int currentIntervalChirpEnd = getLastSample(getSampleTime(currentIntervalChirpStart) + chirpDuration(currentInterval));
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
    //fill with zeroes until total length
    while(currentSample < invalidSample-1)
        out[currentSample] = qMakePair(false,false);

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
    out.insert(QString("ChirpConfigTxMult"),qMakePair(QString::number(data->synthTxMult,'f',1),QString("")));
    out.insert(QString("ChirpConfigAwgMult"),qMakePair(QString::number(data->awgMult,'f',1),QString("")));
    out.insert(QString("ChirpConfigTotalMult"),qMakePair(QString::number(data->totalMult,'f',1),QString("")));
    out.insert(QString("ChirpConfigMixerSideband"),qMakePair(QString::number(data->mixerTxSideband,'f',1),QString("")));
    out.insert(QString("ChirpConfigTxFreq"),qMakePair(QString::number(data->synthTxFreq,'f',3),QString("MHz")));

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

    if(allChirpsIdentical())
    {
        for(int i=0; i<data->chirpList.first().size(); i++)
        {
            BlackChirp::ChirpSegment s = data->chirpList.first().at(i);
            if(s.empty)
                out.append(QString("\nEmptySegment\t%1").arg(s.durationUs,0,'f',4));
            else
                out.append(QString("\nSegment\t%1\t%2\t%3").arg(s.startFreqMHz,0,'f',3).arg(s.endFreqMHz,0,'f',3).arg(s.durationUs,0,'f',4));
        }

        out.append(QString("\n"));
        for(int i=0; i<data->chirpList.first().size(); i++)
        {
            BlackChirp::ChirpSegment s = data->chirpList.first().at(i);
            if(s.empty)
                out.append(QString("\n#EmptySegment\t%1").arg(s.durationUs,0,'f',4));
            else
                out.append(QString("\n#Segment\t%1\t%2\t%3").arg(awgToRealFreq(s.startFreqMHz),0,'f',3).arg(awgToRealFreq(s.endFreqMHz),0,'f',3).arg(s.durationUs,0,'f',4));
        }
    }
    else
    {
        for(int j=0; j<data->chirpList.size(); j++)
        {
            QList<BlackChirp::ChirpSegment> thisList = data->chirpList.at(j);
            for(int i=0; i<thisList.size(); i++)
            {
                BlackChirp::ChirpSegment s = thisList.at(i);
                if(s.empty)
                    out.append(QString("\nEmptySegment-%2\t%1").arg(s.durationUs,0,'f',4).arg(j));
                else
                    out.append(QString("\nSegment-%4\t%1\t%2\t%3").arg(s.startFreqMHz,0,'f',3).arg(s.endFreqMHz,0,'f',3).arg(s.durationUs,0,'f',4).arg(j));
            }

            out.append(QString("\n"));
            for(int i=0; i<thisList.size(); i++)
            {
                BlackChirp::ChirpSegment s = thisList.at(i);
                if(s.empty)
                    out.append(QString("\n#EmptySegment-%2\t%1").arg(s.durationUs,0,'f',4).arg(j));
                else
                    out.append(QString("\n#Segment-%4\t%1\t%2\t%3").arg(awgToRealFreq(s.startFreqMHz),0,'f',3).arg(awgToRealFreq(s.endFreqMHz),0,'f',3).arg(s.durationUs,0,'f',4).arg(j));
            }
        }
    }

    return out;
}

double ChirpConfig::synthTxMult() const
{
    return data->synthTxMult;
}

double ChirpConfig::awgMult() const
{
    return data->awgMult;
}

double ChirpConfig::mixerSideband() const
{
    return data->mixerTxSideband;
}

double ChirpConfig::totalMult() const
{
    return data->totalMult;
}

double ChirpConfig::synthTxFreq() const
{
    return data->synthTxFreq;
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

    if(data->chirpList.isEmpty())
        return false;

    for(int j=0; j<data->chirpList.size(); j++)
    {
        QList<BlackChirp::ChirpSegment> thisList = data->chirpList.at(j);

        if(thisList.isEmpty())
            return false;

        for(int i=0; i<thisList.size();i++)
        {
            if((thisList.at(i).startFreqMHz > awgMaxFreq || thisList.at(i).startFreqMHz < awgMinFreq) && !thisList.at(i).empty)
                return false;

            if((thisList.at(i).endFreqMHz > awgMaxFreq || thisList.at(i).endFreqMHz < awgMinFreq)  && !thisList.at(i).empty)
                return false;
        }

        if(data->numChirps > 0 && data->chirpInterval < data->preChirpProtection + data->preChirpDelay + chirpDuration(j) + data->postChirpProtection + 4.0)
            return false;
    }

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
                setNumChirps(p);
        }
        else if(key.contains(QByteArray("ChirpInterval")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->chirpInterval = p;
        }
        else if(key.contains(QByteArray("TxMult")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->synthTxMult = p;
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
                data->mixerTxSideband = p;
        }
        else if(key.contains(QByteArray("TxFreq")))
        {
            bool ok;
            double p = val.toDouble(&ok);
            if(ok)
                data->synthTxFreq = p;
        }
    }
    else if(line.startsWith(QByteArray("Segment")))
    {
        QByteArrayList l = line.split('\t');
        if(l.size() < 4)
            return;

        int chirpNum = -1;
        bool ok;
        QByteArrayList l2 = l.at(0).split('-');
        if(l2.size() == 2)
        {
            int cn = l2.at(1).toInt(&ok);
            if(ok && cn >=0 && cn < data->numChirps)
                chirpNum = cn;
        }

        double start = l.at(1).trimmed().toDouble(&ok);
        if(ok)
        {
            double stop = l.at(2).trimmed().toDouble(&ok);
            if(ok)
            {
                double dur = l.at(3).trimmed().toDouble(&ok);
                if(ok)
                    addSegment(start,stop,dur,chirpNum);
            }
        }
    }
    else if(line.startsWith(QByteArray("EmptySegment")))
    {
        QByteArrayList l = line.split('\t');
        if(l.size() < 2)
            return;

        int chirpNum = -1;
        bool ok;
        QByteArrayList l2 = l.at(0).split('-');
        if(l2.size() == 2)
        {
            int cn = l2.at(1).toInt(&ok);
            if(ok && cn >=0 && cn < data->numChirps)
                chirpNum = cn;
        }

        double dur = l.at(1).trimmed().toDouble(&ok);
        if(ok)
            addEmptySegment(dur,chirpNum);
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
    if(n > data->chirpList.size())
    {
        if(!data->chirpList.isEmpty())
        {
            for(int i=data->numChirps; i<n; i++)
                data->chirpList.append(data->chirpList.first());
        }
        else
        {
            for(int i=data->numChirps; i<n; i++)
                data->chirpList.append(QList<BlackChirp::ChirpSegment>());
        }
    }
    else if(n < data->chirpList.size())
    {
        while(data->chirpList.size() > n)
            data->chirpList.removeLast();
    }

    data->numChirps = n;    
    validate();
}

void ChirpConfig::setChirpInterval(const double i)
{
    data->chirpInterval = i;
    validate();
}

void ChirpConfig::addSegment(const double startMHz, const double endMHz, const double durationUs, const int chirpNum)
{
    if(startMHz > 0.0 && endMHz > 0.0 && durationUs > 0.0)
    {
        BlackChirp::ChirpSegment seg;
        seg.startFreqMHz = startMHz;
        seg.endFreqMHz = endMHz;
        seg.durationUs = durationUs;
        seg.alphaUs = (endMHz-startMHz)/durationUs;
        seg.empty = false;

        if(chirpNum < 0 || chirpNum > data->chirpList.size())
        {
            for(int i=0; i<data->chirpList.size(); i++)
                data->chirpList[i].append(seg);
        }
        else
            data->chirpList[chirpNum].append(seg);
    }
}

void ChirpConfig::addEmptySegment(const double durationUs, const int chirpNum)
{
    if(durationUs > 0.0)
    {
        BlackChirp::ChirpSegment seg;
        seg.startFreqMHz = 0.0;
        seg.endFreqMHz = 0.0;
        seg.durationUs = durationUs;
        seg.alphaUs = 0.0;
        seg.empty = true;

        if(chirpNum < 0 || chirpNum > data->chirpList.size())
        {
            for(int i=0; i<data->chirpList.size(); i++)
                data->chirpList[i].append(seg);
        }
        else
            data->chirpList[chirpNum].append(seg);
    }
}

void ChirpConfig::setChirpList(const QList<QList<BlackChirp::ChirpSegment>> l)
{
    QList<QList<BlackChirp::ChirpSegment>> newChirpList;
    for(int j=0; j<l.size(); j++)
    {
        newChirpList.append(QList<BlackChirp::ChirpSegment>());
        for(int i=0; i<l.at(j).size(); i++)
        {
            BlackChirp::ChirpSegment seg;
            seg.durationUs = l.at(j).at(i).durationUs;
            seg.empty = l.at(j).at(i).empty;
            if(!l.at(j).at(i).empty)
            {
                seg.startFreqMHz = realToAwgFreq(l.at(j).at(i).startFreqMHz);
                seg.endFreqMHz = realToAwgFreq(l.at(j).at(i).endFreqMHz);
                seg.alphaUs = (seg.endFreqMHz - seg.startFreqMHz)/seg.durationUs;
            }
            else
            {
                seg.startFreqMHz = 0.0;
                seg.endFreqMHz = 0.0;
                seg.alphaUs = 0.0;
            }

            newChirpList[j].append(seg);
        }
    }
    data->chirpList = newChirpList;
    validate();
}

void ChirpConfig::setTxFreq(double f)
{
    data->synthTxFreq = f;
}

void ChirpConfig::setTxMult(double m)
{
    data->synthTxMult = m;
}

void ChirpConfig::setAwgMult(double m)
{
    data->awgMult = m;
}

void ChirpConfig::setTxSideband(double s)
{
    data->mixerTxSideband = s;
}

void ChirpConfig::setTotalMult(double m)
{
    data->totalMult = m;
}

void ChirpConfig::saveToSettings() const
{
    if(!isValid())
        return;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastChirpConfig"));

    s.setValue(QString("preChirpProtection"),preChirpProtection());
    s.setValue(QString("preChirpDelay"),preChirpDelay());
    s.setValue(QString("postChirpProtection"),postChirpProtection());
    s.setValue(QString("numChirps"),numChirps());
    s.setValue(QString("chirpInterval"),chirpInterval());

    s.setValue(QString("txFreq"),synthTxFreq());
    s.setValue(QString("txMult"),synthTxMult());
    s.setValue(QString("awgMult"),awgMult());
    s.setValue(QString("txSideband"),mixerSideband());
    s.setValue(QString("txTotalMult"),totalMult());

    //if all chirps are identical, only write segments list
    if(!allChirpsIdentical())
    {
        s.beginWriteArray(QString("chirps"));
        QList<QList<BlackChirp::ChirpSegment>> cList = chirpList();
        for(int j=0; j<cList.size(); j++)
        {
            s.setArrayIndex(j);
            QList<BlackChirp::ChirpSegment> segmentList = cList.at(j);

            s.beginWriteArray(QString("segments"));
            for(int i=0; i<segmentList.size(); i++)
            {
                s.setArrayIndex(i);
                s.setValue(QString("startFreq"),segmentList.at(i).startFreqMHz);
                s.setValue(QString("endFreq"),segmentList.at(i).endFreqMHz);
                s.setValue(QString("duration"),segmentList.at(i).durationUs);
                s.setValue(QString("empty"),segmentList.at(i).empty);
            }
            s.endArray();
        }
        s.endArray();
        s.remove(QString("segments"));
    }
    else
    {
        QList<BlackChirp::ChirpSegment> segmentList = chirpList().first();

        s.beginWriteArray(QString("segments"));
        for(int i=0; i<segmentList.size(); i++)
        {
            s.setArrayIndex(i);
            s.setValue(QString("startFreq"),segmentList.at(i).startFreqMHz);
            s.setValue(QString("endFreq"),segmentList.at(i).endFreqMHz);
            s.setValue(QString("duration"),segmentList.at(i).durationUs);
            s.setValue(QString("empty"),segmentList.at(i).empty);
        }
        s.endArray();
        s.remove(QString("chirps"));
    }
    s.endGroup();
}

ChirpConfig ChirpConfig::loadFromSettings()
{
    ChirpConfig out;
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    double chirpMin = s.value(QString("rfConfig/chirpMin"),26500.0).toDouble();
    double chirpMax = s.value(QString("rfConfig/chirpMax"),40000.0).toDouble();
    s.beginGroup(QString("lastChirpConfig"));

    out.setPreChirpProtection(s.value(QString("preChirpProtection"),out.preChirpProtection()).toDouble());
    out.setPreChirpDelay(s.value(QString("preChirpDelay"),out.preChirpDelay()).toDouble());
    out.setPostChirpProtection(s.value(QString("postChirpProtection"),out.preChirpProtection()).toDouble());
    out.setNumChirps(s.value(QString("numChirps"),out.numChirps()).toInt());
    out.setChirpInterval(s.value(QString("chirpInterval"),out.chirpInterval()).toDouble());

    out.setTxFreq(s.value(QString("txFreq"),out.synthTxFreq()).toDouble());
    out.setTxMult(s.value(QString("txMult"),out.synthTxMult()).toDouble());
    out.setAwgMult(s.value(QString("awgMult"),out.awgMult()).toDouble());
    out.setTxSideband(s.value(QString("txSideband"),out.mixerSideband()).toDouble());
    out.setTotalMult(s.value(QString("txTotalMult"),out.totalMult()).toDouble());

    //if all chirps are identical, then there will only be a "segments" list
    int num = s.beginReadArray(QString("segments"));
    if(num > 0)
    {
        for(int i=0; i<num; i++)
        {
            s.setArrayIndex(i);

            double startFreqMHz = qBound(chirpMin,s.value(QString("startFreq"),-1.0).toDouble(),chirpMax);
            double endFreqMHz = qBound(chirpMin,s.value(QString("endFreq"),-1.0).toDouble(),chirpMax);
            double durationUs = qBound(0.1,s.value(QString("duration"),-1.0).toDouble(),100000.0);
            bool empty = s.value(QString("empty"),false).toBool();

            if(!empty)
                out.addSegment(startFreqMHz,endFreqMHz,durationUs);
            else
                out.addEmptySegment(durationUs);
        }
    }
    else
    {
        s.endArray();
        int numChirps = s.beginReadArray(QString("chirps"));
        for(int j=0; j<numChirps; j++)
        {
            s.setArrayIndex(j);
            int numSegments = s.beginReadArray(QString("segments"));
            for(int i=0; i<numSegments; i++)
            {
                s.setArrayIndex(i);

                double startFreqMHz = s.value(QString("startFreq"),-1.0).toDouble();
                double endFreqMHz = s.value(QString("endFreq"),-1.0).toDouble();
                double durationUs = s.value(QString("duration"),-1.0).toDouble();
                bool empty = s.value(QString("empty"),false).toBool();

                if(!empty)
                    out.addSegment(startFreqMHz,endFreqMHz,durationUs,j);
                else
                    out.addEmptySegment(durationUs,j);
            }
            s.endArray();
        }
    }
    s.endArray();
    s.endGroup();

    out.validate();
    return out;
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
    if(segment.empty)
        return 0.0;

    return gsl_sf_sin(gsl_sf_angle_restrict_pos(2.0*M_PI*(segment.startFreqMHz + 0.5*segment.alphaUs*t)*t + phase));
}

double ChirpConfig::calculateEndingPhaseRadians(const BlackChirp::ChirpSegment segment, const double endingTime, const double startingPhase) const
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

double ChirpConfig::realToAwgFreq(const double realFreq) const
{
    return data->mixerTxSideband*(realFreq/data->totalMult - data->synthTxMult*data->synthTxFreq)/data->awgMult;
}

double ChirpConfig::awgToRealFreq(const double awgFreq) const
{
    return data->totalMult*(data->mixerTxSideband*data->awgMult*awgFreq + data->synthTxFreq*data->synthTxMult);
}

