#include <data/experiment/rfconfig.h>

#include <QPair>
#include <QTextStream>
#include <QSaveFile>
#include <QFile>

#include <data/storage/settingsstorage.h>
#include <data/storage/blackchirpcsv.h>
#include <hardware/optional/chirpsource/awg.h>

RfConfig::RfConfig() : HeaderStorage(BC::Store::RFC::key), d_currentClockIndex{0}
{
    addChild(&d_chirpConfig);
}

RfConfig::~RfConfig()
{

}

bool RfConfig::prepareForAcquisition()
{
    if(d_chirpConfig.chirpList().isEmpty())
        return false;

    //LO and DR scans populate the clock config list.
    //If empty, there is only 1 config, so set it from the template
    if(d_clockConfigs.isEmpty())
        d_clockConfigs.append(d_clockTemplate);

    d_currentClockIndex = 0;
    d_completedSweeps = 0;

    return true;
}

void RfConfig::setCurrentClocks(const QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks)
{
    if(!d_clockConfigs.isEmpty())
        d_clockConfigs[d_currentClockIndex] = clocks;
    else
        d_clockTemplate = clocks;
}

void RfConfig::setClockDesiredFreq(RfConfig::ClockType t, double f)
{
    if(!d_clockConfigs.isEmpty())
        d_clockConfigs[d_currentClockIndex][t].desiredFreqMHz = f;
    else
        d_clockTemplate[t].desiredFreqMHz = f;
}

void RfConfig::setClockFreqInfo(ClockType t, const ClockFreq &cf)
{
    if(cf.hwKey.isEmpty())
    {
        d_clockTemplate.remove(t);
        return;
    }
    if(d_commonUpDownLO && t == UpLO)
        d_clockTemplate.insert(DownLO,cf);
    if(d_commonUpDownLO && t == DownLO)
        d_clockTemplate.insert(UpLO,cf);
    d_clockTemplate.insert(t,cf);
}

void RfConfig::addClockStep(QHash<ClockType, RfConfig::ClockFreq> h)
{
    d_clockConfigs.append(h);
}

void RfConfig::addLoScanClockStep(double upLoMHz, double downLoMHz)
{
    //make a copy of the clock template
    auto c{d_clockTemplate};

    c[UpLO].desiredFreqMHz = upLoMHz;
    c[DownLO].desiredFreqMHz = downLoMHz;
    d_clockConfigs.append(c);
}

void RfConfig::addDrScanClockStep(double drFreqMHz)
{
    //make a copy of the clock template
    auto c{d_clockTemplate};

    c[DRClock].desiredFreqMHz = drFreqMHz;
    d_clockConfigs.append(c);

}

void RfConfig::clearClockSteps()
{
    d_currentClockIndex = 0;
    d_clockConfigs.clear();
}

void RfConfig::setChirpConfig(const ChirpConfig &cc)
{
    d_chirpConfig = cc;
}

int RfConfig::advanceClockStep()
{
    d_currentClockIndex++;
    if(d_currentClockIndex >= d_clockConfigs.size())
    {
        d_currentClockIndex = 0;
        d_completedSweeps++;
    }

    return d_currentClockIndex;
}

int RfConfig::completedSweeps() const
{
    return d_completedSweeps;
}

quint64 RfConfig::totalShots() const
{
    return static_cast<quint64>(d_shotsPerClockConfig)
            *static_cast<quint64>(d_clockConfigs.size())
            *static_cast<quint64>(d_targetSweeps);
}

quint64 RfConfig::completedSegmentShots() const
{
    quint64 completedSweepShots = static_cast<quint64>(d_completedSweeps)
            *static_cast<quint64>(d_shotsPerClockConfig)
            *static_cast<quint64>(d_clockConfigs.size());

    return completedSweepShots +
            static_cast<quint64>(d_shotsPerClockConfig)
            *static_cast<quint64>(d_currentClockIndex);
}

bool RfConfig::canAdvance(qint64 shots) const
{
    qint64 target = static_cast<qint64>(d_completedSweeps+1)*static_cast<qint64>(d_shotsPerClockConfig);

    //don't return true if this is the last segment! 5/18/21: why? doing this prevents last segment from being stored
//    if(currentClockIndex + 1 == clockConfigList.size()
//            && completedSweeps + 1 == targetSweeps)
//        return false;

    return shots >= target;
}

int RfConfig::numSegments() const
{
    if(d_clockConfigs.isEmpty())
        return 1;

    return d_clockConfigs.size();
}

QVector<QHash<RfConfig::ClockType, RfConfig::ClockFreq> > RfConfig::clockSteps() const
{
    if(d_clockConfigs.isEmpty())
        return {d_clockTemplate};

    return d_clockConfigs;
}

QHash<RfConfig::ClockType, RfConfig::ClockFreq> RfConfig::getClocks() const
{
    if(d_currentClockIndex >=0 && d_currentClockIndex < d_clockConfigs.size())
        return d_clockConfigs.at(d_currentClockIndex);

    return d_clockTemplate;
}

double RfConfig::clockFrequency(ClockType t) const
{
    return d_clockTemplate.value(t).desiredFreqMHz;
}

double RfConfig::rawClockFrequency(ClockType t) const
{
    return getRawFrequency(d_clockTemplate.value(t));
}

QString RfConfig::clockHardware(ClockType t) const
{
    return d_clockTemplate.value(t).hwKey;
}

bool RfConfig::isComplete() const
{
    return d_completedSweeps >= d_targetSweeps;
}

double RfConfig::calculateChirpFreq(double awgFreq) const
{
    double cf = clockFrequency(UpLO);
    double chirp = awgFreq*d_awgMult;
    if(d_upMixSideband == LowerSideband)
        chirp = cf - chirp;
    else
        chirp = cf + chirp;

    return chirp*d_chirpMult;

}

double RfConfig::calculateAwgFreq(double chirpFreq) const
{
    double cf = clockFrequency(UpLO);
    double awg = chirpFreq/d_chirpMult;
    if(d_upMixSideband == LowerSideband)
        awg = cf - awg;
    else
        awg = awg - cf;

    return awg/d_awgMult;

}

double RfConfig::calculateChirpAbsOffset(double awgFreq) const
{
    return qAbs(calculateChirpFreq(awgFreq) - clockFrequency(DownLO));

}

QPair<double, double> RfConfig::calculateChirpAbsOffsetRange() const
{
    QPair<double,double> out(-1.0,-1.0);

    int limit = d_chirpConfig.numChirps();
    if(limit > 1 && d_chirpConfig.allChirpsIdentical())
        limit = 1;

    for(int j=0; j<limit; j++)
    {
        auto cc = d_chirpConfig.chirpList().at(j);
        for(int k=0; k < cc.size(); k++)
        {
            double f1 = calculateChirpAbsOffset(cc.at(k).startFreqMHz);
            double f2 = calculateChirpAbsOffset(cc.at(k).endFreqMHz);
            if(f1 > f2)
                qSwap(f1,f2);
            if((out.first < 0.0) || (out.second < 0.0))
            {
                out.first = f1;
                out.second = f2;
            }
            else
            {
                out.first = qMin(out.first,f1);
                out.second = qMin(out.second,f2);
            }
        }
    }

    return out;
}

bool RfConfig::writeClockFile(int num) const
{
    QDir d(BlackchirpCSV::exptDir(num));
    QSaveFile f(d.absoluteFilePath(BC::CSV::clockFile));
    if(f.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        QTextStream t(&f);
        BlackchirpCSV::writeLine(t,{"Index","ClockType","FreqMHz","Operation","Factor","HwKey","OutputNum"});
        for(int i=0;i<d_clockConfigs.size(); ++i)
        {
            for(auto it=d_clockConfigs.at(i).cbegin(); it!=d_clockConfigs.at(i).cend(); ++it)
            {
                BlackchirpCSV::writeLine(t,{
                                  i,
                                  it.key(),
                                  it.value().desiredFreqMHz,
                                  it.value().op,
                                  QVariant(it.value().factor),
                                  it.value().hwKey,
                                  it.value().output
                              });
            }
        }
        return f.commit();
    }

    return false;
}

void RfConfig::loadClockSteps(BlackchirpCSV *csv, int num, QString path)
{
    QDir d(BlackchirpCSV::exptDir(num,path));
    QFile f(d.absoluteFilePath(BC::CSV::clockFile));
    if(f.open(QIODevice::ReadOnly))
    {
        while(!f.atEnd())
        {
            auto l = csv->readLine(f);
            if(l.isEmpty())
                continue;

            if(l.startsWith("Index"))
                continue;

            if(l.size() == 7)
            {
                bool ok = false;
                int index = l.at(0).toInt(&ok);
                if(!ok)
                    continue;
                ClockType type = l.at(1).value<ClockType>();
                double freq = l.at(2).toDouble(&ok);
                if(!ok)
                    continue;
                MultOperation m = l.at(3).value<MultOperation>();
                double factor = l.at(4).toDouble(&ok);
                if(!ok)
                    continue;
                QString hwKey = l.at(5).toString();
                int output = l.at(6).toInt(&ok);
                if(!ok)
                    continue;

                while(d_clockConfigs.size() <= index)
                    d_clockConfigs.append(QHash<ClockType,ClockFreq>());

                d_clockConfigs[index].insert(type,{freq,m,factor,hwKey,output});
            }
        }
        if(!d_clockConfigs.isEmpty())
            d_clockTemplate = d_clockConfigs.constFirst();
    }
}

double RfConfig::getRawFrequency(RfConfig::ClockFreq f) const
{
    switch(f.op)
    {
    case Divide:
        return f.desiredFreqMHz*f.factor;
    case Multiply:
    default:
        return f.desiredFreqMHz/f.factor;
    }
}



void RfConfig::storeValues()
{
    using namespace BC::Store::RFC;
    store(commonLO,d_commonUpDownLO);
    store(targetSweeps,d_targetSweeps);
    store(shots,d_shotsPerClockConfig);
    store(awgM,d_awgMult);
    store(upSB,d_upMixSideband);
    store(chirpM,d_chirpMult);
    store(downSB,d_downMixSideband);
}

void RfConfig::retrieveValues()
{
    using namespace BC::Store::RFC;
    d_commonUpDownLO = retrieve(commonLO,false);
    d_targetSweeps = retrieve(targetSweeps,1);
    d_shotsPerClockConfig = retrieve(shots,0);
    d_awgMult = retrieve(awgM,1.0);
    d_upMixSideband = retrieve(upSB,UpperSideband);
    d_chirpMult = retrieve(chirpM,1.0);
    d_downMixSideband = retrieve(downSB,UpperSideband);
}
