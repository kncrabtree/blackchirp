#include <data/experiment/rfconfig.h>

#include <QPair>
#include <QTextStream>
#include <QSaveFile>
#include <QFile>

#include <data/storage/settingsstorage.h>
#include <data/storage/blackchirpcsv.h>
#include <hardware/core/chirpsource/awg.h>

RfConfig::RfConfig() : HeaderStorage(BC::Store::RFC::key)
{
}

RfConfig::~RfConfig()
{

}

bool RfConfig::prepareForAcquisition()
{
    if(d_chirps.isEmpty())
        return false;

    for(int i=0; i<d_chirps.size(); i++)
    {
        if(d_chirps.at(i).chirpList().isEmpty())
            return false;
    }

    //LO and DR scans populate the clock config list.
    //If empty, there is only 1 config, so set it from the template
    if(d_clockConfigs.isEmpty())
        d_clockConfigs.append(d_clockTemplate);

    d_currentClockIndex = 0;
    d_completedSweeps = 0;

    return true;
}

void RfConfig::setClockDesiredFreq(ClockType t, double targetFreqMHz)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.desiredFreqMHz = targetFreqMHz;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockFactor(ClockType t, double factor)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.factor = factor;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockOp(ClockType t, RfConfig::MultOperation o)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.op = o;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockOutputNum(ClockType t, int output)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.output = output;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockHwKey(ClockType t, QString key)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.hwKey = key;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockHwInfo(ClockType t, QString hwKey, int output)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.hwKey = hwKey;
    c.output = output;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockFreqInfo(ClockType t, double targetFreqMHz, double factor, RfConfig::MultOperation o, QString hwKey, int output)
{
    ClockFreq f;
    f.desiredFreqMHz = targetFreqMHz;
    f.factor = factor;
    f.op = o;
    f.hwKey = hwKey;
    f.output = output;

    setClockFreqInfo(t,f);
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

    //these functions will modify d_clockTemplate
    setClockDesiredFreq(UpLO,upLoMHz);
    setClockDesiredFreq(DownLO,downLoMHz);
    d_clockConfigs.append(d_clockTemplate);

    //restore template
    d_clockTemplate = c;
}

void RfConfig::addDrScanClockStep(double drFreqMHz)
{
    //make a copy of the clock template
    auto c{d_clockTemplate};

    //modify d_clockTemplate
    setClockDesiredFreq(DRClock,drFreqMHz);
    d_clockConfigs.append(d_clockTemplate);

    //restore
    d_clockTemplate = c;
}

void RfConfig::clearClockSteps()
{
    d_currentClockIndex = 0;
    d_clockConfigs.clear();
}

void RfConfig::clearChirpConfigs()
{
    d_chirps.clear();
}

bool RfConfig::setChirpConfig(const ChirpConfig cc, int num)
{
    if(d_chirps.isEmpty() && num == 0)
    {
        d_chirps.append(cc);
        return true;
    }

    if(num < d_chirps.size())
    {
        d_chirps[num] = cc;
        return true;
    }

    return false;


}

void RfConfig::addChirpConfig(ChirpConfig cc)
{
    if(d_chirps.isEmpty())
    {
        using namespace BC::Key::AWG;
        SettingsStorage s(key,SettingsStorage::Hardware);
        double sr = s.get(rate,16e9);

        if(rawClockFrequency(AwgRef) > 0.0)
            cc.setAwgSampleRate(rawClockFrequency(AwgRef)*1e6);
        else
            cc.setAwgSampleRate(sr);
    }
    d_chirps.append(cc);
}

void RfConfig::setChirpList(const QVector<QVector<ChirpConfig::ChirpSegment> > l, int num)
{
    if(num>=0 && num < d_chirps.size())
        d_chirps[num].setChirpList(l);
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

ChirpConfig RfConfig::getChirpConfig(int num) const
{
    return d_chirps.value(num,ChirpConfig());
}

int RfConfig::numChirpConfigs() const
{
    return d_chirps.size();
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

    for(int i=0; i<d_chirps.size(); i++)
    {
        auto c = d_chirps.at(i);

        int limit = c.numChirps();
        if(limit > 1 && c.allChirpsIdentical())
            limit = 1;

        for(int j=0; j<limit; j++)
        {
            auto cc = c.chirpList().at(j);
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
    }

    return out;
}

bool RfConfig::writeClockFile(int num, QString path) const
{
    QSaveFile f(BlackChirp::getExptFile(num,BlackChirp::ClockFile,path));
    if(f.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        QTextStream t(&f);
        BlackchirpCSV csv;
        csv.writeLine(t,{"Index","ClockType","FreqMHz","Operation","Factor","HwKey","OutputNum"});
        for(int i=0;i<d_clockConfigs.size(); ++i)
        {
            for(auto it=d_clockConfigs.at(i).cbegin(); it!=d_clockConfigs.at(i).cend(); ++it)
            {
                csv.writeLine(t,{
                                  i,
                                  it.key(),
                                  it.value().desiredFreqMHz,
                                  it.value().op,
                                  it.value().factor,
                                  it.value().hwKey,
                                  it.value().output
                              });
            }
        }
        return f.commit();
    }

    return false;
}

void RfConfig::loadClockSteps(int num, QString path)
{
    QFile f(BlackChirp::getExptFile(num,BlackChirp::ClockFile,path));
    if(f.open(QIODevice::ReadOnly))
    {
        while(!f.atEnd())
        {
            auto l = f.readLine();
            if(l.isEmpty() || l.startsWith("Index"))
                continue;

            auto list = QString(l).trimmed().split(',');
            if(list.size() == 7)
            {
                bool ok = false;
                int index = list.at(0).toInt(&ok);
                if(!ok)
                    continue;
                ClockType type = QVariant(list.at(1)).value<ClockType>();
                double freq = list.at(2).toDouble(&ok);
                if(!ok)
                    continue;
                MultOperation m = QVariant(list.at(3)).value<MultOperation>();
                double factor = list.at(4).toDouble(&ok);
                if(!ok)
                    continue;
                QString hwKey = list.at(5);
                int output = list.at(6).toInt(&ok);
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



void RfConfig::prepareToSave()
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

void RfConfig::loadComplete()
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
