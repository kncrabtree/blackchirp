#include "rfconfig.h"

#include <QSettings>
#include <QPair>
#include <QTextStream>
#include <QFile>

class RfConfigData : public QSharedData
{
public:
    RfConfigData() : awgMult(1.0), upMixSideband(BlackChirp::UpperSideband), chirpMult(1.0),
        downMixSideband(BlackChirp::UpperSideband), commonUpDownLO(false),
        currentClockIndex(-1), completedSweeps(-1), targetSweeps(-1), shotsPerClockConfig(-1) {}

    //Upconversion chain
    double awgMult;
    BlackChirp::Sideband upMixSideband;
    double chirpMult;

    //downconversion chain
    BlackChirp::Sideband downMixSideband;

    //Logical clocks:
    QHash<BlackChirp::ClockType,RfConfig::ClockFreq> currentClocks;

    //options
    bool commonUpDownLO;

    //multiple clock setups
    QList<QHash<BlackChirp::ClockType,RfConfig::ClockFreq>> clockConfigList;
    int currentClockIndex;
    int completedSweeps;
    int targetSweeps;
    int shotsPerClockConfig;



    //chirps
    QList<ChirpConfig> chirps;
};

RfConfig::RfConfig() : data(new RfConfigData)
{
}

RfConfig::RfConfig(const RfConfig &rhs) : data(rhs.data)
{

}

RfConfig &RfConfig::operator=(const RfConfig &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

RfConfig::~RfConfig()
{

}

void RfConfig::saveToSettings() const
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("lastRfConfig"));
    s.setValue(QString("awgMult"),awgMult());
    s.setValue(QString("upSideband"),upMixSideband());
    s.setValue(QString("chirpMult"),chirpMult());
    s.setValue(QString("downSideband"),downMixSideband());
    s.setValue(QString("commonLO"),commonLO());
    s.setValue(QString("targetSweeps"),data->targetSweeps);
    s.setValue(QString("shotsPerClockConfig"),data->shotsPerClockConfig);
    s.beginWriteArray(QString("clocks"));
    int index = 0;
    if(!data->currentClocks.isEmpty())
    {
        for(auto it=data->currentClocks.constBegin(); it != data->currentClocks.constEnd(); it++)
        {
            s.setArrayIndex(index);
            auto c = it.value();
            s.setValue(QString("type"),it.key());
            s.setValue(QString("desiredFreqMHz"),c.desiredFreqMHz);
            s.setValue(QString("factor"),c.factor);
            s.setValue(QString("op"),c.op);
            s.setValue(QString("output"),c.output);
            s.setValue(QString("hwKey"),c.hwKey);
            index++;
        }
    }
    s.endArray();
    if(!data->clockConfigList.isEmpty())
    {
        s.beginWriteArray(QString("clockSteps"));
        for(int i=0; i<data->clockConfigList.size(); i++)
        {
            s.setArrayIndex(i);
            auto c = data->clockConfigList.at(i);
            index = 0;
            s.beginWriteArray(QString("clocks"));
            for(auto it=c.constBegin(); it != c.constEnd(); it++)
            {
                s.setArrayIndex(index);
                auto c = it.value();
                s.setValue(QString("type"),it.key());
                s.setValue(QString("desiredFreqMHz"),c.desiredFreqMHz);
                s.setValue(QString("factor"),c.factor);
                s.setValue(QString("op"),c.op);
                s.setValue(QString("output"),c.output);
                s.setValue(QString("hwKey"),c.hwKey);
                index++;
            }
            s.endArray();
        }
        s.endArray();
    }

    for(int i=0; i<data->chirps.size(); i++)
        data->chirps.at(i).saveToSettings(i);

    s.endGroup();
    s.sync();
}

RfConfig RfConfig::loadFromSettings()
{
    RfConfig out;
    out.setCommonLO(false); //this will be set properly after clocks are loaded

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("lastRfConfig"));

    int num = s.beginReadArray(QString("clocks"));
    for(int i=0; i<num; i++)
    {
        s.setArrayIndex(i);
        ClockFreq cf;
        auto type = static_cast<BlackChirp::ClockType>(s.value(QString("type"),BlackChirp::UpConversionLO).toInt());
        cf.desiredFreqMHz = s.value(QString("desiredFreqMHz"),0.0).toDouble();
        cf.factor = s.value(QString("factor"),1.0).toDouble();
        cf.op = static_cast<MultOperation>(s.value(QString("op"),Multiply).toInt());
        cf.hwKey = s.value(QString("hwKey"),QString("")).toString();
        cf.output = s.value(QString("output"),0).toInt();
        out.setClockFreqInfo(type,cf);
    }
    s.endArray();

    out.setAwgMult(s.value(QString("awgMult"),1.0).toDouble());
    out.setUpMixSideband(static_cast<BlackChirp::Sideband>(s.value(QString("upSideband"),BlackChirp::UpperSideband).toInt()));
    out.setChirpMult(s.value(QString("chirpMult"),1.0).toDouble());
    out.setDownMixSideband(static_cast<BlackChirp::Sideband>(s.value(QString("downSideband"),BlackChirp::UpperSideband).toInt()));
    out.setCommonLO(s.value(QString("commonLO"),false).toBool());
    out.setTargetSweeps(s.value(QString("targetSweeps"),-1).toInt());
    out.setShotsPerClockStep(s.value(QString("shotsPerClockConfig"),-1).toInt());

    int num2 = s.beginReadArray(QString("clockSteps"));
    for(int j=0; j<num2; j++)
    {
        s.setArrayIndex(j);
        num = s.beginReadArray(QString("clocks"));
        QHash<BlackChirp::ClockType,ClockFreq> h;
        for(int i=0; i<num; i++)
        {
            s.setArrayIndex(i);
            ClockFreq cf;
            auto type = static_cast<BlackChirp::ClockType>(s.value(QString("type"),BlackChirp::UpConversionLO).toInt());
            cf.desiredFreqMHz = s.value(QString("desiredFreqMHz"),0.0).toDouble();
            cf.factor = s.value(QString("factor"),1.0).toDouble();
            cf.op = static_cast<MultOperation>(s.value(QString("op"),Multiply).toInt());
            cf.hwKey = s.value(QString("hwKey"),QString("")).toString();
            cf.output = s.value(QString("output"),0).toInt();
            h.insert(type,cf);
        }
        s.endArray();
        out.addClockStep(h);
    }
    s.endArray();


    num = s.beginReadArray(QString("chirpConfigs"));
    s.endArray();
    for(int i=0; i<num; i++)
    {
        auto cc = ChirpConfig::loadFromSettings(i);
        out.addChirpConfig(cc);
    }

    s.endGroup();

    return out;
}

QMap<QString, QPair<QVariant, QString> > RfConfig::headerMap() const
{
    QMap<QString, QPair<QVariant, QString> > out;

    QString prefix = QString("RfConfig");
    QString empty = QString("");
    QString upper = QString("Upper");
    QString lower = QString("Lower");
    QString m = QString("Multiply");
    QString d = QString("Divide");

    out.insert(prefix+QString("AwgMult"),qMakePair(awgMult(),empty));
    out.insert(prefix+QString("UpMixSideband"),qMakePair(upMixSideband() == BlackChirp::UpperSideband ? upper : lower,empty));
    out.insert(prefix+QString("ChirpMult"),qMakePair(chirpMult(),empty));
    out.insert(prefix+QString("DownMixSideband"),qMakePair(downMixSideband() == BlackChirp::UpperSideband ? upper : lower,empty));
    out.insert(prefix+QString("CommonLO"),qMakePair(commonLO(),empty));
    out.insert(prefix+QString("TargetSweeps"),qMakePair(targetSweeps(),empty));
    out.insert(prefix+QString("ShotsPerClockStep"),qMakePair(shotsPerClockStep(),empty));
    auto l = getClocks();
    if(!l.isEmpty())
    {
        for(auto it = l.constBegin(); it != l.constEnd(); it++)
        {
            QString p2 = prefix+QString("Clock.")+BlackChirp::clockKey(it.key())+QString(".");
            ClockFreq c = it.value();
            out.insert(p2+QString("Frequency"),qMakePair(c.desiredFreqMHz,QString("MHz")));
            out.insert(p2+QString("Factor"),qMakePair(c.factor,empty));
            out.insert(p2+QString("Op"),qMakePair(c.op == Multiply ? m : d,empty));
            out.insert(p2+QString("Output"),qMakePair(c.output,empty));
            out.insert(p2+QString("HwKey"),qMakePair(c.hwKey,empty));
        }
    }
    ///TODO: Handle multiple chirpconfigs?
    out.unite(getChirpConfig().headerMap());
    return out;
}

void RfConfig::parseLine(const QString key, const QVariant val)
{
    if(key.endsWith(QString("AwgMult")))
        data->awgMult = val.toDouble();
    if(key.endsWith(QString("UpMixSideband")))
        data->upMixSideband = val.toString().startsWith(QString("Upper")) ? BlackChirp::UpperSideband : BlackChirp::LowerSideband;
    if(key.endsWith(QString("ChirpMult")))
        data->chirpMult = val.toDouble();
    if(key.endsWith(QString("DownMixSideband")))
        data->downMixSideband = val.toString().startsWith(QString("Upper")) ? BlackChirp::UpperSideband : BlackChirp::LowerSideband;
    if(key.endsWith(QString("CommonLO")))
        data->commonUpDownLO = val.toBool();
    if(key.endsWith(QString("TargetSweeps")))
        data->targetSweeps = val.toInt();
    if(key.endsWith(QString("CompletedSweeps")))
        data->completedSweeps = val.toInt();
    if(key.endsWith(QString("ShotsPerClockStep")))
        data->shotsPerClockConfig = val.toInt();
    if(key.contains("Clock."))
    {
        auto l = key.split(QString("."));
        if(l.size() < 3)
            return;
        auto type = BlackChirp::clockType(l.at(1));
        auto subkey = l.at(2);
        if(subkey.startsWith(QString("Frequency")))
            setClockDesiredFreq(type,val.toDouble());
        if(subkey.startsWith(QString("Factor")))
            setClockFactor(type,val.toDouble());
        if(subkey.startsWith(QString("Op")))
            setClockOp(type,val.toString().startsWith(QString("Multiply")) ? Multiply : Divide);
        if(subkey.startsWith(QString("Output")))
            setClockOutputNum(type,val.toInt());
        if(subkey.startsWith(QString("HwKey")))
            setClockHwKey(type,val.toString());
    }
}

bool RfConfig::prepareForAcquisition(BlackChirp::FtmwType t)
{
    if(data->chirps.isEmpty())
        return false;

    for(int i=0; i<data->chirps.size(); i++)
    {
        if(data->chirps.at(i).chirpList().isEmpty())
            return false;
    }

    if((t != BlackChirp::FtmwLoScan) && (t != BlackChirp::FtmwDrScan))
        data->clockConfigList.clear();

    if(!data->clockConfigList.isEmpty())
        data->currentClocks = data->clockConfigList.constFirst();

    if(data->currentClocks.isEmpty())
        return false;


    data->currentClockIndex = 0;
    data->completedSweeps = 0;

    return true;
}

void RfConfig::setAwgMult(const double m)
{
    data->awgMult = m;
}

void RfConfig::setUpMixSideband(const BlackChirp::Sideband s)
{
    data->upMixSideband = s;
}

void RfConfig::setChirpMult(const double m)
{
    data->chirpMult = m;
}

void RfConfig::setDownMixSideband(const BlackChirp::Sideband s)
{
    data->downMixSideband = s;
}

void RfConfig::setCommonLO(bool b)
{
    data->commonUpDownLO = b;
}

void RfConfig::setShotsPerClockStep(int s)
{
    data->shotsPerClockConfig = s;
}

void RfConfig::setTargetSweeps(int s)
{
    data->targetSweeps = s;
}

void RfConfig::setClockDesiredFreq(BlackChirp::ClockType t, double targetFreqMHz)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.desiredFreqMHz = targetFreqMHz;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockFactor(BlackChirp::ClockType t, double factor)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.factor = factor;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockOp(BlackChirp::ClockType t, RfConfig::MultOperation o)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.op = o;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockOutputNum(BlackChirp::ClockType t, int output)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.output = output;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockHwKey(BlackChirp::ClockType t, QString key)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.hwKey = key;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockHwInfo(BlackChirp::ClockType t, QString hwKey, int output)
{
    if(!getClocks().contains(t))
        setClockFreqInfo(t);

    auto c = getClocks().value(t);
    c.hwKey = hwKey;
    c.output = output;
    setClockFreqInfo(t,c);
}

void RfConfig::setClockFreqInfo(BlackChirp::ClockType t, double targetFreqMHz, double factor, RfConfig::MultOperation o, QString hwKey, int output)
{
    ClockFreq f;
    f.desiredFreqMHz = targetFreqMHz;
    f.factor = factor;
    f.op = o;
    f.hwKey = hwKey;
    f.output = output;

    setClockFreqInfo(t,f);
}

void RfConfig::setClockFreqInfo(BlackChirp::ClockType t, const ClockFreq &cf)
{
    if(cf.hwKey.isEmpty())
    {
        data->currentClocks.remove(t);
        return;
    }
    if(commonLO() && t == BlackChirp::UpConversionLO)
        data->currentClocks.insert(BlackChirp::DownConversionLO,cf);
    if(commonLO() && t == BlackChirp::DownConversionLO)
        data->currentClocks.insert(BlackChirp::UpConversionLO,cf);
    data->currentClocks.insert(t,cf);
}

void RfConfig::addClockStep(QHash<BlackChirp::ClockType, RfConfig::ClockFreq> h)
{
    data->clockConfigList.append(h);
}

void RfConfig::addLoScanClockStep(double upLoMHz, double downLoMHz)
{
    setClockDesiredFreq(BlackChirp::UpConversionLO,upLoMHz);
    setClockDesiredFreq(BlackChirp::DownConversionLO,downLoMHz);
    data->clockConfigList.append(data->currentClocks);
    data->currentClocks = data->clockConfigList.constFirst();
}

void RfConfig::addDrScanClockStep(double drFreqMHz)
{
    setClockDesiredFreq(BlackChirp::DRClock,drFreqMHz);
    data->clockConfigList.append(data->currentClocks);
    data->currentClocks = data->clockConfigList.constFirst();
}

void RfConfig::clearClockSteps()
{
    data->clockConfigList.clear();
}

void RfConfig::clearChirpConfigs()
{
    data->chirps.clear();
}

bool RfConfig::setChirpConfig(const ChirpConfig cc, int num)
{
    if(data->chirps.isEmpty() && num == 0)
    {
        data->chirps.append(cc);
        return true;
    }

    if(num < data->chirps.size())
    {
        data->chirps[num] = cc;
        return true;
    }

    return false;


}

void RfConfig::addChirpConfig(ChirpConfig cc)
{
    if(data->chirps.isEmpty())
    {
        QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

        s.beginGroup(QString("awg"));
        s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
        double sampleRate = s.value(QString("sampleRate"),1e9).toDouble();
        s.endGroup();
        s.endGroup();

        if(rawClockFrequency(BlackChirp::AwgClock) > 0.0)
            cc.setAwgSampleRate(rawClockFrequency(BlackChirp::AwgClock)*1e6);
        else
            cc.setAwgSampleRate(sampleRate);
    }
    data->chirps.append(cc);
}

int RfConfig::advanceClockStep()
{
    data->currentClockIndex++;
    if(data->currentClockIndex >= data->clockConfigList.size())
    {
        data->currentClockIndex = 0;
        data->completedSweeps++;
    }
    data->currentClocks = data->clockConfigList.at(data->currentClockIndex);

    return data->currentClockIndex;
}

double RfConfig::awgMult() const
{
    return data->awgMult;
}

BlackChirp::Sideband RfConfig::upMixSideband() const
{
    return data->upMixSideband;
}

double RfConfig::chirpMult() const
{
    return data->chirpMult;
}

BlackChirp::Sideband RfConfig::downMixSideband() const
{
    return data->downMixSideband;
}

bool RfConfig::commonLO() const
{
    return data->commonUpDownLO;
}

int RfConfig::targetSweeps() const
{
    return data->targetSweeps;
}

int RfConfig::shotsPerClockStep() const
{
    return data->shotsPerClockConfig;
}

int RfConfig::currentIndex() const
{
    return data->currentClockIndex;
}

int RfConfig::completedSweeps() const
{
    return data->completedSweeps;
}

qint64 RfConfig::totalShots() const
{
    return static_cast<qint64>(data->shotsPerClockConfig)
            *static_cast<qint64>(data->clockConfigList.size())
            *static_cast<qint64>(data->targetSweeps);
}

qint64 RfConfig::completedSegmentShots() const
{
    qint64 completedSweepShots = static_cast<qint64>(data->completedSweeps)
            *static_cast<qint64>(data->shotsPerClockConfig)
            *static_cast<qint64>(data->clockConfigList.size());

    return completedSweepShots +
            static_cast<qint64>(data->shotsPerClockConfig)
            *static_cast<qint64>(data->currentClockIndex);
}

bool RfConfig::canAdvance(qint64 shots) const
{
    qint64 target = static_cast<qint64>(data->completedSweeps+1)*static_cast<qint64>(data->shotsPerClockConfig);

    //don't return true if this is the last segment!
    if(data->currentClockIndex + 1 == data->clockConfigList.size()
            && data->completedSweeps + 1 == data->targetSweeps)
        return false;

    return shots >= target;
}

int RfConfig::numSegments() const
{
    if(data->clockConfigList.isEmpty())
        return 1;

    return data->clockConfigList.size();
}

QHash<BlackChirp::ClockType, RfConfig::ClockFreq> RfConfig::getClocks() const
{
    return data->currentClocks;
}

double RfConfig::clockFrequency(BlackChirp::ClockType t) const
{
    if(data->currentClocks.contains(t))
        return data->currentClocks.value(t).desiredFreqMHz;
    else
        return -1.0;
}

double RfConfig::rawClockFrequency(BlackChirp::ClockType t) const
{
    if(data->currentClocks.contains(t))
        return getRawFrequency(data->currentClocks.value(t));
    else
        return -1.0;
}

QString RfConfig::clockHardware(BlackChirp::ClockType t) const
{
    if(data->currentClocks.contains(t))
        return data->currentClocks.value(t).hwKey;
    else
        return QString("");
}

ChirpConfig RfConfig::getChirpConfig(int num) const
{
    if(num < data->chirps.size())
        return data->chirps.at(num);

    return ChirpConfig();
}

int RfConfig::numChirpConfigs() const
{
    return data->chirps.size();
}

bool RfConfig::isComplete() const
{
    return data->completedSweeps >= data->targetSweeps;
}

double RfConfig::calculateChirpFreq(double awgFreq) const
{
    double cf = clockFrequency(BlackChirp::UpConversionLO);
    double chirp = awgFreq*awgMult();
    if(upMixSideband() == BlackChirp::LowerSideband)
        chirp = cf - chirp;
    else
        chirp = cf + chirp;

    return chirp*chirpMult();

}

double RfConfig::calculateAwgFreq(double chirpFreq) const
{
    double cf = clockFrequency(BlackChirp::UpConversionLO);
    double awg = chirpFreq/chirpMult();
    if(upMixSideband() == BlackChirp::LowerSideband)
        awg = cf - awg;
    else
        awg = awg - cf;

    return awg/awgMult();

}

double RfConfig::calculateChirpAbsOffset(double awgFreq) const
{
    return qAbs(calculateChirpFreq(awgFreq) - clockFrequency(BlackChirp::DownConversionLO));

}

QPair<double, double> RfConfig::calculateChirpAbsOffsetRange() const
{
    QPair<double,double> out(-1.0,-1.0);

    for(int i=0; i<data->chirps.size(); i++)
    {
        auto c = data->chirps.at(i);

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

QString RfConfig::clockStepsString() const
{
    QString o;
    QTextStream out(&o);
    QString nl("\n");
    QString tab("\t");

    out << QString("#The blank line is important! Do not remove it.") << nl;

    for(int i=0; i<data->clockConfigList.size(); i++)
    {
        out << nl;
        auto d = data->clockConfigList.at(i);
        for(auto it = d.constBegin(); it != d.constEnd(); it++)
        {
            out << nl;
            out << static_cast<int>(it.key()) << tab;
            out << it.value().hwKey << tab;
            out << it.value().output << tab;
            out << static_cast<int>(it.value().op) << tab;
            out << it.value().factor << tab;
            out << QString::number(it.value().desiredFreqMHz,'f',6);
        }
    }

    out.flush();
    return o;
}

void RfConfig::loadClockSteps(int num, QString path)
{
    QFile f(BlackChirp::getExptFile(num,BlackChirp::ClockFile,path));
    if(!f.open(QIODevice::ReadOnly))
        return;

    QHash<BlackChirp::ClockType,ClockFreq> thisHash;

    while(!f.atEnd())
    {
        QString line = f.readLine().trimmed();
        if(line.startsWith(QString("#")))
            continue;

        if(line.isEmpty()) //start a new QHash
        {
            if(!thisHash.isEmpty())
            {
                data->clockConfigList.append(thisHash);
                thisHash.clear();
                continue;
            }
            
            continue;
        }

        QStringList l = line.split(QString("\t"));

        //each line should have 6 fields
        if(l.size() < 6)
            continue;

        auto key = static_cast<BlackChirp::ClockType>(l.at(0).trimmed().toInt());
        auto hwKey = l.at(1).trimmed();
        auto output = l.at(2).trimmed().toInt();
        auto op = static_cast<MultOperation>(l.at(3).trimmed().toInt());
        auto factor = l.at(4).trimmed().toDouble();
        auto freq = l.at(5).trimmed().toDouble();

        ClockFreq cf { freq, op, factor, hwKey, output };
        thisHash.insert(key,cf);
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

