#include "rfconfig.h"

#include <QSettings>

class RfConfigData : public QSharedData
{
public:
    RfConfigData() : awgMult(1.0), upMixSideband(BlackChirp::UpperSideband), chirpMult(1.0),
        downMixSideband(BlackChirp::UpperSideband), commonUpDownLO(false) {}

    //Upconversion chain
    double awgMult;
    BlackChirp::Sideband upMixSideband;
    double chirpMult;

    //downconversion chain
    BlackChirp::Sideband downMixSideband;

    //Logical clocks:
    QMap<BlackChirp::ClockType,RfConfig::ClockFreq> clocks;

    //options
    bool commonUpDownLO;

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
    s.setValue(QString("upSideband"),static_cast<int>(upMixSideband()));
    s.setValue(QString("chirpMult"),chirpMult());
    s.setValue(QString("downSideband"),static_cast<int>(downMixSideband()));
    s.setValue(QString("commonLO"),commonLO());
    s.beginWriteArray(QString("clocks"));
    int index = 0;
    if(!data->clocks.isEmpty())
    {
        for(auto it=data->clocks.constBegin(); it != data->clocks.end(); it++)
        {
            s.setArrayIndex(index);
            auto c = it.value();
            s.setValue(QString("type"),static_cast<int>(it.key()));
            s.setValue(QString("desiredFreqMHz"),c.desiredFreqMHz);
            s.setValue(QString("factor"),c.factor);
            s.setValue(QString("op"),static_cast<int>(c.op));
            s.setValue(QString("output"),c.output);
            s.setValue(QString("hwKey"),c.hwKey);
            index++;
        }
    }
    s.endArray();
    for(int i=0; i<data->chirps.size(); i++)
        data->chirps.at(i).saveToSettings(i);

    s.endGroup();
    s.sync();
}

RfConfig RfConfig::loadFromSettings()
{
    RfConfig out;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("lastRfConfig"));
    out.setAwgMult(s.value(QString("awgMult"),1.0).toDouble());
    out.setUpMixSideband(static_cast<BlackChirp::Sideband>(s.value(QString("upSideband"),BlackChirp::UpperSideband).toInt()));
    out.setChirpMult(s.value(QString("chirpMult"),1.0).toDouble());
    out.setDownMixSideband(static_cast<BlackChirp::Sideband>(s.value(QString("downSideband"),BlackChirp::UpperSideband).toInt()));
    out.setCommonLO(s.value(QString("commonLO"),false).toBool());
    int num = s.beginReadArray(QString("clocks"));
    for(int i=0; i<num; i++)
    {
        s.setArrayIndex(i);
        ClockFreq cf;
        auto type = static_cast<BlackChirp::ClockType>(s.value(QString("type"),BlackChirp::UpConversionLO).toInt());
        cf.desiredFreqMHz = s.value(QString("desiredFreqMHz"),0.0).toDouble();
        cf.factor = s.value(QString("factor"),1.0).toDouble();
        cf.op = static_cast<RfConfig::MultOperation>(s.value(QString("op"),RfConfig::Multiply).toInt());
        cf.hwKey = s.value(QString("hwKey"),QString("")).toString();
        cf.output = s.value(QString("output"),0).toInt();
        out.setClockFreqInfo(type,cf);
    }
    s.endArray();


    num = s.beginReadArray(QString("chirpConfigs"));
    s.endArray();
    for(int i=0; i<num; i++)
        out.addChirpConfig(ChirpConfig::loadFromSettings(i));

    s.endGroup();

    return out;
}

bool RfConfig::isValid() const
{
    if(data->chirps.isEmpty())
        return false;

    for(int i=0; i<data->chirps.size(); i++)
    {
        if(data->chirps.at(i).chirpList().isEmpty())
            return false;
    }

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

void RfConfig::setClockFreqInfo(BlackChirp::ClockType t, RfConfig::ClockFreq cf)
{
    data->clocks.insert(t,cf);
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

void RfConfig::addChirpConfig(const ChirpConfig cc)
{
    data->chirps.append(cc);
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

QMap<BlackChirp::ClockType, RfConfig::ClockFreq> RfConfig::getClocks() const
{
    return data->clocks;
}

double RfConfig::clockFrequency(BlackChirp::ClockType t) const
{
    if(data->clocks.contains(t))
        return data->clocks.value(t).desiredFreqMHz;
    else
        return -1.0;
}

double RfConfig::rawClockFrequency(BlackChirp::ClockType t) const
{
    if(data->clocks.contains(t))
        return getRawFrequency(data->clocks.value(t));
    else
        return -1.0;
}

ChirpConfig RfConfig::getChirpConfig(int num) const
{
    if(num < data->chirps.size())
        return data->chirps.at(num);

    return ChirpConfig();
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

