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

void RfConfig::saveToSetting() const
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("lastRfConfig"));


    s.endGroup();
}

RfConfig RfConfig::loadFromSettings()
{
    RfConfig out;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(QString("lastRfConfig"));


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

void RfConfig::setClockFreq(BlackChirp::ClockType t, double targetFreqMHz, double factor, RfConfig::MultOperation o)
{
    ClockFreq f;
    f.desiredFreqMHz = targetFreqMHz;
    f.factor = factor;
    f.op = o;

    data->clocks.insert(t,f);
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

ChirpConfig RfConfig::getChirpConfig(int num)
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

