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

double RfConfig::rawClockFrequency(BlackChirp::ClockType t) const
{
    if(!data->clocks.contains(t))
        return getRawFrequency(data->clocks.value(t));
    else
        return -1.0;
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

