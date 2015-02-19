#include "ftmwconfig.h"

class FtmwConfigData : public QSharedData
{
public:
    FtmwConfigData() : isEnabled(false), type(FtmwConfig::Forever), targetShots(-1), completedShots(0), autoSaveShots(1000), loFreq(0.0), sideband(Fid::UpperSideband) {}

    bool isEnabled;
    FtmwConfig::FtmwType type;
    qint64 targetShots;
    qint64 completedShots;
    QDateTime targetTime;
    int autoSaveShots;

    double loFreq;
    Fid::Sideband sideband;
    QList<Fid> fidList;

    Oscilloscope::ScopeConfig scopeConfig;



};

FtmwConfig::FtmwConfig() : data(new FtmwConfigData)
{

}

FtmwConfig::FtmwConfig(const FtmwConfig &rhs) : data(rhs.data)
{

}

FtmwConfig &FtmwConfig::operator=(const FtmwConfig &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

FtmwConfig::~FtmwConfig()
{

}

bool FtmwConfig::isEnabled() const
{
    return data->isEnabled;
}

FtmwConfig::FtmwType FtmwConfig::type() const
{
    return data->type;
}

qint64 FtmwConfig::targetShots() const
{
    return data->targetShots;
}

qint64 FtmwConfig::completedShots() const
{
    return data->completedShots;
}

QDateTime FtmwConfig::targetTime() const
{
    return data->targetTime;
}

int FtmwConfig::autoSaveShots() const
{
    return data->autoSaveShots;
}

double FtmwConfig::loFreq() const
{
    return data->loFreq;
}

Fid::Sideband FtmwConfig::sideband() const
{
    return data->sideband;
}

QList<Fid> FtmwConfig::fidList() const
{
    return data->fidList;
}

Oscilloscope::ScopeConfig FtmwConfig::scopeConfig() const
{
    return data->scopeConfig;
}

void FtmwConfig::setEnabled()
{
    data->isEnabled = true;
}

void FtmwConfig::setType(const FtmwType type)
{
    data->type = type;
}

void FtmwConfig::setTargetShots(const qint64 target)
{
    data->targetShots = target;
}

void FtmwConfig::increment()
{
    data->completedShots++;
}

void FtmwConfig::setTargetTime(const QDateTime time)
{
    data->targetTime = time;
}

void FtmwConfig::setAutoSaveShots(const int shots)
{
    data->autoSaveShots = shots;
}

void FtmwConfig::setLoFreq(const double f)
{
    data->loFreq = f;
}

void FtmwConfig::setSideband(const Fid::Sideband sb)
{
    data->sideband = sb;
}

void FtmwConfig::setFidList(const QList<Fid> list)
{
    data->fidList = list;
}

void FtmwConfig::setScopeConfig(const Oscilloscope::ScopeConfig &other)
{
    data->scopeConfig = other;
}

bool FtmwConfig::isComplete() const
{
    switch(type())
    {
    case TargetShots:
        return completedShots() >= targetShots();
        break;
    case TargetTime:
        return QDateTime::currentDateTime() > targetTime();
        break;
    case Forever:
    case PeakUp:
    default:
        return false;
        break;
    }

    //not reached
    return false;
}

QHash<QString, QPair<QVariant, QString> > FtmwConfig::headerHash() const
{
    QHash<QString, QPair<QVariant, QString> > out;

    QString prefix = QString("FtmwConfig");
    QString empty = QString("");

    out.insert(prefix+QString("Enabled"),qMakePair(isEnabled(),empty));
    if(!isEnabled())
        return out;

    out.insert(prefix+QString("Type"),qMakePair((int)type(),empty));
    if(type() == TargetShots)
        out.insert(prefix+QString("TargetShots"),qMakePair(targetShots(),empty));
    if(type() == TargetTime)
        out.insert(prefix+QString("TargetTime"),qMakePair(targetTime(),empty));
    out.insert(prefix+QString("LOFrequency"),qMakePair(QString::number(loFreq(),'f',6),QString("MHz")));
    out.insert(prefix+QString("Sideband"),qMakePair((int)sideband(),empty));

    Oscilloscope::ScopeConfig sc = scopeConfig();
    out.unite(sc.headerHash());

    return out;

}

