#include "ftmwconfig.h"

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

FtmwConfig::ScopeConfig FtmwConfig::scopeConfig() const
{
    return data->scopeConfig;
}

Fid FtmwConfig::fidTemplate() const
{
    return data->fidTemplate;
}

int FtmwConfig::numFrames() const
{
    return scopeConfig().summaryFrame ? 1 : scopeConfig().numFrames;
}

QVector<qint64> FtmwConfig::parseWaveform(QByteArray b) const
{

    QVector<qint64> out(numFrames()*scopeConfig().recordLength);
    //read raw data into vector in 64 bit integer form
    for(int i=0;i<numFrames()*scopeConfig().recordLength;i++)
    {
        if(scopeConfig().bytesPerPoint == 1)
        {
            char y = b.at(i);
            out[i] = (static_cast<qint64>(y) + static_cast<qint64>(scopeConfig().yOff));
        }
        else
        {
            char y1 = b.at(2*i);
            char y2 = b.at(2*i + 1);
            qint16 y = 0;
            if(scopeConfig().byteOrder == QDataStream::LittleEndian)
            {
                y += (qint8)y1;
                y += 256*(qint8)y2;
            }
            else
            {
                y += (qint8)y2;
                y += 256*(qint8)y1;
            }
            out[i] = (static_cast<qint64>(y) + static_cast<qint64>(scopeConfig().yOff));
        }
    }

    return out;
}

QString FtmwConfig::errorString() const
{
    return data->errorString;
}

bool FtmwConfig::prepareForAcquisition()
{
    Fid f(scopeConfig().xIncr,loFreq(),QVector<qint64>(0),sideband(),scopeConfig().yMult,1);
    data->fidTemplate = f;

#ifdef BC_CUDA

#endif
    return true;


}

void FtmwConfig::setEnabled()
{
    data->isEnabled = true;
}

void FtmwConfig::setFidTemplate(const Fid f)
{
    data->fidTemplate = f;
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

void FtmwConfig::setFids(const QByteArray newData)
{
#ifndef BC_CUDA
    data->rawData = parseWaveform(newData);
#else

#endif

    if(!data->fidList.isEmpty())
        data->fidList.clear();

    for(int i=0; i<numFrames(); i++)
    {
        Fid f = fidTemplate();
        f.setData(data->rawData.mid(i*scopeConfig().recordLength,scopeConfig().recordLength));
        data->fidList.append(f);
    }

}

void FtmwConfig::addFids(const QByteArray rawData)
{
#ifndef BC_CUDA
    QVector<qint64> newData = parseWaveform(rawData);
    Q_ASSERT(data->rawData.size() == newData.size());
    for(int i=0; i<data->rawData.size(); i++)
        data->rawData[i] += newData.at(i);
#else

#endif

    const qint64 *d = data->rawData.data();
    for(int i=0; i<data->fidList.size(); i++)
    {
        data->fidList.removeFirst();
        Fid f = fidTemplate();
        f.setShots(completedShots());
        f.setData(QVector<qint64>(scopeConfig().recordLength));
        f.copyAdd(d,i*scopeConfig().recordLength);
        data->fidList.append(f);
    }
}

void FtmwConfig::setScopeConfig(const FtmwConfig::ScopeConfig &other)
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
        return QDateTime::currentDateTime() >= targetTime();
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

    FtmwConfig::ScopeConfig sc = scopeConfig();
    out.unite(sc.headerHash());

    return out;

}

