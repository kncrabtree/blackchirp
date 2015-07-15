#include "ftmwconfig.h"

#include <QFile>

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

FtmwConfig::FtmwConfig(int num) : data(new FtmwConfigData)
{
    //eventually, do all the file parsing etc...
    //for now, just build the fid list
    QFile fid(BlackChirp::getExptFile(num,BlackChirp::FidFile));
    if(fid.open(QIODevice::ReadOnly))
    {
        QDataStream d(&fid);
        QByteArray magic;
        d >> magic;
        if(magic.startsWith("BCFID"))
        {
            if(magic.endsWith("v1.0"))
                d >> data->fidList;
        }
        fid.close();
    }
}

FtmwConfig::~FtmwConfig()
{

}

bool FtmwConfig::isEnabled() const
{
    return data->isEnabled;
}

BlackChirp::FtmwType FtmwConfig::type() const
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

BlackChirp::Sideband FtmwConfig::sideband() const
{
    return data->sideband;
}

QList<Fid> FtmwConfig::fidList() const
{
    return data->fidList;
}

BlackChirp::FtmwScopeConfig FtmwConfig::scopeConfig() const
{
    return data->scopeConfig;
}

ChirpConfig FtmwConfig::chirpConfig() const
{
    return data->chirpConfig;
}

Fid FtmwConfig::fidTemplate() const
{
    return data->fidTemplate;
}

int FtmwConfig::numFrames() const
{
    return scopeConfig().summaryFrame ? 1 : scopeConfig().numFrames;
}

QList<Fid> FtmwConfig::parseWaveform(QByteArray b) const
{

    int np = scopeConfig().recordLength;
    QList<Fid> out;
    //read raw data into vector in 64 bit integer form
    for(int j=0;j<numFrames();j++)
    {
        QVector<qint64> d(np);

        for(int i=0; i<np;i++)
        {
            if(scopeConfig().bytesPerPoint == 1)
            {
                char y = b.at(j*np+i);
                d[i] = (static_cast<qint64>(y) + static_cast<qint64>(scopeConfig().yOff));
            }
            else
            {
                unsigned char y1 = b.at(2*(j*np+i));
                unsigned char y2 = b.at(2*(j*np+i) + 1);
                qint16 y = 0;
                if(scopeConfig().byteOrder == QDataStream::LittleEndian)
                {
                    y |= static_cast<quint8>(y1);
                    y |= static_cast<quint8>(y2) << 8;
                }
                else
                {
                    y |= static_cast<quint8>(y1) << 8;
                    y |= static_cast<quint8>(y2);
                }
                d[i] = (static_cast<qint64>(y) + static_cast<qint64>(scopeConfig().yOff));
            }
        }

        Fid f = fidTemplate();
        f.setData(d);
        out.append(f);
    }

    return out;
}

QString FtmwConfig::errorString() const
{
    return data->errorString;
}

double FtmwConfig::ftMin() const
{
    double sign = 1.0;
    if(data->sideband == BlackChirp::LowerSideband)
        sign = -1.0;
    double lastFreq = data->loFreq + sign*data->scopeConfig.sampleRate/(1e6*2.0);
    return qMin(data->loFreq,lastFreq);
}

double FtmwConfig::ftMax() const
{
    double sign = 1.0;
    if(data->sideband == BlackChirp::LowerSideband)
        sign = -1.0;
    double lastFreq = data->loFreq + sign*data->scopeConfig.sampleRate/(1e6*2.0);
    return qMax(data->loFreq,lastFreq);
}

bool FtmwConfig::writeFidFile(int num, int snapNum) const
{
    QFile fid(BlackChirp::getExptFile(num,BlackChirp::FidFile,snapNum));
    if(fid.open(QIODevice::WriteOnly))
    {
        QDataStream d(&fid);
        d << Fid::magicString();
        d << data->fidList;
        fid.close();
        return true;
    }
    else
        return false;
}

bool FtmwConfig::prepareForAcquisition()
{
    Fid f(scopeConfig().xIncr,loFreq(),QVector<qint64>(0),sideband(),scopeConfig().yMult,1);
    data->fidTemplate = f;

    if(!chirpConfig().isValid())
    {
        data->errorString = QString("Invalid chirp configuration.");
        return false;
    }

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

void FtmwConfig::setType(const BlackChirp::FtmwType type)
{
    data->type = type;
}

void FtmwConfig::setTargetShots(const qint64 target)
{
    data->targetShots = target;
}

void FtmwConfig::increment()
{
    if(type() == BlackChirp::FtmwPeakUp)
        data->completedShots = qMin(completedShots()+1,targetShots());
    else
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

void FtmwConfig::setSideband(const BlackChirp::Sideband sb)
{
    data->sideband = sb;
}

bool FtmwConfig::setFidsData(const QList<QVector<qint64> > newList)
{
    if(data->fidList.isEmpty())
    {
        for(int i=0; i<newList.size(); i++)
        {
            Fid f = fidTemplate();
            f.setData(newList.at(i));
            data->fidList.append(f);
        }
    }
    else
    {
        if(newList.size() != data->fidList.size())
        {
            data->errorString = QString("Could not set new FID list data. List sizes are not equal (new = %1, current = %2)")
                    .arg(newList.size()).arg(data->fidList.size());
            return false;
        }

        for(int i=0; i<data->fidList.size(); i++)
        {
            data->fidList[i].setData(newList.at(i));
            if(type() == BlackChirp::FtmwPeakUp)
                data->fidList[i].setShots(qMin(completedShots()+1,targetShots()));
            else
                data->fidList[i].setShots(completedShots()+1);
        }
    }

    return true;
}

bool FtmwConfig::addFids(const QByteArray rawData)
{
    QList<Fid> newList = parseWaveform(rawData);
    if(data->completedShots > 0)
    {
        if(newList.size() != data->fidList.size())
        {
            data->errorString = QString("Could not set new FID list data. List sizes are not equal (new = %1, current = %2)")
                    .arg(newList.size()).arg(data->fidList.size());
            return false;
        }

        if(type() == BlackChirp::FtmwPeakUp)
        {
            for(int i=0; i<data->fidList.size(); i++)
                newList[i].rollingAverage(data->fidList.at(i),targetShots());
        }
        else
        {
            for(int i=0; i<data->fidList.size(); i++)
                newList[i] += data->fidList.at(i);
        }
    }
    data->fidList = newList;

    return true;
}

bool FtmwConfig::subtractFids(const QList<Fid> otherList)
{
    if(otherList.size() != data->fidList.size())
        return false;

    for(int i=0; i<otherList.size(); i++)
    {
        if(otherList.at(i).size() != data->fidList.size())
            return false;

        if(otherList.at(i).shots() > data->fidList.at(i).shots())
            return false;
    }

    for(int i=0; i<data->fidList.size(); i++)
        data->fidList[i] -= otherList.at(i);

    return true;
}

void FtmwConfig::resetFids()
{
    data->fidList.clear();
    data->completedShots = 0;
}

void FtmwConfig::setScopeConfig(const BlackChirp::FtmwScopeConfig &other)
{
    data->scopeConfig = other;
}

void FtmwConfig::setChirpConfig(const ChirpConfig other)
{
    data->chirpConfig = other;
}

bool FtmwConfig::isComplete() const
{
    if(!isEnabled())
        return true;

    switch(type())
    {
    case BlackChirp::FtmwTargetShots:
        return completedShots() >= targetShots();
        break;
    case BlackChirp::FtmwTargetTime:
        return QDateTime::currentDateTime() >= targetTime();
        break;
    case BlackChirp::FtmwForever:
    case BlackChirp::FtmwPeakUp:
    default:
        return false;
        break;
    }

    //not reached
    return false;
}

QMap<QString, QPair<QVariant, QString> > FtmwConfig::headerMap() const
{
    QMap<QString, QPair<QVariant, QString> > out;

    QString prefix = QString("FtmwConfig");
    QString empty = QString("");

    out.insert(prefix+QString("Enabled"),qMakePair(isEnabled(),empty));
    if(!isEnabled())
        return out;

    out.insert(prefix+QString("Type"),qMakePair((int)type(),empty));
    if(type() == BlackChirp::FtmwTargetShots)
        out.insert(prefix+QString("TargetShots"),qMakePair(targetShots(),empty));
    if(type() == BlackChirp::FtmwTargetTime)
        out.insert(prefix+QString("TargetTime"),qMakePair(targetTime(),empty));
    out.insert(prefix+QString("LoFrequency"),qMakePair(QString::number(loFreq(),'f',6),QString("MHz")));
    out.insert(prefix+QString("Sideband"),qMakePair((int)sideband(),empty));
    out.insert(prefix+QString("FidVMult"),qMakePair(QString::number(fidTemplate().vMult(),'g',12),QString("V")));

    BlackChirp::FtmwScopeConfig sc = scopeConfig();
    out.unite(sc.headerMap());
    out.unite(data->chirpConfig.headerMap());

    return out;

}

